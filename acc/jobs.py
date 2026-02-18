"""JobManager — tracks training runs with full loss history.

Jobs are not ephemeral — they persist with full loss history. When the UI
connects or reconnects via SSE, it can request stream(job_id, from_step=N)
to replay missed steps and then continue with live updates.
"""

import uuid
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Iterator

from acc.trainer import Trainer


@dataclass
class JobInfo:
    """A single training run with full history."""

    id: str
    state: str  # "running", "completed", "stopped", "failed"
    total_steps: int
    current_step: int = 0
    task_names: list[str] = field(default_factory=list)
    losses: list[dict] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    checkpoint_id: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize for JSON API.  Omits losses to keep response small;
        use /jobs/{id}/loss_history for chart data."""
        return {
            "id": self.id,
            "state": self.state,
            "total_steps": self.total_steps,
            "current_step": self.current_step,
            "task_names": self.task_names,
            "n_losses": len(self.losses),
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "checkpoint_id": self.checkpoint_id,
            "error": self.error,
        }


class JobManager:
    """Manages training jobs with history and SSE streaming support.

    Jobs run in a background thread so the API remains responsive.
    """

    def __init__(self):
        self._jobs: dict[str, JobInfo] = {}
        self._current_job_id: Optional[str] = None
        self._lock = threading.Lock()
        # SSE subscribers: job_id -> list of threading.Event for new data
        self._subscribers: dict[str, list[threading.Event]] = {}

    def start(
        self,
        trainer: Trainer,
        steps: int,
        checkpoint_id: Optional[str] = None,
        blocking: bool = True,
        task_weights: Optional[dict[str, float]] = None,
    ) -> JobInfo:
        """Start a new training job.

        Args:
            trainer: The Trainer instance.
            steps: Number of training steps.
            checkpoint_id: Which checkpoint this started from (for tracking).
            blocking: If True, run synchronously. If False, run in background thread.
            task_weights: Optional dict mapping task_name -> sampling weight.
                Passed through to Trainer.train().

        Returns:
            The JobInfo for this run.
        """
        job_id = uuid.uuid4().hex[:12]
        task_names = [t.name for t in trainer.tasks if t.enabled]

        job = JobInfo(
            id=job_id,
            state="running",
            total_steps=steps,
            task_names=task_names,
            checkpoint_id=checkpoint_id,
        )

        with self._lock:
            self._jobs[job_id] = job
            self._current_job_id = job_id

        def on_step(step_info: dict):
            with self._lock:
                job.losses.append(step_info)
                job.current_step = step_info["step"]
            # Notify SSE subscribers
            self._notify_subscribers(job_id)

        def run_training():
            try:
                trainer.train(steps=steps, on_step=on_step, task_weights=task_weights)
                with self._lock:
                    if job.state == "running":
                        job.state = "completed"
                    job.completed_at = datetime.now()
            except Exception as e:
                with self._lock:
                    job.state = "failed"
                    job.error = str(e)
                    job.completed_at = datetime.now()
            finally:
                with self._lock:
                    if self._current_job_id == job_id:
                        self._current_job_id = None
                self._notify_subscribers(job_id)

        if blocking:
            run_training()
        else:
            thread = threading.Thread(target=run_training, daemon=True)
            thread.start()

        return job

    def stop(self, job_id: Optional[str] = None) -> Optional[JobInfo]:
        """Stop a running job."""
        with self._lock:
            jid = job_id or self._current_job_id
            if jid is None:
                return None
            job = self._jobs.get(jid)
            if job is None:
                return None
            job.state = "stopped"
        return job

    def get(self, job_id: str) -> Optional[JobInfo]:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    def list(self) -> list[JobInfo]:
        """List all jobs, most recent first."""
        return list(reversed(self._jobs.values()))

    def current(self) -> Optional[JobInfo]:
        """Get the currently running job, if any."""
        with self._lock:
            if self._current_job_id is None:
                return None
            return self._jobs.get(self._current_job_id)

    def stream(self, job_id: str, from_step: int = 0) -> Iterator[dict]:
        """Stream job loss updates, replaying from from_step.

        Yields loss dicts. First replays any missed steps (from_step..current),
        then yields live updates as they arrive. Stops when job completes.
        """
        job = self._jobs.get(job_id)
        if job is None:
            return

        # Replay missed steps
        with self._lock:
            for loss in job.losses:
                if loss["step"] >= from_step:
                    yield loss
            last_seen = job.current_step

        # Live updates
        event = threading.Event()
        if job_id not in self._subscribers:
            self._subscribers[job_id] = []
        self._subscribers[job_id].append(event)

        try:
            while True:
                event.wait(timeout=1.0)
                event.clear()

                with self._lock:
                    new_losses = [l for l in job.losses if l["step"] > last_seen]
                    for loss in new_losses:
                        yield loss
                    if new_losses:
                        last_seen = new_losses[-1]["step"]

                    if job.state in ("completed", "stopped", "failed"):
                        # Yield any remaining losses
                        final = [l for l in job.losses if l["step"] > last_seen]
                        for loss in final:
                            yield loss
                        break
        finally:
            if job_id in self._subscribers:
                self._subscribers[job_id].remove(event)

    def _notify_subscribers(self, job_id: str):
        """Wake up all SSE subscribers for this job."""
        if job_id in self._subscribers:
            for event in self._subscribers[job_id]:
                event.set()
