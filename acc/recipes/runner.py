"""RecipeRunner — executes recipes in a background thread.

Reports progress via RecipeJob which can be polled via API.
"""

import threading
import traceback
import uuid
from typing import Optional

from acc.recipes.base import Recipe, RecipeContext, RecipeJob


class RecipeRunner:
    """Executes a recipe in a background thread."""

    def __init__(self):
        self._current_job: Optional[RecipeJob] = None
        self._current_ctx: Optional[RecipeContext] = None
        self._thread: Optional[threading.Thread] = None

    def start(self, recipe: Recipe, api: "TrainerAPI") -> RecipeJob:  # noqa: F821
        """Start a recipe in a background thread. Returns the RecipeJob."""
        if self._current_job is not None and self._current_job.state == "running":
            raise RuntimeError("A recipe is already running. Stop it first.")

        ctx = RecipeContext(api)
        job = RecipeJob(
            id=uuid.uuid4().hex[:12],
            recipe_name=recipe.name,
        )

        self._current_job = job
        self._current_ctx = ctx

        def _run():
            try:
                recipe.run(ctx)
                job.state = "completed"
                job.current_phase = "complete"
            except Exception as e:
                job.state = "failed"
                job.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                print(f"[RecipeRunner] Recipe '{recipe.name}' failed: {e}")
            finally:
                job.phases_completed = list(ctx._phases_completed)
                job.checkpoints_created = list(ctx._checkpoints_created)

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        return job

    def stop(self) -> None:
        """Signal the current recipe to stop."""
        if self._current_ctx is not None:
            self._current_ctx.stop()
        if self._current_job is not None and self._current_job.state == "running":
            self._current_job.state = "stopped"

    def current(self) -> Optional[RecipeJob]:
        """Get current or most recent recipe job."""
        if self._current_job is None:
            return None
        # Sync phase from context
        if self._current_ctx is not None and self._current_job.state == "running":
            self._current_job.current_phase = self._current_ctx.phase
            self._current_job.phases_completed = list(self._current_ctx._phases_completed)
            self._current_job.checkpoints_created = list(self._current_ctx._checkpoints_created)
        return self._current_job

    def wait(self, timeout: float = None) -> Optional[RecipeJob]:
        """Wait for the current recipe to finish. Returns the job."""
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        return self._current_job
