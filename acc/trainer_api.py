"""Trainer HTTP API — JSON endpoints on localhost:6060.

Everything the UI can do, you can do via HTTP calls.
The trainer API is the source of truth.
"""

import asyncio
import base64
import io
import json
import threading
from datetime import datetime
from typing import Optional

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import torch

from acc.autoencoder import Autoencoder
from acc.model_output import ModelOutput
from acc.trainer import Trainer
from acc.jobs import JobManager
from acc.checkpoints import CheckpointStore
from acc.tasks.base import Task, TaskError
from acc.dataset import AccDataset
from acc.recipes.runner import RecipeRunner
from acc.recipes.registry import RecipeRegistry
from acc.tasks.registry import TaskRegistry
from acc.generators.registry import GeneratorRegistry
from acc.loss_health import compute_loss_summary


class TrainerAPI:
    """HTTP API wrapping the trainer process state.

    Holds references to the model, tasks, datasets, trainer, jobs, and checkpoints.
    All state lives here — the UI is stateless.
    """

    def __init__(self):
        self.app = FastAPI(title="ACC Trainer")

        # CORS — allow the Astro dev server (port 4321) and any origin for dev.
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self.autoencoder: Optional[Autoencoder] = None
        self.trainer: Optional[Trainer] = None
        self.jobs = JobManager()
        self.checkpoints: Optional[CheckpointStore] = None
        self.tasks: dict[str, Task] = {}
        self.datasets: dict[str, AccDataset] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.recipe_runner = RecipeRunner()
        self.recipe_registry = RecipeRegistry()
        self.recipe_registry.start_watcher()  # Hot-reload recipes on file change
        self.task_registry = TaskRegistry()
        self.task_registry.start_watcher()  # Hot-reload tasks on file change
        self.generator_registry = GeneratorRegistry()
        self.generator_registry.start_watcher()  # Hot-reload generators on file change

        self._register_routes()

    def _is_model_busy(self) -> bool:
        """Check if ANY background thread is using the model/CUDA.

        This covers two cases:
        1. A training job is running (jobs.current() is not None)
        2. A recipe is running — even between training jobs, the recipe
           thread does CUDA work (evaluate, checkpoint save/load,
           model.to(device), etc.)
        """
        if self.jobs.current() is not None:
            return True
        rj = self.recipe_runner.current()
        if rj is not None and rj.state == "running":
            return True
        return False

    def _training_guard(self) -> Optional[JSONResponse]:
        """Return a 409 response if model/CUDA is busy, None otherwise.

        CUDA is not thread-safe — running model forward passes from the
        API thread while a background thread (training or recipe) is also
        doing CUDA operations corrupts the CUDA context and kills the process.

        The guard covers:
        - Active training jobs (forward/backward on training thread)
        - Active recipes (evaluate, checkpoint, model creation between jobs)
        """
        if self._is_model_busy():
            return JSONResponse(
                {"error": "Model is busy (training or recipe running). Eval endpoints are disabled."},
                status_code=409,
            )
        return None

    def _register_routes(self):
        app = self.app

        # -- Model --
        @app.get("/model/describe")
        async def model_describe():
            if self.autoencoder is None:
                return JSONResponse({"error": "No model loaded"}, status_code=404)
            desc = str(self.autoencoder)
            # Polymorphic layer counting: Autoencoder has .encoder/.decoder,
            # FactorSlotAutoencoder has .backbone/.decoder_stages
            model = self.autoencoder
            num_enc = (
                len(model.encoder)
                if hasattr(model, "encoder")
                else len(model.backbone) // 2  # Conv2d + ResBlock pairs
                if hasattr(model, "backbone")
                else 0
            )
            num_dec = 0
            if model.has_decoder:
                num_dec = (
                    len(model.decoder)
                    if hasattr(model, "decoder") and model.decoder is not None
                    else len(model.decoder_stages)
                    if hasattr(model, "decoder_stages")
                    else 0
                )
            return {
                "description": desc,
                "has_decoder": model.has_decoder,
                "latent_dim": model.latent_dim,
                "num_encoder_layers": num_enc,
                "num_decoder_layers": num_dec,
                "capabilities": {
                    "eval": True,
                    "reconstructions": model.has_decoder,
                    "traversals": hasattr(model, "factor_groups") and model.has_decoder,
                    "sort_by_factor": hasattr(model, "factor_groups"),
                    "attention_maps": hasattr(model, "cross_attn_stages"),
                },
            }

        @app.post("/model/create")
        async def model_create():
            return JSONResponse(
                {"error": "Model creation via API not yet implemented. Use Python code."},
                status_code=501,
            )

        # -- Datasets --
        @app.get("/datasets")
        async def list_datasets():
            return [ds.describe() for ds in self.datasets.values()]

        @app.get("/datasets/{name}/sample")
        async def dataset_sample(name: str, n: int = 8):
            ds = self.datasets.get(name)
            if ds is None:
                return JSONResponse({"error": f"Dataset '{name}' not found"}, status_code=404)
            samples = ds.sample(n)
            images_b64 = []
            for i in range(samples.shape[0]):
                img = samples[i]
                images_b64.append(_tensor_to_base64(img))
            return {"images": images_b64}

        @app.post("/datasets/load_builtin")
        async def load_builtin_dataset(request: Request):
            data = await request.json()
            name = data.get("name", "mnist")
            image_size = data.get("image_size", 64)
            if name == "mnist":
                from acc.dataset import load_mnist

                ds = load_mnist(image_size=image_size)
                self.datasets[ds.name] = ds
                return ds.describe()
            return JSONResponse({"error": f"Unknown builtin dataset: {name}"}, status_code=400)

        # -- Tasks --
        @app.get("/tasks")
        async def list_tasks():
            result = []
            for task in self.tasks.values():
                info = task.describe()
                result.append(info)
            return result

        @app.post("/tasks/add")
        async def add_task(request: Request):
            guard = self._training_guard()
            if guard:
                return guard
            if self.autoencoder is None:
                return JSONResponse({"error": "No model loaded"}, status_code=400)

            data = await request.json()
            class_name = data.get("class_name")
            task_name = data.get("name")
            dataset_name = data.get("dataset_name")
            weight = data.get("weight", 1.0)

            dataset = self.datasets.get(dataset_name)
            if dataset is None:
                return JSONResponse({"error": f"Dataset '{dataset_name}' not found"}, status_code=400)

            task_class = self.task_registry.get(class_name)
            if task_class is None:
                return JSONResponse({"error": f"Unknown task class: {class_name}"}, status_code=400)

            # Parse optional latent_slice (e.g. "0:4" -> (0, 4))
            latent_slice = None
            slice_str = data.get("latent_slice")
            if slice_str and ":" in str(slice_str):
                parts = str(slice_str).split(":")
                latent_slice = (int(parts[0]), int(parts[1]))

            try:
                task = task_class(task_name, dataset, weight=weight, latent_slice=latent_slice)
                task.attach(self.autoencoder)
            except TaskError as e:
                return JSONResponse({"error": str(e)}, status_code=400)

            self.tasks[task_name] = task
            self._rebuild_trainer()
            return task.describe()

        @app.post("/tasks/{name}/toggle")
        async def toggle_task(name: str):
            task = self.tasks.get(name)
            if task is None:
                return JSONResponse({"error": f"Task '{name}' not found"}, status_code=404)
            task.enabled = not task.enabled
            return {"name": name, "enabled": task.enabled}

        @app.post("/tasks/{name}/set_weight")
        async def set_task_weight(name: str, request: Request):
            task = self.tasks.get(name)
            if task is None:
                return JSONResponse({"error": f"Task '{name}' not found"}, status_code=404)
            data = await request.json()
            weight = data.get("weight")
            if weight is None:
                return JSONResponse({"error": "Missing 'weight' field"}, status_code=400)
            task.weight = float(weight)
            return task.describe()

        @app.post("/tasks/{name}/remove")
        async def remove_task(name: str):
            guard = self._training_guard()
            if guard:
                return guard
            task = self.tasks.pop(name, None)
            if task is None:
                return JSONResponse({"error": f"Task '{name}' not found"}, status_code=404)
            self._rebuild_trainer()
            return {"removed": name}

        # -- Training / Jobs --
        @app.post("/train/start")
        async def train_start(request: Request):
            guard = self._training_guard()
            if guard:
                return guard
            if self.trainer is None:
                return JSONResponse(
                    {"error": "No trainer configured. Add model and tasks first."},
                    status_code=400,
                )

            data = await request.json() if await request.body() else {}
            steps = data.get("steps", 500)
            lr = data.get("lr", self.trainer.lr)
            probe_lr = data.get("probe_lr", self.trainer.probe_lr)

            if lr != self.trainer.lr or probe_lr != self.trainer.probe_lr:
                self.trainer.lr = lr
                self.trainer.probe_lr = probe_lr
                self.trainer._build_optimizers()

            job = self.jobs.start(self.trainer, steps=steps, blocking=False)
            return job.to_dict()

        @app.post("/train/stop")
        async def train_stop():
            if self.trainer is None:
                return JSONResponse({"error": "No trainer"}, status_code=400)
            self.trainer.stop()
            job = self.jobs.current()
            if job is not None:
                job.state = "stopped"
                return job.to_dict()
            return {"status": "no running job"}

        # -- Jobs --
        @app.get("/jobs")
        async def list_jobs():
            return [j.to_dict() for j in self.jobs.list()]

        @app.get("/jobs/current")
        async def current_job():
            job = self.jobs.current()
            if job is None:
                return JSONResponse(content=None)
            return job.to_dict()

        @app.get("/jobs/history")
        async def jobs_history(limit: int = 10):
            """Recent completed/stopped jobs with summary stats."""
            all_jobs = self.jobs.list()
            recent = all_jobs[:limit]
            result = []
            for j in recent:
                summary = {
                    "id": j.id,
                    "state": j.state,
                    "total_steps": j.total_steps,
                    "current_step": j.current_step,
                    "started_at": j.started_at.isoformat(),
                    "completed_at": j.completed_at.isoformat() if j.completed_at else None,
                }
                # Per-task final losses with health (last loss entry for each task)
                final_losses = {}
                for loss_entry in reversed(j.losses):
                    tn = loss_entry["task_name"]
                    if tn not in final_losses:
                        final_losses[tn] = {
                            "loss": loss_entry["task_loss"],
                            "health": loss_entry.get("health", "unknown"),
                        }
                summary["final_losses"] = final_losses
                # Overall health: worst across all tasks
                healths = [v["health"] for v in final_losses.values()]
                if "critical" in healths:
                    summary["overall_health"] = "critical"
                elif "warning" in healths:
                    summary["overall_health"] = "warning"
                elif healths:
                    summary["overall_health"] = "healthy"
                else:
                    summary["overall_health"] = "unknown"
                result.append(summary)
            return result

        @app.get("/jobs/{job_id}")
        async def get_job(job_id: str):
            job = self.jobs.get(job_id)
            if job is None:
                return JSONResponse({"error": f"Job '{job_id}' not found"}, status_code=404)
            return job.to_dict()

        @app.get("/jobs/{job_id}/loss_history")
        async def job_loss_history(job_id: str, max_points: int = 200):
            """Loss history for a job, downsampled per task for chart display.

            Returns at most `max_points` entries per task name.  Always keeps
            the first and last entry for each task so the chart spans the full
            step range.  For live jobs the tail is the most important part so
            we sample uniformly across what we have.
            """
            job = self.jobs.get(job_id)
            if job is None:
                return JSONResponse({"error": f"Job '{job_id}' not found"}, status_code=404)

            losses = job.losses
            if len(losses) <= max_points:
                return losses

            # Group by task_name, downsample each, merge back in step order.
            # Always preserve entries with training_metrics (sparse, high-value).
            from collections import defaultdict
            by_task: dict[str, list[dict]] = defaultdict(list)
            metrics_entries = []
            for entry in losses:
                by_task[entry.get("task_name", "?")].append(entry)
                if "training_metrics" in entry:
                    metrics_entries.append(entry)

            result = []
            for task_entries in by_task.values():
                n = len(task_entries)
                if n <= max_points:
                    result.extend(task_entries)
                else:
                    # Always keep first and last; sample evenly in between
                    step = max(1, n // max_points)
                    sampled = task_entries[::step]
                    if task_entries[-1] not in sampled:
                        sampled.append(task_entries[-1])
                    result.extend(sampled)

            # Always include training_metrics entries (they're rare — ~1 per epoch)
            seen_steps = {e.get("step") for e in result}
            for me in metrics_entries:
                if me.get("step") not in seen_steps:
                    result.append(me)

            result.sort(key=lambda e: e.get("step", 0))
            return result

        @app.get("/jobs/{job_id}/loss_at")
        async def job_loss_at(job_id: str, index: int = -1):
            """Single loss entry by index.  index=0 for first, index=-1 for latest.

            Returns a single step_info dict, or 404 if out of range.
            Lightweight endpoint for quick health checks during training.
            """
            job = self.jobs.get(job_id)
            if job is None:
                return JSONResponse({"error": f"Job '{job_id}' not found"}, status_code=404)
            losses = job.losses
            if not losses:
                return JSONResponse({"error": "No losses yet"}, status_code=404)
            try:
                entry = losses[index]
            except IndexError:
                return JSONResponse({"error": f"Index {index} out of range (have {len(losses)})"}, status_code=404)
            return {
                **entry,
                "n_total_losses": len(losses),
                "job_state": job.state,
                "job_step": job.current_step,
                "job_total_steps": job.total_steps,
            }

        @app.get("/jobs/current/loss_summary")
        async def current_job_loss_summary():
            """Per-task loss summary for the currently running job."""
            job = self.jobs.current()
            if job is None:
                return JSONResponse({"error": "No current job"}, status_code=404)
            summaries = compute_loss_summary(job.losses)
            return {name: s.to_dict() for name, s in summaries.items()}

        @app.get("/jobs/{job_id}/loss_summary")
        async def job_loss_summary(job_id: str):
            """Per-task loss summary with health classification for a job."""
            job = self.jobs.get(job_id)
            if job is None:
                return JSONResponse({"error": f"Job '{job_id}' not found"}, status_code=404)
            summaries = compute_loss_summary(job.losses)
            return {name: s.to_dict() for name, s in summaries.items()}

        @app.get("/jobs/{job_id}/stream")
        async def stream_job(job_id: str, from_step: int = 0):
            # SSE DISABLED — the blocking stream iterator + training thread
            # both contend for jobs._lock on every step, creating a lock
            # convoy that starves the uvicorn event loop and makes the
            # entire trainer unresponsive. Return immediate "done" instead.
            # TODO: fix by replacing lock-based streaming with an async queue.
            async def generate():
                yield 'data: {"done": true}\n\n'

            return StreamingResponse(generate(), media_type="text/event-stream")

        # -- Evaluation --
        @app.post("/eval/run")
        async def run_eval():
            guard = self._training_guard()
            if guard is not None:
                return guard
            if self.trainer is None:
                return JSONResponse({"error": "No trainer"}, status_code=400)
            results = self.trainer.evaluate_all()
            return results

        @app.post("/eval/checkpoint")
        async def eval_checkpoint(request: Request):
            """Run evaluation on a specific checkpoint without changing current state.

            Temporarily loads the checkpoint, runs eval, then restores original state.
            Body: {"checkpoint_id": "abc123"}
            Returns: {"checkpoint_id": "abc123", "tag": "...", "metrics": {task: {metric: val}}}
            """
            guard = self._training_guard()
            if guard is not None:
                return guard
            if self.trainer is None or self.autoencoder is None or self.checkpoints is None:
                return JSONResponse({"error": "No trainer/model/checkpoint store"}, status_code=400)

            data = await request.json()
            cp_id = data.get("checkpoint_id")
            if not cp_id:
                return JSONResponse({"error": "Missing 'checkpoint_id'"}, status_code=400)

            # Save current state to a temp buffer (deep copy to avoid mutation)
            import copy
            original_state = copy.deepcopy(self.trainer.state_dict())

            try:
                # Load the target checkpoint
                cp = self.checkpoints.load(cp_id, self.autoencoder, self.trainer, device=self.device)
                # Run eval
                results = self.trainer.evaluate_all()
                return {
                    "checkpoint_id": cp_id,
                    "tag": cp.tag,
                    "metrics": results,
                }
            except FileNotFoundError as e:
                return JSONResponse({"error": str(e)}, status_code=404)
            finally:
                # Restore original state
                self.trainer.load_state_dict(original_state)
                self.autoencoder.to(self.device)
                for task in self.tasks.values():
                    if task.head is not None:
                        task.head.to(self.device)

        @app.get("/eval/siblings")
        async def eval_siblings():
            """Return eval metrics for all sibling checkpoints (same parent_id).

            Reads stored eval_results from checkpoint metadata — no re-evaluation.
            Returns: {
                "siblings": [
                    {"id": "abc", "tag": "standard-baseline", "description": "...", "eval_results": {...}},
                    {"id": "def", "tag": "gated-cgg", "description": "...", "eval_results": {...}},
                ],
                "current_id": "abc"
            }
            """
            if self.checkpoints is None:
                return {"siblings": [], "current_id": None}

            current_id = self.checkpoints.current_id
            if not current_id:
                return {"siblings": [], "current_id": None}

            # Find current checkpoint's parent_id
            all_cps = self.checkpoints.tree()
            current_cp = None
            for cp in all_cps:
                if cp.id == current_id:
                    current_cp = cp
                    break

            if current_cp is None:
                return {"siblings": [], "current_id": current_id}

            parent_id = current_cp.parent_id

            # Find all siblings: checkpoints with the same parent_id
            # (includes the current checkpoint itself)
            siblings = []
            for cp in all_cps:
                if cp.parent_id == parent_id:
                    eval_results = cp.metrics.get("eval_results", {})
                    siblings.append({
                        "id": cp.id,
                        "tag": cp.tag,
                        "description": cp.description,
                        "eval_results": eval_results,
                    })

            return {"siblings": siblings, "current_id": current_id}

        @app.get("/eval/features")
        async def eval_features(layer_name: str = ""):
            """Extract weight features as base64 PNG images.

            For layers whose weight rows/cols match the model's image shape,
            each row is reshaped to an image and returned as base64 PNG.

            Query params:
                layer_name: specific layer to visualize (default: auto-detect)

            Returns: {
                "layers": {
                    "encoder.0": {
                        "n_features": 64,
                        "weight_shape": [64, 784],
                        "image_shape": [28, 28],
                        "features": ["base64...", ...]
                    }
                }
            }
            """
            if self.autoencoder is None:
                return JSONResponse({"error": "No model loaded"}, status_code=400)

            image_shape = None
            if hasattr(self.autoencoder, '_image_shape') and self.autoencoder._image_shape:
                image_shape = self.autoencoder._image_shape
            if image_shape is None:
                return JSONResponse({"error": "Model has no image_shape"}, status_code=400)

            import numpy as np
            flat_dim = 1
            for d in image_shape:
                flat_dim *= d
            spatial = image_shape[1:] if len(image_shape) == 3 else image_shape  # (H, W)

            result = {}
            for name, module in self.autoencoder.named_modules():
                if layer_name and name != layer_name:
                    continue
                if not hasattr(module, 'weight'):
                    continue
                w = module.weight.detach().cpu()
                # Check if weight rows match flattened image dim
                if w.ndim == 2 and w.shape[1] == flat_dim:
                    features = []
                    for i in range(w.shape[0]):
                        row = w[i].view(*spatial).float()
                        # Normalize to [0, 1]
                        rmin, rmax = row.min(), row.max()
                        if rmax - rmin > 1e-8:
                            row = (row - rmin) / (rmax - rmin)
                        else:
                            row = torch.zeros_like(row) + 0.5
                        # Convert to [1, H, W] for _tensor_to_base64
                        features.append(_tensor_to_base64(row.unsqueeze(0)))
                    result[name] = {
                        "n_features": w.shape[0],
                        "weight_shape": list(w.shape),
                        "image_shape": list(spatial),
                        "features": features,
                    }

            return {"layers": result}

        @app.get("/eval/features/siblings")
        async def eval_features_siblings(layer_name: str = "encoder.0"):
            """Extract weight features for all sibling checkpoints.

            Temporarily loads each sibling checkpoint, extracts features,
            restores original state. Returns features for comparison.

            Returns: {
                "siblings": [
                    {"id": "abc", "tag": "control", "features": ["base64...", ...]},
                    ...
                ],
                "current_id": "abc",
                "n_features": 64,
                "image_shape": [28, 28]
            }
            """
            if self.autoencoder is None or self.trainer is None or self.checkpoints is None:
                return JSONResponse({"error": "No model/trainer/checkpoints"}, status_code=400)

            guard = self._training_guard()
            if guard is not None:
                return guard

            image_shape = None
            if hasattr(self.autoencoder, '_image_shape') and self.autoencoder._image_shape:
                image_shape = self.autoencoder._image_shape
            if image_shape is None:
                return JSONResponse({"error": "Model has no image_shape"}, status_code=400)

            import numpy as np
            import copy
            flat_dim = 1
            for d in image_shape:
                flat_dim *= d
            spatial = image_shape[1:] if len(image_shape) == 3 else image_shape

            current_id = self.checkpoints.current_id
            if not current_id:
                return {"siblings": [], "current_id": None, "n_features": 0, "image_shape": list(spatial)}

            # Find siblings
            all_cps = self.checkpoints.tree()
            current_cp = None
            for cp in all_cps:
                if cp.id == current_id:
                    current_cp = cp
                    break
            if current_cp is None:
                return {"siblings": [], "current_id": current_id, "n_features": 0, "image_shape": list(spatial)}

            sibling_cps = [cp for cp in all_cps if cp.parent_id == current_cp.parent_id]
            if len(sibling_cps) < 2:
                return {"siblings": [], "current_id": current_id, "n_features": 0, "image_shape": list(spatial)}

            # Save current state
            original_state = copy.deepcopy(self.trainer.state_dict())
            original_cp_id = current_id

            siblings = []
            n_features = 0
            try:
                for cp in sibling_cps:
                    # Load checkpoint
                    self.checkpoints.load(cp.id, self.autoencoder, self.trainer, device=self.device)

                    # Extract features from target layer
                    named = dict(self.autoencoder.named_modules())
                    module = named.get(layer_name)
                    if module is None or not hasattr(module, 'weight'):
                        continue
                    w = module.weight.detach().cpu()
                    if w.ndim != 2 or w.shape[1] != flat_dim:
                        continue

                    features = []
                    for i in range(w.shape[0]):
                        row = w[i].view(*spatial).float()
                        rmin, rmax = row.min(), row.max()
                        if rmax - rmin > 1e-8:
                            row = (row - rmin) / (rmax - rmin)
                        else:
                            row = torch.zeros_like(row) + 0.5
                        features.append(_tensor_to_base64(row.unsqueeze(0)))
                    n_features = w.shape[0]
                    siblings.append({
                        "id": cp.id,
                        "tag": cp.tag,
                        "description": cp.description,
                        "features": features,
                    })
            finally:
                # Restore original state
                self.trainer.load_state_dict(original_state)
                self.autoencoder.to(self.device)
                for task in self.tasks.values():
                    if task.head is not None:
                        task.head.to(self.device)
                self.checkpoints._current_id = original_cp_id

            return {
                "siblings": siblings,
                "current_id": current_id,
                "n_features": n_features,
                "image_shape": list(spatial),
            }

        @app.get("/eval/features/snapshots")
        async def eval_features_snapshots(tag: str = "", step: int = -1):
            """Return feature weight snapshot data for timeline visualization.

            The recipe stores FeatureSnapshotRecorder objects on the API as
            ``self.snapshot_recorders[tag]``. This endpoint renders them.

            Query params:
                tag: condition tag (e.g. "pca-k8"). Empty = list available tags.
                step: specific step to render. -1 = return index of all snapshots.

            Returns (tag empty):
                {"tags": ["nbr-k8", "pca-k8", ...]}

            Returns (step == -1, tag set):
                {"tag": "pca-k8", "steps": [{"step": 0, "event": "init"}, ...]}

            Returns (step >= 0, tag set):
                {"tag": "pca-k8", "step": 500, "event": "periodic",
                 "n_features": 64, "image_shape": [28, 28],
                 "features": ["base64...", ...]}
            """
            recorders = getattr(self, 'snapshot_recorders', None)
            if not recorders:
                return JSONResponse(
                    {"error": "No snapshot recorders available. Run a gated recipe first."},
                    status_code=404,
                )

            # List available tags
            if not tag:
                return {"tags": list(recorders.keys())}

            recorder = recorders.get(tag)
            if recorder is None:
                return JSONResponse(
                    {"error": f"No snapshots for tag '{tag}'. Available: {list(recorders.keys())}"},
                    status_code=404,
                )

            # Return step index
            if step < 0:
                return {"tag": tag, "steps": recorder.get_steps()}

            # Render a specific step's features as base64 PNGs
            feature_images = recorder.get_feature_images(step)
            if feature_images is None:
                return JSONResponse(
                    {"error": f"No snapshot at step {step} for tag '{tag}'"},
                    status_code=404,
                )

            # feature_images is [D, H, W], normalize each to [0,1] and encode
            features_b64 = []
            for i in range(feature_images.shape[0]):
                row = feature_images[i].float()
                rmin, rmax = row.min(), row.max()
                if rmax - rmin > 1e-8:
                    row = (row - rmin) / (rmax - rmin)
                else:
                    row = torch.zeros_like(row) + 0.5
                features_b64.append(_tensor_to_base64(row.unsqueeze(0)))

            return {
                "tag": tag,
                "step": step,
                "event": next(
                    (e for s, e, _ in recorder.snapshots if s == step), ""
                ),
                "n_features": feature_images.shape[0],
                "image_shape": list(recorder.image_shape),
                "features": features_b64,
            }

        @app.get("/eval/bcl/diagnostics")
        async def bcl_diagnostics(tag: str = "", mode: str = "scatter"):
            """Return BCL diagnostic data for dashboard panels.

            Query params:
                tag: condition tag (e.g. "bcl-med"). Empty = list available tags.
                mode: "scatter" | "winrate" | "diversity"

            Returns (tag empty):
                {"tags": ["bcl-slow", "bcl-med"]}

            Returns (mode=scatter):
                {"tag": "bcl-med", "entries": [{step, grad_magnitude[D], som_magnitude[D], win_rate[D]}, ...]}

            Returns (mode=winrate):
                {"tag": "bcl-med", "entries": [{step, win_rate[D]}, ...]}

            Returns (mode=diversity):
                {"tag": "bcl-med", "entries": [{step, mean_similarity}, ...]}
            """
            trackers = getattr(self, 'bcl_trackers', None)
            if not trackers:
                return JSONResponse(
                    {"error": "No BCL trackers available. Run a BCL recipe first."},
                    status_code=404,
                )

            if not tag:
                return {"tags": list(trackers.keys())}

            tracker = trackers.get(tag)
            if tracker is None:
                return JSONResponse(
                    {"error": f"No BCL data for tag '{tag}'. Available: {list(trackers.keys())}"},
                    status_code=404,
                )

            if mode == "scatter":
                return {"tag": tag, "entries": tracker.signal_scatter_log}
            elif mode == "winrate":
                return {"tag": tag, "entries": tracker.win_rate_log}
            elif mode == "diversity":
                return {"tag": tag, "entries": tracker.dead_diversity_log}
            else:
                return JSONResponse(
                    {"error": f"Unknown mode '{mode}'. Use scatter/winrate/diversity."},
                    status_code=400,
                )

        @app.post("/eval/reconstructions")
        async def eval_reconstructions(request: Request):
            """Encode + decode N images, return originals and reconstructions side-by-side."""
            guard = self._training_guard()
            if guard is not None:
                return guard
            if self.autoencoder is None:
                return JSONResponse({"error": "No model loaded"}, status_code=400)
            if not self.autoencoder.has_decoder:
                return JSONResponse({"error": "Model has no decoder"}, status_code=400)

            data = await request.json() if await request.body() else {}
            n = data.get("n", 8)

            # Get images from first available dataset
            ds = next(iter(self.datasets.values()), None) if self.datasets else None
            if ds is None:
                return JSONResponse({"error": "No datasets loaded"}, status_code=400)

            self.autoencoder.eval()
            with torch.no_grad():
                images = ds.sample(n).to(self.device)
                model_out = self.autoencoder(images)
                recon = model_out[ModelOutput.RECONSTRUCTION].clamp(0, 1)

                originals = [_tensor_to_base64(images[i]) for i in range(n)]
                reconstructions = [_tensor_to_base64(recon[i]) for i in range(n)]

            return {"originals": originals, "reconstructions": reconstructions}

        # -- Checkpoints --
        @app.get("/checkpoints")
        async def list_checkpoints():
            if self.checkpoints is None:
                return []
            return [cp.to_dict() for cp in self.checkpoints.tree()]

        @app.post("/checkpoints/save")
        async def save_checkpoint(request: Request):
            if self.autoencoder is None or self.trainer is None:
                return JSONResponse({"error": "No model/trainer loaded"}, status_code=400)
            if self.checkpoints is None:
                from acc.checkpoints import CheckpointStore
                self.checkpoints = CheckpointStore("./acc/checkpoints_data")
            data = await request.json() if await request.body() else {}
            tag = data.get("tag", "checkpoint")
            # Build metrics (summary + full history) BEFORE save
            metrics = {}
            recent_jobs = self.jobs.list()
            for j in recent_jobs:
                if j.losses:
                    summaries = compute_loss_summary(j.losses)
                    metrics["loss_summary"] = {
                        name: s.to_dict() for name, s in summaries.items()
                    }
                    metrics["loss_history"] = j.losses
                    break
            cp = self.checkpoints.save(
                self.autoencoder, self.trainer, tag=tag,
                description="Manual save from dashboard",
                metrics=metrics,
            )
            return cp.to_dict()

        @app.post("/checkpoints/load")
        async def load_checkpoint(request: Request):
            guard = self._training_guard()
            if guard:
                return guard
            if (
                self.autoencoder is None
                or self.trainer is None
                or self.checkpoints is None
            ):
                return JSONResponse({"error": "No model/trainer/checkpoint store"}, status_code=400)
            data = await request.json()
            cp_id = data.get("id")
            if not cp_id:
                return JSONResponse({"error": "Missing checkpoint id"}, status_code=400)
            try:
                cp = self.checkpoints.load(cp_id, self.autoencoder, self.trainer, device=self.device)
                cp_dict = cp.to_dict()
                # Inject checkpoint's stored loss history as a synthetic job
                # so the dashboard can show loss curves for this checkpoint
                loss_history = cp_dict.get("metrics", {}).get("loss_history")
                if loss_history:
                    from acc.jobs import JobInfo
                    syn_id = f"cp-{cp_id[:8]}"
                    job = JobInfo(
                        id=syn_id,
                        state="completed",
                        total_steps=max((e.get("step", 0) for e in loss_history), default=0),
                        current_step=max((e.get("step", 0) for e in loss_history), default=0),
                        task_names=list({e["task_name"] for e in loss_history}),
                        losses=loss_history,
                        started_at=datetime.fromisoformat(cp_dict["timestamp"]),
                        completed_at=datetime.fromisoformat(cp_dict["timestamp"]),
                        checkpoint_id=cp_id,
                    )
                    with self.jobs._lock:
                        self.jobs._jobs[syn_id] = job
                return cp_dict
            except FileNotFoundError as e:
                return JSONResponse({"error": str(e)}, status_code=404)

        @app.post("/checkpoints/fork")
        async def fork_checkpoint(request: Request):
            if self.checkpoints is None:
                return JSONResponse({"error": "No checkpoint store"}, status_code=400)
            data = await request.json()
            cp_id = data.get("id")
            new_tag = data.get("new_tag", "fork")
            try:
                cp = self.checkpoints.fork(cp_id, new_tag)
                return cp.to_dict()
            except KeyError as e:
                return JSONResponse({"error": str(e)}, status_code=404)

        # -- Registry (what's available) --
        @app.get("/registry/tasks")
        async def registry_tasks():
            """List available task classes — discovered dynamically by TaskRegistry."""
            return self.task_registry.list()

        @app.get("/registry/layers")
        async def registry_layers():
            return [
                {
                    "class_name": "ConvBlock",
                    "description": "Conv2d + BatchNorm + ReLU",
                },
                {
                    "class_name": "ConvTransposeBlock",
                    "description": "ConvTranspose2d + BatchNorm + ReLU",
                },
                {
                    "class_name": "ResBlock",
                    "description": "GroupNorm + SiLU + Conv residual block",
                },
                {
                    "class_name": "FactorHead",
                    "description": "AdaptivePool + Linear projection for factor slice",
                },
                {
                    "class_name": "CrossAttentionBlock",
                    "description": "Spatial cross-attention to factor embeddings",
                },
                {
                    "class_name": "FactorEmbedder",
                    "description": "Per-factor MLP projection to shared embed space",
                },
            ]

        # -- Generators --
        @app.get("/registry/generators")
        async def registry_generators():
            """List available dataset generators — discovered dynamically by GeneratorRegistry."""
            return self.generator_registry.list()

        @app.post("/generators/generate")
        async def generate_dataset(request: Request):
            """Generate a dataset using a registered generator.

            Body: {"generator_name": "thickness", "params": {"n": 1000, "image_size": 32}}
            """
            data = await request.json()
            gen_name = data.get("generator_name")
            params = data.get("params", {})

            generator = self.generator_registry.get(gen_name)
            if generator is None:
                return JSONResponse({"error": f"Generator '{gen_name}' not found"}, status_code=404)

            try:
                dataset = generator.generate(**params)
                # Register the dataset
                self.datasets[dataset.name] = dataset
                return dataset.describe()
            except Exception as e:
                return JSONResponse({"error": f"Generation failed: {e}"}, status_code=500)

        # -- Recipes --
        # Static routes MUST come before parameterized {name} routes
        @app.get("/recipes")
        async def list_recipes():
            return self.recipe_registry.list()

        @app.get("/recipes/current")
        async def current_recipe():
            job = self.recipe_runner.current()
            if job is None:
                return JSONResponse(content=None)
            return job.to_dict()

        @app.post("/recipes/stop")
        async def stop_recipe():
            self.recipe_runner.stop()
            job = self.recipe_runner.current()
            return job.to_dict() if job else {"status": "no running recipe"}

        @app.get("/recipes/{name}")
        async def get_recipe(name: str):
            recipe = self.recipe_registry.get(name)
            if recipe is None:
                return JSONResponse({"error": f"Recipe '{name}' not found"}, status_code=404)
            return {"name": recipe.name, "description": recipe.description}

        @app.post("/recipes/{name}/run")
        async def run_recipe(name: str):
            recipe = self.recipe_registry.get(name)
            if recipe is None:
                return JSONResponse({"error": f"Recipe '{name}' not found"}, status_code=404)
            try:
                job = self.recipe_runner.start(recipe, self)
                return job.to_dict()
            except RuntimeError as e:
                return JSONResponse({"error": str(e)}, status_code=409)

        # -- Checkpoint tree --
        @app.get("/checkpoints/tree")
        async def checkpoint_tree():
            """Return full tree structure: nodes with parent_id links."""
            if self.checkpoints is None:
                return {"nodes": [], "current_id": None}
            nodes = [cp.to_dict() for cp in self.checkpoints.tree()]
            return {
                "nodes": nodes,
                "current_id": self.checkpoints.current_id,
            }

        @app.get("/checkpoints/current")
        async def current_checkpoint():
            """Return current checkpoint details + the job that trained it.

            The 'relevant job' is the most recent job whose checkpoint_id
            matches this checkpoint, OR (if none) the most recent job overall
            that was running when this checkpoint was saved.
            """
            has_model = self.autoencoder is not None
            cp_id = self.checkpoints.current_id if self.checkpoints else None

            # Current checkpoint metadata
            cp_data = None
            if cp_id and self.checkpoints:
                for cp in self.checkpoints.tree():
                    if cp.id == cp_id:
                        cp_data = cp.to_dict()
                        break

            # Find the most relevant job for this checkpoint:
            # 1. Job whose checkpoint_id matches (started FROM this checkpoint)
            # 2. Job completed closest before the checkpoint was saved
            # 3. Fallback: most recent job with losses
            relevant_job_id = None
            all_jobs = self.jobs.list()
            if cp_id:
                for j in all_jobs:
                    if j.checkpoint_id == cp_id:
                        relevant_job_id = j.id
                        break

            if not relevant_job_id and cp_data and all_jobs:
                # Match by timestamp: checkpoint saved right after job completes
                cp_time = datetime.fromisoformat(cp_data["timestamp"])
                best_job = None
                best_delta = None
                for j in all_jobs:
                    if j.completed_at and j.losses:
                        delta = (cp_time - j.completed_at).total_seconds()
                        if 0 <= delta < 10:  # checkpoint within 10s of job completion
                            if best_delta is None or delta < best_delta:
                                best_delta = delta
                                best_job = j
                if best_job:
                    relevant_job_id = best_job.id

            # Fallback: most recent job with losses
            if not relevant_job_id and all_jobs:
                for j in all_jobs:
                    if j.losses:
                        relevant_job_id = j.id
                        break

            return {
                "has_model": has_model,
                "checkpoint": cp_data,
                "checkpoint_id": cp_id,
                "relevant_job_id": relevant_job_id,
            }

        # -- Eval Visualization --
        @app.get("/eval/traversals")
        async def eval_traversals(
            n_seeds: int = 5,
            n_steps: int = 9,
            range_val: float = 3.0,
            checkpoint_id: str = "",
        ):
            """Generate latent traversal grids for each factor group.

            For each factor group:
              - Encode n_seeds test images -> z
              - Hold other dims fixed, vary this group's dims from -range to +range
              - Decode each -> grid of n_seeds rows x n_steps columns

            Args:
                checkpoint_id: If provided, temporarily load this checkpoint for eval.

            Returns: {"factor_name": [row_of_base64_pngs, ...], ...}
            """
            guard = self._training_guard()
            if guard is not None:
                return guard
            if self.autoencoder is None:
                return JSONResponse({"error": "No model"}, status_code=400)
            if not hasattr(self.autoencoder, "factor_groups"):
                return JSONResponse(
                    {"error": "Model has no factor_groups"}, status_code=400
                )

            def _generate_traversals():
                ds = next(iter(self.datasets.values()), None) if self.datasets else None
                if ds is None:
                    return {"error": "No datasets loaded"}

                model = self.autoencoder
                model.eval()
                with torch.no_grad():
                    seeds = ds.sample(n_seeds).to(self.device)
                    model_out = model(seeds)
                    z_flat = model_out[ModelOutput.MU]  # (B, total_latent_dim)

                    # For spatial bottleneck: reshape to (B, C, h, w) so we can
                    # vary entire channel groups (= factor groups) at once.
                    has_spatial = hasattr(model, "latent_channels")
                    if has_spatial:
                        z_spatial = z_flat.view(
                            n_seeds, model.latent_channels,
                            model._spatial_size, model._spatial_size,
                        )

                    result = {}
                    for fg in model.factor_groups:
                        rows = []
                        for seed_idx in range(n_seeds):
                            row_images = []
                            if has_spatial:
                                # Vary channel group in spatial z
                                z_seed = z_spatial[seed_idx].clone()  # (C, h, w)
                                ch_slice = z_spatial[:, fg.latent_start:fg.latent_end]  # (B, fg_ch, h, w)
                                fg_mean = ch_slice.mean(dim=0)  # (fg_ch, h, w)
                                fg_std = ch_slice.std(dim=0).clamp(min=0.1)
                                for step_i in range(n_steps):
                                    alpha = -range_val + (2 * range_val) * step_i / (n_steps - 1)
                                    z_mod = z_seed.clone().unsqueeze(0)  # (1, C, h, w)
                                    z_mod[0, fg.latent_start:fg.latent_end] = fg_mean + alpha * fg_std
                                    z_mod_flat = z_mod.flatten(1)  # (1, total_latent_dim)
                                    recon = _decode_z(model, z_mod_flat)
                                    row_images.append(_tensor_to_base64(recon[0]))
                            else:
                                # Flat latent: vary flat slice directly
                                z_seed = z_flat[seed_idx].clone()
                                fg_mean = z_flat[:, fg.latent_start:fg.latent_end].mean(dim=0)
                                fg_std = z_flat[:, fg.latent_start:fg.latent_end].std(dim=0).clamp(min=0.1)
                                for step_i in range(n_steps):
                                    alpha = -range_val + (2 * range_val) * step_i / (n_steps - 1)
                                    z_mod = z_seed.clone().unsqueeze(0)
                                    z_mod[0, fg.latent_start:fg.latent_end] = fg_mean + alpha * fg_std
                                    recon = _decode_z(model, z_mod)
                                    row_images.append(_tensor_to_base64(recon[0]))
                            rows.append(row_images)
                        result[fg.name] = rows

                return result

            if checkpoint_id:
                try:
                    with self._load_checkpoint_temporarily(checkpoint_id):
                        result = _generate_traversals()
                except (FileNotFoundError, RuntimeError) as e:
                    return JSONResponse({"error": str(e)}, status_code=400)
            else:
                result = _generate_traversals()

            if "error" in result:
                return JSONResponse(result, status_code=400)
            return result

        @app.get("/eval/sort_by_factor")
        async def eval_sort_by_factor(n_show: int = 20, checkpoint_id: str = ""):
            """Sort test images by mean activation of each factor group.

            For each factor group:
              - Encode all test images -> z
              - Compute mean of this factor slice
              - Sort by this value
              - Return lowest n_show and highest n_show as base64 PNGs

            Args:
                checkpoint_id: If provided, temporarily load this checkpoint for eval.

            Returns: {"factor_name": {"lowest": [...], "highest": [...]}, ...}
            """
            guard = self._training_guard()
            if guard is not None:
                return guard
            if self.autoencoder is None:
                return JSONResponse({"error": "No model"}, status_code=400)
            if not hasattr(self.autoencoder, "factor_groups"):
                return JSONResponse(
                    {"error": "Model has no factor_groups"}, status_code=400
                )

            def _generate_sort():
                ds = next(iter(self.datasets.values()), None) if self.datasets else None
                if ds is None:
                    return {"error": "No datasets loaded"}

                self.autoencoder.eval()
                n_encode = min(500, len(ds))
                images = ds.sample(n_encode).to(self.device)

                with torch.no_grad():
                    model_out = self.autoencoder(images)
                    z = model_out[ModelOutput.MU]

                    result = {}
                    for fg in self.autoencoder.factor_groups:
                        factor_z = z[:, fg.latent_start : fg.latent_end]
                        mean_activation = factor_z.mean(dim=1)
                        sorted_indices = mean_activation.argsort()

                        lowest = [
                            _tensor_to_base64(images[sorted_indices[i].item()])
                            for i in range(min(n_show, len(sorted_indices)))
                        ]
                        highest = [
                            _tensor_to_base64(images[sorted_indices[-(i + 1)].item()])
                            for i in range(min(n_show, len(sorted_indices)))
                        ]
                        result[fg.name] = {"lowest": lowest, "highest": highest}

                return result

            if checkpoint_id:
                try:
                    with self._load_checkpoint_temporarily(checkpoint_id):
                        result = _generate_sort()
                except (FileNotFoundError, RuntimeError) as e:
                    return JSONResponse({"error": str(e)}, status_code=400)
            else:
                result = _generate_sort()

            if "error" in result:
                return JSONResponse(result, status_code=400)
            return result

        @app.get("/eval/attention_maps")
        async def eval_attention_maps(n_images: int = 4, checkpoint_id: str = ""):
            """Extract per-factor attention heatmaps from cross-attention layers.

            Args:
                checkpoint_id: If provided, temporarily load this checkpoint for eval.

            Returns:
                {
                    "factor_name": [base64_heatmap_overlay, ...],
                    ...,
                    "originals": [base64, ...]
                }
            """
            guard = self._training_guard()
            if guard is not None:
                return guard
            if self.autoencoder is None:
                return JSONResponse({"error": "No model"}, status_code=400)
            if not hasattr(self.autoencoder, "cross_attn_stages"):
                return JSONResponse(
                    {"error": "Model has no cross-attention stages"}, status_code=400
                )

            from acc.eval.attention import extract_attention_maps

            def _generate_attention():
                ds = next(iter(self.datasets.values()), None) if self.datasets else None
                if ds is None:
                    return {"error": "No datasets loaded"}

                images = ds.sample(n_images).to(self.device)
                attn_maps = extract_attention_maps(self.autoencoder, images)

                originals = [_tensor_to_base64(images[i]) for i in range(n_images)]
                result = {"originals": originals}
                for factor_name, maps in attn_maps.items():
                    overlays = []
                    for i in range(n_images):
                        overlay = _attention_overlay(images[i], maps[i])
                        overlays.append(_tensor_to_base64(overlay))
                    result[factor_name] = overlays

                return result

            try:
                if checkpoint_id:
                    with self._load_checkpoint_temporarily(checkpoint_id):
                        result = _generate_attention()
                else:
                    result = _generate_attention()
            except Exception as e:
                return JSONResponse({"error": f"Attention extraction failed: {e}"}, status_code=400)

            if "error" in result:
                return JSONResponse(result, status_code=400)
            return result

        @app.post("/eval/ufr")
        async def eval_ufr():
            """Compute UFR disentanglement metrics.

            Returns: {"ufr": float, "disentanglement": float, "completeness": float}
            """
            guard = self._training_guard()
            if guard is not None:
                return guard
            if self.autoencoder is None:
                return JSONResponse({"error": "No model"}, status_code=400)
            if not hasattr(self.autoencoder, "factor_groups"):
                return JSONResponse(
                    {"error": "Model has no factor_groups"}, status_code=400
                )
            if not self.datasets:
                return JSONResponse({"error": "No datasets loaded"}, status_code=400)

            from acc.eval.ufr import compute_ufr

            results = compute_ufr(
                self.autoencoder, self.datasets, self.device
            )
            return results

        # -- Device --
        @app.get("/device")
        async def get_device():
            """Current device and available devices."""
            available = ["cpu"]
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    available.append(f"cuda:{i}")
            return {
                "current": str(self.device),
                "available": available,
            }

        @app.post("/device/set")
        async def set_device(request: Request):
            """Change the active device. Moves model + probe heads.

            Body: {"device": "cuda:1"}
            """
            guard = self._training_guard()
            if guard:
                return guard
            data = await request.json()
            device_str = data.get("device")
            if not device_str:
                return JSONResponse({"error": "Missing 'device' field"}, status_code=400)

            # Validate device string
            try:
                new_device = torch.device(device_str)
                if new_device.type == "cuda" and not torch.cuda.is_available():
                    return JSONResponse({"error": "CUDA not available"}, status_code=400)
                if new_device.type == "cuda" and new_device.index is not None:
                    if new_device.index >= torch.cuda.device_count():
                        return JSONResponse(
                            {"error": f"CUDA device {new_device.index} not found. Have {torch.cuda.device_count()} GPUs."},
                            status_code=400,
                        )
            except Exception as e:
                return JSONResponse({"error": f"Invalid device: {e}"}, status_code=400)

            old_device = self.device
            self.device = new_device

            # Move model
            if self.autoencoder is not None:
                self.autoencoder.to(new_device)

            # Move probe heads
            for task in self.tasks.values():
                if task.head is not None:
                    task.head.to(new_device)

            # Rebuild trainer with new device
            if self.trainer is not None:
                self.trainer.device = new_device
                self.trainer._build_optimizers()

            return {
                "previous": str(old_device),
                "current": str(new_device),
            }

        # -- Health --
        @app.get("/health")
        async def health():
            recipe_job = self.recipe_runner.current()
            return {
                "status": "ok",
                "has_model": self.autoencoder is not None,
                "num_tasks": len(self.tasks),
                "num_datasets": len(self.datasets),
                "num_recipes": len(self.recipe_registry.list()),
                "num_generators": len(self.generator_registry.list()),
                "recipe_running": recipe_job is not None and recipe_job.state == "running",
                "device": str(self.device),
            }

    def _rebuild_trainer(self):
        """Rebuild the Trainer when tasks change."""
        if self.autoencoder is None:
            return
        tasks = list(self.tasks.values())
        if not tasks:
            self.trainer = None
            return
        lr = self.trainer.lr if self.trainer else 1e-3
        probe_lr = self.trainer.probe_lr if self.trainer else 1e-3
        batch_size = self.trainer.batch_size if self.trainer else 64
        self.trainer = Trainer(
            self.autoencoder,
            tasks,
            self.device,
            lr=lr,
            probe_lr=probe_lr,
            batch_size=batch_size,
        )

    def _load_checkpoint_temporarily(self, checkpoint_id: str):
        """Context manager: temporarily load a checkpoint, then restore original state.

        Usage:
            with self._load_checkpoint_temporarily("abc123"):
                # self.autoencoder has the checkpoint's weights
                result = self.autoencoder(images)
            # original weights are restored
        """
        import contextlib
        import copy

        @contextlib.contextmanager
        def _ctx():
            if self.trainer is None or self.autoencoder is None or self.checkpoints is None:
                raise RuntimeError("No trainer/model/checkpoint store")

            original_state = copy.deepcopy(self.trainer.state_dict())
            try:
                self.checkpoints.load(
                    checkpoint_id, self.autoencoder, self.trainer, device=self.device
                )
                yield
            finally:
                self.trainer.load_state_dict(original_state)
                self.autoencoder.to(self.device)
                for task in self.tasks.values():
                    if task.head is not None:
                        task.head.to(self.device)

        return _ctx()

    def setup(
        self,
        autoencoder: Autoencoder,
        tasks: list[Task],
        datasets: dict[str, AccDataset],
        checkpoint_dir: str = "./acc/checkpoints",
    ):
        """Configure the API with a model, tasks, and datasets."""
        self.autoencoder = autoencoder
        self.checkpoints = CheckpointStore(checkpoint_dir)
        for ds_name, ds in datasets.items():
            self.datasets[ds_name] = ds
        for task in tasks:
            self.tasks[task.name] = task
        self.trainer = Trainer(
            autoencoder,
            tasks,
            self.device,
        )

    def run(self, host: str = "0.0.0.0", port: int = 6060):
        """Start the HTTP server. No auto-reload — trainer holds in-memory state.

        Sets a short GIL switch interval so training threads can't starve
        the HTTP event loop. Default is 5ms; we set 0.5ms which guarantees
        the event loop gets CPU time even during heavy training.

        If the Astro dashboard has been built (dashboard/dist/ exists), it's
        mounted at / so the trainer serves its own UI. In dev mode the Astro
        dev server runs separately and proxies API calls here.
        """
        import sys
        import uvicorn

        # Serve the built Astro dashboard at / if available.
        # Must be mounted AFTER all API routes so they take priority.
        dashboard_dist = Path(__file__).resolve().parent.parent / "dashboard" / "dist"
        if dashboard_dist.is_dir():
            self.app.mount("/", StaticFiles(directory=str(dashboard_dist), html=True), name="dashboard")
            print(f"  Dashboard: serving from {dashboard_dist}")
        else:
            print(f"  Dashboard: not built (run 'npm run build' in dashboard/)")

        # Force frequent GIL switching so training threads don't starve HTTP.
        # Default is 5ms (0.005). We set 0.5ms — training throughput impact
        # is negligible (context switches are ~1μs on modern CPUs).
        sys.setswitchinterval(0.0005)

        uvicorn.run(self.app, host=host, port=port)


def _decode_z(model: torch.nn.Module, z: torch.Tensor) -> torch.Tensor:
    """Decode a flat latent vector z through the model's decoder.

    Works with FactorSlotAutoencoder (spatial bottleneck): reshapes flat z
    to spatial, extracts factor slices, embeds them, runs decoder stages.

    z shape: (B, total_latent_dim) where total_latent_dim = latent_channels * h * w
    """
    if hasattr(model, "factor_groups") and hasattr(model, "decoder_input"):
        # FactorSlotAutoencoder with spatial bottleneck
        B = z.shape[0]
        # Reshape flat z back to spatial: (B, latent_channels, h, w)
        z_spatial = z.view(B, model.latent_channels, model._spatial_size, model._spatial_size)

        # Extract factor slices (channel groups, spatially pooled) for cross-attention
        factor_slices = model._extract_factor_slices(z_spatial)
        factor_embeds = model.factor_embedder(factor_slices)

        # Decode from spatial latent
        h = model.decoder_input(z_spatial)
        for stage, cross_attn in zip(model.decoder_stages, model.cross_attn_stages):
            h = stage(h)
            if cross_attn is not None:
                h = cross_attn(h, factor_embeds)
        return model.to_output(h).clamp(0, 1)
    else:
        raise NotImplementedError("Traversals require FactorSlotAutoencoder")


def _tensor_to_base64(tensor: torch.Tensor) -> str:
    """Convert a [C, H, W] tensor to base64-encoded PNG."""
    import PIL.Image
    import numpy as np

    img = tensor.cpu().detach()
    if img.shape[0] == 1:
        img = img.squeeze(0)  # grayscale -> (H, W)
    elif img.shape[0] == 3:
        img = img.permute(1, 2, 0)  # RGB -> (H, W, 3)
    img_np = (img.numpy() * 255).clip(0, 255).astype(np.uint8)
    pil_img = PIL.Image.fromarray(img_np)

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _attention_overlay(
    image: torch.Tensor, heatmap: torch.Tensor, alpha: float = 0.5
) -> torch.Tensor:
    """Overlay a [H, W] attention heatmap on a [C, H, W] image.

    Returns a [3, H, W] RGB tensor with the heatmap blended in.
    Uses a red-yellow colormap: low attention = transparent, high = red/yellow.
    """
    img = image.cpu().detach().float()
    heat = heatmap.cpu().detach().float()

    # Normalize heatmap to [0, 1]
    h_min, h_max = heat.min(), heat.max()
    if h_max > h_min:
        heat = (heat - h_min) / (h_max - h_min)

    # Convert grayscale to RGB if needed
    if img.shape[0] == 1:
        img = img.expand(3, -1, -1)

    # Simple red-yellow colormap: R=1, G=heat, B=0 (low=red, high=yellow)
    H, W = heat.shape
    color = torch.zeros(3, H, W)
    color[0] = 1.0  # R
    color[1] = heat  # G (0=red, 1=yellow)
    color[2] = 0.0  # B

    # Blend: where heat is high, show color; where low, show image
    blend_alpha = heat.unsqueeze(0) * alpha
    result = img * (1 - blend_alpha) + color * blend_alpha

    return result.clamp(0, 1)
