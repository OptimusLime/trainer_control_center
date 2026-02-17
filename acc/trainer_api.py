"""Trainer HTTP API — JSON endpoints on localhost:6060.

Everything the UI can do, you can do via HTTP calls.
The trainer API is the source of truth.
"""

import asyncio
import base64
import io
import json
import threading
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

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
            task = self.tasks.pop(name, None)
            if task is None:
                return JSONResponse({"error": f"Task '{name}' not found"}, status_code=404)
            self._rebuild_trainer()
            return {"removed": name}

        # -- Training / Jobs --
        @app.post("/train/start")
        async def train_start(request: Request):
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

            # Group by task_name, downsample each, merge back in step order
            from collections import defaultdict
            by_task: dict[str, list[dict]] = defaultdict(list)
            for entry in losses:
                by_task[entry.get("task_name", "?")].append(entry)

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
            async def generate():
                # Run the blocking stream() iterator in a thread
                loop = asyncio.get_event_loop()
                stream_iter = self.jobs.stream(job_id, from_step=from_step)
                while True:
                    try:
                        step_data = await loop.run_in_executor(
                            None, lambda: next(stream_iter, None)
                        )
                        if step_data is None:
                            break
                        yield f"data: {json.dumps(step_data)}\n\n"
                    except StopIteration:
                        break
                yield 'data: {"done": true}\n\n'

            return StreamingResponse(generate(), media_type="text/event-stream")

        # -- Evaluation --
        @app.post("/eval/run")
        async def run_eval():
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

        @app.post("/eval/reconstructions")
        async def eval_reconstructions(request: Request):
            """Encode + decode N images, return originals and reconstructions side-by-side."""
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
                recon = model_out[ModelOutput.RECONSTRUCTION]

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
            if (
                self.autoencoder is None
                or self.trainer is None
                or self.checkpoints is None
            ):
                return JSONResponse({"error": "No model/trainer/checkpoint store"}, status_code=400)
            data = await request.json() if await request.body() else {}
            tag = data.get("tag", "checkpoint")
            cp = self.checkpoints.save(self.autoencoder, self.trainer, tag=tag)
            # Persist loss summary from the most recent completed job
            recent_jobs = self.jobs.list()
            for j in recent_jobs:
                if j.losses:
                    summaries = compute_loss_summary(j.losses)
                    cp.metrics["loss_summary"] = {
                        name: s.to_dict() for name, s in summaries.items()
                    }
                    break
            return cp.to_dict()

        @app.post("/checkpoints/load")
        async def load_checkpoint(request: Request):
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
                return cp.to_dict()
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

                self.autoencoder.eval()
                with torch.no_grad():
                    seeds = ds.sample(n_seeds).to(self.device)
                    model_out = self.autoencoder(seeds)
                    z_base = model_out[ModelOutput.MU]

                    result = {}
                    for fg in self.autoencoder.factor_groups:
                        rows = []
                        for seed_idx in range(n_seeds):
                            row_images = []
                            z_seed = z_base[seed_idx].clone()
                            # Get the mean activation for this factor group across seeds
                            # to use as the center of the traversal range
                            fg_mean = z_base[:, fg.latent_start : fg.latent_end].mean(dim=0)
                            fg_std = z_base[:, fg.latent_start : fg.latent_end].std(dim=0).clamp(min=0.1)
                            for step_i in range(n_steps):
                                # Vary relative to population stats: mean +/- range * std
                                alpha = -range_val + (2 * range_val) * step_i / (n_steps - 1)
                                z_mod = z_seed.clone().unsqueeze(0)
                                z_mod[0, fg.latent_start : fg.latent_end] = fg_mean + alpha * fg_std
                                recon = _decode_z(self.autoencoder, z_mod)
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

            if checkpoint_id:
                try:
                    with self._load_checkpoint_temporarily(checkpoint_id):
                        result = _generate_attention()
                except (FileNotFoundError, RuntimeError) as e:
                    return JSONResponse({"error": str(e)}, status_code=400)
            else:
                result = _generate_attention()

            if "error" in result:
                return JSONResponse(result, status_code=400)
            return result

        @app.post("/eval/ufr")
        async def eval_ufr():
            """Compute UFR disentanglement metrics.

            Returns: {"ufr": float, "disentanglement": float, "completeness": float}
            """
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
        """Start the HTTP server. No auto-reload — trainer holds in-memory state."""
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)


def _decode_z(model: torch.nn.Module, z: torch.Tensor) -> torch.Tensor:
    """Decode a latent vector z through the model's decoder.

    Works with FactorSlotAutoencoder: splits z into factor slices,
    embeds them, and runs through the decoder stages.
    Falls back to a full forward pass with zeros if model doesn't
    have the expected decoder structure.
    """
    if hasattr(model, "factor_groups") and hasattr(model, "decoder_init"):
        # FactorSlotAutoencoder decoder path
        B = z.shape[0]
        factor_slices = {}
        for fg in model.factor_groups:
            factor_slices[fg.name] = z[:, fg.latent_start : fg.latent_end]

        factor_embeds = model.factor_embedder(factor_slices)
        h = model.decoder_init(z).view(
            B,
            model._decoder_init_ch,
            model._decoder_init_spatial,
            model._decoder_init_spatial,
        )
        for stage, cross_attn in zip(model.decoder_stages, model.cross_attn_stages):
            h = stage(h)
            if cross_attn is not None:
                h = cross_attn(h, factor_embeds)
        return model.to_output(h)
    else:
        # Fallback: run full forward with a zero image, not ideal
        # but works for simple autoencoders
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
