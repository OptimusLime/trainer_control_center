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
from acc.trainer import Trainer
from acc.jobs import JobManager
from acc.checkpoints import CheckpointStore
from acc.tasks.base import Task, TaskError
from acc.dataset import AccDataset


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

        self._register_routes()

    def _register_routes(self):
        app = self.app

        # -- Model --
        @app.get("/model/describe")
        async def model_describe():
            if self.autoencoder is None:
                return JSONResponse({"error": "No model loaded"}, status_code=404)
            desc = str(self.autoencoder)
            return {
                "description": desc,
                "has_decoder": self.autoencoder.has_decoder,
                "latent_dim": self.autoencoder.latent_dim,
                "num_encoder_layers": len(self.autoencoder.encoder),
                "num_decoder_layers": len(self.autoencoder.decoder)
                if self.autoencoder.has_decoder
                else 0,
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

            task_class = _resolve_task_class(class_name)
            if task_class is None:
                return JSONResponse({"error": f"Unknown task class: {class_name}"}, status_code=400)

            try:
                task = task_class(task_name, dataset, weight=weight)
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

        @app.get("/jobs/{job_id}")
        async def get_job(job_id: str):
            job = self.jobs.get(job_id)
            if job is None:
                return JSONResponse({"error": f"Job '{job_id}' not found"}, status_code=404)
            return job.to_dict()

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
                cp = self.checkpoints.load(cp_id, self.autoencoder, self.trainer)
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
            """List available task classes."""
            return [
                {
                    "class_name": "ClassificationTask",
                    "description": "Linear probe classification with cross-entropy loss",
                },
                {
                    "class_name": "ReconstructionTask",
                    "description": "Reconstruction via decoder with L1 loss",
                },
                {
                    "class_name": "RegressionTask",
                    "description": "Linear probe regression with MSE loss, MAE eval",
                },
            ]

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

        # -- Health --
        @app.get("/health")
        async def health():
            return {
                "status": "ok",
                "has_model": self.autoencoder is not None,
                "num_tasks": len(self.tasks),
                "num_datasets": len(self.datasets),
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
        """Start the HTTP server."""
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)


def _resolve_task_class(class_name: str):
    """Resolve a task class name to the actual class."""
    from acc.tasks.classification import ClassificationTask
    from acc.tasks.reconstruction import ReconstructionTask
    from acc.tasks.regression import RegressionTask

    classes = {
        "ClassificationTask": ClassificationTask,
        "ReconstructionTask": ReconstructionTask,
        "RegressionTask": RegressionTask,
    }
    return classes.get(class_name)


def _tensor_to_base64(tensor: torch.Tensor) -> str:
    """Convert a [C, H, W] tensor to base64-encoded PNG."""
    import PIL.Image
    import numpy as np

    img = tensor.cpu().detach()
    if img.shape[0] == 1:
        img = img.squeeze(0)  # grayscale
    img_np = (img.numpy() * 255).clip(0, 255).astype(np.uint8)
    pil_img = PIL.Image.fromarray(img_np)

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
