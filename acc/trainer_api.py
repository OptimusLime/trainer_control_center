"""Trainer HTTP API — JSON endpoints on localhost:8787.

Everything the UI can do, you can do via HTTP calls.
The trainer API is the source of truth.
"""

import base64
import io
import json
import threading
from typing import Optional

from flask import Flask, request, jsonify, Response

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
        self.app = Flask(__name__)
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
        @app.route("/model/describe", methods=["GET"])
        def model_describe():
            if self.autoencoder is None:
                return jsonify({"error": "No model loaded"}), 404
            desc = str(self.autoencoder)
            return jsonify(
                {
                    "description": desc,
                    "has_decoder": self.autoencoder.has_decoder,
                    "latent_dim": self.autoencoder.latent_dim,
                    "num_encoder_layers": len(self.autoencoder.encoder),
                    "num_decoder_layers": len(self.autoencoder.decoder)
                    if self.autoencoder.has_decoder
                    else 0,
                }
            )

        @app.route("/model/create", methods=["POST"])
        def model_create():
            # For M1: model creation is done from Python code, not via API
            # This endpoint exists for completeness
            return jsonify(
                {
                    "error": "Model creation via API not yet implemented. Use Python code."
                }
            ), 501

        # -- Datasets --
        @app.route("/datasets", methods=["GET"])
        def list_datasets():
            return jsonify([ds.describe() for ds in self.datasets.values()])

        @app.route("/datasets/<name>/sample", methods=["GET"])
        def dataset_sample(name):
            ds = self.datasets.get(name)
            if ds is None:
                return jsonify({"error": f"Dataset '{name}' not found"}), 404
            n = request.args.get("n", 8, type=int)
            samples = ds.sample(n)
            # Convert to base64 PNG images
            images_b64 = []
            for i in range(samples.shape[0]):
                img = samples[i]
                images_b64.append(_tensor_to_base64(img))
            return jsonify({"images": images_b64})

        @app.route("/datasets/load_builtin", methods=["POST"])
        def load_builtin_dataset():
            data = request.get_json()
            name = data.get("name", "mnist")
            image_size = data.get("image_size", 64)
            if name == "mnist":
                from acc.dataset import load_mnist

                ds = load_mnist(image_size=image_size)
                self.datasets[ds.name] = ds
                return jsonify(ds.describe())
            return jsonify({"error": f"Unknown builtin dataset: {name}"}), 400

        # -- Tasks --
        @app.route("/tasks", methods=["GET"])
        def list_tasks():
            result = []
            for task in self.tasks.values():
                info = task.describe()
                # Include latest eval metrics if available
                result.append(info)
            return jsonify(result)

        @app.route("/tasks/add", methods=["POST"])
        def add_task():
            if self.autoencoder is None:
                return jsonify({"error": "No model loaded"}), 400

            data = request.get_json()
            class_name = data.get("class_name")
            task_name = data.get("name")
            dataset_name = data.get("dataset_name")
            weight = data.get("weight", 1.0)

            dataset = self.datasets.get(dataset_name)
            if dataset is None:
                return jsonify({"error": f"Dataset '{dataset_name}' not found"}), 400

            # Resolve task class
            task_class = _resolve_task_class(class_name)
            if task_class is None:
                return jsonify({"error": f"Unknown task class: {class_name}"}), 400

            try:
                task = task_class(task_name, dataset, weight=weight)
                task.attach(self.autoencoder)
            except TaskError as e:
                return jsonify({"error": str(e)}), 400

            self.tasks[task_name] = task
            self._rebuild_trainer()
            return jsonify(task.describe())

        @app.route("/tasks/<name>/toggle", methods=["POST"])
        def toggle_task(name):
            task = self.tasks.get(name)
            if task is None:
                return jsonify({"error": f"Task '{name}' not found"}), 404
            task.enabled = not task.enabled
            return jsonify({"name": name, "enabled": task.enabled})

        @app.route("/tasks/<name>/remove", methods=["POST"])
        def remove_task(name):
            task = self.tasks.pop(name, None)
            if task is None:
                return jsonify({"error": f"Task '{name}' not found"}), 404
            self._rebuild_trainer()
            return jsonify({"removed": name})

        # -- Training / Jobs --
        @app.route("/train/start", methods=["POST"])
        def train_start():
            if self.trainer is None:
                return jsonify(
                    {"error": "No trainer configured. Add model and tasks first."}
                ), 400

            data = request.get_json() or {}
            steps = data.get("steps", 500)
            lr = data.get("lr", self.trainer.lr)
            probe_lr = data.get("probe_lr", self.trainer.probe_lr)

            # Update learning rates if changed
            if lr != self.trainer.lr or probe_lr != self.trainer.probe_lr:
                self.trainer.lr = lr
                self.trainer.probe_lr = probe_lr
                self.trainer._build_optimizers()

            job = self.jobs.start(self.trainer, steps=steps, blocking=False)
            return jsonify(job.to_dict())

        @app.route("/train/stop", methods=["POST"])
        def train_stop():
            if self.trainer is None:
                return jsonify({"error": "No trainer"}), 400
            self.trainer.stop()
            job = self.jobs.current()
            if job is not None:
                job.state = "stopped"
                return jsonify(job.to_dict())
            return jsonify({"status": "no running job"})

        # -- Jobs --
        @app.route("/jobs", methods=["GET"])
        def list_jobs():
            return jsonify([j.to_dict() for j in self.jobs.list()])

        @app.route("/jobs/current", methods=["GET"])
        def current_job():
            job = self.jobs.current()
            if job is None:
                return jsonify(None)
            return jsonify(job.to_dict())

        @app.route("/jobs/<job_id>", methods=["GET"])
        def get_job(job_id):
            job = self.jobs.get(job_id)
            if job is None:
                return jsonify({"error": f"Job '{job_id}' not found"}), 404
            return jsonify(job.to_dict())

        @app.route("/jobs/<job_id>/stream", methods=["GET"])
        def stream_job(job_id):
            from_step = request.args.get("from_step", 0, type=int)

            def generate():
                for step_data in self.jobs.stream(job_id, from_step=from_step):
                    yield f"data: {json.dumps(step_data)}\n\n"
                yield 'data: {"done": true}\n\n'

            return Response(generate(), mimetype="text/event-stream")

        # -- Evaluation --
        @app.route("/eval/run", methods=["POST"])
        def run_eval():
            if self.trainer is None:
                return jsonify({"error": "No trainer"}), 400
            results = self.trainer.evaluate_all()
            return jsonify(results)

        # -- Checkpoints --
        @app.route("/checkpoints", methods=["GET"])
        def list_checkpoints():
            if self.checkpoints is None:
                return jsonify([])
            return jsonify([cp.to_dict() for cp in self.checkpoints.tree()])

        @app.route("/checkpoints/save", methods=["POST"])
        def save_checkpoint():
            if (
                self.autoencoder is None
                or self.trainer is None
                or self.checkpoints is None
            ):
                return jsonify({"error": "No model/trainer/checkpoint store"}), 400
            data = request.get_json() or {}
            tag = data.get("tag", "checkpoint")
            cp = self.checkpoints.save(self.autoencoder, self.trainer, tag=tag)
            return jsonify(cp.to_dict())

        @app.route("/checkpoints/load", methods=["POST"])
        def load_checkpoint():
            if (
                self.autoencoder is None
                or self.trainer is None
                or self.checkpoints is None
            ):
                return jsonify({"error": "No model/trainer/checkpoint store"}), 400
            data = request.get_json()
            cp_id = data.get("id")
            if not cp_id:
                return jsonify({"error": "Missing checkpoint id"}), 400
            try:
                cp = self.checkpoints.load(cp_id, self.autoencoder, self.trainer)
                return jsonify(cp.to_dict())
            except FileNotFoundError as e:
                return jsonify({"error": str(e)}), 404

        @app.route("/checkpoints/fork", methods=["POST"])
        def fork_checkpoint():
            if self.checkpoints is None:
                return jsonify({"error": "No checkpoint store"}), 400
            data = request.get_json()
            cp_id = data.get("id")
            new_tag = data.get("new_tag", "fork")
            try:
                cp = self.checkpoints.fork(cp_id, new_tag)
                return jsonify(cp.to_dict())
            except KeyError as e:
                return jsonify({"error": str(e)}), 404

        # -- Registry (what's available) --
        @app.route("/registry/tasks", methods=["GET"])
        def registry_tasks():
            """List available task classes."""
            from acc.tasks.classification import ClassificationTask
            from acc.tasks.reconstruction import ReconstructionTask
            from acc.tasks.regression import RegressionTask

            classes = [
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
            return jsonify(classes)

        @app.route("/registry/layers", methods=["GET"])
        def registry_layers():
            return jsonify(
                [
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
            )

        # -- Health --
        @app.route("/health", methods=["GET"])
        def health():
            return jsonify(
                {
                    "status": "ok",
                    "has_model": self.autoencoder is not None,
                    "num_tasks": len(self.tasks),
                    "num_datasets": len(self.datasets),
                    "device": str(self.device),
                }
            )

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

    def run(self, host: str = "0.0.0.0", port: int = 8787):
        """Start the HTTP server."""
        self.app.run(host=host, port=port, threaded=True)


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
