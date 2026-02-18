"""Test: Does the training guard have gaps during recipe execution?

HYPOTHESIS: During a recipe, the _training_guard() only checks
self.jobs.current() (i.e., whether a JobManager job is running).
Between sequential ctx.train() calls in a recipe, the job completes
and _current_job_id is set to None — but the recipe thread is still
doing CUDA work (evaluate, checkpoint, create_model, fork). During
this window, eval endpoints pass the guard and hit the model on the
async thread, causing concurrent CUDA access.

TEST: Run a recipe while polling _is_training() and recipe_runner.current().
Log the gaps where _is_training() returns False but the recipe is still running.
If gaps exist, any eval endpoint could slip through during those gaps.

PREDICTION: If hypothesis is correct, we will see windows where:
  - recipe_runner.current().state == "running"  (recipe IS active)
  - jobs.current() is None                       (training guard would NOT block)

This proves the guard is insufficient without needing to crash anything.
"""

import threading
import time
import sys
import os

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")


def test_guard_gap():
    """Detect windows where recipe is running but training guard would not fire."""
    from acc.trainer_api import TrainerAPI

    api = TrainerAPI()

    # We need a simple recipe that does train -> evaluate -> train cycles
    # Use the actual recipe if available, or build a minimal one
    from acc.recipes.base import Recipe, RecipeContext
    from acc.conv_vae import ConvVAE
    from acc.dataset import load_mnist
    from acc.tasks.reconstruction import ReconstructionTask
    from acc.tasks.kl_divergence import KLDivergenceTask

    class GapTestRecipe(Recipe):
        name = "gap_test"
        description = "Minimal recipe that does train/eval cycles to expose guard gaps"

        def run(self, ctx: RecipeContext):
            # Branch 1
            ctx.phase = "branch_1"
            ctx.create_model(lambda: ConvVAE(latent_dim=8))
            ds = ctx.load_dataset("mnist", load_mnist)
            ctx.attach_task(ReconstructionTask("recon", dataset=ds))
            ctx.attach_task(KLDivergenceTask("kl", dataset=ds))
            ctx.train(steps=50)  # Short training
            # --- GAP HERE: job completed, but recipe thread still running ---
            ctx.log("Branch 1: evaluating (CUDA work on recipe thread)")
            m = ctx.evaluate()   # CUDA forward passes
            ctx.log(f"Branch 1 eval: {m}")
            ctx.save_checkpoint("branch_1")
            # Simulate some CUDA work (loading checkpoint state)
            time.sleep(0.5)

            # Branch 2
            ctx.phase = "branch_2"
            ctx.create_model(lambda: ConvVAE(latent_dim=16))
            ctx.attach_task(ReconstructionTask("recon", dataset=ds))
            ctx.attach_task(KLDivergenceTask("kl", dataset=ds))
            ctx.train(steps=50)
            # --- GAP HERE again ---
            ctx.log("Branch 2: evaluating (CUDA work on recipe thread)")
            m = ctx.evaluate()
            ctx.log(f"Branch 2 eval: {m}")
            ctx.save_checkpoint("branch_2")

    # Monitor thread: rapidly polls both guards
    gaps_detected = []
    monitor_stop = threading.Event()

    def monitor():
        """Poll _is_model_busy and recipe state at high frequency."""
        poll_count = 0
        while not monitor_stop.is_set():
            recipe_job = api.recipe_runner.current()

            recipe_running = recipe_job is not None and recipe_job.state == "running"
            # This is what _is_model_busy() checks now (the fix)
            guard_would_block = api._is_model_busy()

            if recipe_running and not guard_would_block:
                gaps_detected.append({
                    "time": time.time(),
                    "poll_count": poll_count,
                    "recipe_phase": recipe_job.current_phase if recipe_job else None,
                })

            poll_count += 1
            time.sleep(0.005)  # 200Hz polling

    # Start monitor
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()

    # Start recipe
    recipe = GapTestRecipe()
    print("Starting gap test recipe...")
    recipe_job = api.recipe_runner.start(recipe, api)

    # Wait for recipe to finish
    api.recipe_runner.wait(timeout=120)
    monitor_stop.set()
    monitor_thread.join(timeout=2)

    # Report results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    if gaps_detected:
        print(f"\nGAPS DETECTED: {len(gaps_detected)} polling intervals where:")
        print("  - Recipe thread IS running (doing CUDA work)")
        print("  - Training guard would NOT block (jobs.current() is None)")
        print(f"\nSample gaps:")
        # Group by phase
        phases = {}
        for g in gaps_detected:
            p = g["recipe_phase"]
            if p not in phases:
                phases[p] = 0
            phases[p] += 1
        for phase, count in phases.items():
            print(f"  Phase '{phase}': {count} gap polls detected")

        print(f"\n>>> HYPOTHESIS CONFIRMED: Training guard has gaps during recipe execution.")
        print(f">>> Any eval endpoint called during these gaps will access CUDA")
        print(f">>> concurrently with the recipe thread, causing context corruption.")
    else:
        print("\nNo gaps detected.")
        print(">>> HYPOTHESIS REJECTED: Training guard covers all recipe CUDA work.")

    return len(gaps_detected) > 0


if __name__ == "__main__":
    result = test_guard_gap()
    sys.exit(0 if result else 1)
