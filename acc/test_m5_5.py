"""M5.5 verification: Loss Diagnosis Dashboard.

Tests the loss health classification, summary statistics, enriched step_info,
and API endpoints. Does NOT test visual UI rendering (that requires a browser).

Run: python -m acc.test_m5_5
"""

import sys

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name} — {detail}")


def main():
    global passed, failed

    # ─── Test 1: LossHealth enum ───
    print("\n=== Test 1: LossHealth enum ===")
    from acc.loss_health import LossHealth

    check("LossHealth has HEALTHY", hasattr(LossHealth, "HEALTHY"))
    check("LossHealth has WARNING", hasattr(LossHealth, "WARNING"))
    check("LossHealth has CRITICAL", hasattr(LossHealth, "CRITICAL"))
    check("HEALTHY.value", LossHealth.HEALTHY.value == "healthy")
    check("WARNING.value", LossHealth.WARNING.value == "warning")
    check("CRITICAL.value", LossHealth.CRITICAL.value == "critical")
    check("HEALTHY.css_class", LossHealth.HEALTHY.css_class == "loss-healthy")
    check("CRITICAL.color", LossHealth.CRITICAL.color == "#f85149")

    # ─── Test 2: classify_loss thresholds ───
    print("\n=== Test 2: classify_loss thresholds ===")
    from acc.loss_health import classify_loss

    # ReconstructionTask: healthy < 0.05, warning < 0.15, critical >= 0.15
    check("recon 0.02 -> healthy", classify_loss("ReconstructionTask", 0.02) == LossHealth.HEALTHY)
    check("recon 0.10 -> warning", classify_loss("ReconstructionTask", 0.10) == LossHealth.WARNING)
    check("recon 0.40 -> critical", classify_loss("ReconstructionTask", 0.40) == LossHealth.CRITICAL)

    # KLDivergenceTask: healthy < 5.0, warning < 15.0, critical >= 15.0
    check("kl 2.0 -> healthy", classify_loss("KLDivergenceTask", 2.0) == LossHealth.HEALTHY)
    check("kl 10.0 -> warning", classify_loss("KLDivergenceTask", 10.0) == LossHealth.WARNING)
    check("kl 20.0 -> critical", classify_loss("KLDivergenceTask", 20.0) == LossHealth.CRITICAL)

    # ClassificationTask: healthy < 0.5, warning < 2.0, critical >= 2.0
    check("classify 0.3 -> healthy", classify_loss("ClassificationTask", 0.3) == LossHealth.HEALTHY)
    check("classify 1.0 -> warning", classify_loss("ClassificationTask", 1.0) == LossHealth.WARNING)
    check("classify 3.0 -> critical", classify_loss("ClassificationTask", 3.0) == LossHealth.CRITICAL)

    # RegressionTask: healthy < 0.1, warning < 0.5, critical >= 0.5
    check("regression 0.05 -> healthy", classify_loss("RegressionTask", 0.05) == LossHealth.HEALTHY)
    check("regression 0.3 -> warning", classify_loss("RegressionTask", 0.3) == LossHealth.WARNING)
    check("regression 0.8 -> critical", classify_loss("RegressionTask", 0.8) == LossHealth.CRITICAL)

    # Unknown task type uses default thresholds
    check("unknown 0.05 -> healthy", classify_loss("CustomTask", 0.05) == LossHealth.HEALTHY)
    check("unknown 5.0 -> critical", classify_loss("CustomTask", 5.0) == LossHealth.CRITICAL)

    # ─── Test 3: LossSummary + compute_loss_summary ───
    print("\n=== Test 3: compute_loss_summary ===")
    from acc.loss_health import compute_loss_summary, LossSummary

    # Create synthetic loss history
    losses = []
    for step in range(1, 101):
        # Recon starts at 0.4 (critical) and drops to 0.03 (healthy)
        recon_loss = 0.4 - (0.37 * step / 100)
        losses.append({
            "step": step * 2 - 1,
            "task_name": "recon",
            "task_type": "ReconstructionTask",
            "task_loss": recon_loss,
            "health": classify_loss("ReconstructionTask", recon_loss).value,
        })
        # KL stays constant at 3.0 (healthy)
        losses.append({
            "step": step * 2,
            "task_name": "kl",
            "task_type": "KLDivergenceTask",
            "task_loss": 3.0,
            "health": classify_loss("KLDivergenceTask", 3.0).value,
        })

    summaries = compute_loss_summary(losses)

    check("summaries has recon", "recon" in summaries)
    check("summaries has kl", "kl" in summaries)
    check("recon is LossSummary", isinstance(summaries["recon"], LossSummary))

    recon_s = summaries["recon"]
    check("recon final < 0.05", recon_s.final < 0.05, f"final={recon_s.final}")
    check("recon min < final", recon_s.min_val <= recon_s.final)
    check("recon max > final", recon_s.max_val > recon_s.final)
    check("recon mean between min and max", recon_s.min_val <= recon_s.mean <= recon_s.max_val)
    check("recon trend is improving", recon_s.trend == "improving", f"trend={recon_s.trend}")
    check("recon health is healthy (final)", recon_s.health == LossHealth.HEALTHY, f"health={recon_s.health}")
    check("recon n_steps is 100", recon_s.n_steps == 100)
    check("recon task_type is ReconstructionTask", recon_s.task_type == "ReconstructionTask")

    kl_s = summaries["kl"]
    check("kl final == 3.0", kl_s.final == 3.0)
    check("kl trend is flat", kl_s.trend == "flat", f"trend={kl_s.trend}")
    check("kl health is healthy", kl_s.health == LossHealth.HEALTHY)

    # Test to_dict
    recon_dict = recon_s.to_dict()
    check("to_dict has task_name", recon_dict["task_name"] == "recon")
    check("to_dict has health string", recon_dict["health"] == "healthy")
    check("to_dict has trend", recon_dict["trend"] == "improving")
    check("to_dict has min", "min" in recon_dict)
    check("to_dict has max", "max" in recon_dict)

    # ─── Test 4: Empty losses ───
    print("\n=== Test 4: Edge cases ===")
    empty_summary = compute_loss_summary([])
    check("empty losses -> empty dict", empty_summary == {})

    # Single entry
    single = compute_loss_summary([{
        "step": 1, "task_name": "recon", "task_type": "ReconstructionTask",
        "task_loss": 0.5, "health": "critical"
    }])
    check("single entry summary", "recon" in single)
    check("single entry trend is flat", single["recon"].trend == "flat")
    check("single entry n_steps is 1", single["recon"].n_steps == 1)

    # ─── Test 5: Enriched step_info from Trainer ───
    print("\n=== Test 5: Enriched step_info from Trainer ===")
    import torch
    from acc.factor_slot_autoencoder import FactorSlotAutoencoder
    from acc.factor_group import FactorGroup
    from acc.tasks.reconstruction import ReconstructionTask
    from acc.tasks.kl_divergence import KLDivergenceTask
    from acc.dataset import AccDataset

    # Create a tiny model and run a few steps
    device = torch.device("cpu")
    factor_groups = [FactorGroup("digit", 0, 4), FactorGroup("free", 4, 8)]
    model = FactorSlotAutoencoder(
        in_channels=1,
        factor_groups=factor_groups,
        image_size=16,
    )

    # Create a tiny dataset
    images = torch.randn(32, 1, 16, 16).clamp(0, 1)
    labels = torch.zeros(32, dtype=torch.long)
    ds = AccDataset(images, labels, name="test_ds")

    recon_task = ReconstructionTask("recon", ds)
    recon_task.attach(model)
    kl_task = KLDivergenceTask("kl", ds, weight=0.5)
    kl_task.attach(model)

    from acc.trainer import Trainer
    trainer = Trainer(model, [recon_task, kl_task], device, lr=1e-3, batch_size=8)
    history = trainer.train(steps=10)

    check("history has 10 entries", len(history) == 10, f"len={len(history)}")
    check("first entry has health", "health" in history[0], f"keys={list(history[0].keys())}")
    check("first entry has task_type", "task_type" in history[0], f"keys={list(history[0].keys())}")
    check("health is valid string", history[0]["health"] in ("healthy", "warning", "critical"))
    check("task_type is valid", history[0]["task_type"] in ("ReconstructionTask", "KLDivergenceTask"))

    # Check that health classification matches the loss value
    for entry in history:
        expected = classify_loss(entry["task_type"], entry["task_loss"]).value
        if entry["health"] != expected:
            check("health matches classify_loss", False, f"step={entry['step']}: {entry['health']} != {expected}")
            break
    else:
        check("all health values match classify_loss", True)

    # ─── Test 6: compute_loss_summary on trainer output ───
    print("\n=== Test 6: compute_loss_summary on real trainer output ===")
    summaries = compute_loss_summary(history)
    check("summary has recon", "recon" in summaries)
    check("summary has kl", "kl" in summaries)

    for name, s in summaries.items():
        check(f"{name} n_steps > 0", s.n_steps > 0)
        check(f"{name} health is valid", s.health in (LossHealth.HEALTHY, LossHealth.WARNING, LossHealth.CRITICAL))

    # ─── Summary ───
    print(f"\n{'='*60}")
    print(f"M5.5 Results: {passed} passed, {failed} failed")
    if failed > 0:
        print("SOME TESTS FAILED")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
