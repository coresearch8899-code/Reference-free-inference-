# -*- coding: utf-8 -*-
"""Run the analysis in order. --skip-fit reuses the fitted state-space models in ../results (step 01 takes 1-2 h)."""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
steps = ["01_fit_state_space.py", "02_validate_state_space.py", "03_recalibration_policies.py", "04_posterior_results.py",
         "05_noise_structure.py", "06_calibrated_injection_online.py", "06b_step_handling_full_process.py", "06c_block_sensitivity.py", "07_baselines_dependence.py", "08_nist_descriptive.py", "09_nist_forward_evaluation.py",
         "10_reset_criterion_cleaning_bias.py", "11_sensitivity_coverage.py", "12_figures.py"]
if "--skip-fit" in sys.argv:
    steps = steps[1:]
for s in steps:
    if s[:2] in ("06", "07"):                  # both noise bounds
        for mode in ("detrended", "raw"):
            print(f"\n=== {s} ({mode}) ===", flush=True)
            subprocess.run([sys.executable, os.path.join(HERE, s)], cwd=HERE, check=True, env=dict(os.environ, MODE=mode))
        continue
    print(f"\n=== {s} ===", flush=True)
    subprocess.run([sys.executable, os.path.join(HERE, s)], cwd=HERE, check=True)
print("\nAll steps finished.")
