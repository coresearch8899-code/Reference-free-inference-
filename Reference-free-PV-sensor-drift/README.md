# Reference-free inference in multi-zone irradiance measurement networks

Code and data for the manuscript *"Reference-free inference in multi-zone irradiance measurement networks:
identifiability, threshold reliability and decision bias"* (T. Saengsuwan, C. Naunchan).

## Availability
This repository holds the code only. The site data and reference outputs are available from the corresponding author on reasonable request (thanapon.sae@crru.ac.th); place them in `data/` and `results/` before running. `data/DATA_DICTIONARY.md` describes the files. External data are public, and `code/optional_fetch_nist.py` rebuilds the NIST file.

## Run
```bash
pip install -r requirements.txt
cd code
python run_all.py --skip-fit   # reuses fitted models in ../results
python run_all.py              # also refits step 01 (1-2 h)
```

## Steps
| Script | Purpose | Manuscript |
|---|---|---|
| ss_data.py, ss_model.py, detector.py, injection.py | daily panel; state-space model; calibrated three-state detector | Sections 2, 4 |
| 01_fit_state_space.py | maximum-likelihood fit, main model and variants (supplementary) | Tables S1, S2 |
| 02_validate_state_space.py | semi-synthetic test of the estimator | Table S3 |
| 03_recalibration_policies.py | online state-space alarm, recalibration scenarios | Table S8 |
| 04_posterior_results.py | sensor and array posteriors, interventions | Section 7, Table S9 |
| 05_noise_structure.py | noise of every signal, both bounds | Table S15 |
| 06_calibrated_injection_online.py | calibrated injection, fault scenarios, repeated looks, monitoring with frozen thresholds | Tables 3–4, Figs 2, 5c |
| 06b_step_handling_full_process.py | step treatments, whole procedure in each replicate | Tables 5–6 |
| 06c_block_sensitivity.py | bootstrap block length | Table S14 |
| 07_baselines_dependence.py | single-reference baselines, dependence | Fig 3, Section 6.5 |
| 08_nist_descriptive.py | steps and descriptive analysis of the external network | Fig 4a |
| 09_nist_forward_evaluation.py | forward evaluation, nominal and calibrated | Table 7, Fig 4b |
| 10_reset_criterion_cleaning_bias.py | recovery ratio, satellite check, interval bias | Section 7, Fig 6, Table S16 |
| 11_sensitivity_coverage.py | reset sensitivity, cleaning inputs, coverage by stratum, hyper-parameters | Tables S1, S4–S7 |
| 12_figures.py | figures and graphical abstract | Figs 1–6 |
| optional_fetch_nist.py | rebuilds data/nist_daily.csv from the public archive (about 1 h) | – |

Seeds are fixed in the scripts. `results/*.pkl` are pickles written by step 01; load them only from this repository.

## External data
PVDAQ (CC BY 4.0, doi:10.25984/1846021); NIST campus arrays (doi:10.18434/M3S67G); NASA POWER (CERES SYN1deg).

Correspondence: Thanapon Saengsuwan, thanapon.sae@crru.ac.th, ORCID 0000-0003-1283-5454.
