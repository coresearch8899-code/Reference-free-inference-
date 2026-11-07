# -*- coding: utf-8 -*-
"""Step F12: does a longer bootstrap block repair the false-alarm rate when thresholds are learned from one development period?
N = 3, 18 months of development and 18 months of evaluation data, no fault. Same noise for every block length within a replicate."""
import os
import numpy as np
import pandas as pd
from detector import loo, remove_season, step_search, pipeline, calibrate, decide
from injection import RES, SUF, DEV, X, noise

rng = np.random.default_rng(20261112)
FP = int(os.environ.get("FP", 300)); FPB = int(os.environ.get("FPB", 300)); test = 78; BL = (13, 26, 39); sc = np.ones(3)
rec = []
for i in range(FP):
    L = noise(DEV + test, 3, sc, rng)
    Cd, _ = step_search(remove_season(loo(L[:DEV]), X[:DEV], DEV))
    E = Cd - Cd.mean(0)
    r = pipeline(loo(L), DEV, test, X)
    for b in BL:
        q = calibrate(E, DEV, test, X, rng, B=FPB, block=b)
        rec.append(dict(block=b, threshold=q["s"], alarm=int(decide(r, q)[0] > 0)))
R = pd.DataFrame(rec)
S = R.groupby("block").agg(replicates=("alarm", "size"), false_alarm=("alarm", "mean"), threshold_mean=("threshold", "mean")).reset_index()
S.to_csv(os.path.join(RES, f"full_process_block{SUF}.csv"), index=False)
print(S.round(3).to_string())
