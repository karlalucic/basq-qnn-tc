"""Preregistration for the 300-material fresh-holdout QPU run.

Run BEFORE final_qpu_submit.py and commit the outputs. The holdout
(data/fresh_holdout_pls.npz) has never been used for training, model
selection, or tuning; this script freezes the protocol and the ideal
predictions so the hardware result can be verified against a public
commit that predates the job.

Outputs:
  data/ideal_evs300_holdout.npy   exact statevector predictions, champion weights
  HOLDOUT_PREREG.md               protocol + SHA-256 hashes + ideal RMSE
"""

import hashlib
from datetime import date

import numpy as np

from baselines import rmse_kelvin
from qnn import make_qnn

WEIGHTS = "data/trained_weights_q4_l2_pls.npy"
HOLDOUT = "data/fresh_holdout_pls.npz"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


d = dict(np.load(HOLDOUT))
w = np.load(WEIGHTS)

qnn, _ = make_qnn(4, 2, 4)
ideal = np.asarray(qnn.forward(d["X_hold"], w)).ravel()
np.save("data/ideal_evs300_holdout.npy", ideal)

rm, ma = rmse_kelvin(d["y_hold"], ideal, d)
print(f"ideal statevector on 300-material holdout: RMSE {rm:.2f} K  MAE {ma:.2f} K")

md = f"""# Preregistration: fresh-holdout hardware run

Date committed: {date.today().isoformat()}

The 300 materials in `data/fresh_holdout_pls.npz` were split off during data
preparation and have never influenced training, architecture selection, or
tuning. Before submitting the hardware job we commit the exact protocol, the
input hashes, and the ideal predictions. The job ID and hardware result will
be added below after the run, whatever the outcome.

## Protocol (frozen)

- Model: champion QNN, 4 qubits x 2 layers, 32 parameters, PLS-4 inputs
  (`data/trained_weights_q4_l2_pls.npy`)
- Backend: `ibm_fez`, tiled inference, K <= 25 calibration-ranked buffered
  tiles, one EstimatorV2 PUB, 300 points -> ceil(300/K) parameter rows
- Shots 2048; resilience_level 1 (TREX), dynamical decoupling XY4, gate
  twirling; identical options to the recorded July 11 tiled run
- Submitted in the same Batch as the 200-point test-set job, so both share
  one calibration
- Comparison: hardware RMSE vs the ideal statevector RMSE below, computed
  and committed before submission

## Frozen inputs

| artifact | SHA-256 |
|---|---|
| `data/fresh_holdout_pls.npz` | `{sha256(HOLDOUT)}` |
| `data/trained_weights_q4_l2_pls.npy` | `{sha256(WEIGHTS)}` |
| `data/ideal_evs300_holdout.npy` | `{sha256('data/ideal_evs300_holdout.npy')}` |

Ideal statevector result on the holdout: **RMSE {rm:.2f} K, MAE {ma:.2f} K**.

## Hardware result

To be filled in after the job completes: job ID, RMSE, MAE, QPU seconds.
"""

with open("HOLDOUT_PREREG.md", "w") as f:
    f.write(md)
print("wrote HOLDOUT_PREREG.md")
