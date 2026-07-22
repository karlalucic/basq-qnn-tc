# Preregistration: fresh-holdout hardware run

Date committed: 2026-07-22

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
| `data/fresh_holdout_pls.npz` | `6dfa53cbc330c305fe19cc2513bb4b89c62f93ec4793adb3049264eddb0c0500` |
| `data/trained_weights_q4_l2_pls.npy` | `2125c5e96620c44aaf0c3eb336ee5f74d27a0bdc835be47818d09887e4450433` |
| `data/ideal_evs300_holdout.npy` | `b0edc63ac631007d5d9d463944c9ecc5a93b98e1e6815c508596cf1afed1590b` |

Ideal statevector result on the holdout: **RMSE 21.30 K, MAE 14.84 K**.

## Hardware result

To be filled in after the job completes: job ID, RMSE, MAE, QPU seconds.
