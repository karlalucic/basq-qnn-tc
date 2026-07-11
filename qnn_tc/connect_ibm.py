"""IBM Quantum connection helper.

One-time setup (after you have an API key from quantum.cloud.ibm.com):

    .venv/bin/python connect_ibm.py save <API_KEY> [CRN]

Then verify and list backends:

    .venv/bin/python connect_ibm.py check

Accounts are stored in ~/.qiskit/qiskit-ibm.json. A second account for
the hackathon instance (ibm_basquecountry) can be saved under the name
"basq" once BasQ shares the instance CRN:

    .venv/bin/python connect_ibm.py save <API_KEY> <BASQ_CRN> basq
"""

import sys

from qiskit_ibm_runtime import QiskitRuntimeService


def save(token, instance=None, name="default"):
    kwargs = dict(channel="ibm_quantum_platform", token=token,
                  set_as_default=(name == "default"), name=name,
                  overwrite=True)
    if instance:
        kwargs["instance"] = instance
    QiskitRuntimeService.save_account(**kwargs)
    print(f"Saved account '{name}'. Run: connect_ibm.py check")


def check(name=None):
    service = QiskitRuntimeService(name=name) if name else QiskitRuntimeService()
    print("Connected. Backends visible to this account:")
    for b in service.backends():
        status = b.status()
        print(f"  {b.name:24s} {b.num_qubits:4d} qubits | "
              f"pending jobs: {status.pending_jobs} | "
              f"operational: {status.operational}")
    lb = service.least_busy(operational=True, simulator=False)
    print("Least busy:", lb.name)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "save":
        save(sys.argv[2],
             sys.argv[3] if len(sys.argv) > 3 else None,
             sys.argv[4] if len(sys.argv) > 4 else "default")
    elif len(sys.argv) >= 2 and sys.argv[1] == "check":
        check(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        print(__doc__)
