#!/usr/bin/env python3
"""Seal the *identity* of the fragility monitor's signals: the gauge labels,
sub-labels, and the analyst note. The raw frag/score numbers stay public
because the QQQ/QLD dashboards compute V/T from them and bare numbers don't
reveal which signal is which. Encryption: PBKDF2-SHA256 (200k) -> AES-256-GCM,
key from the MONITOR_PASSPHRASE env var / GitHub secret.

Run AFTER the data build writes data.json and BEFORE the commit step.
Fails closed: no passphrase -> non-zero exit, nothing published.

    python seal_current.py [path/to/data.json]
"""
import json, os, sys, base64
try:
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except ImportError:
    sys.exit("seal_current: run `pip install cryptography` (add it to the workflow)")

ITER = 200_000
PASS = os.environ.get("MONITOR_PASSPHRASE")
if not PASS:
    sys.exit("seal_current: MONITOR_PASSPHRASE not set - refusing to publish plaintext")

path = sys.argv[1] if len(sys.argv) > 1 else "data.json"
d = json.load(open(path, encoding="utf-8"))
if "sealed" in d and "commentary" not in d:
    print("seal_current: already sealed, nothing to do"); raise SystemExit(0)

cur = d.get("current") or {}
def split(arr):
    labs = []
    for r in arr:
        labs.append({"label": r.get("label"), "sub": r.get("sub")})
        r.pop("label", None); r.pop("sub", None)   # keep frag/score, drop identity
    return labs
payload = json.dumps({"vulnerability": split(cur.get("vulnerability", [])),
                      "trigger":       split(cur.get("trigger", [])),
                      "commentary":    d.get("commentary")},
                     separators=(",", ":")).encode("utf-8")
salt, iv = os.urandom(16), os.urandom(12)
key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=ITER).derive(PASS.encode())
ct = AESGCM(key).encrypt(iv, payload, None)      # ciphertext || 16-byte GCM tag
d["sealed"] = {"v": 1, "iter": ITER,
               "salt": base64.b64encode(salt).decode(),
               "iv":   base64.b64encode(iv).decode(),
               "ct":   base64.b64encode(ct).decode()}
d["current"] = cur                # frag/score only
d.pop("commentary", None)
json.dump(d, open(path, "w", encoding="utf-8"), separators=(",", ":"), ensure_ascii=False)
print("seal_current: sealed gauge labels + commentary; frag/score kept public")
