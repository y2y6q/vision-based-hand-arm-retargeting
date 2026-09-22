# dex-retargeting 0.4.6 runtime wheel

This directory contains the unmodified official `dex_retargeting-0.4.6-py3-none-any.whl` wheel used by the project-local external solver process.

- Distribution: `dex-retargeting==0.4.6`
- Source: https://pypi.org/project/dex-retargeting/0.4.6/
- License: MIT
- SHA-256: `cf08b93e204af21b7146f12a83d55f0a0d227f991d202655c475535621bb62f7`

The robosuite process does not import this wheel directly. It starts the configured isolated dex Python process, which imports this pure-Python wheel together with that environment's native Pinocchio and NLopt libraries. This prevents mixed native DLLs in MuJoCo / robosuite's interpreter.
