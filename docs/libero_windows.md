# LIBERO on this Windows workstation

LIBERO is isolated from the robosuite mainline because the official release
uses Python 3.8 and robosuite 1.4.0. Do not install these packages into
`.venv-robosuite`.

The local environment is `.venv-libero` and currently contains Python 3.8.0,
LIBERO 0.1.0 from `third_party/LIBERO`, robosuite 1.4.0, MuJoCo 3.2.3, and CPU
PyTorch 1.11.0. The preview script supplies two Windows-only compatibility
assets inside that isolated environment: the MuJoCo DLL expected by
robosuite 1.4 and a WGL render macro. It does not alter the mainline venv.

Run an official deterministic scene preview and record it with:

```powershell
Set-Location "E:\hand_arm_retargeting"
& .\.venv-libero\Scripts\python.exe .\scripts\run_libero_task_preview.py --suite libero_spatial --task-id 0 --record .\outputs\libero\libero_spatial_task0_preview.mp4
```

The preview loads an official BDDL scene and its stored initial state, then
uses a 7-dimensional zero action only. It verifies scene loading and video
recording; it does not measure task success and it does not yet connect a
SpaceMouse. `outputs/libero/*.json` records this distinction.

`robomimic` imports successfully. `pip check` still reports missing
`egl-probe`: its package uses Unix `make -j` and cannot compile on this
Windows setup. This is a GPU-device probe for robomimic and does not block the
WGL offscreen preview above.
