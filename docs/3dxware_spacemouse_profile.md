# 3DxWare profile for robosuite SpaceMouse teleoperation

This project reads the SpaceMouse directly through `hidapi`.  3DxWare can
still independently turn device events into virtual mouse-wheel, radial-menu,
or driver-window events.  The two paths are separate.

The local installation was inspected on 2026-09-07:

- 3Dconnexion 3DxWare 10: `10.9.14.745`
- 3DxWinCore: `17.9.14.3006`
- install root: `E:\3DxWare`
- device: SpaceMouse Wireless `256F:C62E`
- user profile directory: `%APPDATA%\3Dconnexion\3DxWare\Cfg`

The installed `AppDefCfg_KMJ.xml` enables `HIDMultiAxis_Rx` and maps it to
`HIDMouse_Wheel` with reversed direction.  C62E's left and right physical
buttons are parented to `V3DK_MENU_1` and `V3DK_MENU_2`, which the default
profile maps to radial menus.  This is the observed source of scrolling and
driver menus; the teleoperation Python code does not generate those events.

## Read-only inspection

From the repository root, run:

```powershell
& .\.venv-robosuite\Scripts\python.exe .\scripts\manage_3dxware_profile.py inspect --report .\outputs\3dxware_profile\inspection.json
```

This command reads the service-state XML and installed configuration. It does
not change 3DxWare, restart a process, or touch the SpaceMouse device.

## Optional reversible profile installation

The tool creates only user-local profile files and always makes a manifest
backup before writing. It follows the installed driver's observed naming
convention (`python-KMJ.xml`, `pycharm64-KMJ.xml`) so 3DxWare can discover the
profile by executable; its XML ID and contents remain project specific. It
never writes under `E:\3DxWare`, stops a service, or changes another
application profile.

For the GLFW MuJoCo window, use `python.exe` only after accepting its scope:
3DxWare matches profiles by executable name, so this suppresses virtual
3DxWare mouse/menu output while *any* `python.exe` GUI is foreground. It does
not affect ordinary non-Python applications.

The project wrapper covers the two locally observed hosts, `python.exe` and
`pycharm64.exe`:

```powershell
& .\scripts\install_3dxware_spacemouse_profile.ps1
```

The command prints a manifest path below
`outputs\3dxware_profiles\backups`. To restore exactly the prior state:

```powershell
& .\scripts\restore_3dxware_spacemouse_profile.ps1 -Manifest <printed-manifest-path>
```

The restore refuses to overwrite a profile modified after installation unless
`--force` is provided. It removes a generated profile only if its contents
still match the recorded installed digest.

The generated C62E profile disables all six 3DxWare virtual axis mappings and
sets `V3DK_MENU_1`, `V3DK_MENU_2`, and their direct logical aliases to the
locally defined `Driver_Disabled` action. It does not alter the physical HID
reports consumed by `hidapi`; validate that behavior manually after refocusing
the target application.

## GUI fallback / verification procedure

Use this procedure when 3DxWare does not adopt the user-local XML immediately
or when a different foreground executable owns the viewer window.

1. Start the teleoperation process and click its MuJoCo viewer. In 3Dconnexion
   Settings, confirm that the active application is `python.exe`. If the viewer
   is embedded in PyCharm or a terminal, select that foreground executable
   instead.
2. Select **SpaceMouse Wireless**. In the axis assignments, disable every
   virtual axis assignment. In particular, remove the installed default
   `HIDMultiAxis_Rx -> HIDMouse_Wheel` mapping.
3. In button assignments, set the left button (`V3DK_MENU_1`) and right button
   (`V3DK_MENU_2`) to **Disabled**. Do not assign a radial menu, macro,
   click, driver window, or application action.
4. Save the application profile, refocus the viewer, then run the project
   SpaceMouse diagnostic. Confirm six raw HID axes still change, the buttons
   only appear in diagnostic logs, and no other window scrolls or shows a menu.
5. If the result is wrong, use the restore command above or reset only this
   application profile in 3Dconnexion Settings. Do not disable or uninstall
   3DxWare globally.

The driver and hardware effects remain **pending manual verification** until a
person moves the device and presses both buttons with the target window focused.
