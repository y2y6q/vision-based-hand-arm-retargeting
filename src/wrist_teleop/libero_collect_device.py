"""The collection entry's SpaceMouse: the neutral learned at startup survives a layout reset.

``SpaceMouseHid.reset_input_state()`` runs on every new layout, M and task
switch. It drops the motion and buttons from before the reset (good), but it
also forgets the zero bias learned when the device opened. Learning that bias
again needs ``zero_bias_calibration_frames`` (8) motion reports in a row inside
the deadzone, and until then every sample maps to zero. The SpaceMouse
Wireless sends no reports at all while it rests (the 2026-09-17 diagnostic:
``valid`` false for 10 s untouched), and a push leaves the deadzone after one
or two reports, which empties the count again. So after each layout the arm
ignored the SpaceMouse until a few light touches happened to give 8 quiet
reports in a row: the 3-4 s the operator waited between attempts (2026-11-10).

The bias belongs to the device, not to the layout (``SpaceMouseHid`` never
adapts it once locked), so this class puts it back after the reset. Everything
else the reset does stays: old motion and buttons are dropped, the device is
disarmed, and it arms again ``neutral_startup_s`` (0.3 s) after the reset once
the cap is seen at rest. A cap that sends nothing counts as at rest; one still
pushed through the reset keeps reporting and arms only after it is let go.
A device that never locked a neutral (or was closed and opened again)
calibrates exactly as before.
"""

from __future__ import annotations

from .device import SpaceMouseHid


class KeepNeutralSpaceMouse(SpaceMouseHid):
    """``SpaceMouseHid`` whose ``reset_input_state()`` keeps a locked startup zero bias."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.kept_neutral_resets = 0  # resets that kept the startup neutral (for the run log)

    def open(self):
        self._zero_bias_locked = False  # an opened (or opened again) device always calibrates afresh
        return super().open()

    def reset_input_state(self) -> None:
        bias = self._zero_bias.copy()
        keep = bool(self._zero_bias_locked and self._device is not None)
        super().reset_input_state()
        if keep:
            self._zero_bias[:] = bias
            self._zero_bias_locked = True
            self._calibration_state = "neutral_locked"
            self.kept_neutral_resets += 1
