from __future__ import annotations
from pathlib import Path
from typing import Optional
import yaml
import sounddevice as sd
from rich import print

CONFIG_PATH = Path(__file__).parent / "config.yaml"

def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def resolve_device(sel: Optional[object]) -> Optional[int]:
    """Return a sounddevice device index for None/int/str selector.
    - None => use default device
    - int  => treat as device index
    - str  => substring match on device name (first match wins)
    """
    if sel is None:
        return None
    if isinstance(sel, int):
        return sel
    # substring match by name
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        name = dev.get("name", "")
        if sel.lower() in str(name).lower():
            return idx
    print(f"[yellow]Device selector '{sel}' not found. Using default.[/yellow]")
    return None
