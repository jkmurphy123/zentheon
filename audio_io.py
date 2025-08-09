from __future__ import annotations
import numpy as np
import sounddevice as sd
import soundfile as sf
from typing import Optional
from utils import load_config, resolve_device
from rich import print

cfg = load_config()
SR = int(cfg["audio"]["sample_rate"])
IN_DEV = resolve_device(cfg["audio"]["input_device"])   # may be None
OUT_DEV = resolve_device(cfg["audio"]["output_device"]) # may be None


def list_devices() -> None:
    print("\n[aqua]Audio devices[/aqua]:")
    for i, d in enumerate(sd.query_devices()):
        print(f"[{i:2d}] {d['name']}  (in:{d['max_input_channels']} out:{d['max_output_channels']})")


def record_seconds(seconds: float) -> np.ndarray:
    print(f"[green]Recording {seconds}s @ {SR} Hz...[/green]")
    audio = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="float32", device=IN_DEV)
    sd.wait()
    return audio.squeeze(-1)


def play(audio: np.ndarray, sr: Optional[int] = None) -> None:
    sr = int(sr or SR)
    print(f"[green]Playing audio ({len(audio)/sr:.2f}s) @ {sr} Hz...[/green]")
    sd.play(audio, sr, device=OUT_DEV)
    sd.wait()


def save_wav(path: str, audio: np.ndarray, sr: Optional[int] = None) -> None:
    sr = int(sr or SR)
    sf.write(path, audio, sr)
