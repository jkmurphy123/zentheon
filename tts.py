from __future__ import annotations
import subprocess
import shutil
from pathlib import Path
import shlex
import soundfile as sf
import numpy as np
from rich import print
from utils import load_config
from audio_io import play

cfg = load_config()


class PiperTTS:
    def __init__(self, voice_path: str, out_wav: str):
        self.voice = Path(voice_path)
        self.out_wav = Path(out_wav)
        if not self.voice.exists():
            raise FileNotFoundError(f"Piper voice not found: {self.voice}")
        # require 'piper' binary in PATH
        if not shutil.which("piper"):
            raise RuntimeError("'piper' binary not found in PATH. Install Piper.")

    def speak(self, text: str) -> None:
        print("[green]Synthesizing with Piper...[/green]")
        # Send text via stdin (avoids shell quoting issues)
        proc = subprocess.run(
            ["piper", "-m", str(self.voice), "-f", str(self.out_wav)],
            input=text,
            text=True,
            capture_output=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Piper failed: {proc.stderr}")

        audio, sr = sf.read(self.out_wav)
        if audio.ndim > 1:
            audio = audio[:, 0]
        play(audio.astype(np.float32), sr)


def sh_quote(s: str) -> str:
    # Kept for compatibility if you later switch back to a shell pipeline
    return shlex.quote(s)


def make_tts():
    p = cfg["tts"]["piper"]
    return PiperTTS(p["voice_path"], p["output_wav"])
