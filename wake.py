from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional
from rich import print
from utils import load_config

cfg = load_config()

@dataclass
class WakeResult:
    keyword: str

class KeyboardWakeDetector:
    def __init__(self, keyword: str = "wake"):
        self.keyword = keyword
    def listen(self) -> WakeResult:
        input(f"Press [Enter] to simulate wake word ('{self.keyword}')...")
        return WakeResult(keyword=self.keyword)

class PorcupineWakeDetector:
    def __init__(self, keyword: str = "jarvis", access_key_env: str = "PORCUPINE_ACCESS_KEY", ppn_path: str | None = None, sensitivity: float = 0.6):
        try:
            import pvporcupine
            from pvrecorder import PvRecorder
        except Exception as e:
            raise RuntimeError("Porcupine not installed: pip install pvporcupine pvrecorder") from e
        self.pvporcupine = pvporcupine
        self.PvRecorder = PvRecorder
        self.keyword = keyword
        self.access_key = os.getenv(access_key_env)
        if not self.access_key:
            raise RuntimeError(f"Set env var {access_key_env} with your Picovoice AccessKey")
        # If a custom .ppn is provided, use it; otherwise fall back to built-in keyword
        if ppn_path and os.path.exists(ppn_path):
            self.handle = self.pvporcupine.create(
                access_key=self.access_key,
                keyword_paths=[ppn_path],
                sensitivities=[sensitivity],
            )
            self.keyword = os.path.basename(ppn_path)
        else:
            self.handle = self.pvporcupine.create(
                access_key=self.access_key,
                keywords=[self.keyword],
                sensitivities=[sensitivity],
            )
        self.recorder = self.PvRecorder(device_index=-1, frame_length=self.handle.frame_length)
    def listen(self) -> WakeResult:
        print("[green]Listening for Porcupine keyword...[/green]")
        self.recorder.start()
        try:
            while True:
                pcm = self.recorder.read()
                result = self.handle.process(pcm)
                if result >= 0:
                    return WakeResult(keyword=self.keyword)
        finally:
            self.recorder.stop()
            self.recorder.delete()
            self.handle.delete()

class OpenWakeWordDetector:
    def __init__(self, model_path: str):
        try:
            from openwakeword.model import Model
        except Exception as e:
            raise RuntimeError("openwakeword not installed: pip install openwakeword onnxruntime") from e
        self.model = Model(wakeword_model_paths=[model_path])
        # Skeleton: leaving actual streaming audio hookup as a TODO.
    def listen(self) -> WakeResult:
        raise NotImplementedError("OpenWakeWord streaming hookup is TODO in skeleton")


def make_wake_detector():
    b = cfg["wake"]["backend"]
    if b == "keyboard":
        return KeyboardWakeDetector(cfg["wake"]["porcupine"]["keyword"])  # reuse name
    if b == "porcupine":
        p = cfg["wake"]["porcupine"]
        return PorcupineWakeDetector(keyword=p["keyword"], access_key_env=p["access_key_env"], ppn_path=p.get("ppn_path"), sensitivity=p.get("sensitivity", 0.6))
    if b == "openwakeword":
        p = cfg["wake"]["openwakeword"]
        return OpenWakeWordDetector(model_path=p["model_path"])
    raise ValueError(f"Unknown wake backend: {b}")
