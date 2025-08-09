from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from rich import print
from utils import load_config

cfg = load_config()

@dataclass
class ASRResult:
    text: str

class FasterWhisperASR:
    def __init__(self, model_id: str, compute_type: str = "int8"):
        try:
            from faster_whisper import WhisperModel
        except Exception as e:
            raise RuntimeError("Install faster-whisper: pip install faster-whisper") from e
        print(f"[green]Loading faster-whisper model: {model_id} ({compute_type})[/green]")
        # device="cuda" may or may not be supported on your Jetson; try auto
        self.model = WhisperModel(model_id, device="auto", compute_type=compute_type)
    def transcribe(self, audio: np.ndarray, sr: int) -> ASRResult:
        segments, info = self.model.transcribe(audio, language="en", beam_size=1)
        text = " ".join(seg.text for seg in segments).strip()
        return ASRResult(text=text)

class SherpaASR:
    def __init__(self, model_dir: str):
        try:
            import sherpa_onnx
        except Exception as e:
            raise RuntimeError("Install sherpa-onnx: pip install sherpa-onnx") from e
        raise NotImplementedError("Skeleton: wire sherpa-onnx offline/streaming models here.")


def make_asr():
    back = cfg["asr"]["backend"]
    if back == "faster_whisper":
        fw = cfg["asr"]["faster_whisper"]
        return FasterWhisperASR(fw["model_id"], fw.get("compute_type", "int8"))
    if back == "sherpa":
        sh = cfg["asr"]["sherpa"]
        return SherpaASR(sh["model_dir"])
    raise ValueError(f"Unknown ASR backend: {back}")
