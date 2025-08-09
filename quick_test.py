#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import sys
import time

def run_openwakeword(device: int, model_path: str, sr: int, block: int, threshold: float):
    import numpy as np
    import sounddevice as sd
    from openwakeword.model import Model

    print(f"[oww] device={device} sr={sr} block={block} model={model_path} threshold={threshold}")
    openwakeword.utils.download_models()   
    model = Model(wakeword_model_paths=[model_path]) if model_path else Model()  # loads bundled models if None
    with sd.InputStream(channels=1, samplerate=sr, dtype="float32",
                        blocksize=block, device=device) as stream:
        print("Listening (OpenWakeWord)...")
        while True:
            audio, _ = stream.read(block)
            scores = model.predict(audio.flatten())
            # scores is a dict: {model_name: probability}
            name, prob = max(scores.items(), key=lambda kv: kv[1])
            if prob >= threshold:
                print(f"Detected: {name} (p={prob:.2f})")
                break

def run_porcupine(device: int, keyword: str | None, ppn_path: str | None, sensitivity: float):
    try:
        import pvporcupine
        from pvrecorder import PvRecorder
    except Exception as e:
        sys.exit(f"[porcupine] missing deps: {e}\nTry: pip install pvporcupine pvrecorder")

    access_key = os.getenv("PORCUPINE_ACCESS_KEY")
    if not access_key:
        sys.exit("[porcupine] Set PORCUPINE_ACCESS_KEY")

    try:
        if ppn_path:
            porc = pvporcupine.create(
                access_key=access_key,
                keyword_paths=[ppn_path],
                sensitivities=[sensitivity],
            )
            disp_name = os.path.basename(ppn_path)
        else:
            if not keyword:
                keyword = "jarvis"
            porc = pvporcupine.create(
                access_key=access_key,
                keywords=[keyword],
                sensitivities=[sensitivity],
            )
            disp_name = keyword
    except NotImplementedError as e:
        sys.exit(f"[porcupine] {e}\nThis CPU isn't supported by the wheel. Use --backend openwakeword instead.")

    rec = PvRecorder(device_index=device, frame_length=porc.frame_length)
    print(f"[porcupine] device={device} keyword={disp_name} sens={sensitivity}")
    print("Listening (Porcupine)...")
    rec.start()
    try:
        while True:
            if porc.process(rec.read()) >= 0:
                print(f"Detected: {disp_name}")
                break
    finally:
        rec.stop(); rec.delete(); porc.delete()

def main():
    p = argparse.ArgumentParser(description="Quick wake-word test (OpenWakeWord or Porcupine)")
    p.add_argument("--backend", choices=["openwakeword", "porcupine"], default="openwakeword")
    p.add_argument("--device", type=int, required=True, help="Input device index (from list_devices.py)")
    # OpenWakeWord params
    p.add_argument("--model-path", default="", help="OWW model .tflite/.onnx (empty loads bundled models)")
    p.add_argument("--threshold", type=float, default=0.5, help="OWW detection threshold")
    p.add_argument("--sr", type=int, default=16000, help="Sample rate")
    p.add_argument("--block", type=int, default=512, help="Frames per read (~32ms at 16k)")
    # Porcupine params
    p.add_argument("--keyword", default="jarvis", help="Built-in Porcupine keyword")
    p.add_argument("--ppn-path", default="", help="Custom Porcupine .ppn path")
    p.add_argument("--sensitivity", type=float, default=0.6, help="Porcupine sensitivity 0..1")
    args = p.parse_args()

    if args.backend == "openwakeword":
        run_openwakeword(args.device, args.model_path or None, args.sr, args.block, args.threshold)
    else:
        run_porcupine(args.device, args.keyword, args.ppn_path or None, args.sensitivity)

if __name__ == "__main__":
    main()
