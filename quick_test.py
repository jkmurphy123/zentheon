#!/usr/bin/env python3
"""
Quick wake-word test for OpenWakeWord (default) and Porcupine.

Examples (mic device 24):
  # OWW with bundled models (downloads internals on first run)
  python quick_wake_test.py --backend openwakeword --device 24

  # OWW with a specific model file + live debug of a target name
  python quick_wake_test.py --backend openwakeword --device 24 \
    --model-path models/openwakeword/hey_jarvis.tflite \
    --threshold 0.35 --sr 16000 --block 1280 --target-substr mycroft

  # Porcupine with custom .ppn
  export PORCUPINE_ACCESS_KEY="..."
  python quick_wake_test.py --backend porcupine --device 24 \
    --ppn-path models/porcupine/my_custom.ppn --sensitivity 0.6
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from typing import Optional


def run_openwakeword(
    device: int,
    model_path: Optional[str],
    sr: int,
    block: int,
    threshold: float,
    target_substr: str,
):
    import numpy as np
    import sounddevice as sd
    import openwakeword
    from openwakeword.model import Model

    # Ensure internal resources (mel/vad/emb) are present on first run
    openwakeword.utils.download_models()

    model = Model(wakeword_model_paths=[model_path]) if model_path else Model()
    names = list(model.models.keys())
    print(f"[oww] device={device} sr={sr} block={block} threshold={threshold}")
    print("Loaded models:", names)

    target = None
    if target_substr:
        target = next((n for n in names if target_substr.lower() in n.lower()), None)
        if target:
            print(f"Debugging probability for: {target!r}")
        else:
            print(f"(No loaded model name contains {target_substr!r}; live debug disabled)")

    with sd.InputStream(
        channels=1,
        samplerate=sr,
        dtype="float32",
        blocksize=block,
        device=device,
    ) as stream:
        print("Listening (OpenWakeWord)...")
        while True:
            audio, _ = stream.read(block)
            x = audio.flatten()
            rms = (x**2).mean() ** 0.5

            scores = model.predict(x)  # dict: name -> prob
            top3 = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
            # live meter: rms and the target model prob
            tgt_p = scores.get(target, 0.0) if target else 0.0
            print(f"rms={rms:.3f}  {('p('+target+')=%.3f' % tgt_p) if target else ''}   top3={[(n, round(p,3)) for n,p in top3]}", end="\r")

            name, prob = top3[0]
            if prob >= threshold:
                print()  # newline after the \r prints
                print(f"Detected: {name} (p={prob:.3f})")
                break


def run_porcupine(
    device: int,
    keyword: Optional[str],
    ppn_path: Optional[str],
    sensitivity: float,
):
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
        sys.exit(
            f"[porcupine] {e}\nThis CPU isn't supported by the wheel. "
            f"Use --backend openwakeword instead."
        )

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
        rec.stop()
        rec.delete()
        porc.delete()


def main():
    p = argparse.ArgumentParser(description="Quick wake-word test (OpenWakeWord or Porcupine)")
    p.add_argument("--backend", choices=["openwakeword", "porcupine"], default="openwakeword")
    p.add_argument("--device", type=int, required=True, help="Input device index (from list_devices.py)")

    # OpenWakeWord params
    p.add_argument("--model-path", default="", help="OWW model .tflite/.onnx (empty loads bundled models)")
    p.add_argument("--threshold", type=float, default=0.35, help="OWW detection threshold")
    p.add_argument("--sr", type=int, default=16000, help="Sample rate")
    p.add_argument("--block", type=int, default=1280, help="Frames per read (~80 ms at 16 kHz)")
    p.add_argument("--target-substr", default="mycroft", help="Substring of model name to print live probability for")

    # Porcupine params
    p.add_argument("--keyword", default="jarvis", help="Built-in Porcupine keyword")
    p.add_argument("--ppn-path", default="", help="Custom Porcupine .ppn path")
    p.add_argument("--sensitivity", type=float, default=0.6, help="Porcupine sensitivity 0..1")
    args = p.parse_args()

    if args.backend == "openwakeword":
        run_openwakeword(
            device=args.device,
            model_path=args.model_path or None,
            sr=args.sr,
            block=args.block,
            threshold=args.threshold,
            target_substr=args.target_substr,
        )
    else:
        run_porcupine(
            device=args.device,
            keyword=args.keyword,
            ppn_path=args.ppn_path or None,
            sensitivity=args.sensitivity,
        )


if __name__ == "__main__":
    main()
