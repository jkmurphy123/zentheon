from __future__ import annotations
import argparse
from rich import print
from utils import load_config
from audio_io import list_devices, record_seconds, play
from wake import make_wake_detector
from llm_client import make_llm
from tts import make_tts

cfg = load_config()


def cmd_check_audio(args):
    list_devices()
    audio = record_seconds(3.0)
    play(audio)
    print("[green]Audio record/playback OK[/green]")


def cmd_check_wake(args):
    det = make_wake_detector()
    res = det.listen()
    print(f"[green]Wake detected: {res.keyword}[/green]")


def cmd_check_asr(args):
    try:
        from asr import make_asr
        import numpy as np
    except Exception as e:
        print(f"[red]ASR import failed: {e}[/red]")
        return
    asr = make_asr()
    print("Speak a short sentence after pressing Enter...")
    input()
    audio = record_seconds(4.0)
    out = asr.transcribe(audio, cfg["audio"]["sample_rate"])  # type: ignore
    print(f"[green]ASR text:[/green] {out.text}")


def cmd_check_llm(args):
    llm = make_llm()
    text = llm.chat("Say 'hello from the Jetson skeleton' in five words or fewer.")
    print(f"[green]LLM reply:[/green] {text}")


def cmd_check_tts(args):
    tts = make_tts()
    tts.speak("Hello from the Jetson voice skeleton.")


def cmd_check_loop(args):
    # Minimal: keyboard wake -> prompt -> LLM -> TTS
    det = make_wake_detector()
    llm = make_llm()
    tts = make_tts()
    print("[cyan]Waiting for wake...[/cyan]")
    det.listen()
    prompt = input("Enter a prompt to send to the LLM: ")
    reply = llm.chat(prompt)
    print(f"[green]LLM:[/green] {reply}")
    tts.speak(reply)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check:audio").set_defaults(func=cmd_check_audio)
    sub.add_parser("check:wake").set_defaults(func=cmd_check_wake)
    sub.add_parser("check:asr").set_defaults(func=cmd_check_asr)
    sub.add_parser("check:llm").set_defaults(func=cmd_check_llm)
    sub.add_parser("check:tts").set_defaults(func=cmd_check_tts)
    sub.add_parser("check:loop").set_defaults(func=cmd_check_loop)
    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
