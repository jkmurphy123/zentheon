# zentheon

# Jetson Voice Assistant — Skeleton

This is a minimal scaffold to validate each component on a Jetson device.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# enumerate audio devices to set in config.yaml
python scripts/list_devices.py

# run checks
python app.py check:audio
python app.py check:wake    # default is keyboard (press Enter)
python app.py check:asr     # needs a small model; downloads on first run (faster-whisper)
python app.py check:llm     # needs a GGUF at llm.model_path
python app.py check:tts     # needs Piper binary in PATH and a voice .onnx

# quick end-to-end (keyboard wake -> mock ASR -> LLM -> Piper)
python app.py check:loop
