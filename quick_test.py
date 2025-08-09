# save as porcupine_quicktest.py and run `python porcupine_quicktest.py 24`
import os, sys
import pvporcupine
from pvrecorder import PvRecorder
key = os.getenv("PORCUPINE_ACCESS_KEY") or sys.exit("Set PORCUPINE_ACCESS_KEY")
ppn = os.getenv("PPN_PATH")  # or leave None to use a built-in like 'jarvis'
porc = pvporcupine.create(access_key=key, keyword_paths=[ppn] if ppn else None, keywords=None if ppn else ["jarvis"])
rec = PvRecorder(device_index=int(sys.argv[1]), frame_length=porc.frame_length)
print("listening...")
rec.start()
try:
    while True:
        if porc.process(rec.read()) >= 0:
            print("detected!")
            break
finally:
    rec.stop(); rec.delete(); porc.delete()
