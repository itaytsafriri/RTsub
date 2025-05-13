import sounddevice as sd
import numpy as np
import queue
import whisper
import os
import sys
import arabic_reshaper
from bidi.algorithm import get_display
import scipy.signal
import openai
import time

# Parameters
SAMPLERATE = 16000
CHANNELS = 1
BLOCK_SECONDS = 9  # Window size in seconds
STEP_SECONDS = 3   # Step size in seconds (how often to process)
WORDS_PER_LINE = 7

# Initialize Whisper model
print("Loading Whisper model...")
model = whisper.load_model("large")  # or "medium" for better accuracy

# Audio queue
q = queue.Queue()

# Set up OpenAI client using environment variable for API key
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def audio_callback(indata, frames, time, status):
    q.put(indata.copy())

def print_subtitle(text, words_per_line=WORDS_PER_LINE):
    words = text.strip().split()
    lines = []
    for i in range(0, len(words), words_per_line):
        line = ' '.join(words[i:i+words_per_line])
        # Reshape and apply bidi for proper RTL display
        reshaped_text = arabic_reshaper.reshape(line)
        bidi_text = get_display(reshaped_text)
        lines.append(bidi_text)
    # Only show the last two lines (like TV subs)
    subtitle = '\n'.join(lines[-2:])
    # Clear console (works on Windows and Unix)
    os.system('cls' if os.name == 'nt' else 'clear')
    print(subtitle)
    sys.stdout.flush()

def postprocess_hebrew(text, max_words=7):
    prompt = (
        "תקן שגיאות כתיב, שגיאות תחביר, הוסף סימני פיסוק, "
        f"ופרוס את הטקסט לשורות של עד {max_words} מילים, מבלי לחתוך משפטים באמצע. "
        "החזר רק את הטקסט המתוקן והמפוסק."
        "\n\n"
        f"טקסט:\n{text}\n\n"
        "פלט:"
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()

def main():
    # List devices to help user select the right one
    print("Available audio input devices:")
    print(sd.query_devices())
    device = 0  # <--- Set your desired device index here

    # Check supported sample rates
    supported_samplerate = None
    for rate in [16000, 22050, 32000, 44100, 48000]:
        try:
            sd.check_input_settings(device=device, samplerate=rate)
            print(f"  {rate} Hz: supported")
            if supported_samplerate is None:
                supported_samplerate = rate
        except Exception as e:
            print(f"  {rate} Hz: NOT supported ({e})")
    if supported_samplerate is None:
        raise RuntimeError("No supported sample rates found for the selected device.")
    print(f"Using sample rate: {supported_samplerate}")

    blocksize = supported_samplerate * BLOCK_SECONDS
    audio_buffer = []
    chunk_samples = blocksize
    chunk_count = 0
    start_time = time.time()

    def fixed_audio_callback(indata, frames, time_info, status):
        audio_buffer.append(indata.copy())

    with sd.InputStream(samplerate=supported_samplerate, channels=CHANNELS, callback=fixed_audio_callback, blocksize=blocksize, device=device):
        print("Recording and transcribing... (Press Ctrl+C to stop)")
        try:
            while True:
                # Wait until we have enough audio for a chunk
                if len(audio_buffer) == 0:
                    time.sleep(0.1)
                    continue
                audio_chunk = audio_buffer.pop(0)
                audio = np.squeeze(audio_chunk).astype(np.float32)
                # Whisper expects audio in range [-1, 1]
                if np.max(np.abs(audio)) > 1:
                    audio = audio / np.max(np.abs(audio))
                # Resample if needed
                if supported_samplerate != 16000:
                    num_samples = int(len(audio) * 16000 / supported_samplerate)
                    audio = scipy.signal.resample(audio, num_samples)
                # Transcribe
                result = model.transcribe(audio, language="he", fp16=False, task="transcribe")
                hebrew_text = result['text']
                hebrew_text = postprocess_hebrew(hebrew_text, max_words=WORDS_PER_LINE)
                # Calculate when to display this chunk
                chunk_start = start_time + chunk_count * BLOCK_SECONDS
                display_time = chunk_start + BLOCK_SECONDS
                now = time.time()
                if now < display_time:
                    time.sleep(display_time - now)
                print_subtitle(hebrew_text)
                chunk_count += 1
        except KeyboardInterrupt:
            print("\nStopped.")

if __name__ == "__main__":
    main()
