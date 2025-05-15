import os
import sounddevice as sd
from google.cloud import speech
import numpy as np
import sys

RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

print(sd.query_devices())

# client = speech.SpeechClient() # Old way
# New way: Initialize client with service account credentials
credentials_path = "C:\\Users\\Admin\\Desktop\\repos\\RTsub\\realtime-subs-459714-42a5bd7fdfcf.json"
client = speech.SpeechClient.from_service_account_file(credentials_path)

config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=RATE,
    language_code="he-IL",  # Hebrew
    enable_word_time_offsets=True  # Enable word timestamps
)
streaming_config = speech.StreamingRecognitionConfig(
    config=config,
    interim_results=True,
)

def audio_generator():
    try:
        with sd.RawInputStream(samplerate=RATE, blocksize=CHUNK, dtype='int16', channels=1, device=0) as stream:
            print("Speak into your microphone...")
            count = 0
            while count < 50:  # Only send 50 chunks for testing
                raw_buffer, overflowed = stream.read(CHUNK)
                if overflowed:
                    print("Warning: audio input overflowed!", file=sys.stderr)
                np_data = np.frombuffer(raw_buffer, dtype='int16')
                print(f"Processing audio chunk: shape={np_data.shape}, dtype={np_data.dtype}")
                yield np_data.tobytes()
                count += 1
    except Exception as e:
        print("Error in audio_generator:", e)
        raise

requests = (speech.StreamingRecognizeRequest(audio_content=chunk) for chunk in audio_generator())

responses = client.streaming_recognize(streaming_config, requests)

try:
    for response in responses:
        for result in response.results:
            if result.is_final:
                print("\nFinal Transcript:", result.alternatives[0].transcript)
                print("Word timings:")
                for word_info in result.alternatives[0].words:
                    word = word_info.word
                    start_time = word_info.start_time.total_seconds()
                    end_time = word_info.end_time.total_seconds()
                    print(f"  Word: '{word}', Start: {start_time:.2f}s, End: {end_time:.2f}s")
                print("-----")
            else:
                print("Interim:", result.alternatives[0].transcript, end='\r')
except Exception as e:
    print("Error in main loop:", e)

