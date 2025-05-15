import sounddevice as sd
import numpy as np
# import queue # No longer needed for faster-whisper direct processing
import os
import sys
import arabic_reshaper
from bidi.algorithm import get_display
# import scipy.signal
import time
import threading
import queue
import openai # Added for GPT post-processing
from google.cloud import speech # Added for Google Speech-to-Text
import re
from collections import deque

# Parameters
SAMPLERATE = 16000
CHANNELS = 1
STREAMING_CHUNK_MS = 100 # ms, for microphone read and Google STT
GOOGLE_CHUNK_SAMPLES = int(SAMPLERATE / (1000 / STREAMING_CHUNK_MS)) # Samples per chunk
WORDS_PER_LINE = 6  # Reduced from 7 to 5-7 range
LINGER_DURATION = 2.0  # Reduced from 3.0 to make it more dynamic
MIN_WORDS_FOR_DISPLAY = 1  # Debug: show any words
# MAX_PROCESSING_BUDGET_PER_BLOCK: Time allowed for whisper + GPT.
# If total desired delay is 70s, and BLOCK_SECONDS is 30s, then this is 40s.
MAX_PROCESSING_BUDGET_PER_BLOCK = 40.0 # seconds 
MAX_BUFFERED_SENTENCES = 2
SENTENCE_END_PATTERN = r'[.!?]\s*'

# Add these constants for sentence and time-based buffering
SUB_BLOCK_SECONDS = 10.0  # Buffer duration in seconds
SENTENCE_END_REGEX = re.compile(r'[.!?…]$')
MIN_WORDS_FOR_FLUSH_ON_SENTENCE_END = 3
MIN_WORDS_FOR_FLUSH_ON_TIMEOUT = 1

# Constants for block-based processing
BLOCK_COLLECTION_SECONDS = 35.0
DEFAULT_SUB_SEGMENT_ONSCREEN_DURATION = 2.5  # Fallback display time for a 1-2 line segment

# --- Google Cloud Speech-to-Text Configuration ---
GOOGLE_CREDENTIALS_PATH = "C:\\\\Users\\\\Admin\\\\Desktop\\\\repos\\\\RTsub\\\\realtime-subs-459714-42a5bd7fdfcf.json"
try:
    gcp_speech_client = speech.SpeechClient.from_service_account_file(GOOGLE_CREDENTIALS_PATH)
    print("Google Speech-to-Text client initialized with service account.")
except Exception as e:
    print(f"Error initializing Google Speech-to-Text client: {e}")
    print(f"Ensure the path '{GOOGLE_CREDENTIALS_PATH}' is correct and the file is accessible.")
    sys.exit(1)

GCP_RECOGNITION_CONFIG = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=SAMPLERATE,
    language_code="he-IL",
    enable_word_time_offsets=True,
    # model="latest_long",  # Removed: Not supported for he-IL / iw-IL
    enable_automatic_punctuation=True  # Keep: Generally beneficial
)
GCP_STREAMING_CONFIG = speech.StreamingRecognitionConfig(
    config=GCP_RECOGNITION_CONFIG,
    interim_results=True, 
)
# --- End Google Cloud Speech-to-Text Configuration ---

# Initialize OpenAI client
# Ensure OPENAI_API_KEY environment variable is set
API_KEY = "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"
try:
    client = openai.OpenAI(api_key=API_KEY) # Hardcoded API key
    # Perform a dummy call to check API key validity early
    client.models.list() 
    print("OpenAI client initialized and API key appears valid.")
except openai.APIConnectionError as e:
    print(f"OpenAI API connection error: {e}. Please check your network connection and API key.")
    sys.exit(1)
except openai.AuthenticationError as e:
    print(f"OpenAI API authentication error: {e}. The provided API key might be invalid or revoked.")
    sys.exit(1)
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    sys.exit(1)

# Audio queue - No longer needed for this direct processing approach
# q = queue.Queue()

# Global buffer for audio data
audio_buffer_np = np.zeros((0, CHANNELS), dtype=np.float32)

# Queues and Event for thread communication and shutdown
audio_stream_queue = queue.Queue(maxsize=100) # For raw audio bytes from recorder to transcriber (increased maxsize)
segments_queue = queue.Queue(maxsize=50)     # For processed word lists (text, start, end) for playback
shutdown_event = threading.Event()

class SentenceBuffer:
    def __init__(self, max_sentences=2):
        self.buffer = deque(maxlen=max_sentences)
        self.current_sentence = []
        self.last_update_time = time.monotonic()
    
    def add_words(self, words):
        current_time = time.monotonic()
        for word in words:
            self.current_sentence.append(word)
            text = ' '.join([w['word'] for w in self.current_sentence])
            if re.search(SENTENCE_END_PATTERN, text):
                self.buffer.append(list(self.current_sentence))
                self.current_sentence = []
                self.last_update_time = current_time
    
    def get_sentences(self):
        return list(self.buffer)
    
    def clear(self):
        self.buffer.clear()
        self.current_sentence = []
        self.last_update_time = time.monotonic()

def format_subtitle_line(words_on_line):
    line = ' '.join(words_on_line)
    reshaped_text = arabic_reshaper.reshape(line)
    return get_display(reshaped_text)

def split_into_subtitle_lines(words, max_words_per_line=WORDS_PER_LINE):
    lines = []
    current_line = []
    
    for word in words:
        current_line.append(word)
        if len(current_line) >= max_words_per_line:
            lines.append(current_line)
            current_line = []
    
    if current_line and len(current_line) >= MIN_WORDS_FOR_DISPLAY:
        lines.append(current_line)
    
    return lines

def postprocess_hebrew_with_gpt(text_to_correct):
    if not text_to_correct.strip():
        return "" # Return empty if nothing to correct
    try:
        # print(f"GPT: Sending text for correction: '{text_to_correct[:100]}...'") # Debug
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "אתה מערכת לתיקוני שגיאות עבור כתוביות בשידורי חדשות בשידור חי, הכתוביות הן תמלול של הצהרות ונאומים יש לתקן טעויות בעברית, להוסיף ניקוד ופיסוק אם נדרש ובעיקר לשמור על הניסוח המקורי ללא שינויים בטקסט, רק תיקונים. חשוב ביותר: מספר המילים בפלט המתוקן חייב להיות זהה לחלוטין למספר המילים בקלט המקורי. אין להוסיף, למחוק או למזג מילים. תקן כל מילה במקומה."},
                {"role": "user", "content": f"Please correct the following Hebrew text: \n{text_to_correct}"}
            ],
            temperature=0.2 # Lower temperature for more deterministic corrections
        )
        corrected_text = completion.choices[0].message.content.strip()
        # print(f"GPT: Received corrected text: '{corrected_text[:100]}...'") # Debug
        return corrected_text
    except openai.APIError as e:
        print(f"GPT API Error: {e}")
    except Exception as e:
        print(f"Error during GPT post-processing: {e}")
    return text_to_correct # Fallback to original text on error

def play_segments_sequentially(segments_q_local, shutdown_ev_local):
    print("\n--- Starting Subtitle Playback (Block Synchronized TV Style) ---")
    display_pair = ["", ""]
    last_subtitle_final_clear_time_monotonic = 0

    try:
        while not shutdown_ev_local.is_set():
            try:
                # Get the entire block of words collected by the transcriber
                segment_data = segments_q_local.get(timeout=0.5) 
                if not segment_data or not segment_data.get('words'):
                    if segments_q_local.empty(): # only task_done if we actually got an empty item
                        segments_q_local.task_done()
                    continue

                all_block_words = segment_data['words']
                if not all_block_words:
                    segments_q_local.task_done()
                    continue

                block_display_phase_start_monotonic = time.monotonic()
                gcp_offset_of_first_word_in_block = None
                for word_obj in all_block_words:
                    if 'start' in word_obj:
                        gcp_offset_of_first_word_in_block = word_obj['start']
                        break
                
                # Helper to get min start and max end GCP times from a list of word objects
                def get_segment_gcp_times(word_list):
                    s_time, e_time = None, None
                    for w_obj in word_list:
                        if 'start' in w_obj and (s_time is None or w_obj['start'] < s_time):
                            s_time = w_obj['start']
                        if 'end' in w_obj and (e_time is None or w_obj['end'] > e_time):
                            e_time = w_obj['end']
                    return s_time, e_time

                # Split the entire block into displayable lines first
                potential_lines_with_word_objects = split_into_subtitle_lines(all_block_words, WORDS_PER_LINE)
                if not potential_lines_with_word_objects:
                    segments_q_local.task_done()
                    continue
                
                line_idx = 0
                while line_idx < len(potential_lines_with_word_objects) and not shutdown_ev_local.is_set():
                    current_sub_segment_word_objects = []
                    current_sub_segment_word_objects.extend(potential_lines_with_word_objects[line_idx])
                    line_idx_for_display = 0 # For display_pair indexing
                    display_pair[line_idx_for_display] = format_subtitle_line([w['word'] for w in potential_lines_with_word_objects[line_idx]])
                    display_pair[1] = "" # Clear second line initially

                    if line_idx + 1 < len(potential_lines_with_word_objects):
                        current_sub_segment_word_objects.extend(potential_lines_with_word_objects[line_idx+1])
                        display_pair[1] = format_subtitle_line([w['word'] for w in potential_lines_with_word_objects[line_idx+1]])
                        line_idx_to_advance = 2
                    else:
                        line_idx_to_advance = 1

                    sub_segment_gcp_start, sub_segment_gcp_end = get_segment_gcp_times(current_sub_segment_word_objects)
                    
                    target_display_start_monotonic = block_display_phase_start_monotonic
                    target_clear_monotonic = block_display_phase_start_monotonic + DEFAULT_SUB_SEGMENT_ONSCREEN_DURATION

                    if gcp_offset_of_first_word_in_block is not None and \
                        sub_segment_gcp_start is not None and sub_segment_gcp_end is not None:
                        start_delay_from_block_start = sub_segment_gcp_start - gcp_offset_of_first_word_in_block
                        target_display_start_monotonic = block_display_phase_start_monotonic + max(0, start_delay_from_block_start)
                        
                        end_delay_from_block_start = sub_segment_gcp_end - gcp_offset_of_first_word_in_block
                        target_clear_monotonic = block_display_phase_start_monotonic + max(0, end_delay_from_block_start)
                        
                        if target_clear_monotonic <= target_display_start_monotonic: # Ensure some display time
                            target_clear_monotonic = target_display_start_monotonic + DEFAULT_SUB_SEGMENT_ONSCREEN_DURATION
                   
                    # Wait until it's time to show this sub-segment
                    while time.monotonic() < target_display_start_monotonic and not shutdown_ev_local.is_set():
                        time.sleep(0.02)
                    if shutdown_ev_local.is_set(): break

                    os.system('cls' if os.name == 'nt' else 'clear')
                    print(display_pair[0])
                    print(display_pair[1])
                    sys.stdout.flush()
                    last_subtitle_final_clear_time_monotonic = target_clear_monotonic # Update for linger

                    # Wait until it's time to clear this sub-segment
                    while time.monotonic() < target_clear_monotonic and not shutdown_ev_local.is_set():
                        time.sleep(0.02)
                    if shutdown_ev_local.is_set(): break
                    
                    line_idx += line_idx_to_advance

                # After processing all sub-segments in a block, clear the screen
                os.system('cls' if os.name == 'nt' else 'clear')
                display_pair = ["", ""]
                sys.stdout.flush()
                segments_q_local.task_done()

            except queue.Empty:
                # Linger logic: if LINGER_DURATION has passed since the last sub-segment of a block was cleared
                if display_pair[0] == "" and display_pair[1] == "": # ensure screen is already clear
                     if time.monotonic() - last_subtitle_final_clear_time_monotonic > LINGER_DURATION:
                        # Already cleared, just ensuring we don't re-clear unnecessarily if linger already passed
                        pass 
                elif time.monotonic() - last_subtitle_final_clear_time_monotonic > LINGER_DURATION : 
                    os.system('cls' if os.name == 'nt' else 'clear')
                    display_pair = ["", ""]
                    sys.stdout.flush()
                continue
            except Exception as e:
                print(f"Playback thread error: {e}")
                # import traceback; traceback.print_exc()
                if shutdown_ev_local.is_set(): break
                time.sleep(0.1)
    finally:
        print("Playback thread finishing.")

def recorder_thread_func(audio_q_local, shutdown_ev_local, samplerate, chunk_size_samples):
    print("Recorder thread started (streaming mode).")
    try:
        # Using default device, RawInputStream with dtype='int16'
        with sd.RawInputStream(samplerate=samplerate, blocksize=chunk_size_samples,
                               dtype='int16', channels=CHANNELS, device=None) as stream:
            print(f"Recorder: Listening with {samplerate}Hz, {chunk_size_samples} samples/chunk ({STREAMING_CHUNK_MS}ms).")
            while not shutdown_ev_local.is_set():
                raw_buffer, overflowed = stream.read(chunk_size_samples)
                if overflowed:
                    print("Recorder: Audio input overflowed!", file=sys.stderr)
                
                # Convert raw buffer to numpy array, then to bytes (as per google_stream_test.py)
                np_data = np.frombuffer(raw_buffer, dtype='int16')
                try:
                    audio_q_local.put(np_data.tobytes(), timeout=0.1) 
                except queue.Full:
                    if shutdown_ev_local.is_set(): break
                    # print("Recorder: Audio stream queue full, frame dropped.") # Can be very noisy
                    time.sleep(0.01) # Brief pause
                    continue
    except AttributeError as e:
        if "'NoneType' object has no attribute 'read'" in str(e):
             print("Recorder Error: Could not open audio stream. No input device found or device busy?")
        else:
            print(f"Recorder thread AttributeError: {e}")
    except Exception as e:
        print(f"Recorder thread error: {e}")
        # import traceback; traceback.print_exc()
    finally:
        print("Recorder thread finishing.")

def transcriber_thread_func(audio_q_local, segments_q_local, speech_client, streaming_config_gcp, shutdown_ev_local):
    print("Transcriber thread started (Google streaming - Block Collection Mode).")
    word_buffer = []
    last_block_flush_time = time.monotonic()

    def audio_generator_from_queue(q, ev):
        while not ev.is_set():
            try:
                chunk_bytes = q.get(timeout=0.1)
                if chunk_bytes is None: return
                yield speech.StreamingRecognizeRequest(audio_content=chunk_bytes)
            except queue.Empty:
                if ev.is_set(): return
                continue
            except Exception as e_gen:
                print(f"Audio generator error: {e_gen}")
                if ev.is_set(): return
                break
    stream_active = False
    while not shutdown_ev_local.is_set():
        if not stream_active:
            print("Transcriber: Attempting to start new Google STT stream...")
            requests_iterable = audio_generator_from_queue(audio_q_local, shutdown_ev_local)
            try:
                responses = speech_client.streaming_recognize(streaming_config_gcp, requests_iterable)
                print("Transcriber: Google STT stream active.")
                stream_active = True
            except Exception as e_init_stream:
                print(f"Transcriber: Failed to start Google STT stream: {e_init_stream}")
                stream_active = False
                if shutdown_ev_local.is_set(): break
                time.sleep(2)
                continue
        try:
            for response in responses:
                if shutdown_ev_local.is_set():
                    print("Transcriber: Shutdown signal during STT response processing.")
                    stream_active = False
                    break
                if not response.results:
                    continue
                result = response.results[0]
                if not result.alternatives:
                    continue
                transcript = result.alternatives[0].transcript
                
                current_words_from_response = []
                if result.alternatives[0].words:
                    for word_info in result.alternatives[0].words:
                        current_words_from_response.append({
                            'word': word_info.word,
                            'start': word_info.start_time.total_seconds(),
                            'end': word_info.end_time.total_seconds()
                        })
                elif transcript.strip() and result.is_final: # Only use transcript for final if no words
                    for w_idx, w_text in enumerate(transcript.strip().split()):
                        # Create approximate timings if only final transcript is available
                        # This is a rough estimation, assuming even distribution
                        # A more sophisticated approach might be needed if this path is common
                        stability_offset = result.stability * 0.1 # Fictional use of stability for demo
                        approx_start = stability_offset + w_idx * 0.3 # Placeholder
                        approx_end = approx_start + 0.25 # Placeholder
                        current_words_from_response.append({
                            'word': w_text,
                            'start': approx_start + (time.monotonic() - last_block_flush_time), # Make relative to current block period
                            'end': approx_end + (time.monotonic() - last_block_flush_time)
                        })
                
                if current_words_from_response:
                    word_buffer.extend(current_words_from_response)
                
                now = time.monotonic()
                if now - last_block_flush_time >= BLOCK_COLLECTION_SECONDS and word_buffer:
                    print(f"Transcriber: Flushing block of {len(word_buffer)} words after {BLOCK_COLLECTION_SECONDS}s.")
                    
                    # GPT Post-processing for the entire block
                    block_to_process_raw = list(word_buffer) # Important: Copy for processing
                    original_text_for_gpt = " ".join([w['word'] for w in block_to_process_raw])
                    corrected_text_from_gpt = ""
                    if original_text_for_gpt.strip():
                        # print(f"DEBUG: Sending to GPT: {original_text_for_gpt[:100]}...") # Optional: for debugging GPT input
                        corrected_text_from_gpt = postprocess_hebrew_with_gpt(original_text_for_gpt)
                        # print(f"DEBUG: Received from GPT: {corrected_text_from_gpt[:100]}...") # Optional: for debugging GPT output
                    
                    final_block_for_playback = []
                    if corrected_text_from_gpt and original_text_for_gpt.strip():
                        corrected_word_list = corrected_text_from_gpt.split()
                        if len(corrected_word_list) == len(block_to_process_raw):
                            for i, original_word_obj in enumerate(block_to_process_raw):
                                final_block_for_playback.append({
                                    'word': corrected_word_list[i],
                                    'start': original_word_obj.get('start'), # Use .get for safety
                                    'end': original_word_obj.get('end')      # Use .get for safety
                                })
                            # print(f"Transcriber: GPT applied to block. Word count matched.") # Optional debug
                        else:
                            print(f"Transcriber: Word count mismatch after GPT. Original: {len(block_to_process_raw)}, Corrected: {len(corrected_word_list)}. Using Google's output for this block.")
                            final_block_for_playback = list(block_to_process_raw) # Fallback
                    else: # Fallback if GPT fails or original text was empty or only whitespace
                        # print("Transcriber: GPT fallback - using original words.") # Optional debug
                        final_block_for_playback = list(block_to_process_raw)
                    
                    try:
                        # print(f"DEBUG: Putting block of {len(final_block_for_playback)} words to queue.") # Optional debug
                        segments_q_local.put({'words': final_block_for_playback, 'is_final': True}, timeout=1.0) 
                        word_buffer.clear() # Clear original buffer
                        last_block_flush_time = now 
                    except queue.Full:
                        print(f"Transcriber: Segments queue full. A {BLOCK_COLLECTION_SECONDS}s block may be lost. Playback might be too slow.")
                        # If queue is full, this block is lost. word_buffer was already copied to block_to_process_raw.
                        # We should clear word_buffer anyway to start fresh for the next block.
                        word_buffer.clear() 
                        last_block_flush_time = now # Still update time to prevent immediate re-flush of empty buffer
                        pass 

        except StopIteration: # Happens if audio generator ends
            print("Transcriber: Audio generator stopped.")
            stream_active = False
            if word_buffer and not shutdown_ev_local.is_set():
                print(f"Transcriber: Flushing final partial block of {len(word_buffer)} words due to stream end (with GPT).")
                block_to_process_raw = list(word_buffer)
                original_text_for_gpt = " ".join([w['word'] for w in block_to_process_raw])
                corrected_text_from_gpt = ""
                if original_text_for_gpt.strip():
                    corrected_text_from_gpt = postprocess_hebrew_with_gpt(original_text_for_gpt)
                final_block_for_playback = []
                if corrected_text_from_gpt and original_text_for_gpt.strip():
                    corrected_word_list = corrected_text_from_gpt.split()
                    if len(corrected_word_list) == len(block_to_process_raw):
                        for i, original_word_obj in enumerate(block_to_process_raw):
                            final_block_for_playback.append({
                                'word': corrected_word_list[i],
                                'start': original_word_obj.get('start'),
                                'end': original_word_obj.get('end')
                            })
                    else:
                        final_block_for_playback = list(block_to_process_raw) 
                else:
                    final_block_for_playback = list(block_to_process_raw) 
                try:
                    segments_q_local.put({'words': final_block_for_playback, 'is_final': True}, timeout=1.0)
                    word_buffer.clear()
                except queue.Full:
                    print("Transcriber: Segments queue full on final partial flush. Block lost.")
            if shutdown_ev_local.is_set(): break 
        except queue.Empty: 
             if shutdown_ev_local.is_set(): break
        except Exception as e_stream_proc:
            print(f"Transcriber: Error processing Google STT response stream: {e_stream_proc}")
            stream_active = False
            if shutdown_ev_local.is_set(): break
            time.sleep(1)
    # Final flush if application is shutting down and buffer has content
    if word_buffer and not shutdown_ev_local.is_set(): 
        print(f"Transcriber: Flushing any remaining words ({len(word_buffer)}) on transcriber exit (with GPT).")
        block_to_process_raw = list(word_buffer)
        original_text_for_gpt = " ".join([w['word'] for w in block_to_process_raw])
        corrected_text_from_gpt = ""
        if original_text_for_gpt.strip():
            corrected_text_from_gpt = postprocess_hebrew_with_gpt(original_text_for_gpt)
        final_block_for_playback = []
        if corrected_text_from_gpt and original_text_for_gpt.strip():
            corrected_word_list = corrected_text_from_gpt.split()
            if len(corrected_word_list) == len(block_to_process_raw):
                for i, original_word_obj in enumerate(block_to_process_raw):
                    final_block_for_playback.append({
                        'word': corrected_word_list[i],
                        'start': original_word_obj.get('start'),
                        'end': original_word_obj.get('end')
                    })
            else:
                final_block_for_playback = list(block_to_process_raw) 
        else:
            final_block_for_playback = list(block_to_process_raw) 
        try:
            segments_q_local.put({'words': final_block_for_playback, 'is_final': True}, timeout=1.0)
        except queue.Full:
            print("Transcriber: Segments queue full on very final flush. Words lost.")
    print("Transcriber thread finishing.")

def main():
    print("Initializing Real-Time Hebrew Subtitle Generator (Google Streaming Mode)...")
    print("Available audio input devices:")
    try:
        print(sd.query_devices())
        print(f"Using default input device with samplerate {SAMPLERATE}Hz for streaming.")
    except Exception as e:
        print(f"Error querying audio devices: {e}. Ensure microphone is connected.")
        print("Ensure OPENAI_API_KEY is set for GPT, and Google Cloud credentials are correct.")
        return

    recorder = threading.Thread(target=recorder_thread_func, 
                                args=(audio_stream_queue, shutdown_event, SAMPLERATE, GOOGLE_CHUNK_SAMPLES))
    transcriber = threading.Thread(target=transcriber_thread_func, 
                                   args=(audio_stream_queue, segments_queue, gcp_speech_client, GCP_STREAMING_CONFIG, shutdown_event))
    playback_thread = threading.Thread(target=play_segments_sequentially, args=(segments_queue, shutdown_event))

    print("Starting threads...")
    recorder.start()
    transcriber.start()
    playback_thread.start()

    print(f"Main thread: System initialized. Press Ctrl+C to exit.")
    
    try:
        # Keep main thread alive to catch Ctrl+C and manage shutdown
        while not shutdown_event.is_set():
            time.sleep(0.5)
            if not recorder.is_alive() and not shutdown_event.is_set():
                print("Error: Recorder thread died unexpectedly. Shutting down.")
                shutdown_event.set()
            if not transcriber.is_alive() and not shutdown_event.is_set():
                print("Error: Transcriber thread died unexpectedly. Shutting down.")
                shutdown_event.set()
            if not playback_thread.is_alive() and not shutdown_event.is_set():
                print("Error: Playback thread died unexpectedly. Shutting down.")
                shutdown_event.set()

    except KeyboardInterrupt:
        print("\nCtrl+C pressed. Initiating graceful shutdown...")
    finally:
        print("Main: Setting shutdown event for all threads...")
        shutdown_event.set()

        print("Main: Waiting for recorder thread to join...")
        recorder.join(timeout=3) 
        if recorder.is_alive(): print("Main: Recorder thread did not join in time.")

        print("Main: Waiting for transcriber thread to join...")
        transcriber.join(timeout=7) # Google stream might take a moment to close
        if transcriber.is_alive(): print("Main: Transcriber thread did not join in time.")

        print("Main: Waiting for playback thread to join...")
        playback_thread.join(timeout=3)
        if playback_thread.is_alive(): print("Main: Playback thread did not join in time.")
        
        print("Main: Attempting to clear queues...")
        while not audio_stream_queue.empty():
            try: audio_stream_queue.get_nowait() 
            except queue.Empty: break
            except Exception: break 
        while not segments_queue.empty():
            try: segments_queue.get_nowait(); segments_queue.task_done()
            except queue.Empty: break
            except Exception: break
        
        print("All threads should be finished. Exiting script.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An unhandled error occurred in __main__: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sd.stop() # Ensure sounddevice is stopped if an error occurred before normal exit.
        print("Script has ended.")
