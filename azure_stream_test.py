import azure.cognitiveservices.speech as speechsdk

# Fill in your Azure Speech key and region below
speech_key = "YCFHHDtJp9rUwneoaJwmgXbn7932GxOZq90NfQZ4iewepAXFoOJFKJQQJ99BEAC5RqLJXJ3w3AAAYACOGkjTB"
service_region = "westeurope"  # e.g., "westeurope"
language = "he-IL"  # Hebrew

speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
speech_config.speech_recognition_language = language
audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)

speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

def recognized(evt):
    print("Recognized: {}".format(evt.result.text))

def recognizing(evt):
    print("Recognizing (interim): {}".format(evt.result.text))

speech_recognizer.recognized.connect(recognized)
speech_recognizer.recognizing.connect(recognizing)

print("Recognizer is set up and listening...")

print("Speak into your microphone...")
speech_recognizer.start_continuous_recognition()
input("Press Enter to stop...\n")
speech_recognizer.stop_continuous_recognition() 