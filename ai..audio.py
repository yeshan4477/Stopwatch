import io
import wave
from google import genai
import numpy as np
import sounddevice as sd
import speech_recognition as sr
import pygame
from gtts import gTTS

pygame.mixer.init()

def speak(text):
  print(f"\n--> Speaking:{text}")
  tts = gTTS(text=text, lang="en", tld="co.in", slow=False)
  fp = io.BytesIO()
  tts.write_to_fp(fp)
  fp.seek(0)

  pygame.mixer.music.load(fp)
  pygame.mixer.music.play()
  while pygame.mixer.music.get_busy():
    pygame.time.Clock().tick(10)

recorded_frames = []
sampling_rate = 44100
chunk_size = 1024
loudness_threshold = 0.05
silent_chunks = 0
has_started_speaking = False

stream = sd.InputStream(samplerate=sampling_rate, blocksize=chunk_size, channels=1)
stream.start()
print("Recording... Speak into your mic:")

while True:
  chunk, overflow = stream.read(chunk_size)
  recorded_frames.append(chunk)
  volume = abs(chunk).max()

  if volume >= loudness_threshold:
    has_started_speaking = True
    silent_chunks = 0
  else:
    if has_started_speaking:
      silent_chunks += 1

  print(f"Volume: {volume:.4f} | Silent Chunks: {silent_chunks}/80", end="\r", flush=True,)

  if has_started_speaking and silent_chunks >= 80:
    print("\nRecording finished! Processing audio...")
    break

stream.stop()
stream.close()

# Audio conversion
audio_array = np.concatenate(recorded_frames, axis=0)
audio_int16 = (audio_array * 32767).astype(np.int16)

wav_filename = "recorded_voice.wav"
with wave.open(wav_filename, "wb") as wav_file:
  wav_file.setnchannels(1)
  wav_file.setsampwidth(2)
  wav_file.setframerate(sampling_rate)
  wav_file.writeframes(audio_int16.tobytes())

# Transcribe with Google STT
recognizer = sr.Recognizer()
with sr.AudioFile(wav_filename) as source:
  audio_data = recognizer.record(source)

client = genai.Client(api_key="api key")

try:
  print("Sending to speech recognizer...")
  text = recognizer.recognize_google(audio_data, language="en-IN")
  print(f"\n--> You said: {text}")

  print("Thinking...")
  chat = client.chats.create(model="gemini-3.6-flash")
  response = chat.send_message(text)

  print(f"\n--> Gemini: {response.text}")

except sr.UnknownValueError:
  print("\nGoogle Speech Recognition could not understand the audio.")
except sr.RequestError as e:
  print(f"\nCould not request results from Google service: {e}")





