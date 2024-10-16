import openai
from openai import OpenAI
import os
import tempfile
import sounddevice as sd
import soundfile as sf
import numpy as np

class AudioHandler:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def speak_openai(self, text):
        response = self.client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )
        
        # Save the audio to a temporary file and play it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            for chunk in response.iter_bytes(chunk_size=1024):
                temp_audio.write(chunk)
        
        # Play the audio file
        data, samplerate = sf.read(temp_audio.name)
        sd.play(data, samplerate)
        sd.wait()  # Wait until the audio is finished playing
        
        print(f"Speaking: {text}")

        # Remove the temporary file after playing
        os.unlink(temp_audio.name)

    def listen_openai(self, duration=5):
        print("Listening... Speak now.")
        
        # Record audio
        fs = 44100  # Sample rate
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()  # Wait until recording is finished

        # Save recording to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            sf.write(temp_audio.name, recording, fs)
        
        # Transcribe the audio file
        with open(temp_audio.name, "rb") as file:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1", 
                file=file
            )
        
        # Remove the temporary file
        os.unlink(temp_audio.name)
        
        print(f"Transcribed: {transcript.text}")  # Print the transcribed text
        return transcript.text

    # Keep the original speak and listen methods as fallback
    def speak(self, text):
        print(f"Speaking: {text}")  # Placeholder for the original speak method

    def listen(self):
        return input("Listening (type your input): ")  # Placeholder for the original listen method