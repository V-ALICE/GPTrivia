# SillyAiTrivia

Inspired by DougDoug's stupid (in a good way) AI streams

**Required**
- `pip install -r requirements.txt`

**Optional**
- If using ElevenLabs (remote) for TTS: `pip install elevenlabs soundfile sounddevice`
- If using Bark (local) for TTS: `pip install sounddevice git+https://github.com/suno-ai/bark.git`
- If using Azure (remote) for TTS and/or STT: `pip install azure-cognitiveservices-speech`
- If using Sphinx (local) for STT: `pip install PyAudio pocketsphinx pySpeechRecognition`

**Environment**
- Add `OPENAI_API_KEY` and `OPENAI_ORG_ID` to .env file (or environment variables)
- If using ElevenLabs, add `ELEVEN_API_KEY` to .env file (or environment variables)
- If using Azure, add `SPEECH_KEY` and `SPEECH_REGION` to .env file (or environment variables)

**Run**
- `python ai_trivia.py <path-to-config>`

**Notes**
- Mostly only tested on Windows
- Speech-to-Text will use the default mic unless an override is set
- May need to install different versions of torch depending on CUDA versions

**TODO**
- Add option to quiet the local AI gen printing
- Add option to import custom prompt from file
