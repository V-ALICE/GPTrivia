# GPTrivia

Play stupid trivia with GPT-4, inspired by DougDoug's ridiculous AI trivia stream.

**Notes**
- I threw this together just for fun and just for me, so the code is messy and might not work correctly sometimes. These instructions may not be complete or accurate either
- Only tested on Windows. The resources *could* work on Linux, but would require extra setup/packages that aren't noted here
- Speech-to-Text will only use the current default mic

**Required**
- Python >=3.8
- `pip install -r requirements.txt`

**Optional**
- If using any of these make sure to set the `enabled` flags in the config file you're using
- If using ElevenLabs (remote) for text-to-speech: `pip install elevenlabs`
- If using Bark (local) for text-to-speech: `pip install git+https://github.com/suno-ai/bark.git`
- If using Azure (remote) for speech-to-text: `pip install azure-cognitiveservices-speech`

**Environment**
- Add `OPENAI_API_KEY` and `OPENAI_ORG_ID` to .env file (or environment variables)
- If using ElevenLabs, add `ELEVEN_API_KEY` to .env file (or environment variables)
- If using Azure, add `SPEECH_KEY` and `SPEECH_REGION` to .env file (or environment variables)

**Run**
- `python ai_trivia.py <path-to-config>`

**Regarding Bark**
- See https://github.com/suno-ai/bark
- Bark is a local Text-to-Speech model and takes time to run even on high-end hardware. If you don't have a decent GPU with 8+ GB of VRAM, consider sticking to remote options
- You may need to install a different version of torch depending on your CUDA version
- The full Bark model requires 12GB of VRAM. This can be lowered to 8GB (4GB?) by setting `SUNO_USE_SMALL_MODELS=True` in your environment

**TODO**
- Add local Whisper STT?
- Add option to quiet the local AI logging?
