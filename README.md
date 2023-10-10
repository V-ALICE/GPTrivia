# SillyAiTrivia

Inspired by DougDoug's stupid (in a good way) AI streams

Should theoretically work on Windows and Linux, although I've only tested on Windows/WSL

**Required**
- `pip install -r requirements.txt`

**Optional**
- If using ElevenLabs for voice: `pip install -r voice_requirements/elevenlabs_reqs.txt`
- If using Bark for voice: `pip install -r voice_requirements/bark_reqs.txt`

**Environment**
- Add `OPENAI_API_KEY` and `OPENAI_ORG_ID` to .env file (or environment variables)
- If using ElevenLabs, add `ELEVEN_API_KEY` to .env file (or environment variables)

**Run**
- `python ai_trivia.py <path-to-config>`

**Notes**
- May need to install different versions of torch depending on CUDA versions
- Experimental Tortoise-TTS support (requires local build/install of tortoise-tts known to python)

**TODO**
- Improve Tortoise-TTS support
- Add option to quiet the local AI gen outputs
- Add Speech-to-Text
