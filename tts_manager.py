import io
import logging
import os
import threading
from queue import Empty, Queue

import openai
import sounddevice
import soundfile

try:
    import elevenlabs

    _ELEVEN_LOAD = True
except ImportError or ModuleNotFoundError:
    _ELEVEN_LOAD = False

try:
    import bark

    _BARK_LOAD = True
except ImportError or ModuleNotFoundError:
    _BARK_LOAD = False


class TextToSpeechManager:
    def __init__(self, config: dict, log_level: str, client: openai.OpenAI) -> None:
        self._config = config
        self._logger = logging.getLogger("tts-manager")
        self._logger.setLevel(log_level)
        self._client = client  # TODO: add OpenAI TTS support

        if sum([config[key]["enabled"] for key in config.keys()]) > 1:
            self._logger.error("Cannot have multiple Text-to-Speech modules enabled at once. Please edit your config")
            exit(1)

        # ElevenLabs audio
        self._use_eleven = _ELEVEN_LOAD and config["eleven"]["enabled"]
        if self._use_eleven and os.getenv("ELEVEN_API_KEY") is None:
            self._logger.error("ELEVEN_API_KEY must be set in .env or environment variables when eleven is enabled")
            exit(1)
        if config["eleven"]["enabled"] and not _ELEVEN_LOAD:
            self._logger.warning("ElevenLabs TTS is enabled but could not be loaded. Audio will not be played.")

        # Bark audio
        self._use_bark = _BARK_LOAD and config["bark"]["enabled"]
        if config["bark"]["enabled"] and not _BARK_LOAD:
            self._logger.warning("Bark is enabled but could not be loaded. Audio will not be played.")
        if self._use_bark:
            self._logger.info("Loading Bark models... If the models aren't cached yet this will take a while")
            bark.preload_models()

        # OpenAI audio
        self._use_openai = config["openai"]["enabled"]

    def _smart_split(self, content: str, max_len: int = 200) -> list[str]:
        tier1_splits = [".", "?", "!"]  # Ideal to split at these characters
        tier2_splits = [":", ";", "—"]  # If no tier 1 splits are available, use these
        tier3_splits = [","]  # If no tier 1 or 2 splits are available, use these
        tier4_splits = [" "]  # If no other splits are available, these can used as a worst-case scenario
        # These might come after a tier 1 split character, in which case they should be included
        tier1_additions = ['"', "?", "!"]

        segments: list[str] = []
        while len(content) > max_len:
            idx = max([content.rfind(x, 0, max_len) for x in tier1_splits])
            if idx == -1:
                idx = max([content.rfind(x, 0, max_len) for x in tier2_splits])
            else:
                while idx + 1 < len(content) and content[idx + 1] in tier1_additions:
                    idx += 1
            if idx == -1:
                idx = max([content.rfind(x, 0, max_len) for x in tier3_splits])
            if idx == -1:
                idx = max([content.rfind(x, 0, max_len) for x in tier4_splits])
            if idx == -1:  # This text has no useful places to split within the given max_len
                break
            idx += 1  # Add one so the split character is included with first chunk
            segments.append(content[:idx].strip())
            content = content[idx:]

        segments.append(content.strip())
        return segments

    def _clean_input(self, content: str, avoid_ellipses: bool = False) -> str:
        trash = ["\r", "\n", "\t"]  # Characters that shouldn't be in voice synth text
        for char in trash:
            content = content.replace(char, " ")
        if avoid_ellipses:
            content.replace("...", ",")  # ellipses sometimes break voice synth
            content.replace("…", ",")
        return content

    def _speak_eleven(self, content: str) -> bool:
        # TODO: add stream support?
        self._logger.info("Requesting voice synth from ElevenLabs...")
        message_audio = elevenlabs.generate(
            text=self._clean_input(content),
            voice=self._config["eleven"]["voice_name"],
            model=self._config["eleven"]["model_type"],
        )
        elevenlabs.play(message_audio, use_ffmpeg=self._config["eleven"]["ffmpeg_available"])
        return True

    def _speak_bark(self, content: str) -> bool:
        def keep_playing(q: Queue, gen_done: threading.Event) -> None:
            while not gen_done.is_set() or not q.empty():
                try:
                    audio = q.get(timeout=1)
                    sounddevice.wait()
                    sounddevice.play(audio, bark.SAMPLE_RATE)
                except Empty:
                    pass
                except KeyboardInterrupt:
                    break

        contents = self._smart_split(self._clean_input(content), 150)
        if len(contents) > 1:
            self._logger.debug(f"Text was split into segments: {contents}")
        if self._config["bark"]["stream"]:
            self._logger.info(
                "Generating streaming voice synth with Bark... Depending how fast segments generate, "
                "there may be gaps in the audio"
            )
            audio_queue = Queue()
            gen_done = threading.Event()
            cycle = threading.Thread(target=keep_playing, args=(audio_queue, gen_done))
            for _ in range(min(self._config["bark"]["stream_preload"], len(contents))):
                audio_queue.put(bark.generate_audio(contents.pop(0), history_prompt=self._config["bark"]["voice_name"]))
            cycle.start()
            for text in contents:
                audio_queue.put(bark.generate_audio(text, history_prompt=self._config["bark"]["voice_name"]))
            gen_done.set()
            cycle.join()
            sounddevice.wait()
        else:
            self._logger.info("Generating voice synth with Bark... This may take a while")
            audios: list = []
            for text in contents:
                audios.append(bark.generate_audio(text, history_prompt=self._config["bark"]["voice_name"]))
            for audio in audios:
                sounddevice.play(audio, bark.SAMPLE_RATE)
                sounddevice.wait()
        return True

    def _speak_openai(self, content: str) -> bool:
        model = "tts-1-hd" if self._config["openai"]["high_quality"] else "tts-1"
        response = self._client.audio.speech.create(
            model=model,
            voice=self._config["openai"]["voice_name"],
            input=content,
        )

        data, sample_rate = soundfile.read(io.BytesIO(response.read()))
        sounddevice.play(data, sample_rate)
        sounddevice.wait()

    def speak(self, content: str) -> bool:
        try:
            if self._use_eleven:
                return self._speak_eleven(content)
            if self._use_bark:
                return self._speak_bark(content)
            if self._use_openai:
                return self._speak_openai(content)
        except Exception as e:
            self._logger.warning(f'Voice synthesis failed with: "{e}"')
        return False
