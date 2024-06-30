import logging
import os
import queue
import tempfile
import threading

import openai
import sounddevice
import soundfile
from pynput import keyboard

try:
    import azure.cognitiveservices.speech as speechsdk

    _AZURE_LOAD = True
except ImportError or ModuleNotFoundError:
    _AZURE_LOAD = False


class SpeechToTextManager:
    def __init__(self, config: dict, log_level: str, client: openai.OpenAI) -> None:
        self._config = config
        self._logger = logging.getLogger("stt-manager")
        self._logger.setLevel(log_level)
        self._client = client

        if sum([config[key]["enabled"] for key in config.keys()]) > 1:
            self._logger.error("Cannot have multiple Speech-to-Text modules enabled at once. Please edit your config")
            exit(1)

        # Azure
        self._use_azure = _AZURE_LOAD and config["azure"]["enabled"]
        if self._use_azure and (os.getenv("SPEECH_KEY") is None or os.getenv("SPEECH_REGION") is None):
            self._logger.error(
                "SPEECH_KEY and SPEECH_REGION must be set in .env or environment variables when azure is enabled"
            )
            exit(1)
        if config["azure"]["enabled"] and not _AZURE_LOAD:
            self._logger.warning("Azure STT is enabled but could not be loaded. Voice input will not be used.")
        if self._use_azure:
            speech_config = speechsdk.SpeechConfig(
                subscription=os.environ.get("SPEECH_KEY"),
                region=os.environ.get("SPEECH_REGION"),
            )
            speech_config.speech_recognition_language = "en-US"
            audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
            self._azure = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

        # OpenAI
        self._use_openai = config["openai"]["enabled"]

    def _get_push_to_talk(self, start_func, end_func, outofline_start=False) -> None:
        print("> System: Speech-to-Text will be active while holding shift")

        def on_press(key):
            if key == keyboard.Key.shift:
                if not outofline_start:
                    start_func()
                self._logger.debug("Started recording...")
                return False
            if isinstance(key, keyboard.KeyCode) and bytes(key.char, "utf-8") == b"\x03":
                self._logger.debug("Ctrl+C detected")
                return False
            return True

        def on_release(key):
            if key == keyboard.Key.shift:
                end_func()
                self._logger.debug("Stopped recording")
                return False
            return True

        with keyboard.Listener(on_press=on_press) as listener:  # Start recording
            listener.join()
        with keyboard.Listener(on_release=on_release) as listener:  # Stop recording
            if outofline_start:
                start_func()
            listener.join()

    def _get_azure(self) -> str | None:
        speech_recognition_result = None

        def get_cb(evt):
            nonlocal speech_recognition_result
            speech_recognition_result = evt.result

        self._azure.recognized.connect(get_cb)
        self._get_push_to_talk(self._azure.start_continuous_recognition, self._azure.stop_continuous_recognition)

        if speech_recognition_result is None:
            return None
        if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
            self._logger.debug(f"Microphone input interpretted as: {speech_recognition_result.text}")
            if len(speech_recognition_result.text) == 0:
                return None
            return speech_recognition_result.text
        if speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
            self._logger.warning(f"No speech could be recognized: {speech_recognition_result.no_match_details}")
        elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = speech_recognition_result.cancellation_details
            self._logger.warning(f"Speech Recognition canceled: {cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                self._logger.warning(f"Error details: {cancellation_details.error_details}")
        return None

    def _get_rec_as_file_name(self) -> str:
        flag = threading.Event()
        temp_file_name: str = None

        def start_rec():
            nonlocal temp_file_name
            sample_rate = 16000
            recording = queue.Queue()

            def cb(indata, frames, time, status):
                recording.put(indata.copy())

            fd, temp_file_name = tempfile.mkstemp(prefix="stt_input_", suffix=".wav")
            with soundfile.SoundFile(fd, mode="w", samplerate=sample_rate, channels=1, format="WAV") as file:
                with sounddevice.InputStream(samplerate=sample_rate, channels=1, callback=cb):
                    while not flag.is_set():
                        file.write(recording.get())

        def end_rec():
            flag.set()

        self._get_push_to_talk(start_rec, end_rec, True)

        return temp_file_name

    def _filter_openai_junk(self, input: str) -> str:
        # Whisper likes to make up things when input doesn't sound like speech
        junk_strings = [
            "MBC 뉴스 이덕영입니다.",
            "워싱턴에서 MBC 뉴스 이덕영입니다.",
            "😍😍😍😍",
            "😆 😆",
            "ご視聴ありがとうございました。",
            "Thank you so much for watching !",
            "지금까지 신선한 경제였습니다.",
            "ご視聴ありがとうございました",
            "시청해 주셔서 감사합니다.",
            "Thank you for watching.",
            "You",
            "You.",
            "できましたできました ご視聴ありがとうございました",
            "KATHRYN",
        ]
        if input in junk_strings:
            self._logger.warning(f'Tossing presumably junk response from OpenAI STT: "{input}"')
            return "None"
        return input

    def _get_openai(self) -> str:
        file_name = self._get_rec_as_file_name()
        file = open(file_name, "rb")
        transcript = self._client.audio.transcriptions.create(file=file, model="whisper-1")
        file.close()
        os.remove(file_name)
        return self._filter_openai_junk(transcript.text)

    def get_next(self) -> str | None:
        try:
            if self._use_azure:
                return self._get_azure()
            if self._use_openai:
                return self._get_openai()
        except Exception as e:
            self._logger.warning(f'Speech recognition failed with: "{e}"')
        return None

    def using_stt(self) -> bool:
        return self._use_azure or self._use_openai
