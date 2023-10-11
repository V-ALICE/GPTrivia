import logging
import os
from pynput import keyboard

try:
    import azure.cognitiveservices.speech as speechsdk
    _AZURE_LOAD = True
except ImportError or ModuleNotFoundError:
    _AZURE_LOAD = False

try:
    import speech_recognition
    _SPHINX_LOAD = True
except ImportError or ModuleNotFoundError:
    _SPHINX_LOAD = False

class SpeechToTextManager:
    def __init__(self, config: dict, log_level: str) -> None:
        self._config = config
        self._logger = logging.getLogger("stt-manager")
        self._logger.setLevel(log_level)

        if sum([config[key]["enabled"] for key in config.keys()]) > 1:
            self._logger.error("Cannot have multiple Speech-to-Text modules enabled at once. Please edit your config")
            exit(1)

        # Azure voice
        self._use_azure = _AZURE_LOAD and config["azure"]["enabled"]
        if self._use_azure and (os.getenv("SPEECH_KEY") is None or os.getenv("SPEECH_REGION") is None):
            self._logger.error("SPEECH_KEY and SPEECH_REGION must be set in .env or environment variables when azure is enabled")
            exit(1)
        if config["azure"]["enabled"] and not _AZURE_LOAD:
             self._logger.warning("Azure STT is enabled but could not be loaded. Voice input will not be used.")
        if self._use_azure:
            speech_config = speechsdk.SpeechConfig(
                subscription=os.environ.get('SPEECH_KEY'),
                region=os.environ.get('SPEECH_REGION')
            )
            speech_config.speech_recognition_language="en-US"
            device_override = self._config["azure"]["device_id"] if self._config["azure"]["override_device_id"] else None
            audio_config = speechsdk.audio.AudioConfig(
                use_default_microphone=(not self._config["azure"]["override_device_id"]),
                device_name=device_override
            )
            self._azure = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

        # PocketSphinx voice
        self._use_sphinx = _SPHINX_LOAD and config["sphinx"]["enabled"]
        if config["sphinx"]["enabled"] and not _SPHINX_LOAD:
             self._logger.warning("Sphinx STT is enabled but could not be loaded. Voice input will not be used.")
        if self._use_sphinx:
            self._sphinx = speech_recognition.Recognizer()
            self._sphinx.energy_threshold = config["sphinx"]["energy_threshold"]
            self._sphinx.pause_threshold = config["sphinx"]["pause_threshold"]

    def _get_push_to_talk(self, start_func, end_func) -> None:
        print("> System: Speech-to-Text will be active while holding shift")
        def on_press(key):
            if key == keyboard.Key.shift:
                start_func()
                self._logger.debug("Started listening...")
                return False
            if isinstance(key, keyboard.KeyCode) and bytes(key.char, 'utf-8') == b'\x03': 
                self._logger.debug("Ctrl+C detected")
                return False
            return True
        def on_release(key):
            if key == keyboard.Key.shift:
                end_func()
                self._logger.debug("Stopped listening")
                return False
            return True
        with keyboard.Listener(on_press=on_press) as listener: # Start listening
            listener.join()
        with keyboard.Listener(on_release=on_release) as listener: # Stop listening
            listener.join()

    def _get_azure(self) -> str | None:
        speech_recognition_result = None
        if self._config["azure"]["push_to_talk"]:
            def get_cb(evt):
                nonlocal speech_recognition_result
                speech_recognition_result = evt.result
            self._azure.recognized.connect(get_cb)
            self._get_push_to_talk(
                self._azure.start_continuous_recognition,
                self._azure.stop_continuous_recognition
            )
        else:
            print("> System: Speech-to-Text is now active, and will conttinue for 15 seconds or until you stop talking")
            speech_recognition_result = self._azure.recognize_once_async().get()

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

    def _get_sphinx(self) -> str | None:
        # TODO: push-to-talk support?
        device_override = self._config["sphinx"]["device_index"] if self._config["sphinx"]["override_device_index"] else None
        with speech_recognition.Microphone(device_index=device_override) as source:
            if self._config["sphinx"]["auto_calibrate"]:
                self._logger.info("Calibrating audio input for ambient noise...")
                self._sphinx.adjust_for_ambient_noise(source)
            print("> System: Speech-to-Text is now active, and will continue until you stop talking")
            audio = self._sphinx.listen(source, timeout=10)
        try:
            text = self._sphinx.recognize_sphinx(audio)
            self._logger.debug(f"Microphone input interpretted as: {text}")
            return text
        except speech_recognition.UnknownValueError:  
            self._logger.warning("No speech could be recognized")  
        except speech_recognition.RequestError as e:  
            self._logger.warning(f"Sphinx error: {e}") 
        return None

    def get_next(self) -> str | None:
        try:
            if self._use_azure:
                return self._get_azure()
            if self._use_sphinx:
                return self._get_sphinx()
        except Exception as e:
            self._logger.warning(f'Speech recognition failed with: "{e}"')
        return None
    
    def using_stt(self) -> bool:
        return self._use_azure or self._use_sphinx