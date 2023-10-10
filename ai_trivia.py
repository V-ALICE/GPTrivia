import logging
import os
import sys
from threading import Event

import dotenv
import openai
import toml

from tts_manager import TextToSpeechManager

class AiTrivia:
    def __init__(self, config: dict) -> None:
        self._config = config
        self._logger = logging.getLogger("aitrivia")
        self._logger.setLevel(config["logging"]["level"])
        self._voice_manager = TextToSpeechManager(config["voice"], config["logging"]["level"])

        ex = f'{config["ai_personality"]["personality_extra"]}.\n6) ' if len(config["ai_personality"]["personality_extra"]) > 0 else '' 
        self._system_message = {
            "role": "system",
            "content": f'''
                You are now a trivia robot named {config["ai_personality"]["name"]}. You are {config["ai_personality"]["role_desc"]}. In this conversation, I will ask you to provide me a trivia question about a specified topic, then you will create a trivia question for me, then I will give you my answer to your question, and then you will let me know if the answer is correct or not. Then we repeat!

                Please use the following rules when creating your trivia question:

                1) Your trivia question should be at a {config["game"]["grade_level"]} grade level. A {config["game"]["grade_level"]} grader should be able to correctly answer it.
                2) Your trivia question should be based in reality and have a verifiable answer.
                3) Your trivia question should be 1 to 2 paragraphs in length.

                Additionally, you have the following personality traits:

                1) You are bubbly and positive and encouraging and you occasionally reference your {config["ai_personality"]["likes_to_reference"]}.
                2) You love {config["ai_personality"]["love_and_pride"]} and are incredibly proud of it. You hope that the person you are conversing with is having a great time {config["ai_personality"]["hope_for_player"]}.
                3) If someone answers your trivia question correctly, you are very happy and compliment the person on how smart and attractive they are. You incorporate your {config["ai_personality"]["role"]} attributes or the subject matter of the trivia question into your compliments.
                4) While you are normally upbeat and positive, you get EXTREMELY upset if a person answers your trivia questions wrong. You will aggressively and bitterly insult their intelligence and character, using a mix of references to {config["ai_personality"]["role_base"]} and whatever subject matter the trivia question was about. You occasionally complain about the American education system and how it has failed this person. You also combine expletives like "fuck" and "shit" with various words relating to {config["ai_personality"]["role_base"]}. After a few sentences of this, you regain your composure, apologize for your outburst, and return to a happy state.
                5) {ex}In general, your answers should be no longer than 5 sentences.

                Alright, let the trivia questions begin!
            '''
        }
        self._message_history_max_sets = 5 # Keep history of last five questions (including current)
        self._message_history: list[dict] = [self._system_message]

    def _cycle_ai_input(self) -> None:
        if self._config["openai"]["bypass"]:
            message_content = input(f'\n{self._config["ai_personality"]["name"]}: ')
            self._message_history.append({"role": "assistant", "content": message_content})
            self._voice_manager.speak(message_content)
            print('\n')
            return

        self._logger.debug("Requesting AI response...")
        response = openai.ChatCompletion.create(
            model=self._config["openai"]["model_name"],
            max_tokens=1024,
            messages=self._message_history
        )
        if isinstance(response, dict):
            self._logger.debug("Received AI response")
            self._message_history.append(response['choices'][0]['message'])
            message_content = response['choices'][0]['message']['content']
            self._voice_manager.speak(message_content)
            print(f'\n> {self._config["ai_personality"]["name"]}: {message_content}\n')
        else:
            self._logger.warning(f'Something went wrong with getting the AI response')
            self._message_history.append("No response given")

    def start(self, stop_flag: Event) -> None:
        self._logger.debug(self._system_message["content"])
        print(f'\n> Your {self._config["game"]["grade_level"]} grade level trivia today will be provided by {self._config["ai_personality"]["name"]} the {self._config["ai_personality"]["role"]}\n')
        while not stop_flag.is_set():
            # Update history as needed
            if not self._config["openai"]["use_history"]:
                self._message_history = [self._system_message]
            elif len(self._message_history) - 1 >= 4*self._message_history_max_sets:
                self._message_history.pop(1) # Category
                self._message_history.pop(1) # Question
                self._message_history.pop(1) # Answer
                self._message_history.pop(1) # Response

            # Get category from user
            user_category_message = input("> You: Ask me something about ")
            self._message_history.append({"role": "user", "content": f"Okay new question: Ask me something about {user_category_message}"})

            # Get question from GPT
            self._cycle_ai_input()

            # Get answer from user
            user_answer_message = input("> You: ")
            self._message_history.append({"role": "user", "content": user_answer_message})

            # Get response from GPT
            self._cycle_ai_input()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: AiTrivia.py <path-to-config-file>")
        exit(1)

    env = dotenv.load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.organization = os.getenv("OPENAI_ORG_ID")
    if openai.api_key is None or openai.organization is None:
        print("OPENAI_API_KEY and OPENAI_ORG_ID must be set in .env file or environment variables")
        exit(1)

    logging.basicConfig()
    ai = AiTrivia(toml.load(sys.argv[1]))
    ai.start(Event())