import sys

from explaignn.library.utils import get_config
from explaignn.question_understanding.question_understanding import QuestionUnderstanding


class NaiveConcat(QuestionUnderstanding):
    """
    Prepend various parts of the ongoing conversation to the current turn.
    A turn refers to the question and the answers.
    Answers can be predicted answers or generated answers.
        - Option 1 (init): Prepend initial turn
        - Option 2 (prev): Prepend previous turn
        - Option 3 (init_prev): Prepend initial and previous turn
        - Option 4 (all): Prepend ALL previous turns
    The option can be set in the config file.
    """

    def inference_on_turn(self, turn, history_turns):
        """Run model on single turn and add predictions."""
        intent_explicit = self._preprend_history(history_turns, turn)
        turn["structured_representation"] = intent_explicit
        return turn

    def inference_on_conversation(self, conversation):
        """Run inference on a single conversation."""
        history_turns = list()
        for turn in conversation["questions"]:
            # concat history to question
            question = self._preprend_history(history_turns, turn)
            turn["structured_representation"] = question

            # append to history
            history_turns.append(turn)
        return conversation

    def _preprend_history(self, history_turns, current_turn):
        """
        Transform the relevant turns and current turn into the input text.
        """
        ## consider last turn and first turn only
        if self.config["naive_concat"] == "init_prev":
            if len(history_turns) > 2:
                history_turns = [history_turns[0], history_turns[-1]]

        ## consider first turn only
        elif self.config["naive_concat"] == "init":
            if len(history_turns) > 1:
                history_turns = [history_turns[0]]

        ## consider last turn only
        elif self.config["naive_concat"] == "prev":
            if len(history_turns) > 1:
                history_turns = [history_turns[-1]]

        ## consider ALL turns
        elif self.config["naive_concat"] == "all":
            history_turns = history_turns

        ## consider only current turn
        elif self.config["naive_concat"] == "none":
            history_turns = []

        else:
            raise Exception("Unknown value for naive_concat!")

        # create history text
        history_text = " ".join(
            [self._history_turn_to_text(history_turn) for history_turn in history_turns]
        )

        # create input
        current_question = current_turn["question"]
        input_text = f"{history_text} {current_question}"
        return input_text

    def _history_turn_to_text(self, history_turn):
        """
        Transform the given history turn to text.
        """
        question = history_turn["question"]

        # use predicted answer in end-to-end evaluation
        if self.use_gold_answers:
            answers = history_turn["answers"]
            answers_text = ", ".join([answer["label"] for answer in answers])
        else:
            answer = history_turn["pred_answers"][0]
            answers_text = answer["label"]

        history_turn_text = f"{question} {answers_text}"
        return history_turn_text


#######################################################################################################################
#######################################################################################################################
if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception(
            "Invalid number of options provided.\nUsage: python explaignn/question_understanding/naive_concat/naive_concat.py <PATH_TO_CONFIG>"
        )

    # load config
    config_path = sys.argv[1]
    config = get_config(config_path)
    naive_concat = NaiveConcat(config, use_gold_answers=True)
    naive_concat.inference()
