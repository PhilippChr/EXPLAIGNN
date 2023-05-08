# Question Understanding (QU)

Module to create an intent-explicit form of the current question and the corresponding conversational history.

- [Create your own QU module](#create-your-own-qu-module)
  - [`inference_on_turn` function](#inference_on_turn-function)
  - [`inference_on_conversation` function](#inference_on_conversation-function)
  - [`train` function](#optional-train-function)

## Create your own QU module
You can inherit from the [QuestionUnderstanding](question_understanding.py) class and create your own QU module. Implementing the functions `inference_on_turn` and `inference_on_conversation` is sufficient for the pipeline to run properly. You might want to implement your own training procedure for your module via the `train` function though.

Further, you need to instantiate a logger in the class, which will be used in the parent class.
Alternatively, you can call the __init__ method of the parent class.  
Also, make sure to use the `use_gold_answers` parameter properly in your derived class.
This parameter will be given as parameter when initializing the module.

## `inference_on_turn` function

**Inputs**:
- `turn`: the current turn for which the intent-explicit representation should be generated.
- `history_turn_texts`: the previous turns in the conversation. Can be used to generate the intent-explicit form. List of the texts of the turns. Includes previous questions and answer texts.

**Description**:  
This method is supposed to generate an intent-explicit form of the current question, given the conversational history.
Please make sure that the class parameter `use_gold_answers` controls whether the gold answer(s) (in `turn["answers"]`) or the predicted answer(s) (in `turn["pred_answers"]`) are used.

**Output**:  
Returns the turn. Make sure to store the intent-explicit representation of the information need in `turn["structured_representation"]`. 

## `inference_on_conversation` function

**Inputs**:
- `conversation`: the conversation for which the intent-explicit representation should be generated.

**Description**:  
This method is supposed to generate intent-explicit forms for all turns in the conversation. In the method, you can keep track of the conversational history (e.g. using a list). Please make sure that the class parameter `use_gold_answers` controls whether the gold answer(s) (in `turn["answers"]`) or the predicted answer(s) (in `turn["pred_answers"]`) are used.

**Output**:  
Returns the conversation. Make sure to store the intent-explicit representation of the information need for every turn in the conversation, in `turn["structured_representation"]`. 

## [Optional] `train` function

**Inputs**: NONE

**Description**:  
If required, you can train your QU module here. You can make use of whatever parameters are stored in your .yml file.

**Output**: NONE
