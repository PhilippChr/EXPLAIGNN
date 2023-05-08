# Heterogeneous Answering (HA)

Module to answer the intent-explicit representation of the question, using the top-*e* retrieved evidences.

- [Create your own Ha module](#create-your-own-ha-module)
  - [`inference_on_turn` function](#inference_on_turn-function)
  - [`inference_on_conversation` function](#inference_on_conversation-function)
  - [`train` function](#optional-train-function)
- [Answer format](#answer-format)

## Create your own HA module
You can inherit from the [HeterogeneousAnswering](heterogeneous_answering.py) class and create your own QU module. Implementing the function `inference_on_turn` is sufficient for the pipeline to run properly. You might want to implement your own training procedure for your module via the `train` function though.

Further, you need to instantiate a logger in the class, which will be used in the parent class.
Alternatively, you can call the __init__ method of the parent class.  

## `inference_on_turn` function

**Inputs**:
- `turn`: the turn, for which the answer should be predicted. You can access the intent-explicit representation of the information need via `turn["structured_representation"]`, and the top-*e* evidences via `turn["top_evidences"]`.

**Description**:  
Run the HA module on the information need, and predict the answer(s).

**Output**:
Returns the turn. Make sure to add the predicted answers to `turn["pred_answers"]`. You can find additional information on the [expected answer format](#answer-format) below.

## [Optional] `train` function

**Inputs**:
- `sources`: list of sources for which the HA module should be trained. The default setting is to train a single model for all sources (and combinations of sources) for generalizability.

**Description**:  
If required, you can train your HA module here. You can make use of whatever parameters are stored in your .yml file.

**Output**: NONE

## Answer format
The predicted answers are given as list of answer-dictionaries, and should be stored in `turn["pred_answers"]`.
Note, that the answers should be normalized to Wikidata. This allows for fair comparison beyond plain string matching.
Further, in a real use-case this has the advantage that knowledge cards can be shown for the given KB items.
In case a date or year is returned, give the corresponding timestamp as the ID ("2011-04-17T00:00:00Z"; standard format in Wikidata, and the CLOCQ API), and a verbalized version as label ("17 April 2011"). You can make use of the timestamp-related functions in the [StringLibrary](../library/string_library.py).
``` json
[{
	"id": "<WIKIDATA ITEM ID>",
	"label": "<WIKIDATA ITEM LABEL>",
	"rank": "<RANK INTEGER>" 
}]
```

`rank` starts with 1, and has exactly one answer at every rank (for comparison on ConvMix).

Example:
``` json
[
  {
    "id": "Q23633",
    "label": "HBO",
    "rank": "1" 
  },
  {
    "id": "2011-04-17T00:00:00Z",
    "label": "17 April 2011",
    "rank": "2" 
  }
]
```

