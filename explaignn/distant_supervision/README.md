# Distant Supervision

- [Usage](#usage)
- [Input format](#input-format)
- [Output format](#output-format)

## Usage
For running the distant supervision on a given dataset, simply run:
```
	bash scripts/silver_annotation.sh [<PATH_TO_CONFIG>]
```
from the ROOT directory of the project.  
The paths to the input files will be read from the given config values for `train_input_path`, `dev_input_path`, and `test_input_path`.
This will create annotated versions of the benchmark in `_intermediate_representations/<BENCHMARK>/`.

## Input format
The annotation script expects the benchmark in the following (minimal) format:
```
[
	// first conversation
	{	
		"conversation_id": "<INT>",
		"questions": [
			// question 1 (complete)
			{
				"turn": 0, 
				"question_id": "<QUESTION-ID>", 
				"question": "<QUESTION>", 
				"answers": [
					{
						"id": "<Wikidata ID>",
						"label": "<Item Label>
					},
				]
			},
			// question 2 (incomplete)
			{
				"turn": 1,
				"question_id": "<QUESTION-ID>", 
				"question": "<QUESTION>",
				"answers": [
				{
					"id": "<Wikidata ID>",
					"label": "<Item Label>
				},
				]
		]
	},
	// second conversation
	{
		...
	},
	// ...
]
```
Any other keys can be provided, and will be written to the output.
You can see [here](../heterogeneous_answering#answer-format) for additional information of the expected format of the answer IDs and labels.

## Output format
The result will be stored in a .json file:

```
[
	// first conversation
	{	
		"conversation_id": "<INT>",
		"questions": [
			// question 1 (complete)
			{
				"turn": 0, 
				"question_id": "<QUESTION-ID>", 
				"question": "<QUESTION>", 
				"answers": [
				{
					"id": "<Wikidata ID>",
					"label": "<Item Label>
				},
				// supervision signals from weak supervision
				"silver_SR": [
					// SR 1
					["<STR1>", "<STR2>",]
				],
				"silver_relevant_turns": [
					// list of integers referring to the relevant turns
					// -> this data is not used in current framework
					0
				]
			},
			// question 2 (incomplete)
			{
				"turn": 1,
				"question_id": "<QUESTION-ID>", 
				"question": "<QUESTION>", 
				"completed_question": "<QUESTION>", 
				"answers": [
				{
					"id": "<Wikidata ID>",
					"label": "<Item Label>
				},
				// supervision signals from weak supervision
				"silver_SR": [
					// SR 1
					["<STR1>", "<STR2>",]
				],
				"silver_relevant_turns": [
					// list of integers referring to the relevant turns
					// -> this data is not used in current framework
					0
				]
			},
			// ...
		]
		// ...
	},
	// second conversation
	{
		...
	},
	// ...
]
```
