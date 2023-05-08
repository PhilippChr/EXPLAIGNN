EXPLAIGNN
============
Explainable Conversational Question Answering over Heterogeneous Sources via Iterative Graph Neural Networks
---------------


- [Description](#description)
- [Code](#code)
    - [System requirements](#system-requirements)
	- [Installation](#installation)
	- [Reproduce paper results](#reproduce-paper-results)
	- [Training the pipeline](#training-the-pipeline)
	- [Testing the pipeline](#testing-the-pipeline)
	- [Using the pipeline](#using-the-pipeline)
- [ConvMix](#convmix)
	- [Loading ConvMix](#loading-convmix)
	- [Comparing on ConvMix](#comparing-on-convmix)
- [Data](#data)
	- [Wikidata](#wikidata)
	- [Wikipedia](#wikipedia)
- [Feedback](#feedback)
- [License](#license)

# Description
This repository contains the code and data for our [SIGIR 2023 paper](https://arxiv.org/abs/2305.01548) on "Explainable Conversational Question Answering over Heterogeneous Sources via Iterative Graph Neural Networks", and builds upon the [CONVINSE code](https://github.com/PhilippChr/CONVINSE) for our [SIGIR 2022 paper](https://dl.acm.org/doi/10.1145/3477495.3531815).

Our new approach, **EXPLAIGNN** follows the following general pipeline:
1) Question Understanding (QU) -- creating an intent-explicit structured representation of a question and its conversational context
2) Evidence Retrieval (ER) -- harnessing this frame-like representation to uniformly capture relevant evidences from different sources
3) Heterogeneous Answering (HA) -- deriving the answer from this set of evidences from heterogeneous sources.

The focus of this work is on the answering phase. 
In this stage, a heterogeneous graph is constructed from the retrieved evidences and corresponding entities,
as the basis for applying graph neural networks (GNNs).
The GNNs are iteratively applied for computing the best answers and supporting evidences in a small number of steps.

Further details can be found on the [EXPLAIGNN website](https://explaignn.mpi-inf.mpg.de) and in the corresponding [paper pre-print](https://arxiv.org/abs/2305.01548).
An interactive demo will also follow soon!

If you use this code, please cite:
```bibtex
@article{christmann2023explainable,
  title={Explainable Conversational Question Answering over
Heterogeneous Sources via Iterative Graph Neural Networks},
  author={Christmann, Philipp and Roy, Rishiraj Saha and Weikum, Gerhard},
  journal={SIGIR},
  year={2023}
}
```

# Code

## System requirements
All code was tested on Linux only.
- [Conda](https://docs.conda.io/projects/conda/en/latest/index.html)
- [PyTorch](https://pytorch.org)
- GPU (suggested)

## Installation
Clone the repo via:
We recommend the installation via conda, and provide the corresponding environment file in [conda-explaignn.yml](conda-explaignn.yml):

```bash
    git clone https://github.com/PhilippChr/EXPLAIGNN.git
    cd EXPLAIGNN/
    conda env create --file conda-explaignn.yml
    conda activate explaignn
    pip install -e .
```

Alternatively, you can also install the requirements via pip, using the [requirements.txt](requirements.txt) file (not tested). In this case, for running the code on a GPU, further packages might be required.

### Install dependencies
EXPLAIGNN makes use of [CLOCQ](https://github.com/PhilippChr/CLOCQ) for retrieving relevant evidences.
CLOCQ can be conveniently integrated via the [publicly available API](https://clocq.mpi-inf.mpg.de), using the client from [the repo](https://github.com/PhilippChr/CLOCQ). If efficiency is a primary concern, it is recommended to directly run the CLOCQ code on the local machine (details are given in the repo).  
In either case, it can be installed via:
```bash
    make install_clocq
```

Optional: If you want to use or compare with [QuReTeC](https://github.com/nickvosk/sigir2020-query-resolution) or [FiD](https://github.com/facebookresearch/FiD) please follow the installation guides in the [CONVINSE repo](https://github.com/PhilippChr/CONVINSE).

To initialize the repo (download data, benchmark, models), run:
```bash
    bash scripts/initialize.sh
```


## Reproduce paper results

If you would like to reproduce the results of EXPLAIGNN for all sources (Table 1 in the SIGIR 2023 paper), or a specific source combination, run:
``` bash
    bash scripts/pipeline.sh --gold-answers config/convmix/explaignn.yml kb_text_table_info
```
the last parameter (`kb_text_table_info`) specifies the sources to be used, separated by an underscore. E.g. "kb_text_info" would evaluate EXPLAIGNN using evidences from KB, text and infoboxes.
Note, that EXPLAIGNN retrieves evidences on-the-fly by default.
Given that the evidences in the information sources can change quickly (e.g. Wikipedia has many updates every day),
results can easily change.
A cache was implemented to improve the reproducability, and we provide a benchmark-related subset of Wikipedia (see details [below](#wikipedia)).

For reproducing the results of EXPLAIGNN using the predicted answers for previous turns (Table 2 in the SIGIR 2023 paper), run:
``` bash
    bash scripts/pipeline.sh --pred-results config/convmix/explaignn.yml kb_text_table_info
```

Finally, for reproducing the results of EXPLAIGNN with different input source combinations (Table 3 in the SIGIR 2023 paper), run:
``` bash
    bash scripts/pipeline.sh --source-combinations config/convmix/explaignn.yml kb_text_table_info
```

The results will be logged in the `out/convmix` directory, and the metrics written to `_results/convmix/explaignn.res`.

## Training the pipeline

To train a pipeline, just choose the config that represents the pipeline you would like to train, and run:
``` bash
    bash scripts/pipeline.sh --train [<PATH_TO_CONFIG>] [<SOURCES_STR>]
```

Example:
``` bash
    bash scripts/pipeline.sh --train config/convmix/explaignn.yml kb_text_table_info
```


## Testing the pipeline

If you create your own pipeline, it is recommended to test it once on an example, to verify that everything runs smoothly.  
You can do that via:
``` bash
    bash scripts/pipeline.sh --example [<PATH_TO_CONFIG>]
```
and see the output file in `out/<benchmark>` for potential errors.

For standard evaluation, you can simply run:
``` bash
    bash scripts/pipeline.sh --gold-answers [<PATH_TO_CONFIG>] [<SOURCES_STR>]
```
Example:
``` bash
    bash scripts/pipeline.sh --gold-answers config/convmix/explaignn.yml kb_text_table_info
```
<br/>

For evaluating with all source combinations, run:
``` bash
    bash scripts/pipeline.sh --source-combinations [<PATH_TO_CONFIG>] [<SOURCES_STR>]
```

Example:
``` bash
    bash scripts/pipeline.sh --source-combinations config/convmix/explaignn.yml kb_text_table_info
```
<br/>

If you want to evaluate using the predicted answers of previous turns, you can run:
``` bash
    bash scripts/pipeline.sh --pred-answers [<PATH_TO_CONFIG>] [<SOURCES_STR>]
```

Example:
``` bash
    bash scripts/pipeline.sh --pred-answers config/convmix/explaignn.yml kb_text_table_info
```
By default, the EXPLAIGNN config and all sources will be used.


The results will be logged in the following directory: `out/<DATA>/<CMD>-<FUNCTION>-<CONFIG_NAME>.out`,  
and the metrics are written to `_results/<DATA>/<CONFIG_NAME>.res`.

## Using the pipeline
For using the pipeline, e.g. for improving individual parts of the pipeline, you can simply implement your own method that inherits from the respective part of the pipeline, create a corresponding config file, and add the module to the pipeline.py file. You can then use the commands outlined above to train and test the pipeline. 
Please see the documentation of the individual modules for further details:
- [Distant Supervision](explaignn/distant_supervision/README.md)
- [Question Understanding (QU)](explaignn/question_understanding/README.md)
- [Evidence Retrieval and Scoring (ERS)](explaignn/evidence_retrieval_scoring/README.md)
- [Heterogeneous Answering (HA)](explaignn/heterogeneous_answering/README.md)

# ConvMix
## Loading ConvMix
The ConvMix dataset can be downloaded (if not already done so via the initialize-script) via:
```
	bash scripts/download.sh convmix
```

Then, the individual ConvMix data splits can be loaded via:
```
	import json
	with open ("_benchmarks/convmix/train_set_ALL.json", "r") as fp:
		train_data = json.load(fp)
	with open ("_benchmarks/convmix/dev_set_ALL.json", "r") as fp:
		dev_data = json.load(fp)
	with open ("_benchmarks/convmix/test_set_ALL.json", "r") as fp:
		test_data = json.load(fp)
```

You could also load domain-specific versions, by replacing "ALL" by either "books", "movies", "music", "soccer" or "tvseries".    
The data will have the following format:
```
[
	// first conversation
	{	
		"conv_id": "<INT>",
		"domain": "<STRING>",
		"questions": [
			// question 1 (complete)
			{
				"turn": 0,
				"question_id": "<STRING: QUESTION-ID>", 
				"question": "<STRING: QUESTION>", 
				"answers": [
					{
						"id": "<STRING: Wikidata ID of answer>",
						"label": "<STRING: Item Label of answer>
					},
				]
				"answer_text": "<STRING: textual form of answer>",
				"answer_src": "<STRING: source the worker found the answer>",
				"entities": [
					{
						"id": "<STRING: Wikidata ID of question entity>",
						"label": "<STRING: Item Label of question entity>
					},
				],
				"paraphrase": "<STRING: paraphrase of current question>"
				
			},
			// question 2 (incomplete)
			{
				"turn": 1,
				"question_id": "<STRING: QUESTION-ID>", 
				"question": "<STRING: QUESTION>", 
				"answers": [
					{
						"id": "<STRING: Wikidata ID of answer>",
						"label": "<STRING: Item Label of answer>
					},
				]
				"answer_text": "<STRING: textual form of answer>",
				"answer_src": "<STRING: source the worker found the answer>",
				"entities": [
					{
						"id": "<STRING: Wikidata ID of question entity>",
						"label": "<STRING: Item Label of question entity>
					},
				],
				"paraphrase": "<STRING: paraphrase of current question>",
				"completed": "<STRING: completed version of current incomplete question>"
		]
	},
	// second conversation
	{
		...
	},
	// ...
]
```


## Comparing on ConvMix
Please make sure that...
- ...you use our dedicated Wikipedia dump, to have a comparable Wikipedia version (see further details below).
- ...you use the same Wikidata dump (2022-01-31), which can be conveniently accessed using the CLOCQ API available at https://clocq.mpi-inf.mpg.de (see further details below).
- ...you use the same evaluation method as EXPLAIGNN (as defined in explaignn/evaluation.py).


# Data
## Wikidata
For the EXPLAIGNN and CONVINSE projects, the Wikidata dump with the timestamp 2022-01-31 was used, which is currently also accessible via the [CLOCQ API](https://clocq.mpi-inf.mpg.de).
Further information on how to retrieve evidences from Wikidata can be found in the [ERS documentation](explaignn/evidence_retrieval_scoring/README.md#wikidata-access).


## Wikipedia
Wikipedia evidences can be retrieved on-the-fly using the [`WikipediaRetriever`]() package. However, we provide a ConvMix-related subset, that can be downloaded via:
``` bash
    bash scripts/download.sh wikipedia
```
Note that this data is also downloaded by the default [initialize script](scripts/initialize.sh).   
The dump is provided as a .pickle file, and provides a mapping from Wikidata item IDs (e.g. Q38111) to Wikipedia evidences.

This ConvMix-related subset has been created as follows. We added evidences retrieved from Wikipedia in 2022-03/04 for the following Wikidata items:
1) all answer entities in the ConvMix benchmark,
2) all question entities in the ConvMix benchmark (as specified by crowdworkers),
3) the top-20 disambiguations for each entity mention detected by CLOCQ, with the input strings being the intent-explicit forms generated for the ConvMix dataset by the EXPLAIGNN pipeline, or the baseline built upon the 'Prepend all' QU method,
4) and whenever new Wikidata entities occured (e.g. for the dynamic setup running the pipeline with predicted answers), we added the corresponding evidences to the dump.

We aim to maximize the number of entities (and thus of evidences) here, to allow for fair (as far as possible) comparison with dense retrieval methods. Crawling the whole Wikipedia dump was out of scope (also Wikimedia strongly discourages this). In total we collected \~230,000 entities, for which we tried retrieving Wikipedia evidences. Note that for some of these, the result was empty.

Further information on how to retrieve evidences from Wikipedia can be found in the [ERS documentation](explaignn/evidence_retrieval_scoring/README.md#wikipedia-access)

# Feedback
We tried our best to document the code of this project, and for making it accessible for easy usage, and for testing your custom implementations of the individual pipeline-components. However, our strengths are not in software engineering, and there will very likely be suboptimal parts in the documentation and code.
If you feel that some part of the documentation/code could be improved, or have other feedback,
please do not hesitate and let use know! You can e.g. contact us via mail: pchristm@mpi-inf.mpg.de.
Any feedback (also positive ;) ) is much appreciated!

# License
The EXPLAIGNN project by [Philipp Christmann](https://people.mpi-inf.mpg.de/~pchristm/), [Rishiraj Saha Roy](https://people.mpi-inf.mpg.de/~rsaharo/) and [Gerhard Weikum](https://people.mpi-inf.mpg.de/~weikum/) is licensed under [MIT license](LICENSE).

