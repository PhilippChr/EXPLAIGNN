# Evidence Retrieval and Scoring (ERS)

Module to retrieve relevant evidences from (heterogeneous) information sources.

- [Create your own ERS module](#create-your-own-ers-module)
	- [`inference_on_turn` function](#inference_on_turn-function)
	- [`store_cache` function](#optional-store_cache-function)
	- [`train` function](#optional-train-function)
- [Available information sources](#available-information-sources)
- [Wikidata access](#wikidata-access)
- [Wikipedia access](#wikipedia-access)
- [Evidences format](#evidences-format)


## Create your own ERS module
You can inherit from the [`EvidenceRetrievalScoring`](evidence_retrieval_scoring.py) class and create your own ERS module.
Implementing the function `inference_on_turn` is sufficient for the pipeline to run.
In case you would like to store intermediate retrieval results, make sure to implement the `store_cache` function, which is called after the ERS module is run to store any data.
Further, you need to instantiate a logger in the class, which will be used in the parent class.
Alternatively, you can call the __init__ method of the parent class. 
Please find further details below.


## `inference_on_turn` function

**Inputs**:
- `turn`: turn that evidences are retrieved for. Will have the intent-explicit form of the current question in `turn["structured_representation"]`.
- `sources`: list of input sources.

**Description**:  
For the given intent-explicit representation of the question, retrieve relevant evidences from heterogeneous information sources. Depending on your individual module implemented, the initially retrieved evidences require to be scored to identify the top-*e* most relevant ones (*e* is defined by `evs_max_evidences` in the config).

**Output**:  
Returns the top-*e* evidences. However, the current pipeline does not make use of the return value.
Make sure to also store these evidences in `turn["top_evidences"]`. In your implementation, make sure that the config parameter `evs_max_evidences` controls the amount of evidences going into the HA part.


## [Optional] `store_cache` function

**Inputs**: NONE

**Description**:  
Whatever intermediate retrieval results you obtain in your implementation of the class, you can store these on disk to re-use them in a future run (e.g. for efficiency or reproducability). The default implementation (in [`EvidenceRetrievalScoring`](evidence_retrieval_scoring.py)) does not do anything. If you do not require storing any data, you can simply skip this function.

**Output**: NONE

## [Optional] `train` function

**Inputs**: NONE

**Description**:  
If required, you can train your ERS module here. You can make use of whatever parameters are stored in your .yml file.

**Output**: NONE

## Available information sources
The following information sources are implemented in the native CONVINSE pipeline:
- `"kb"`: KB-facts from Wikidata,
- `"text"`: text-snippets (sentence-level) from Wikipedia,
- `"table"`: table-records (row-level) from Wikipedia,
- `"info"`: infobox-entries (attribute-value-pairs) from Wikipedia.

For specifying specific combinations of information sources (e.g. for retrieval, training,...), you can either adjust the respective parameters in the config, or provide them as argument to the bash script. E.g. giving the option "kb_text_info" specifies that "kb", "text" and "info" should be set.

## Wikidata access
For accessing Wikidata, you can make use of the [`ClocqRetriever`](clocq_er.py) class.
You can:
1) retrieve relevant KB-facts for a given input snippet, using [CLOCQ](https://clocq.mpi-inf.mpg.de)'s search space retrieval functionality via the `retrieve_evidences` function, specifying the desired [input information sources](#available-information-sources) in a list,
2) retrieve relevant KB-facts for a given input snippet, using [CLOCQ](https://clocq.mpi-inf.mpg.de)'s search space retrieval functionality via the `retrieve_KB_facts` function, which will only return evidences from the KB, or 
3) retrieve KB-facts for a given Wikidata item ID via the `retrieve_kb_facts_for_item` function.

The CLOCQ parameters in the config will be used as input for the CLOCQ functions.
For quickly getting started, you can make use of the publicly available [CLOCQ API](https://clocq.mpi-inf.mpg.de), which is the default setup.
For more efficient access, you can run the CLOCQ algorithm on your local machine. Note, that this comes with quite some memory requirements of \~400 GB.

## Wikipedia access
For accessing Wikipedia text, tables and infoboxes, you can use the [`ClocqRetriever`](clocq_er.py) class, or directly use the [`WikipediaRetriever`](wikipedia_retriever/wikipedia_retriever.py) package.
You can:
1) retrieve facts from Wikipedia for a given input snippet, using [CLOCQ](https://clocq.mpi-inf.mpg.de)'s search space retrieval functionality via the `retrieve_evidences` function, specifying the desired [input information sources](#available-information-sources) in a list,
2) retrieve facts from Wikipedia for a given Wikidata item ID, using the `retrieve_wikipedia_evidences` function in the [`ClocqRetriever`](clocq_er.py) class,
3) retrieve facts from Wikipedia for a given Wikidata item ID, using the `retrieve_wp_evidences` function in the [`WikipediaRetriever`](wikipedia_retriever/wikipedia_retriever.py) package. You can adjust this function as required. Make sure to include the `retrieved_for_entity` key to the resulting evidences (not taken care of in this function).

Either way, the pipeline would try to read evidences from the cache, or the Wikipedia dump (specified with the `ers_wikipedia_dump` keyword in the config).
The parameter `ers_on_the_fly` controls, whether the Wikipedia API is called on-the-fly to retrieve evidences for entities that are not included in the specified Wikipedia dump. If `ers_on_the_fly=False`, an empty list of evidences will be returned in case an entity is not included.

## Evidences format
Evidences are stored and processed in the following format. If you plan your own implementation of the ERS module, make sure that you match this format.

``` json
{
	"evidence_text": "<TEXT>",
	"source": "kb|text|table|infobox",
	"disambiguations": [["<SURFACE_FORM_IN_EVIDENCE>", "ITEM_ID>"], ],
	"wikidata_entities": [{"id": "<ITEM_ID>", "label": "<LABEL>"}, ],
	"retrieved_for_entity": {"id": "<ITEM_ID>", "label": "<LABEL>"}
}
```

