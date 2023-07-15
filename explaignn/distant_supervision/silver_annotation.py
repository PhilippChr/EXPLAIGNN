import json
import matplotlib.pyplot as plt
import networkx as nx
import os
import pickle
import random
import sys
from clocq.CLOCQ import CLOCQ
from clocq.interface.CLOCQInterfaceClient import CLOCQInterfaceClient
from networkx.drawing.nx_agraph import graphviz_layout
from pathlib import Path
from tqdm import tqdm

from conv_flow_annotator import ConvFlowAnnotator
from explaignn.library.string_library import StringLibrary
from explaignn.library.utils import get_config, get_logger
from structured_representation_annotator import StructuredRepresentationAnnotator
from turn_relevance_annotator import TurnRelevanceAnnotator


class NoFlowGraphFoundException(Exception):
    pass

class SilverAnnotation:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__, config)

        # initialize clocq
        if config["clocq_use_api"]:
            self.clocq = CLOCQInterfaceClient(host=config["clocq_host"], port=config["clocq_port"])
        else:
            self.clocq = CLOCQ()

        # initialize annotators
        self.sr_annotator = StructuredRepresentationAnnotator(self.clocq, config)
        self.fg_annotator = ConvFlowAnnotator(self.clocq, config)
        self.tr_annotator = TurnRelevanceAnnotator(config)

        #  open labels
        labels_path = config["path_to_labels"]
        with open(labels_path, "rb") as fp:
            self.labels_dict = pickle.load(fp)

    def process_dataset(self, dataset_path, output_path, tr_data_path):
        """
        Annotate the given dataset and store the output in the specified path.
        """
        self.logger.info(f"Starting to annotate {dataset_path}...")
        with open(dataset_path, "r") as fp:
            dataset = json.load(fp)

        # initialize
        total_question_count = 0
        fail_count = 0
        tr_dataset = list()

        # process data
        for conversation in tqdm(dataset):
            # initialize
            for turn in conversation["questions"]:
                turn["silver_SR"] = []
                turn["silver_relevant_turns"] = None

            # annotate data
            try:
                # create conversation flow graph
                flow_graph = self.fg_annotator.get_conv_flow_graph(conversation)
                if not flow_graph:
                    raise NoFlowGraphFoundException("No flow graph found for conversation!")

                # add annotations
                self.sr_annotator.annotate_structured_representations(flow_graph, conversation)
                tr_data = self.tr_annotator.annotate_turn_relevances(flow_graph, conversation)
                tr_dataset += tr_data

                # count questions
                question_count = self.fg_annotator.get_question_count(flow_graph)
                total_question_count += question_count

            except NoFlowGraphFoundException as _:
                pass

        # log
        self.logger.info(f"Done with annotating on {dataset_path}.")
        self.logger.info(f"- No. SRs extracted: {total_question_count}")
        self.logger.info(f"- No. Fails: {fail_count}")

        # store annotated dataset
        with open(output_path, "w") as fp:
            fp.write(json.dumps(dataset, indent=4))

        # store turn relevance dataset
        if self.config["tr_extract_dataset"]:
            with open(tr_data_path, "w") as fp:
                fp.write(json.dumps(tr_dataset, indent=4))

        self.logger.info(f"Done with annotating {dataset_path}!")

    def _plot_graph(self, flow_graph, metadata=False, structured_representations=None):
        """
        For development purposes: Plot the flow graph using nx and plt.
        """
        if structured_representations:
            structured_representations = {
                turn["question"]: turn["structured_representation"]
                for turn in structured_representations
            }
        if self.config["log_level"] == "DEBUG":
            di_graph = nx.DiGraph()
            leafs = flow_graph["leafs"]
            while leafs:
                for node in leafs:
                    turn = node["turn"]
                    question = node["question"]
                    label = str(turn) + ": " + question
                    if node["type"] == "question" and metadata:
                        if structured_representations:
                            label += "\n" + str(structured_representations[node["question"]])
                        else:
                            label += "\nDIS=" + self._print_diambiguation_triples(
                                node["relevant_disambiguations"]
                            )
                            label += "\nCXT=" + self._print_diambiguation_triples(
                                node["relevant_context"]
                            )
                    di_graph.add_node(label)
                    parents = node["parents"]
                    for parent in parents:
                        parent_question = parent["question"]
                        parent_turn = parent["turn"]
                        parent_label = str(parent_turn) + ": " + parent_question
                        if parent["type"] == "question" and metadata:
                            if structured_representations:
                                parent_label += "\n" + str(
                                    structured_representations[parent["question"]]
                                )
                            else:
                                parent_label += "\nDIS=" + self._print_diambiguation_triples(
                                    parent["relevant_disambiguations"]
                                )
                                parent_label += "\nCXT=" + self._print_diambiguation_triples(
                                    parent["relevant_context"]
                                )
                        di_graph.add_node(parent_label)
                        di_graph.add_edge(parent_label, label)
                leafs = [node for leaf in leafs for node in leaf["parents"]]

            # add questions that could not be answered to graph
            if flow_graph.get("not_answered"):
                not_answered_label = ""
                for turn in flow_graph["not_answered"]:
                    not_answered_label += str(turn) + "\n"
                not_answered_label = not_answered_label.strip()
                di_graph.add_node(not_answered_label)

            # add conversation id
            if flow_graph.get("conv_id"):
                conv_id = str(flow_graph["conv_id"])
                di_graph.add_node(conv_id)

            nx.nx_agraph.write_dot(
                di_graph, os.path.join(self.config["silver_annotation_path"], "examples", "test.dot")
            )

            # same layout using matplotlib with no labels
            pos = graphviz_layout(di_graph, prog="dot")
            pos = pos
            plt.figure(figsize=(18, 20))
            nx.draw(di_graph, pos, with_labels=True, arrows=True, node_size=100)
            plt.xlim([-1, 800])
            plt.show()

    def _print_diambiguation_triples(self, disambiguated_triples):
        """
        For development purposes: Transform a list of disambiguation triples into a string for printing in graph.
        """
        disambiguations = dict()
        for item, surface_forms, label in disambiguated_triples:
            for surface_form in surface_forms:
                if disambiguations.get(surface_form):
                    disambiguations[surface_form].append(label)
                else:
                    disambiguations[surface_form] = [label]
        string = ""
        for surface_form in disambiguations:
            string += surface_form + ": " + str(disambiguations[surface_form]) + "\n"
        return string.strip()

    def random_example(self, dataset_path, index=None):
        """
        For development purposes: Compute the flow graph + structured representations for a random example.
        """
        with open(dataset_path, "r") as fp:
            data = json.load(fp)

        # (pseudo) random index
        if index:
            random_index = index
        else:
            random_index = random.randint(0, len(data) - 1)
        self.logger.info(f"Random_index: {random_index}")

        # run example
        for conversation in data[random_index:]:
            # initialize
            for turn in conversation["questions"]:
                turn["silver_SR"] = []
                turn["silver_relevant_turns"] = None

            self.logger.info(conversation["questions"][0]["question"])

            # create conversation flow graph
            flow_graph = self.fg_annotator.get_conv_flow_graph(conversation)
            if not flow_graph:
                continue

            # add annotations
            self.sr_annotator.annotate_structured_representations(flow_graph, conversation)
            self.fg_annotator.print_dict(conversation)
            self.fg_annotator.print_dict(flow_graph)
            self.tr_annotator.annotate_turn_relevances(flow_graph, conversation)
            self.fg_annotator.print_dict(conversation)

            # plot graph
            self._plot_graph(flow_graph, metadata=True, structured_representations=structured_representations)
            break

def main():
    if len(sys.argv) != 3:
        raise Exception(
            "Usage: python explaignn/distant_supervision/silver_annotation.py --FUNCTION <PATH_TO_CONFIG>"
        )

    # load options
    args = sys.argv[1:]
    function = args[0]
    config_path = args[1]
    config = get_config(config_path)
    benchmark_path = config["benchmark_path"]

    # create annotator
    annotator = SilverAnnotation(config)

    if function == "--example":
        input_path = os.path.join(benchmark_path, config["train_input_path"])
        annotator.random_example(input_path, index=1679)
    else:
        output_dir = config["path_to_intermediate_results"]
        tr_output_dir = os.path.join(config["path_to_intermediate_results"], "tr")
        Path(tr_output_dir).mkdir(parents=True, exist_ok=True)
        method_name = config["name"]

        # process data
        for split in ["train", "dev", "test"]:
            input_path = os.path.join(benchmark_path, config[f"{split}_input_path"])
            output_path = os.path.join(output_dir, f"annotated_{split}.json")
            tr_data_path = os.path.join(tr_output_dir, f"{split}.json")
            annotator.process_dataset(input_path, output_path, tr_data_path)

if __name__ == "__main__":
    main()