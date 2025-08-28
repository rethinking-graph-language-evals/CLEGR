import uuid
import os
import sys
from absl import logging
import random
from tqdm import tqdm
from collections import Counter
import torch
from torch_geometric.data import Data
from pathlib import Path
import pickle
from typing import Tuple, List, Dict, Optional
import numpy as np

from .questions import question_forms, QuestionForm
from .generate_graph import GraphGenerator
from .types_ import GraphSpec
from .args import *

# --- Helper Functions ---

def filter_question_forms(args) -> List[QuestionForm]:
    """Filters question forms based on command line arguments."""
    forms_to_use = []
    has_filter = args.group is not None or args.type_prefix is not None

    for form in question_forms:
        matches = True
        if not has_filter:
            forms_to_use.append(form)
            continue

        # Apply filters if specified
        group_match = (args.group is None) or (form.group == args.group)
        prefix_match = True
        if args.type_prefix is not None:
             prefix_match = any(form.type_string.startswith(p) for p in args.type_prefix)

        if group_match and prefix_match:
            forms_to_use.append(form)

    if not forms_to_use:
        logging.error("No question forms match the specified filters (--group/--type-prefix).")
        sys.exit(1) # Exit if no questions can be generated

    logging.info(f"Using {len(forms_to_use)} question forms based on filters.")
    return forms_to_use


def generate_pyg_dataset(args, forms_to_use: List[QuestionForm], output_dir: Path) -> Tuple[List[Data], Dict]:
    """Generates the dataset as a list of PyG Data objects."""

    dataset: List[Data] = []
    no_of_graphs = args.count
    question_instances_per_graph = args.questions_per_graph

    # Statistics counters
    form_try_count = Counter()
    form_success_count = Counter()
    graph_generation_failures = 0
    question_generation_failures = 0
    total_graphs_generated = 0
    total_questions_generated = 0

    # Global feature mappers (build incrementally or pre-scan?)
    # Pre-scanning is hard without generating graphs first.
    # Build incrementally and ensure consistency later? Risky.
    # Simplest: Generate one graph, get mappers, use for all. Less diverse mapping.
    # Better: Generate N graphs, aggregate mappers, then convert all. Memory intensive.
    # Chosen approach: Generate first graph, get initial mappers, update if new values seen (WARN).
    feature_mappers: Dict[str, Dict[str, Dict[str, int]]] = {"node": {}, "edge": {}, "line": {}}
    mappers_initialized = False

    logging.info(f"Targeting {no_of_graphs} graphs.")
    pbar = tqdm(total=no_of_graphs, desc="Generating GQA pairs")

    while total_graphs_generated < no_of_graphs:
        graph_spec: Optional[GraphSpec] = None
        pyg_data: Optional[Data] = None

        # --- 1. Generate Graph ---
        try:
            graph_generator = GraphGenerator(args)
            graph_generator.generate() # Generates graph_spec internally
            graph_spec = graph_generator.graph_spec

            if not graph_spec or not graph_spec.nodes:
                 raise ValueError("Generated graph is empty or invalid.")
            total_graphs_generated += 1

            # --- 2. Initialize or Update Feature Mappers ---
            current_mappers = graph_spec.get_feature_mappers()
            if not mappers_initialized:
                feature_mappers = current_mappers
                mappers_initialized = True
                logging.info("Initialized feature mappers from the first graph.")
            else:
                # Check for new categories and update (log warnings)
                for domain in ["node", "edge", "line"]:
                     for prop, current_mapping in current_mappers[domain].items():
                         if prop not in feature_mappers[domain]:
                              logging.warning(f"New property '{prop}' found in domain '{domain}'. Adding to mappers.")
                              feature_mappers[domain][prop] = current_mapping
                         else:
                              existing_mapping = feature_mappers[domain][prop]
                              for value, index in current_mapping.items():
                                   if value not in existing_mapping:
                                       new_index = len(existing_mapping)
                                       logging.warning(f"New value '{value}' for '{domain}.{prop}'. Adding with index {new_index}.")
                                       existing_mapping[value] = new_index


            # --- 3. Convert Graph to PyG Data object ---
            pyg_data = graph_spec.to_pyg_data(feature_mappers)
            if pyg_data is None or pyg_data.num_nodes == 0:
                 raise ValueError("Failed to convert graph to valid PyG data.")

             # Add lists to store QA pairs for this graph
            # pyg_data.questions = [] # List[str] - English questions
            # pyg_data.answers = []   # List[Any] - Answers (can be int, bool, str, list)
            # pyg_data.question_types = [] # List[str] - Type strings
            # pyg_data.question_funcs = [] # List[Dict] - Functional representations

        except Exception as e:
            logging.warning(f"Graph generation or PyG conversion failed: {e}", exc_info=args.log_level.upper()=="DEBUG")
            graph_generation_failures += 1
            # Avoid infinite loop if graph gen always fails
            if graph_generation_failures > max(10, no_of_graphs*3):
                 logging.error("Too many graph generation failures. Stopping.")
                 raise RuntimeError("Stopping due to excessive graph generation failures.") from e
            continue # Skip to next graph attempt
        
        logging.info(f"Generated graph {graph_spec.id[:8]} with {len(graph_spec.nodes)} nodes and {len(graph_spec.edges)} edges.")

        # --- 4. Generate Questions for this Graph ---
        questions_added_for_this_graph = 0
        attempts_for_this_graph = 0
        max_attempts_per_question = 3 # Try each form a few times

        form_indices = list(range(len(forms_to_use)))
        random.shuffle(form_indices) # Try forms in random order per graph

        for i in range(question_instances_per_graph):
        # while questions_added_for_this_graph < questions_per_graph and attempts_for_this_graph < max_attempts_per_graph:
            form_idx = form_indices[attempts_for_this_graph % len(forms_to_use)]
            for form_idx in range(len(forms_to_use)):
                form = forms_to_use[form_idx]
                attempts_for_this_graph += 1
                form_try_count[form.type_string] += 1
                
                data_obj = pyg_data.clone() # Clone the PyG Data object for each question
                for attempt in range(max_attempts_per_question):
                    try:
                        q_spec, answer = form.generate(graph_spec, args)

                        if q_spec is not None and answer is not None:
                            # Store QA pair within the PyG Data object
                            data_obj.question = q_spec.english
                            data_obj.question_group = q_spec.group
                            data_obj.question_subgroup = q_spec.subgroup
                            data_obj.label = str(answer)
                            data_obj.question_type = q_spec.type_string

                            # if q_spec.type_string == "CompareLineDisabledAccess":
                            #     print(q_spec.english, str(answer))

                            dataset.append(data_obj)

                            form_success_count[form.type_string] += 1
                            questions_added_for_this_graph += 1
                            total_questions_generated += 1
                            pbar.update(1)
                            break
                        else:
                            # Generation failed expectedly (e.g., invalid args/answer, ValueError)
                            # print(f"Question generation failed for form {form.type_string}: {q_spec} -> {answer}")
                            logging.debug(f"Question generation failed for form {form.type_string}: {q_spec} -> {answer}")
                            question_generation_failures += 1

                    except Exception as e:
                        # Unexpected error during question generation
                        # print(f"Unexpected error generating question type {form.type_string}: {e}")
                        logging.error(f"Unexpected error generating question type {form.type_string}: {e}", exc_info=True)
                        question_generation_failures += 1 # Count as failure
             
        logging.info(f"Graph {graph_spec.id[:8]} generated {questions_added_for_this_graph} questions.")
    

        # Optional: Draw graph image if requestedz
        if args.draw and total_graphs_generated <= 10: # Limit drawing
                draw_filename = os.path.join(output_dir, f"graph_{graph_spec.id[:8]}_{total_graphs_generated}.png")
                graph_generator.draw(str(draw_filename))
        # elif pyg_data and not pyg_data.questions:
        #      logging.debug(f"Graph {graph_spec.id[:8]} generated no valid questions.")


    pbar.close()
    logging.info("-" * 30)
    logging.info("Generation Summary:")
    logging.info(f"  Target graphs pairs: {no_of_graphs}")
    logging.info(f"  Generated GQA pairs: {total_questions_generated}")
    logging.info(f"  Generated Graphs: {total_graphs_generated}")
    logging.info(f"  Generated PyG Data objects: {len(dataset)}")
    logging.info(f"  Graph Generation Failures: {graph_generation_failures}")
    logging.info(f"  Question Generation Failures (Expected): {question_generation_failures}")
    logging.info("-" * 30)
    logging.info("Success Rate per Question Type:")
    for form_type in sorted(form_try_count.keys()):
         tries = form_try_count[form_type]
         successes = form_success_count.get(form_type, 0)
         rate = (successes / tries * 100) if tries > 0 else 0
         logging.info(f"  - {form_type}: {successes}/{tries} ({rate:.1f}%)")
         if tries > 0 and successes == 0:
             logging.warning(f"    Question type {form_type} never succeeded.")
         elif tries > 0 and successes < tries * 0.5: # Warn if low success rate
              logging.warning(f"    Question type {form_type} has low success rate ({rate:.1f}%).")
    logging.info("-" * 30)
    
    # example datapoint
    if dataset:
        example_graph = dataset[0]
        logging.info(f"Example graph {example_graph}:")
    logging.info("-" * 30)
    logging.info("Dataset generation completed.")
    logging.info("Feature mappers:")
    for domain, mappers in feature_mappers.items():
        logging.info(f"  {domain}:")
        for prop, mapping in mappers.items():
            logging.info(f"    {prop}: {len(mapping)} unique values")
    logging.info("-" * 30)

    return dataset, feature_mappers

def gen():
    args = get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # --- Logging Setup ---
    log_level_name = args.log_level.upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    logging.set_verbosity(log_level)
    # Optionally set lower level for specific modules if needed for debugging
    # logging.getlogging('gqa.generate_graph').setLevel(log_level)
    # logging.getlogging('gqa.functional').setLevel(log_level)

    logging.info(f"Starting dataset generation with args: {args}")

    # --- Prepare Output ---
    output_dir = Path("./data_pyg") # New directory for PyG output
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = args.name if args.name else f"clegr_pyg_{uuid.uuid4().hex[:8]}"
    dataset_filename = output_dir / f"{dataset_name}.pt"
    mappers_filename = output_dir / f"{dataset_name}_mappers.pkl"

    logging.info(f"Output dataset will be saved to: {dataset_filename}")
    logging.info(f"Feature mappers will be saved to: {mappers_filename}")

    # --- Filter Question Forms ---
    forms_to_use = filter_question_forms(args)

    # --- Generate Dataset ---
    try:
        pyg_dataset, feature_mappers = generate_pyg_dataset(args, forms_to_use, output_dir)

        if not pyg_dataset:
            logging.warning("Generation finished, but the dataset is empty.")
            sys.exit(0) # Exit cleanly if no data was generated

        # --- Save Dataset and Mappers ---
        logging.info(f"Saving PyG dataset ({len(pyg_dataset)} graphs) to {dataset_filename}...")
        torch.save(pyg_dataset, dataset_filename)
        logging.info("Dataset saved.")

        logging.info(f"Saving feature mappers to {mappers_filename}...")
        with open(mappers_filename, 'wb') as f:
            pickle.dump(feature_mappers, f)
        logging.info("Mappers saved.")

        logging.info("Generation process completed successfully.")

    except Exception as e:
        logging.critical(f"Dataset generation failed with an unhandled exception: {e}", exc_info=True)
        sys.exit(1)

# --- Main Execution ---
if __name__ == "__main__":
    gen()