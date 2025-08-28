# -*- coding: utf-8 -*-
import os
import json
import torch
import numpy as np
from types import SimpleNamespace
import dataclasses
from absl import app, flags, logging
# Correct DataLoader import
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from typing import Type, List, Dict, Any, Union, Optional

# Import project modules
from src.utils import (load_dataset, split_dataset, initialize_model,
                       load_llm_params, remove_hook_from_module)
from src.data import get_dataset
from src.models.llm.llm_wrapper import LLM, BOS, EOS_USER
# Import explanation tools
from explanation_tools.hooked_glm import HookedGLM, LLMAdapter, Phi3Adapter, LLaMAAdapter
# Import model loading utilities
from src.models import get_glm_lightning_module

# WandB import
try:
    import wandb
except ImportError:
    wandb = None
    logging.warning("wandb not installed. WandB logging will be skipped.")

# Flags configuration
FLAGS = flags.FLAGS
flags.DEFINE_string("method", "tea-glm", "Method used during training.")
flags.DEFINE_string("dataset", "clegr-reasoning", "Dataset name used during training.")
flags.DEFINE_string("llm_model_name", "phi-3.5", "LLM model name identifier.")
flags.DEFINE_integer("seed", 42, "Random seed used during training.")
flags.DEFINE_string("save_dir", os.path.expandvars("/scratch/$USER/new_models"), "Base directory for trained models.")
flags.DEFINE_string("datasets_root", os.path.expandvars("/scratch/$USER/datasets"), "Root directory for datasets.")
flags.DEFINE_string("llm_params_file", "src/llm_params.json", "Path to LLM parameters file.")
flags.DEFINE_float("lr", 1e-4, "Learning rate used during training.")
flags.DEFINE_integer("scheduler_step_size", 3, "Scheduler step size.")
flags.DEFINE_float("scheduler_gamma", 0.1, "Scheduler gamma.")
flags.DEFINE_integer("gnn_hidden_channels", 64, "GNN hidden channels.")
flags.DEFINE_integer("gnn_num_layers", 2, "GNN number of layers.")
flags.DEFINE_integer("n_gnn_toks", 10, "Number of GNN tokens.")
flags.DEFINE_string("hf_cache", "./hf_cache/", "HuggingFace cache directory.")
flags.DEFINE_integer("gpu", 0, "GPU index (-1 for CPU).")
flags.DEFINE_integer("num_examples", 200, "Total examples to analyze.")
flags.DEFINE_integer("start_example", 0, "Skip first N examples.")
flags.DEFINE_bool("use_wandb", False, "Enable WandB logging.")
flags.DEFINE_string("wandb_project", "glm-activations", "WandB project.")
flags.DEFINE_string("wandb_entity", None, "WandB entity.")
flags.DEFINE_enum("wandb_mode", "online", ["online", "offline", "disabled"], "WandB mode.")
flags.DEFINE_integer("trial", 1, "WandB trial number.")
flags.DEFINE_string("modifier", "default", "WandB modifier.")
flags.DEFINE_float("modify_fraction", 1.0, "WandB modification fraction.")
flags.DEFINE_bool("save_compressed", True, "Save activations in compressed format.")

def preprocess_context(context, batch_size):
    """Preprocess the context to a list of strings."""
    if context is None:
        return None
    if isinstance(context, list) and len(context) == batch_size and all(isinstance(x, str) for x in context):
        return context
    if isinstance(context, torch.Tensor):
        return [str(c.item()) if c.numel() == 1 else str(c.tolist()) for c in context]
    if isinstance(context, dict):
        processed = []
        for i in range(batch_size):
            parts = []
            for key, val in context.items():
                current_val = val[i] if isinstance(val, (list, torch.Tensor)) and len(val) > i else val
                if isinstance(current_val, torch.Tensor):
                    current_val_str = str(current_val.item()) if current_val.numel() == 1 else str(current_val.tolist())
                else:
                    current_val_str = str(current_val)
                parts.append(f"{key}: {current_val_str}")
            processed.append("\n".join(parts))
        return processed
    if isinstance(context, str):
        return [context] * batch_size
    if isinstance(context, list):
        return [str(c) for c in context]
    else:
        return [str(context)] * batch_size

def load_and_extract_model(checkpoint_path, device, dataset_for_init=None):
    """Loads the G-Retriever model from checkpoint and extracts key components."""
    graph_data_arg = {}
    if dataset_for_init and hasattr(dataset_for_init, 'graph_data'):
        graph_data_arg['graph_data'] = dataset_for_init.graph_data

    logging.info("Initializing model structure...")
    init_model_wrapper = initialize_model(FLAGS.lr, **graph_data_arg)
    logging.info("Model structure initialized.")

    if not os.path.exists(checkpoint_path):
        logging.warning(f"Checkpoint not found at: {checkpoint_path}")
        alt_filename = os.path.basename(checkpoint_path).replace("projector_", "projector__")
        alt_checkpoint_path = os.path.join(os.path.dirname(checkpoint_path), alt_filename)
        if os.path.exists(alt_checkpoint_path):
            logging.warning(f"Trying alternative checkpoint path: {alt_checkpoint_path}")
            checkpoint_path = alt_checkpoint_path
        else:
            logging.error(f"Alternative checkpoint also not found: {alt_checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path} or {alt_checkpoint_path}")

    logging.info(f"Loading model weights from checkpoint: {checkpoint_path}")
    loaded_model_wrapper = get_glm_lightning_module(method=FLAGS.method).load_from_checkpoint(
        checkpoint_path,
        model=init_model_wrapper.model,
        lr=FLAGS.lr,
        graph_data=graph_data_arg.get('graph_data'),
        scheduler_step_size=FLAGS.scheduler_step_size,
        scheduler_gamma=FLAGS.scheduler_gamma,
        map_location=device
    )
    logging.info("Model weights loaded.")
    loaded_model_wrapper.eval()

    glm_model = loaded_model_wrapper.model
    llm_wrapper = glm_model.llm
    core_llm_model = llm_wrapper.llm
    tokenizer = llm_wrapper.tokenizer

    logging.info("Extracted GLM, LLM Wrapper, Core LLM, and Tokenizer.")
    print('GLM, LLM Wrapper, Core LLM, Tokenizer loaded and extracted.')
    return loaded_model_wrapper, llm_wrapper, core_llm_model, tokenizer

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        if isinstance(obj, SimpleNamespace):
            return vars(obj)
        if isinstance(obj, bytes):
            return obj.decode('utf-8', errors='ignore')
        return super(NpEncoder, self).default(obj)

def get_adapter_cls(llm_model_name: str) -> Type[LLMAdapter]:
    """Selects the appropriate LLMAdapter based on the model name."""
    model_name_lower = llm_model_name.lower()
    if "phi-3" in model_name_lower:
        logging.info("Using Phi3Adapter.")
        return Phi3Adapter
    elif "llama" in model_name_lower:
        logging.info("Using LLaMAAdapter.")
        return LLaMAAdapter
    else:
        logging.warning(f"No specific adapter found for '{llm_model_name}'. Defaulting to Phi3Adapter.")
        return Phi3Adapter



def process_graph_data(batch, model_wrapper, dataset, device, i):
    """
    Process graph data and extract the relevant embeddings for the current example.
    Handle soft-prompt mode where no graph encoder exists.
    """
    xs = None
    
    try:
        # Check if we're in soft-prompt mode (no graph encoder)
        if FLAGS.method == "soft-prompt":
            logging.info(f"Example {i}: Soft-prompt mode detected. Skipping graph processing.")
            # In soft-prompt mode, we don't process graph data
            # The model will use learned soft prompts instead
            return None
        
        # Extract node_id from batch
        node_id = getattr(batch, 'node_id', None)
        
        if isinstance(node_id, torch.Tensor):
            node_id_for_indexing = node_id.cpu()
        else:
            node_id_for_indexing = node_id
            
        logging.info(f"Example {i}: Node ID from batch: {node_id_for_indexing} (original type: {type(node_id)})")
        
        # Check if the model has a graph encoder
        if not hasattr(model_wrapper.model, 'encode_graph') or model_wrapper.model.encode_graph is None:
            logging.warning(f"Example {i}: Model does not have a graph encoder. Skipping graph processing.")
            return None
            
        # Additional check for GNN existence in the model
        if hasattr(model_wrapper.model, 'gnn') and model_wrapper.model.gnn is None:
            logging.warning(f"Example {i}: Model GNN is None. Skipping graph processing.")
            return None
        
        # Determine the source of graph data
        graph_input_candidate = None
        source_description = ""

        if hasattr(dataset, 'graph_data') and dataset.graph_data is not None:
            logging.info(f"Example {i}: Using graph data from dataset.graph_data.")
            graph_input_candidate = dataset.graph_data
            source_description = "dataset.graph_data"
        elif hasattr(batch, 'x') and hasattr(batch, 'edge_index'):
            logging.info(f"Example {i}: Using graph data from batch object (batch.x and batch.edge_index exist).")
            graph_input_candidate = batch
            source_description = "batch object with x/edge_index"
        else:
            logging.info(f"Example {i}: No dataset.graph_data found, and batch does not explicitly show .x/.edge_index. "
                         f"Attempting to use the raw batch as graph input.")
            graph_input_candidate = batch
            source_description = "raw batch object (fallback)"

        graph_input = None
        if graph_input_candidate is not None:
            if hasattr(graph_input_candidate, 'to') and callable(getattr(graph_input_candidate, 'to')):
                graph_input = graph_input_candidate.to(device)
                logging.info(f"Example {i}: Moved {source_description} (type: {type(graph_input_candidate)}) to device '{device}'.")
            else:
                graph_input = graph_input_candidate
                logging.warning(f"Example {i}: {source_description} (type: {type(graph_input_candidate)}) "
                                f"has no .to() method. Using as is. Ensure it's on the correct device.")
        
        if graph_input is None:
            raise ValueError(f"Example {i}: Graph input source is None. Cannot proceed with graph encoding.")

        with torch.no_grad():
            # Encode the graph
            graph_emb = model_wrapper.model.encode_graph(graph_input)
            logging.info(f"Example {i}: Raw graph embeddings shape: {graph_emb.shape}")
            
            # Filter based on node_id
            if node_id_for_indexing is not None:
                current_node_embeddings = graph_emb[node_id_for_indexing]
                
                if current_node_embeddings.dim() == 1:
                    current_node_embeddings = current_node_embeddings.unsqueeze(0)
                
                graph_emb = current_node_embeddings
                logging.info(f"Example {i}: Node-filtered embeddings shape: {graph_emb.shape}")
            else:
                logging.info(f"Example {i}: No node_id provided or node_id is invalid for filtering. Using all graph embeddings.")
            
            # Project the embeddings
            projected_emb = model_wrapper.model.projector(graph_emb)
            logging.info(f"Example {i}: Projected embeddings shape: {projected_emb.shape}")
            
            # Handle specific reshaping if node_id was a list
            if isinstance(node_id, list):
                if projected_emb.dim() == 2:
                    projected_emb = projected_emb.unsqueeze(0)
                    logging.info(f"Example {i}: Reshaped projected_emb for list node_id to: {projected_emb.shape}")
                elif projected_emb.dim() == 3 and projected_emb.shape[0] == 1:
                    logging.info(f"Example {i}: projected_emb for list node_id already in suitable shape: {projected_emb.shape}")
                else:
                    logging.warning(f"Example {i}: projected_emb shape is {projected_emb.shape} for list node_id. "
                                    "Review if this shape is expected before split, or adjust projector output/reshaping logic.")

            # Split into token embeddings
            xs = [hi.squeeze(0) for hi in projected_emb.split(1, dim=0)]
            logging.info(f"Example {i}: Created {len(xs)} token embedding sequence(s). Each of shape ~ (seq_len, embed_dim).")
            
            # Ensure embeddings match the language model's expected dtype
            if xs and len(xs) > 0:
                llm_param = next(model_wrapper.model.llm.llm.parameters(), None)
                if llm_param is not None:
                    model_dtype = llm_param.dtype
                    xs = [x.to(model_dtype) for x in xs]
                    logging.info(f"Example {i}: Converted token embeddings to model dtype: {model_dtype}")
                else:
                    logging.warning(f"Example {i}: Could not determine model dtype from LLM parameters. Embeddings dtype not changed.")
            
    except Exception as e:
        logging.error(f"Error processing graph data for example {i}: {e}", exc_info=True)
        xs = None
            
    return xs


def debug_dataloader(loader, max_samples=5):
    """Debug function to check if dataloader contains batches and print first few samples"""
    try:
        print(f"Dataloader length: {len(loader)}")
        print(f"Dataset length: {len(loader.dataset)}")
        
        # Try to get the first batch
        for i, batch in enumerate(loader):
            print(f"Batch {i} shape/keys: {batch}")
            if hasattr(batch, 'question'):
                print(f"  Question: {batch.question}")
            if hasattr(batch, 'context'):
                print(f"  Context: {batch.context}")
            if i >= max_samples - 1:
                break
        print("Successfully loaded batches from dataloader")
        return True
    except Exception as e:
        print(f"Error examining dataloader: {e}")
        return False

def convert_inputs_to_model_dtype(inputs_embeds, attention_mask, core_llm_model):
    """Convert input tensors to the same dtype as the model weights."""
    param = next(core_llm_model.parameters(), None)
    if param is not None:
        model_dtype = param.dtype
        logging.info(f"Model's parameter dtype: {model_dtype}")
        inputs_embeds = inputs_embeds.to(model_dtype)
        logging.info(f"Converted inputs_embeds to dtype: {inputs_embeds.dtype}")
    return inputs_embeds, attention_mask

def extract_activations(hook_outputs):
    """Extract pure activations from hook outputs."""
    activations = {}
    
    for key, hook_output in hook_outputs.items():
        if not key.startswith("layer_"):
            continue
            
        # Extract the actual activation tensor
        activation = None
        if isinstance(hook_output, tuple):
            # For transformer models, the first element is usually the hidden states
            activation = hook_output[0]
        elif isinstance(hook_output, torch.Tensor):
            activation = hook_output
        
        if activation is not None:

            if activation.dim() == 3 and activation.shape[1] > 10:
                activation = activation[:, :10, :]

            # Convert to CPU and detach to save memory
            activation_cpu = activation.detach().cpu()
            activations[key] = activation_cpu

            logging.info(f"Extracted activation for {key}: shape {activation_cpu.shape}")
    
    return activations

def save_activations(activations, save_path, compressed=True):
    """Save activations to file."""
    if compressed:
        # Save as compressed numpy arrays
        np.savez_compressed(
            save_path,
            **{k: v.to(torch.float32).cpu().numpy() for k, v in activations.items()}
        )

        logging.info(f"Saved compressed activations to {save_path}")
    else:
        # Save as regular numpy arrays
        np.savez(
            save_path,
            **{k: v.to(torch.float32).cpu().numpy() for k, v in activations.items()}
        )

        logging.info(f"Saved activations to {save_path}")

def main(argv):
    del argv
    logging.set_verbosity(logging.INFO)

    # Setup
    os.environ["HF_HOME"] = FLAGS.hf_cache
    if FLAGS.use_wandb and FLAGS.wandb_mode != "disabled" and wandb is not None:
        wandb.init(
            project=FLAGS.wandb_project,
            entity=FLAGS.wandb_entity,
            config=FLAGS.flag_values_dict(),
            mode=FLAGS.wandb_mode,
            name=f"{FLAGS.method}_{FLAGS.dataset}_seed{FLAGS.seed}_trial{FLAGS.trial}_{FLAGS.modifier}_frac_{FLAGS.modify_fraction}_activations",
        )
        logging.info("WandB initialized.")
    else:
        logging.info("WandB disabled or not installed.")

    device = torch.device(f"cuda:{FLAGS.gpu}" if torch.cuda.is_available() and FLAGS.gpu >= 0 else "cpu")
    logging.info(f"Using device: {device}")

    # Data Loading and Preparation
    logging.info("Loading dataset...")
    dataset = load_dataset(root=FLAGS.datasets_root, dataset=FLAGS.dataset)
    logging.info(f"Dataset loaded: {dataset}")
    if not dataset or len(dataset) == 0:
        logging.error("Failed to load dataset or dataset is empty.")
        return

    # Debug dataset
    logging.info("Dataset details:")
    try:
        first_sample = dataset[0]
        logging.info(f"First sample example:")
        for key, value in vars(first_sample).items():
            if not key.startswith('_'):
                logging.info(f"  {key}: {value}")
    except Exception as e:
        logging.error(f"Error accessing dataset items: {e}")

    logging.info("Splitting dataset to extract test set only...")
    _, _, test_dataset_list = split_dataset(dataset, seed=FLAGS.seed)

    # Truncate the test set to the specified number of examples
    num_examples = FLAGS.num_examples
    test_dataset_list = test_dataset_list[:num_examples]

    # Log test set info
    logging.info(f"Using only {len(test_dataset_list)} examples from the test set.")

    if not test_dataset_list or len(test_dataset_list) == 0:
        logging.error("Test dataset is empty after slicing.")
        return

    # Create and debug dataloader
    logging.info("Creating DataLoader for the test set (batch_size=1)...")
    test_loader = DataLoader(test_dataset_list, batch_size=1, shuffle=False, num_workers=0)

    # Debug the dataloader
    if not debug_dataloader(test_loader, max_samples=1):
        logging.error("Test dataloader debugging failed! Check your dataset and dataloader.")
        return

    # Load Model
    checkpoint_filename = f"projector_{FLAGS.llm_model_name}_{FLAGS.dataset}_seed_{FLAGS.seed}.ckpt"
    checkpoint_path = os.path.join(FLAGS.save_dir, FLAGS.method, checkpoint_filename)
    logging.info(f"Attempting to load G-Retriever model from: {checkpoint_path}")
    
    try:
        model_wrapper, llm_wrapper, core_llm_model, tokenizer = load_and_extract_model(
            checkpoint_path, device, dataset_for_init=dataset
        )
    except FileNotFoundError as e:
        logging.error(f"Model loading failed: {e}")
        return

    # Determine adapter class based on LLM
    AdapterClass = get_adapter_cls(FLAGS.llm_model_name)

    # Create output directory for activations
    base_save_dir = os.path.join(os.path.expandvars("/scratch/$USER/activations"), FLAGS.method, FLAGS.llm_model_name, FLAGS.dataset, f"seed_{FLAGS.seed}")
    os.makedirs(base_save_dir, exist_ok=True)
    logging.info(f"Saving activations to: {base_save_dir}")

    # Process examples
    logging.info(f"Processing 'g-retriever' on '{FLAGS.dataset}' dataset for activation extraction.")
    examples_processed = 0
    
    for i, batch in enumerate(test_loader):
        if i < FLAGS.start_example:
            continue
        if examples_processed >= FLAGS.num_examples:
            logging.info(f"Reached target number of examples ({FLAGS.num_examples}). Stopping.")
            break

        logging.info(f"\n--- Processing Example {i} ({examples_processed + 1}/{FLAGS.num_examples}) ---")
        batch = batch.to(device)
        
        # Extract data from batch
        question = getattr(batch, 'question', None)
        if question is None:
            logging.warning(f"Skipping example {i}: 'question' attribute not found in batch.")
            continue
            
        raw_context = getattr(batch, 'context', None)
        preprocessed_context = preprocess_context(raw_context, batch_size=1)
        
        logging.info(f"Question: {question}")
        logging.info(f"Context: {preprocessed_context}")
        
        # # Process graph data
        # xs = process_graph_data(batch, model_wrapper, dataset, device, i)
        # # if xs is None:
        # #     logging.warning(f"Failed to process graph data for example {i}. Skipping.")
        # #     continue

        # Process graph data (will return None for soft-prompt)
        xs = process_graph_data(batch, model_wrapper, dataset, device, i)
        
        # If in soft-prompt mode, generate the soft-prompt embeddings directly
        if FLAGS.method == "soft-prompt":
            logging.info(f"Example {i}: Generating soft-prompts from the model's projector.")
            with torch.no_grad():
                # This dummy input should match what your model's forward pass expects.
                # Based on your provided forward pass, it's a single zero tensor.
                dummy_input = torch.LongTensor([0]).to(device)
                projected_emb = model_wrapper.model.projector(dummy_input)
                
                # Split the embeddings into a list as expected by _get_embeds
                xs = [hi.squeeze(0) for hi in projected_emb.split(1, dim=0)]
                
                # Ensure embeddings match the language model's expected dtype
                llm_param = next(core_llm_model.parameters(), None)
                if llm_param is not None:
                    model_dtype = llm_param.dtype
                    xs = [x.to(model_dtype) for x in xs]
                    logging.info(f"Example {i}: Created {len(xs)} soft-prompt embedding(s) with shape {xs[0].shape} and dtype {model_dtype}.")

        if xs is None:
            logging.warning(f"Embeddings (xs) are None for example {i} and method is not 'soft-prompt'. Skipping.")
            continue

        
        # Build embeddings for hooking
        try:
            with torch.no_grad():
                q_list = question if isinstance(question, list) else [question]
                inputs_embeds, attention_mask, _ = llm_wrapper._get_embeds(q_list, preprocessed_context, xs)
                llm_data = {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}
                logging.info(f"inputs_embeds shape: {inputs_embeds.shape}, attention_mask shape: {attention_mask.shape}")
        except Exception as e:
            logging.error(f"Error preparing LLM inputs for example {i}: {e}", exc_info=True)
            continue
        
        # Extract activations
        activations = {}
        hooked_glm_model = None
        
        try:
            # Initialize hooked model
            hooked_glm_model = HookedGLM(
                llm_model=core_llm_model,
                tokenizer=tokenizer,
                adapter_cls=AdapterClass,
                device=device
            )
            hooked_glm_model.eval()
            
            # Convert inputs to correct dtype
            inputs_embeds, attention_mask = convert_inputs_to_model_dtype(inputs_embeds, attention_mask, core_llm_model)
            llm_data = {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}
            
            # Run forward pass with hooks to capture activations
            with torch.no_grad():
                _ = hooked_glm_model(llm_data, output_attentions=False)
            logging.info("Ran hooked model successfully")
            
            # Extract pure activations
            activations = extract_activations(hooked_glm_model.hook_outputs)
            logging.info(f"Extracted activations for {len(activations)} layers")
                
        except Exception as e:
            logging.error(f"Error during activation extraction for example {i}: {e}", exc_info=True)
        
        finally:
            # Clean up hooks
            if hooked_glm_model is not None and hasattr(hooked_glm_model, 'adapter') and hasattr(hooked_glm_model.adapter, 'llm_model'):
                try:
                    remove_hook_from_module(hooked_glm_model.adapter.llm_model)
                    logging.debug(f"Removed hooks for example {i}")
                except Exception as cleanup_err:
                    logging.error(f"Error cleaning up hooks: {cleanup_err}")
        
        # Save activations
        if activations:
            # Save activations as numpy arrays
            activation_file_path = os.path.join(base_save_dir, f"activations_example_{i}.npz")
            try:
                save_activations(activations, activation_file_path, compressed=FLAGS.save_compressed)
                
                # Also save metadata
                metadata = {
                    "example_index": i,
                    "question": question[0] if isinstance(question, list) else question,
                    "context": preprocessed_context[0] if preprocessed_context else None,
                    "ground_truth": getattr(batch, 'label', ["N/A"])[0] if hasattr(batch, 'label') else "N/A",
                    "activation_shapes": {k: list(v.shape) for k, v in activations.items()},
                    "num_layers": len(activations)
                }
                
                metadata_file_path = os.path.join(base_save_dir, f"metadata_example_{i}.json")
                with open(metadata_file_path, "w") as f:
                    json.dump(metadata, f, indent=2, cls=NpEncoder)
                logging.info(f"Saved metadata to {metadata_file_path}")
                
                # Log to WandB if enabled
                if FLAGS.use_wandb and FLAGS.wandb_mode != "disabled" and wandb is not None and wandb.run is not None:
                    try:
                        wandb_log_data = {
                            f"example_{i}/question": metadata["question"],
                            f"example_{i}/ground_truth": metadata["ground_truth"],
                            f"example_{i}/num_layers": metadata["num_layers"]
                        }
                        wandb.log(wandb_log_data, step=i)
                        wandb.save(activation_file_path, base_path=base_save_dir)
                        wandb.save(metadata_file_path, base_path=base_save_dir)
                    except Exception as wandb_err:
                        logging.warning(f"Failed to log example {i} to WandB: {wandb_err}")
                        
            except Exception as e:
                logging.error(f"Error saving activations for example {i}: {e}", exc_info=True)
        else:
            logging.warning(f"No activations extracted for example {i}")
        
        examples_processed += 1
    
    logging.info(f"Finished processing. Total examples analyzed: {examples_processed}")
    if wandb is not None and wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    app.run(main)
