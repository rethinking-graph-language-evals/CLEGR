import os
import pickle
from typing import Any, Dict, Tuple, List, Union

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.io import fs
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from pcst_fast import pcst_fast

from src.data.base import BaseDataset


def retrieval_via_pcst(
    data: Data,
    q_emb: torch.Tensor,
    textual_nodes: List[List[str]],
    textual_edges: Dict[str, List[Any]],
    topk: int = 6,
    topk_e: int = 6,
    cost_e: float = 0.5,
) -> Tuple[Data, str]:
    """
    Retrieval function using Prize-Collecting Steiner Tree (PCST) algorithm.
    
    Args:
        data: Full graph data
        q_emb: Question embedding
        textual_nodes: Textual node information
        textual_edges: Textual edge information
        topk: Top-k nodes to consider
        topk_e: Top-k edges to consider
        cost_e: Edge cost parameter
        
    Returns:
        Tuple of (retrieved_subgraph, textual_description)
    """
    c = 0.01
    dev = data.x.device  # Use the same device as input data

    root = -1
    num_clusters = 1
    pruning = "gw"
    verbosity_level = 0
    
    # Validate input data
    if not hasattr(data, 'x') or data.x is None or data.num_nodes == 0:
        return Data(x=torch.empty(0, 1, device=dev, dtype=torch.float32), 
                    edge_index=torch.empty(2, 0, dtype=torch.long, device=dev), 
                    edge_attr=torch.empty(0, 1, device=dev, dtype=torch.float32)), "Nodes:\nnode_id,node_attr\n\nEdges:\nsrc,edge_attr,dst\n"
    
    # Handle case with no edges but possibly nodes
    if not hasattr(data, 'edge_index') or data.edge_index.numel() == 0:
        # If no edges, but there are nodes, return a subgraph with only those nodes
        if data.num_nodes > 0:
            # All nodes are "involved" if there are no edges to filter by
            involved_nodes = torch.arange(data.num_nodes, device=dev)
            node_lines = [f"{node_id},{textual_nodes[0][node_id]}" for node_id in involved_nodes.tolist() if node_id < len(textual_nodes[0])]
            desc = (
                "Nodes:\n"
                "node_id,node_attr\n"
                + "\n".join(node_lines)
                + "\n\n"
                "Edges:\n"
                "src,edge_attr,dst\n"
            )
            return Data(x=data.x, 
                        edge_index=torch.empty(2, 0, dtype=torch.long, device=dev), 
                        edge_attr=torch.empty(0, data.edge_attr.shape[1] if hasattr(data, 'edge_attr') and data.edge_attr is not None else 1, device=dev, dtype=data.edge_attr.dtype if hasattr(data, 'edge_attr') and data.edge_attr is not None else torch.float32)), desc
        else:
            return Data(x=torch.empty(0, data.x.shape[1], device=dev, dtype=data.x.dtype), 
                        edge_index=torch.empty(2, 0, dtype=torch.long, device=dev), 
                        edge_attr=torch.empty(0, data.edge_attr.shape[1] if hasattr(data, 'edge_attr') and data.edge_attr is not None else 1, device=dev, dtype=data.edge_attr.dtype if hasattr(data, 'edge_attr') and data.edge_attr is not None else torch.float32)), "Nodes:\nnode_id,node_attr\n\nEdges:\nsrc,edge_attr,dst\n"

    # Node prize calculation
    if topk > 0 and hasattr(data, 'x') and data.x is not None:
        q_emb = q_emb.to(dev)
        x = data.x
        n_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, x)
        topk = min(topk, data.num_nodes)
        _, topk_n_indices = torch.topk(n_prizes, topk, largest=True)

        n_prizes_new = torch.zeros_like(n_prizes, device=dev)
        n_prizes_new[topk_n_indices] = torch.arange(topk, 0, -1, device=dev, dtype=n_prizes.dtype)
        n_prizes = n_prizes_new
    else:
        n_prizes = torch.zeros(data.num_nodes, device=dev)

    # Edge prize calculation
    if topk_e > 0 and hasattr(data, 'edge_attr') and data.edge_attr is not None:
        q_emb = q_emb.to(dev)
        edge_attr = data.edge_attr
        e_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, edge_attr)
        topk_e = min(topk_e, e_prizes.unique().size(0))

        if topk_e > 0:
            topk_e_values, _ = torch.topk(e_prizes.unique(), topk_e, largest=True)
            e_prizes[e_prizes < topk_e_values[-1]] = 0.0
            last_topk_e_value = topk_e
            for k in range(topk_e):
                indices = e_prizes == topk_e_values[k]
                # FIX: Ensure value distribution doesn't lead to negative costs later
                value = min((topk_e - k) / (sum(indices) + 1e-8), last_topk_e_value - c)
                e_prizes[indices] = value
                last_topk_e_value = value * (1 - c)
            cost_e = min(cost_e, e_prizes.max().item() * (1 - c / 2))
    else:
        e_prizes = torch.zeros(data.num_edges, device=dev)

    # PCST algorithm setup
    costs = []
    edges = []
    virtual_n_prizes = []
    virtual_edges = []
    virtual_costs = []
    mapping_n = {} # Maps virtual node ID to original edge index
    mapping_e = {} # Maps PCST edge index to original edge index
    
    # FIX: Validate edge indices before processing
    if hasattr(data, 'edge_index') and data.edge_index.numel() > 0:
        edge_index = data.edge_index
        # Check for negative or out-of-bounds indices
        if torch.any(edge_index < 0) or torch.any(edge_index >= data.num_nodes):
            print(f"Warning: Invalid edge indices detected. Min: {edge_index.min()}, Max: {edge_index.max()}, Num nodes: {data.num_nodes}")
            # Filter out invalid edges
            valid_mask = (edge_index[0] >= 0) & (edge_index[0] < data.num_nodes) & (edge_index[1] >= 0) & (edge_index[1] < data.num_nodes)
            if torch.any(valid_mask):
                edge_index = edge_index[:, valid_mask]
                if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                    e_prizes = e_prizes[valid_mask]
            else:
                # No valid edges
                return Data(x=torch.empty(0, data.x.shape[1], device=dev, dtype=data.x.dtype), 
                            edge_index=torch.empty(2, 0, dtype=torch.long, device=dev), 
                            edge_attr=torch.empty(0, data.edge_attr.shape[1] if hasattr(data, 'edge_attr') and data.edge_attr is not None else 1, device=dev, dtype=data.edge_attr.dtype if hasattr(data, 'edge_attr') and data.edge_attr is not None else torch.float32)), "Nodes:\nnode_id,node_attr\n\nEdges:\nsrc,edge_attr,dst\n"
        
        num_original_edges = data.edge_index.shape[1] # Use original count for iteration
        next_virtual_node_id = data.num_nodes  # Start virtual nodes from here
        
        # Iterate over the original edges to decide if they are "real" or "virtualized"
        for i in range(num_original_edges):
            src, dst = data.edge_index[:, i].tolist()
            
            # Double-check edge validity (redundant if filtered above but good for safety)
            if src < 0 or dst < 0 or src >= data.num_nodes or dst >= data.num_nodes:
                continue
                
            prize_e = e_prizes[i].item() # Use the prize corresponding to the original edge index 'i'

            if prize_e <= cost_e:
                mapping_e[len(edges)] = i # Map PCST internal edge index to original edge index
                edges.append((src, dst))
                costs.append(cost_e - prize_e)
            else:
                virtual_node_id = next_virtual_node_id
                next_virtual_node_id += 1
                mapping_n[virtual_node_id] = i # Map virtual node ID to original edge index
                virtual_edges.extend([(src, virtual_node_id), (virtual_node_id, dst)])
                virtual_costs.extend([0, 0])
                virtual_n_prizes.append(prize_e - cost_e)

    # Combine real and virtual node prizes
    if len(virtual_n_prizes) > 0:
        prizes = torch.cat([n_prizes, torch.tensor(virtual_n_prizes, device=dev)])
    else:
        prizes = n_prizes
    
    num_real_edges = len(edges) # Number of edges that remain "real" in PCST
    
    if len(virtual_costs) > 0:
        costs.extend(virtual_costs)
        edges.extend(virtual_edges)

    if not edges:
        # If no edges, and we didn't return above (meaning data.num_nodes > 0),
        # then return a graph with just the nodes
        if data.num_nodes > 0:
            involved_nodes = torch.arange(data.num_nodes, device=dev)
            node_lines = [f"{node_id},{textual_nodes[0][node_id]}" for node_id in involved_nodes.tolist() if node_id < len(textual_nodes[0])]
            desc = (
                "Nodes:\n"
                "node_id,node_attr\n"
                + "\n".join(node_lines)
                + "\n\n"
                "Edges:\n"
                "src,edge_attr,dst\n"
            )
            return Data(x=data.x, 
                        edge_index=torch.empty(2, 0, dtype=torch.long, device=dev), 
                        edge_attr=torch.empty(0, data.edge_attr.shape[1] if hasattr(data, 'edge_attr') and data.edge_attr is not None else 1, device=dev, dtype=data.edge_attr.dtype if hasattr(data, 'edge_attr') and data.edge_attr is not None else torch.float32)), desc
        else:
            return Data(x=torch.empty(0, data.x.shape[1], device=dev, dtype=data.x.dtype), 
                        edge_index=torch.empty(2, 0, dtype=torch.long, device=dev), 
                        edge_attr=torch.empty(0, data.edge_attr.shape[1] if hasattr(data, 'edge_attr') and data.edge_attr is not None else 1, device=dev, dtype=data.edge_attr.dtype if hasattr(data, 'edge_attr') and data.edge_attr is not None else torch.float32)), "Nodes:\nnode_id,node_attr\n\nEdges:\nsrc,edge_attr,dst\n"

    #Additional validation before calling pcst_fast
    edges_np = np.array(edges, dtype=np.int32)
    
    # Check for negative values in edges_np
    if np.any(edges_np < 0):
        print(f"Error: Negative edge endpoints detected: {edges_np[edges_np < 0]}")
        return Data(x=torch.empty(0, data.x.shape[1], device=dev, dtype=data.x.dtype), 
                    edge_index=torch.empty(2, 0, dtype=torch.long, device=dev), 
                    edge_attr=torch.empty(0, data.edge_attr.shape[1] if hasattr(data, 'edge_attr') and data.edge_attr is not None else 1, device=dev, dtype=data.edge_attr.dtype if hasattr(data, 'edge_attr') and data.edge_attr is not None else torch.float32)), "Nodes:\nnode_id,node_attr\n\nEdges:\nsrc,edge_attr,dst\n"
    
    # Detach only for numpy conversion (PCST algorithm doesn't need gradients)
    prizes_np = prizes.detach().cpu().numpy()
    costs_np = np.array(costs, dtype=np.float32)
    
    # Ensure all arrays have the right shape and type
    if len(prizes_np) == 0 or len(costs_np) != len(edges_np):
        print(f"Error: Array size mismatch. Prizes: {len(prizes_np)}, Costs: {len(costs_np)}, Edges: {len(edges_np)}")
        return Data(x=torch.empty(0, data.x.shape[1], device=dev, dtype=data.x.dtype), 
                    edge_index=torch.empty(2, 0, dtype=torch.long, device=dev), 
                    edge_attr=torch.empty(0, data.edge_attr.shape[1] if hasattr(data, 'edge_attr') and data.edge_attr is not None else 1, device=dev, dtype=data.edge_attr.dtype if hasattr(data, 'edge_attr') and data.edge_attr is not None else torch.float32)), "Nodes:\nnode_id,node_attr\n\nEdges:\nsrc,edge_attr,dst\n"

    try:
        # Run PCST algorithm
        vertices, edges_selected_indices = pcst_fast(
            edges_np, prizes_np, costs_np,
            root, num_clusters, pruning, verbosity_level
        )
    except Exception as e:
        print(f"PCST algorithm failed: {e}")
        return Data(x=torch.empty(0, data.x.shape[1], device=dev, dtype=data.x.dtype), 
                    edge_index=torch.empty(2, 0, dtype=torch.long, device=dev), 
                    edge_attr=torch.empty(0, data.edge_attr.shape[1] if hasattr(data, 'edge_attr') and data.edge_attr is not None else 1, device=dev, dtype=data.edge_attr.dtype if hasattr(data, 'edge_attr') and data.edge_attr is not None else torch.float32)), "Nodes:\nnode_id,node_attr\n\nEdges:\nsrc,edge_attr,dst\n"
    
    # Extract selected components
    selected_nodes_mask_pcst = vertices < data.num_nodes # Mask for real nodes selected by PCST
    selected_nodes_indices_pcst = vertices[selected_nodes_mask_pcst]
    
    real_edges_from_pcst = [mapping_e[e_idx] for e_idx in edges_selected_indices if e_idx < num_real_edges and e_idx in mapping_e]
    virtual_nodes_from_pcst = vertices[~selected_nodes_mask_pcst]
    virtual_edges_from_pcst = [mapping_n[v_idx] for v_idx in virtual_nodes_from_pcst if v_idx in mapping_n]
    
    combined_selected_edges_indices = sorted(list(set(real_edges_from_pcst + virtual_edges_from_pcst)))
    
    # Determine all nodes that are part of the final subgraph
    if combined_selected_edges_indices:
        # Get the original edge_index based on the combined_selected_edges_indices
        final_edge_index_original_nodes = data.edge_index[:, combined_selected_edges_indices]
        involved_nodes_from_edges = torch.unique(final_edge_index_original_nodes.flatten())
        all_involved_nodes = torch.unique(torch.cat([
            torch.tensor(selected_nodes_indices_pcst, dtype=torch.long, device=dev),
            involved_nodes_from_edges
        ]))
    else:
        final_edge_index_original_nodes = torch.empty((2, 0), dtype=torch.long, device=dev)
        all_involved_nodes = torch.tensor(selected_nodes_indices_pcst, dtype=torch.long, device=dev)

    # --- FIX: Create boolean masks to preserve gradients ---
    # Initialize full masks
    node_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=dev)
    if all_involved_nodes.numel() > 0:
        node_mask[all_involved_nodes] = True

    edge_mask = torch.zeros(data.num_edges, dtype=torch.bool, device=dev)
    if combined_selected_edges_indices:
        edge_mask[torch.tensor(combined_selected_edges_indices, dtype=torch.long, device=dev)] = True

    # --- Create node mapping for re-indexing the subgraph ---
    # This mapping is crucial for PyG's Data object to have contiguous node IDs from 0
    node_map = -torch.ones(data.num_nodes, dtype=torch.long, device=dev)
    if all_involved_nodes.numel() > 0:
        node_map[all_involved_nodes] = torch.arange(all_involved_nodes.size(0), device=dev)
    
    # Re-index the edge_index for the new subgraph
    if final_edge_index_original_nodes.numel() > 0:
        edge_index_mapped = node_map[final_edge_index_original_nodes]
    else:
        edge_index_mapped = torch.empty((2, 0), dtype=torch.long, device=dev)

    # Build textual description
    if all_involved_nodes.numel() > 0:
        node_lines = [f"{node_id},{textual_nodes[0][node_id]}" for node_id in all_involved_nodes.tolist() if node_id < len(textual_nodes[0])]
    else:
        node_lines = []
    
    if combined_selected_edges_indices:
        edge_lines = [f"{textual_edges['src'][ei]},{textual_edges['attr'][ei]},{textual_edges['dst'][ei]}" for ei in combined_selected_edges_indices if ei < len(textual_edges['src'])]
    else:
        edge_lines = []
    
    desc = (
        "Nodes:\n"
        "node_id,node_attr\n"
        + "\n".join(node_lines)
        + "\n\n"
        "Edges:\n"
        "src,edge_attr,dst\n"
        + "\n".join(edge_lines)
    )

    # Create retrieved subgraph using the masks - this preserves gradients
    retrieved_data = Data(
        x=data.x[node_mask],
        edge_index=edge_index_mapped,
        # Ensure edge_attr is handled correctly if it might be None or empty
        edge_attr=data.edge_attr[edge_mask] if hasattr(data, 'edge_attr') and data.edge_attr is not None and data.edge_attr.numel() > 0 else torch.empty(0, data.edge_attr.shape[1] if hasattr(data, 'edge_attr') and data.edge_attr is not None else 1, device=dev, dtype=data.edge_attr.dtype if hasattr(data, 'edge_attr') and data.edge_attr is not None else torch.float32),
    )

    return retrieved_data, desc



class CLEGRRetrievalDataset(BaseDataset):
    def __init__(
        self,
        root: str,
        transform=None,
        force_reload: bool = False,
        size: str = "small",
        seed: int = 42,
        topk_nodes: int = 5,
        topk_edges: int = 5,
        cost_e: float = 0.5,
        source_dataset: str = "clegr-reasoning",
        
    ):
        self.source_dataset = source_dataset
        self.source_processed_dir = os.path.join(root, source_dataset, "processed")
        self.draw_dir = os.path.join(root, source_dataset, "drawings")
        prefix = size + "-" if size != "small" else ""
        self.size = size
        self.seed = seed
        self.topk_nodes = topk_nodes
        self.topk_edges = topk_edges
        self.cost_e = cost_e

        retrieval_dataset_name = prefix + f"clegr-retrieval-{source_dataset.split('-')[-1]}" #FIX: data-loading
        
        super().__init__(
            dataset_name = retrieval_dataset_name,
            # dataset_name=prefix + "clegr-retrieval",
            root=root,
            transform=transform,
            force_reload=force_reload,
        )
        self.type = "graph"
        
        path = self.processed_paths[0]
        out = fs.torch_load(path, map_location="cpu")
        
        if isinstance(out, tuple) and (len(out) == 2 or len(out) == 3):
            data, self.slices = out[:2]
            data_cls = out[2] if len(out) == 3 else Data
            self.data = data if not isinstance(data, dict) else data_cls.from_dict(data)
        else: # Handle list of data objects
            self.data, self.slices = self.collate(out)


    @property
    def processed_file_names(self) -> List[str]:
        return ["retrieval_data_list.pt", "retrieval_mappers.pkl"]

    def process(self):
        print("### Starting CLEGRRetrievalDataset processing ###")
        print("Loading pre-processed data from source:", self.source_dataset)

        # FIX: Construct the correct path to the source dataset's processed files.
        source_root = os.path.dirname(self.root)
        base_processed_dir = os.path.join(source_root, self.source_dataset, "processed")
        reasoning_data_path = os.path.join(base_processed_dir, "data_list.pt")
        reasoning_mappers_path = os.path.join(base_processed_dir, "mappers.pkl")

        if not os.path.exists(reasoning_data_path):
            raise FileNotFoundError(
                f"Processed reasoning data not found at {reasoning_data_path}. "
                f"Please run the source dataset processing first."
            )
        
        reasoning_out = fs.torch_load(reasoning_data_path, map_location="cpu")
        
        # Unpack data
        if isinstance(reasoning_out, tuple):
            # print("reasoning_out : ", reasoning_out)
            reasoning_data, reasoning_slices = reasoning_out[:2]
            data_cls = reasoning_out[2] if len(reasoning_out) == 3 else Data
            base_data = data_cls.from_dict(reasoning_data) if isinstance(reasoning_data, dict) else reasoning_data

            # Add this right after loading the reasoning data to understand the structure
            print(f"base_data keys: {list(base_data.keys())}")
            print(f"reasoning_slices keys: {list(reasoning_slices.keys())}")

            if 'graph_id' in base_data:
                print(f"base_data['graph_id'] shape/length: {len(base_data['graph_id']) if hasattr(base_data['graph_id'], '__len__') else 'scalar'}")
                print(f"base_data['graph_id'] first 10 values: {base_data['graph_id'][:10]}")

            if 'graph_id' in reasoning_slices:
                print(f"graph_id in reasoning_slices")
                print(f"reasoning_slices['graph_id']: {reasoning_slices['graph_id'][:10]}")
                        
            # FIX: Reverted to the original manual de-collation loop.
            # The standard `base_data.get(i)` method was failing and producing 'None'
            # for all samples, indicating an incompatibility with this dataset's format.
            # This manual loop correctly reconstructs the individual Data objects.
            print("Extracting individual samples manually...")
            base_samples = []
            slice_key = next(iter(reasoning_slices.keys()))
            num_samples = len(reasoning_slices[slice_key]) - 1

            for i in tqdm(range(num_samples), desc="De-collating source data"):
                sample = data_cls()
                node_offset = 0 # To re-index nodes for the subgraph
                
                for key in base_data.keys():
                    if key in reasoning_slices:
                        start_idx = reasoning_slices[key][i]
                        end_idx = reasoning_slices[key][i + 1]
                        value = base_data[key]
                        
                        # Store the node offset to correct the edge_index later
                        if key == 'x':
                            node_offset = start_idx

                        # Slice the data attribute
                        per_sample_keys = {
                            'graph_id', 
                            'question', 
                            'label', 
                            'context', 
                            'question_group', 
                            'question_subgroup', 
                            'question_type'
                        }
                        if key == 'edge_index' and value.dim() == 2:
                            sample[key] = value[:, start_idx:end_idx]
                        elif key in per_sample_keys:
                            sample[key] = value[i] #FIX :To make the them a single string and not a list of string
                        else:
                            sample[key] = value[start_idx:end_idx]
                    else:
                        # Copy attributes that are not sliced (i.e., graph-level attributes)
                        sample[key] = base_data[key]

                
                # FIX: Proper re-indexing of edge_index
                if hasattr(sample, 'edge_index') and sample.edge_index.numel() > 0:
                    sample.edge_index = sample.edge_index - node_offset
                    
                    # Additional validation: ensure all edge indices are non-negative and within bounds
                    if hasattr(sample, 'x') and sample.x.size(0) > 0:
                        num_nodes = sample.x.size(0)
                        valid_mask = (sample.edge_index[0] >= 0) & (sample.edge_index[0] < num_nodes) & (sample.edge_index[1] >= 0) & (sample.edge_index[1] < num_nodes)
                        if torch.any(valid_mask):
                            sample.edge_index = sample.edge_index[:, valid_mask]
                            if hasattr(sample, 'edge_attr') and sample.edge_attr is not None:
                                sample.edge_attr = sample.edge_attr[valid_mask]
                        else:
                            # No valid edges
                            sample.edge_index = torch.empty((2, 0), dtype=torch.long)
                            if hasattr(sample, 'edge_attr'):
                                sample.edge_attr = torch.empty((0, sample.edge_attr.shape[1] if sample.edge_attr.dim() > 1 else 1))
                
                # Set the number of nodes for the individual sample
                if hasattr(sample, 'x'):
                    sample.num_nodes = sample.x.size(0)
                else:
                    sample.num_nodes = 0

                base_samples.append(sample)

        else: # Assume it's already a list of Data objects
            base_samples = reasoning_out

        print(f"Extracted {len(base_samples)} samples from source.")
    
        # Filter out any None samples that may have been created in error.
        original_count = len(base_samples)
        base_samples = [s for s in base_samples if s is not None and hasattr(s, 'graph_id')]
        if len(base_samples) < original_count:
            print(f"⚠️ Warning: Filtered out {original_count - len(base_samples)} invalid entries from the source dataset.")
        
        if not base_samples:
            raise ValueError("After filtering, the source dataset is empty. Cannot proceed.")

        # --- The rest of the function continues as before ---
        print("Initializing BERT model for question embedding...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

        graph_groups = self._group_by_graph_id(base_samples)
        print(f"Processing {len(graph_groups)} unique graphs with retrieval...")

        retrieval_data_list = []
        # Process samples in their original order, not grouped by graph
        for sample in tqdm(base_samples, desc="Processing samples for retrieval"):
            try:
                graph_id = sample.graph_id

                # Get the full graph data for this graph_id
                full_graph_data = graph_groups[graph_id][0]  # First sample has the full graph
                textual_nodes, textual_edges = self._extract_textual_info(full_graph_data)
                
                retrieved_data = self._process_question_with_retrieval(
                    sample, full_graph_data, textual_nodes, textual_edges
                )
                retrieval_data_list.append(retrieved_data)
            except Exception as e:
                print(f"Error processing sample with graph {graph_id}: {e}")
                continue

        print(f"Created {len(retrieval_data_list)} retrieval-enhanced samples")

        if os.path.exists(reasoning_mappers_path):
            print(f"Loading mappers from {reasoning_mappers_path}")
            with open(reasoning_mappers_path, "rb") as f:
                feature_mappers = pickle.load(f)
        else:
            feature_mappers = {}

        fs.torch_save(retrieval_data_list, self.processed_paths[0])
        with open(self.processed_paths[1], "wb") as f:
            pickle.dump(feature_mappers, f)

        print("### CLEGRRetrievalDataset processing finished ###")

    def _group_by_graph_id(self, dataset: List[Data]) -> Dict[Union[int, str], List[Data]]:
        groups = {}
        for sample in dataset:
            # FIX: Assumes graph_id is a tensor/list with one item. Extracts the scalar.
            # graph_id = sample.graph_id[0].item() if torch.is_tensor(sample.graph_id) else sample.graph_id[0]
            graph_id = sample.graph_id
            if graph_id not in groups:
                groups[graph_id] = []
            groups[graph_id].append(sample)
        return groups


    def _extract_textual_info(self, sample: Data) -> Tuple[List[List[str]], Dict[str, List]]:
        textual_nodes = sample.node_texts

        edge_src_ids, edge_dst_ids, edge_attrs = [], [], []

        if hasattr(sample, 'edge_index') and sample.edge_index.numel() > 0:
            edge_texts = sample.edge_texts
            # FIX: Robustly handle nested list for edge_texts
            if isinstance(edge_texts, list) and len(edge_texts) > 0 and isinstance(edge_texts[0], list):
                edge_texts = edge_texts[0]

            num_edges = min(sample.edge_index.size(1), len(edge_texts))
            for i in range(num_edges):
                src, dst = sample.edge_index[:, i]
                edge_src_ids.append(src.item())
                edge_dst_ids.append(dst.item())
                edge_attrs.append(edge_texts[i])

        textual_edges = {'src': edge_src_ids, 'dst': edge_dst_ids, 'attr': edge_attrs}
        return textual_nodes, textual_edges

    def _process_question_with_retrieval(
        self, 
        sample: Data, 
        full_graph: Data, 
        textual_nodes: List[str], 
        textual_edges: Dict[str, List[str]]
    ) -> Data:
        # FIX: `sample.question` is already a list, no need for extra `[]`
        question_emb = self.embed_sentences(sample.question)[0]
        
        retrieved_subgraph, retrieved_context = retrieval_via_pcst(
            data=full_graph,
            q_emb=question_emb,
            textual_nodes=textual_nodes,
            textual_edges=textual_edges,
            topk=self.topk_nodes,
            topk_e=self.topk_edges,
            cost_e=self.cost_e,
        )
        
        final_sample = sample.clone()
        final_sample.x = retrieved_subgraph.x
        final_sample.edge_index = retrieved_subgraph.edge_index
        final_sample.edge_attr = retrieved_subgraph.edge_attr
        final_sample.context = retrieved_context
        final_sample.num_nodes = retrieved_subgraph.num_nodes
        final_sample.num_edges = retrieved_subgraph.num_edges
        final_sample.retrieval_params = {
            'topk_nodes': self.topk_nodes, 'topk_edges': self.topk_edges, 'cost_e': self.cost_e
        }
        
        return final_sample


    def embed_sentences(self, sentences: List[str]) -> torch.Tensor:
    # FIX: Tokenizer expects a list of strings, which `sentences` now correctly is.
    # The `[0]` was removed.
        inputs = self.tokenizer(
            sentences, 
            return_tensors="pt",
            padding=True, 
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state

        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        masked_hidden = last_hidden_state * attention_mask
        sum_hidden = masked_hidden.sum(dim=1)
        lengths = attention_mask.sum(dim=1).clamp(min=1e-9) # Avoid division by zero
        embedding = sum_hidden / lengths

        return embedding.cpu() # Explicitly move to CPU and detach to avoid device mismatches

    def split_by_graph(self, seed: int, split_ratio: tuple = (0.6, 0.2, 0.2)):
        # FIX: Extracts the scalar graph ID, making it hashable for use in sets.
        # This resolves the TypeError: unhashable type: 'list'.
        # graph_ids = [d.graph_id[0].item() if torch.is_tensor(d.graph_id) else d.graph_id[0] for d in self]
        graph_ids = [d.graph_id for d in self]

        unique_graph_ids = np.unique(graph_ids)
        rng = np.random.RandomState(seed)
        shuffled_graph_ids = rng.permutation(unique_graph_ids)
        
        n = len(shuffled_graph_ids)
        train_end = int(n * split_ratio[0])
        val_end = int(n * (split_ratio[0] + split_ratio[1]))
        
        train_graphs = set(shuffled_graph_ids[:train_end])
        val_graphs = set(shuffled_graph_ids[train_end:val_end])
        test_graphs = set(shuffled_graph_ids[val_end:])
        
        print(f"Train graphs: {len(train_graphs)}, Val graphs: {len(val_graphs)}, Test graphs: {len(test_graphs)}")
        
        train_indices = [i for i, gid in enumerate(graph_ids) if gid in train_graphs]
        val_indices = [i for i, gid in enumerate(graph_ids) if gid in val_graphs]
        test_indices = [i for i, gid in enumerate(graph_ids) if gid in test_graphs]
        
        return {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices,
        }


    def split(self, seed: int, split_ratio: Tuple[float, float, float] = (0.6, 0.2, 0.2)):
        # Redirecting to the corrected split_by_graph logic
        return self.split_by_graph(seed, split_ratio)

# These classes remain the same but will now inherit the corrected logic.
class CLEGRRetrieveFactsDataset(CLEGRRetrievalDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, source_dataset="clegr-facts", **kwargs)

class CLEGRRetrieveReasoningDataset(CLEGRRetrievalDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, source_dataset="clegr-reasoning", **kwargs)

class CLEGRRetrieveFull(CLEGRRetrievalDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, source_dataset="clegr-full", **kwargs)