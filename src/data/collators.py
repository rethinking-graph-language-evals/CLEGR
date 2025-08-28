from torch_geometric.data import Batch
from typing import Any
import torch
import pickle

class NodeDataLoaderCollator:
    def __call__(self, data_list):
        # Get the first item's full graph data
        x = data_list[0].x
        edge_index = data_list[0].edge_index
        y = data_list[0].y
        
        # Standard batch collation
        batch = Batch.from_data_list(data_list)
        
        # Add the full graph data just once
        batch.x = x
        batch.edge_index = edge_index
        batch.y = y
        
        return batch
    

class CLEGRNodeDataLoaderCollator:
    def __init__(self, preprocessor_llm: Any = None):
        self.preprocessor_llm = preprocessor_llm.llm
        self.preprocessor_tokenizer = preprocessor_llm.tokenizer
        self.word_embedding = preprocessor_llm.word_embedding

        self.mappers = pickle.load(open("/home/datasets/clegr-facts/processed/mappers.pkl", "rb"))

        self.node_feature_order = [
            'disabled_access', 'has_rail', # Numerical/Boolean first
            'architecture', 'cleanliness', 'music', 'size' # Categorical
        ]

        self.mappers['node']['disabled_access'] = {'false': 0, 'true': 1}
        self.mappers['node']['has_rail'] = {'false': 0, 'true': 1}

        self.cache_embeddings = {}

        def get_embedding(words):
            # Use the preprocessor LLM and tokenizer to get the embedding
            # tokenize the words
            device = self.preprocessor_llm.device
            input_ids = self.preprocessor_tokenizer(
                words,
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids.squeeze(1).to(device)

            # get the embeddings
            with torch.no_grad():
                embeddings = self.word_embedding(input_ids).to(device)
                
            if embeddings.ndim == 3:
                embeddings = embeddings.mean(axis=1)
            # print(embeddings.shape)
            return embeddings

        for i in range(6):
            self.cache_embeddings[i] = {}
            for pair in self.mappers['node'][self.node_feature_order[i]].items():
                words = pair[0]
                self.cache_embeddings[i][pair[1]] = get_embedding(words)
            words = "null"
            id = len(self.cache_embeddings[i])
            self.cache_embeddings[i][id] = get_embedding(words)

        print(self.cache_embeddings[0])

    def __call__(self, data_list):
        print("hi22")
        # Extract graph data from the first item
        first_item = data_list[0]
        x, edge_index, y = first_item.x, first_item.edge_index, first_item.y
        device = x.device

        # Standard batch collation (create the batch on CPU and then later move fields to the correct device)
        batch = Batch.from_data_list(data_list)

        num_samples = x.shape[0]
        num_features = 6
        embedding_dim = 3072

        # Preallocate the tensor on the same device as x for efficiency.
        x_embed = torch.zeros((num_samples, num_features * embedding_dim), device=device)

        # Fill in the embedding tensor for each feature column block
        for feature_idx in range(num_features):
            embeddings = []
            for sample_idx in range(num_samples):
                # Ensure the index is obtained from a CPU tensor
                idx = int(x[sample_idx, feature_idx].detach().cpu().item())
                # Retrieve the cached embedding and move it to the target device
                emb = self.cache_embeddings[feature_idx][idx].to(device)
                embeddings.append(emb)
            # Stack the embeddings and assign them to the appropriate slice of x_embed
            x_embed[:, feature_idx * embedding_dim:(feature_idx + 1) * embedding_dim] = torch.stack(embeddings, dim=0)

        batch.x = x_embed
        batch.edge_index = edge_index.to(device)
        batch.y = y.to(device)

        print(batch)
        print(batch.shape)
        return batch


    
    