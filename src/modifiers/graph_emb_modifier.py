from transformers import AutoTokenizer

from .base_modifier import Modifier
import torch

class GraphEmbModifier(Modifier):
    def __init__(self, llm, filler=None, emb_idx: int = -1, fraction: float = 1.0):
        self.llm_emb = llm.word_embedding
        self.emb_idx = emb_idx
        self.filler = filler
        self.fraction = fraction

    def __call__(self, question_inputs, context_inputs, graph_embs_list, tokenizer: AutoTokenizer):
        new_graph_embs = []
        device = graph_embs_list[0].device

        with torch.no_grad():
            filler_inputs = tokenizer(self.filler, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)
            filler_emb = self.llm_emb(filler_inputs)
            if filler_emb.dim() > 1:
                filler_emb = torch.mean(filler_emb, dim=0)
            filler_emb = filler_emb.to(device)

        for graph_embs in graph_embs_list:
            num_embs = len(graph_embs)
            num_replace = int(self.fraction * num_embs)
            indices = torch.randperm(num_embs)[:num_replace]
            
            graph_embs[indices] = filler_emb.to(dtype=graph_embs.dtype)
            new_graph_embs.append(graph_embs)

        return question_inputs, context_inputs, new_graph_embs
