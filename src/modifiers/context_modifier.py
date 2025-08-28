import torch
from transformers import AutoTokenizer
from .base_modifier import Modifier

class ContextModifier(Modifier):
    def __init__(self, filler: str, to_modify="context", fraction: float = 1.0):
        self.replacement = filler
        self.to_modify = to_modify
        self.fraction = fraction

    def __call__(self, question_inputs, context_inputs, graph_embs, tokenizer: AutoTokenizer):
        replacement_id = tokenizer(self.replacement, add_special_tokens=False)["input_ids"][0]
        inputs = context_inputs if self.to_modify == "context" else question_inputs

        new_input_ids = []
        for input_ids in inputs["input_ids"]:
            input_len = len(input_ids)
            num_replace = int(self.fraction * input_len)
            replace_indices = torch.randperm(input_len)[:num_replace]

            input_ids_tensor = torch.tensor(input_ids)
            input_ids_tensor[replace_indices] = replacement_id
            new_input_ids.append(input_ids_tensor.tolist())

        inputs["input_ids"] = new_input_ids

        if self.to_modify == "context":
            context_inputs = inputs
        else:
            question_inputs = inputs

        return question_inputs, context_inputs, graph_embs