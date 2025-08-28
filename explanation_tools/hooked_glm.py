import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any, Dict, Callable, Type

# --------------------------
# Adapter Interface and Implementations
# --------------------------

class LLMAdapter(nn.Module):
    """
    A generic adapter interface for Large Language Models.
    Subclasses must implement:
      - get_transformer_layers()
      - get_lm_head()
      - get_input_embeddings()
    """
    def __init__(self, llm_model: nn.Module) -> None:
        super().__init__()
        self.llm_model = llm_model

    def get_transformer_layers(self) -> Any:
        raise NotImplementedError("Subclasses must implement get_transformer_layers.")

    def get_lm_head(self) -> Any:
        raise NotImplementedError("Subclasses must implement get_lm_head.")

    def get_input_embeddings(self) -> Any:
        raise NotImplementedError("Subclasses must implement get_input_embeddings.")


class Phi3Adapter(LLMAdapter):
    """
    Adapter for microsoft/Phi-3.5-mini-instruct.
    It locates the transformer layers, LM head, and input embeddings.
    """
    def get_transformer_layers(self) -> Any:
        if hasattr(self.llm_model, "model") and hasattr(self.llm_model.model, "layers"):
            return self.llm_model.model.layers
        elif hasattr(self.llm_model, "model") and hasattr(self.llm_model.model, "transformer"):
            return self.llm_model.model.transformer.h
        elif hasattr(self.llm_model, "transformer"):
            return self.llm_model.transformer.h
        else:
            raise AttributeError("Could not find transformer layers for the Phi3 model.")

    def get_lm_head(self) -> Any:
        if hasattr(self.llm_model, "lm_head"):
            return self.llm_model.lm_head
        return None

    def get_input_embeddings(self) -> Any:
        return self.llm_model.get_input_embeddings()
    
    def get_final_layer_norm(self) -> Any:
        if hasattr(self.llm_model, "model") and hasattr(self.llm_model.model, "norm"):
            return self.llm_model.model.norm
        return None
        




class LLaMAAdapter(LLMAdapter):
    """
    Adapter for LLaMA-like models.
    """
    def get_transformer_layers(self) -> Any:
        if hasattr(self.llm_model, "model") and hasattr(self.llm_model.model, "layers"):
            return self.llm_model.model.layers
        else:
            raise AttributeError("Could not find transformer layers for the LLaMA model.")

    def get_lm_head(self) -> Any:
        if hasattr(self.llm_model, "lm_head"):
            return self.llm_model.lm_head
        return None

    def get_input_embeddings(self) -> Any:
        return self.llm_model.get_input_embeddings()

    def get_final_layer_norm(self) -> Any:
        if hasattr(self.llm_model, "model") and hasattr(self.llm_model.model, "norm"):
            return self.llm_model.model.norm
        return None


# --------------------------
# Hooked Graph Language Model
# --------------------------

class HookedGLM(nn.Module):
    """
    A Hooked Graph Language Model (GLM) that wraps a preloaded LLM model and its tokenizer.
    It registers forward hooks on the transformer's layers and the LM head to capture intermediate outputs
    for analysis, such as logit lens computation.
    
    Args:
        llm_model (nn.Module): The preloaded language model.
        tokenizer (Any): The tokenizer associated with the language model.
        adapter_cls (Type[LLMAdapter]): Adapter class to interface with the LLM.
        device (torch.device): The device on which the model is running.
    """
    def __init__(
        self,
        llm_model: nn.Module,
        tokenizer: Any,
        adapter_cls: Type[LLMAdapter],
        device: torch.device,
    ) -> None:
        super().__init__()
        self.device = device
        self.tokenizer = tokenizer
        self.adapter = adapter_cls(llm_model)
        self.hook_outputs: Dict[str, Any] = {}
        self._register_hooks()

    def _register_hooks(self) -> None:
        """
        Registers forward hooks on each transformer layer and the LM head (if available).
        """
        self.hook_outputs.clear()
        layers = self.adapter.get_transformer_layers()
        for i, layer in enumerate(layers):
            hook_name = f"layer_{i}"
            layer.register_forward_hook(self._make_hook(hook_name))
        lm_head = self.adapter.get_lm_head()
        if lm_head is not None:
            lm_head.register_forward_hook(self._make_hook("lm_head"))

    def _make_hook(self, name: str) -> Callable:
        """
        Creates a hook function that stores the detached output of the module.
        
        Args:
            name (str): Identifier for the hook.
        
        Returns:
            Callable: The hook function.
        """
        def hook(module: nn.Module, inputs: Any, output: Any) -> None:
            if isinstance(output, tuple):
                self.hook_outputs[name] = tuple(
                    o.detach().cpu() if isinstance(o, torch.Tensor) else o
                    for o in output
                )
            elif isinstance(output, torch.Tensor):
                self.hook_outputs[name] = output.detach().cpu()
            else:
                self.hook_outputs[name] = output
        return hook

    def clear_hooks(self) -> None:
        """
        Clears the stored hook outputs.
        """
        self.hook_outputs.clear()

    def forward(self, llm_data: Any, output_attentions: bool = False) -> Any:
        """
        Performs a forward pass through the LLM model while capturing intermediate outputs via hooks.
        
        Args:
            llm_data (Any): The input data for the LLM, expected as a dict or an object with attributes.
            output_attentions (bool, optional): Whether to output attentions. Defaults to False.
        
        Returns:
            Any: The output from the LLM's forward pass.
        """
        self.clear_hooks()
        
        # Prepare model inputs from llm_data (either a dict or an object with attributes).
        if isinstance(llm_data, dict):
            model_inputs = llm_data.copy()
        else:
            model_inputs = vars(llm_data).copy()
        
        # Remap "inputs_ids" to "input_ids" if present.
        if "inputs_ids" in model_inputs:
            model_inputs["input_ids"] = model_inputs.pop("inputs_ids")
        
        # If inputs_embeds exist, remove token IDs to avoid conflict.
        if "inputs_embeds" in model_inputs and "input_ids" in model_inputs:
            del model_inputs["input_ids"]
        
        # Remove any extra keys that are not expected by the LLM.
        for extra_key in ["labels_tensor", "token_modalities", "modality_ids"]:
            model_inputs.pop(extra_key, None)
        
        outputs = self.adapter.llm_model(
            **model_inputs,
            output_attentions=output_attentions,
            output_hidden_states=True
        )
        return outputs
