import torch
from absl import logging
from src.modifiers import Modifier
from src.models.llm.llm_wrapper import LLM


class BaseGLM(torch.nn.Module):
    r"""
    Base class for all Graph Language Models (GLMs).

    Args:
        llm (LLM): The language model to use.
        gnn (torch.nn.Module): The graph neural network to use.
        use_lora (bool): Whether to use LORA or not.
        projector_out_channels (int): The number of output channels of the projector. Should equal the LLM embedding size.
        n_gnn_toks (int): The number of GNN tokens to use in the LLM prompt.
    """

    def __init__(
        self,
        llm: LLM,
        gnn: torch.nn.Module,
        use_lora: bool = False,
        projector_out_channels: int = 64,
        n_gnn_toks: int = 1,
    ) -> None:
        super().__init__()
        self.llm = llm
        self.word_embedding = self.llm.word_embedding
        self.llm_generator = llm.llm
        self.gnn = gnn
        self.use_lora = use_lora
        self.n_gnn_toks = n_gnn_toks
        self.projector_out_channels = projector_out_channels
        self.projector = None
        self.EOS_TOKEN_ID = llm.tokenizer("[/s]", add_special_tokens=False).input_ids[0]
        self.LLM_EOS_TOKEN_ID = llm.tokenizer.eos_token_id

        if use_lora:
            from peft import (
                LoraConfig,
                get_peft_model,
                prepare_model_for_kbit_training,
            )

            self.llm_generator = prepare_model_for_kbit_training(self.llm_generator)
            lora_r: int = 8
            lora_alpha: int = 16
            lora_dropout: float = 0.05
            lora_target_modules = ["q_proj", "v_proj"]
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llm_generator = get_peft_model(self.llm_generator, config)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  llm={self.llm},\n"
            f"  gnn={self.gnn},\n"
            f")"
        )

    def get_context(self, data):
        context = data.context
        con = []
        for i in range(len(data.question)):
            tempcon = ""
            if context is not None:
                if type(context) == dict:
                    for key, value in context.items():
                        tempcon += key + ": " + value[i] + "\n"
                elif type(context[i]) == str:
                    tempcon = context[i]
                else:
                    raise TypeError(
                        f"Unsupported context type: {type(context)}. Expected dict or list."
                    )
            con.append(tempcon)
            
        return con

    def _in_house_generate(self, inputs_embeds, attention_mask, max_length):
        # Cache the embedding layer outside the loop.
        embedding_layer = self.llm_generator.get_input_embeddings()

        # First forward pass with full input embeddings.
        outputs = self.llm_generator(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)

        generated_tokens = [next_token_id]

        # For subsequent iterations, only process the newly generated token.
        current_token_embedding = embedding_layer(next_token_id)

        # Loop from 1 to max tokens.
        for _ in range(1, max_length):
            outputs = self.llm_generator(
                inputs_embeds=current_token_embedding,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)

            # check if the next token is the end of sequence token
            if (
                next_token_id.item() == self.EOS_TOKEN_ID
                or next_token_id.item() == self.LLM_EOS_TOKEN_ID
            ):
                break
            generated_tokens.append(next_token_id)

            # Get embedding for next token.
            current_token_embedding = embedding_layer(next_token_id)

        # Concatenate generated token ids.
        generated_sequence = torch.cat(generated_tokens, dim=1)

        logging.debug(f"Number of generated tokens: {len(generated_tokens)}")

        return generated_sequence

    def generate(self, inputs_embeds, attention_mask, max_length, **kwargs):
        # if "llama" in self.llm.model_name.lower():
        #     return self.llm_generator.generate(
        #         inputs_embeds=inputs_embeds,
        #         attention_mask=attention_mask,
        #         max_length=max_length,
        #         **kwargs,
        #     )
        # else:
        return self._in_house_generate(inputs_embeds, attention_mask, max_length)

    def prepare_projector(self):
        r"""
        Prepares the projector for the GNN.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError

    def load_projector(self, path):
        r"""
        Loads the projector from a file.
        """
        self.projector.load_state_dict(torch.load(path, weights_only=False))
        logging.info("Loaded projector from %s.", path)

    def encode_graph(self, data):
        r"""
        Encodes the graph into a single vector.

        Args:
            data (Batch): A torch_geometric Batch containing at least 'x', 'edge_index', and 'batch'.

        Returns:
            torch.Tensor: The encoded graph representation.
        """
        raise NotImplementedError

    def forward(self, data):
        r"""
        Forward pass of the model.

        Expected keys in the batch:
            - question: List[str]
            - x: torch.Tensor (node features)
            - edge_index: torch.Tensor
            - batch: torch.Tensor (node-to-graph mapping)
            - label: List[str]
            - node_id (optional): List[int]
            - context (optional): Dict[str, str]

        Args:
            data (Batch): A Batch object containing all necessary inputs.

        Returns:
            The model loss (or other training signal).
        """
        raise NotImplementedError

    def inference(
        self, data, max_out_tokens: int, graph_data=None, modifier: Modifier = None
    ):
        r"""
        Inference pass of the model.

        Expected keys in the batch:
            - question: List[str]
            - x: torch.Tensor (node features)
            - edge_index: torch.Tensor
            - batch: torch.Tensor (node-to-graph mapping)
            - node_id (optional): List[int]
            - context (optional): Dict[str, str]

        Args:
            data (Batch): A Batch object containing the inference inputs.
            max_out_tokens (int): The maximum number of tokens to generate.

        Returns:
            The decoded output from the language model.
        """
        raise NotImplementedError
