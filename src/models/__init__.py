from .gnn.gat import GAT
from .gnn.graph_sage import GraphSAGE
from .llm.llm_wrapper import LLM
from .glm.teaglm import TEA_GLM, TEA_GLM_LightningModule
from .glm.g_retriever import GRetriever, GRetrieverLightningModule
from .glm.llm_only import LLMOnly, LLMOnlyLightningModule
from .glm.softprompt import Soft_Prompt, Soft_Prompt_LightningModule

def get_glm(method):
    mapper = {
        "tea-glm": TEA_GLM,
        "g-retriever": GRetriever,
        "llm-only": LLMOnly,
        "soft-prompt": Soft_Prompt,
    }
    return mapper[method]

def get_glm_lightning_module(method):
    mapper = {
        "tea-glm": TEA_GLM_LightningModule,
        "g-retriever": GRetrieverLightningModule,
        "llm-only": LLMOnlyLightningModule,
        "soft-prompt": Soft_Prompt_LightningModule,

    }
    return mapper[method]