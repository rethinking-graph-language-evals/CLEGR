import os
import torch
from transformers import AutoModelForCausalLM
import numpy as np
from sklearn.decomposition import PCA

llm_to_ncomp = {
    "gpt2": 400,
    "Meta-Llama-3-8B-Instruct": 1000,
    "Phi-3.5-mini-instruct": 1000,
    "phi-4": 1000
}

def get_pc(
    llm_name: str,
    datasets_root: str = "../../../datasets/",
    force_reprocess: bool = False,
):
    for llm_str in llm_to_ncomp:
        if llm_str.lower() in llm_name.lower():
            n_components = llm_to_ncomp[llm_str]
            break
    else:
        raise ValueError(f"LLM name {llm_name} not recognized")

    cache_path = os.path.join(datasets_root, f"teaglm_pca/pca_{n_components}_{llm_str}.pt")
    if not force_reprocess and os.path.exists(cache_path):
        return llm_str, torch.load(cache_path, weights_only=False)
    
    # create path
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    llm = AutoModelForCausalLM.from_pretrained(llm_name, device_map="cpu")

    llm_embeds = llm.get_input_embeddings().weight.data

    numpy_matrix = llm_embeds.numpy()

    pca = PCA(n_components=n_components)
    pca.fit(numpy_matrix)

    explained_variance_ratio = pca.explained_variance_ratio_

    ratio_sum = 0
    for i, ratio in enumerate(explained_variance_ratio):
        ratio_sum += ratio
    print(ratio_sum)

    components = pca.components_

    components_float16 = components
    # components_float16 = components.astype(np.float)

    tensor_components_float16 = torch.tensor(components_float16)
    torch.save(tensor_components_float16, cache_path)
    return llm_str, tensor_components_float16