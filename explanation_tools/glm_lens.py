import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from absl import logging

from matplotlib import cm, colors

def get_contrasting_text_color(value, cmap_name="magma", vmin=None, vmax=None):
    """
    Given a value and colormap, returns 'white' or 'black' depending on brightness.
    """
    norm = plt.Normalize(vmin, vmax)
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(norm(value))  # Get RGBA tuple
    r, g, b = rgba[:3]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b  # Perceived brightness
    return 'black' if luminance > 0.5 else 'white'


def compute_decoded_logit_lens(hook_outputs: dict, adapter, tokenizer, target_token_ids=None) -> dict:
    """
    Computes, for each transformer layer, the predicted token (via argmax over logits)
    and its associated maximum logit value at each token position.
    """
    lm_head = adapter.get_lm_head()
    if lm_head is None:
        raise ValueError("LM head is not available from the adapter.")

    lm_weight = lm_head.weight.detach()#.cpu()  # (vocab_size, hidden_dim)
    lm_bias = lm_head.bias.detach() if lm_head.bias is not None else None #.cpu() if lm_head.bias is not None else None

    decoded_dict = {}
    layer_keys = sorted([k for k in hook_outputs.keys() if k.startswith("layer_")],
                        key=lambda x: int(x.split("_")[1]))
    for key in layer_keys:
        hidden_states = hook_outputs[key]
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        hidden_states = hidden_states.to('cuda')
        lm_weight = lm_weight.to('cuda')

        final_ln = adapter.get_final_layer_norm() if hasattr(adapter, "get_final_layer_norm") else None
        if final_ln is not None:
            final_ln = final_ln.to('cuda')

        if final_ln is not None:
            print('applying norm')
            hidden_states = final_ln(hidden_states)

        logits = torch.matmul(hidden_states, lm_weight.T)
        if lm_bias is not None:
            lm_bias = lm_bias.to('cuda')
            logits += lm_bias

        batch_size, seq_len, _ = logits.shape
        layer_result = []
        for b in range(batch_size):
            row = []
            for t in range(seq_len):
                pred_token_id = torch.argmax(logits[b, t]).item()
                pred_logit = torch.max(logits[b, t]).item()
                pred_token_str = tokenizer.decode([pred_token_id], skip_special_tokens=True).strip()
                row.append((pred_token_str, pred_logit))
            layer_result.append(row)
        decoded_dict[key] = layer_result

    return decoded_dict

def compute_decoded_prob_lens(hook_outputs: dict, adapter, tokenizer, target_token_ids=None) -> dict:
    """
    For each transformer layer, this function applies final layer normalization
    (if available) to the hooked hidden states, projects them using the LM head,
    applies softmax to obtain probabilities, and then decodes the token with the 
    highest probability at each position.
    """
    lm_head = adapter.get_lm_head()
    if lm_head is None:
        raise ValueError("LM head is not available from the adapter.")

    final_ln = adapter.get_final_layer_norm() if hasattr(adapter, "get_final_layer_norm") else None

    # Move weights and bias to CUDA once
    lm_weight = lm_head.weight.detach().to('cuda')
    lm_bias = lm_head.bias.detach().to('cuda') if lm_head.bias is not None else None
    if final_ln is not None:
        print('applying norm')
        final_ln = final_ln.to('cuda')

    decoded_dict = {}
    layer_keys = sorted([k for k in hook_outputs.keys() if k.startswith("layer_")],
                        key=lambda x: int(x.split("_")[1]))

    for key in layer_keys:
        hidden_states = hook_outputs[key]
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        hidden_states = hidden_states.to('cuda')
        if final_ln is not None:
            print('applying norm')
            hidden_states = final_ln(hidden_states)

        logits = torch.matmul(hidden_states, lm_weight.T)
        if lm_bias is not None:
            logits += lm_bias

        probs = torch.nn.functional.softmax(logits, dim=-1)

        batch_size, seq_len, _ = logits.shape
        layer_result = []
        for b in range(batch_size):
            row = []
            for t in range(seq_len):
                max_prob, token_id = torch.max(probs[b, t], dim=-1)
                pred_token_str = tokenizer.decode([token_id.item()], skip_special_tokens=True).strip()
                row.append((pred_token_str, max_prob.item()))
            layer_result.append(row)
        decoded_dict[key] = layer_result

    return decoded_dict



def save_decoded_logit_lens_json(decoded_dict: dict, output_path: str) -> None:
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    serializable = {}
    for layer, batch_data in decoded_dict.items():
        serializable[layer] = []
        for row in batch_data:
            serializable[layer].append([[tok, val] for tok, val in row])
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=4)
    logging.info("Decoded logit lens data saved to %s", output_path)


def plot_decoded_heatmap(decoded_dict: dict, output_png_path: str, input_tokens: list = None, title: str = "Decoded Logit Lens Heatmap") -> None:
    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    
    layer_keys = sorted([k for k in decoded_dict.keys() if k.startswith("layer_")],
                        key=lambda x: int(x.split("_")[1]))
    num_layers = len(layer_keys)
    seq_len = len(decoded_dict[layer_keys[0]][0])
    
    heatmap_data = np.empty((num_layers, seq_len))
    heatmap_data[:] = np.nan
    token_overlay = [['' for _ in range(seq_len)] for _ in range(num_layers)]
    
    for i, key in enumerate(layer_keys):
        row_data = decoded_dict[key][0]
        for j, (token_str, token_val) in enumerate(row_data):
            heatmap_data[i, j] = token_val if token_val is not None else np.nan
            token_overlay[i][j] = token_str if token_str is not None else ""
    
    # Compute the overall minimum and maximum for normalization
    vmin = np.nanmin(heatmap_data)
    vmax = np.nanmax(heatmap_data)
    
    fig_width = max(10, seq_len * 0.6)
    fig_height = max(8, num_layers * 0.7)
    plt.figure(figsize=(fig_width, fig_height))
    
    im = plt.imshow(heatmap_data, aspect='auto', cmap=plt.cm.magma, origin='lower')
    plt.colorbar(im, label="Logit Value")
    plt.ylabel("Transformer Layer")
    plt.title(title)
    
    if input_tokens is not None and len(input_tokens) == seq_len:
        plt.xticks(ticks=range(seq_len), labels=input_tokens, rotation=60, fontsize=10)
    else:
        plt.xticks(ticks=range(seq_len), fontsize=10)
    
    plt.yticks(ticks=range(num_layers), labels=layer_keys, fontsize=10)
    
    # Use computed vmin and vmax to ensure text contrast is adjusted properly
    for i in range(num_layers):
        for j in range(seq_len):
            txt = token_overlay[i][j]
            if txt:
                text_color = get_contrasting_text_color(heatmap_data[i, j], cmap_name="magma", vmin=vmin, vmax=vmax)
                plt.text(j, i, txt, ha="center", va="center", color=text_color, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_png_path)
    plt.close()
    logging.info("Full decoded heatmap saved to %s", output_png_path)


def plot_graph_tokens_heatmap(decoded_dict: dict, token_modalities: torch.Tensor, output_png_path: str, graph_input_tokens: list = None, title: str = "Graph Token Logit Lens Heatmap") -> None:
    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    
    graph_indices = (token_modalities[0] == 0).nonzero(as_tuple=False).squeeze(1).tolist()
    if len(graph_indices) == 0:
        logging.info("No graph tokens found in token_modalities.")
        return

    layer_keys = sorted([k for k in decoded_dict.keys() if k.startswith("layer_")],
                        key=lambda x: int(x.split("_")[1]))
    num_layers = len(layer_keys)
    seq_len_graph = len(graph_indices)
    
    heatmap_data = np.empty((num_layers, seq_len_graph))
    heatmap_data[:] = np.nan
    token_overlay = [['' for _ in range(seq_len_graph)] for _ in range(num_layers)]
    
    for i, key in enumerate(layer_keys):
        full_row = decoded_dict[key][0]
        for j, pos in enumerate(graph_indices):
            token_str, token_val = full_row[pos]
            heatmap_data[i, j] = token_val if token_val is not None else np.nan
            token_overlay[i][j] = token_str if token_str is not None else ""
    
    if graph_input_tokens is not None and len(graph_input_tokens) > 0:
        x_labels = [graph_input_tokens[pos] for pos in graph_indices]
    else:
        x_labels = [str(pos) for pos in graph_indices]
    
    # Compute normalization parameters for the graph tokens heatmap
    vmin = np.nanmin(heatmap_data)
    vmax = np.nanmax(heatmap_data)
    
    cell_width = 1.2
    cell_height = 1.0
    plt.figure(figsize=(seq_len_graph * cell_width, num_layers * cell_height))
    im = plt.imshow(heatmap_data, aspect='auto', cmap=plt.cm.magma, origin='lower')
    plt.colorbar(im, label="Logit Value")
    plt.xlabel("Graph Token Position")
    plt.ylabel("Transformer Layer")
    plt.title(title)
    plt.xticks(ticks=range(seq_len_graph), labels=x_labels, rotation=90, fontsize=10)
    plt.yticks(ticks=range(num_layers), labels=layer_keys)
    
    for i in range(num_layers):
        for j in range(seq_len_graph):
            txt = token_overlay[i][j]
            if txt:
                text_color = get_contrasting_text_color(heatmap_data[i, j], cmap_name="magma", vmin=vmin, vmax=vmax)
                plt.text(j, i, txt, ha="center", va="center", color=text_color, fontsize=8)
    
    plt.tight_layout(pad=3.0)
    plt.savefig(output_png_path)
    plt.close()
    logging.info("Graph tokens heatmap saved to %s", output_png_path)
