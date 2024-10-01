import argparse
import os
import shutil

import torch
from mistral_inference.transformer import Transformer
from safetensors.torch import save_model
from transformers import LlavaForConditionalGeneration


def inverse_permute_for_rope(value: torch.Tensor, n_heads: int, hidden_size: int) -> torch.Tensor:
    dim1 = value.shape[0]
    dim2 = hidden_size
    return value.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(2, 1).reshape(dim1, dim2)


def transform_hf_pixtral_to_vllm_pixtral(base_folder: str, original_pixtral_folder: str, finetuned_pixtral_folder: str) -> None:
    modified_pixtral = Transformer.from_folder(os.path.join(base_folder, original_pixtral_folder))
    finetuned_model = LlavaForConditionalGeneration.from_pretrained(os.path.join(base_folder, finetuned_pixtral_folder)).to('cuda')

    # Overwrite state dict
    state_dict_copy = modified_pixtral.state_dict().copy()
    
    # Copy vision module
    for i in range(24):
        state_dict_copy[f'vision_encoder.transformer.layers.{i}.attention.wq.weight'] = inverse_permute_for_rope(
            finetuned_model.vision_tower.transformer.layers[i].attention.q_proj.weight.data,
            n_heads=16,
            hidden_size=1024
        )
        state_dict_copy[f'vision_encoder.transformer.layers.{i}.attention.wk.weight'] = inverse_permute_for_rope(
            finetuned_model.vision_tower.transformer.layers[i].attention.k_proj.weight.data,
            n_heads=16,
            hidden_size=1024
        )
        state_dict_copy[f'vision_encoder.transformer.layers.{i}.attention.wv.weight'] = finetuned_model.vision_tower.transformer.layers[i].attention.v_proj.weight.data
        state_dict_copy[f'vision_encoder.transformer.layers.{i}.attention.wo.weight'] = finetuned_model.vision_tower.transformer.layers[i].attention.o_proj.weight.data
        state_dict_copy[f'vision_encoder.transformer.layers.{i}.attention_norm.weight'] = finetuned_model.vision_tower.transformer.layers[i].attention_norm.weight.data
        state_dict_copy[f'vision_encoder.transformer.layers.{i}.ffn_norm.weight'] = finetuned_model.vision_tower.transformer.layers[i].ffn_norm.weight.data
        state_dict_copy[f'vision_encoder.transformer.layers.{i}.attention_norm.weight'] = finetuned_model.vision_tower.transformer.layers[i].attention_norm.weight.data
        state_dict_copy[f'vision_encoder.transformer.layers.{i}.feed_forward.w1.weight'] = finetuned_model.vision_tower.transformer.layers[i].feed_forward.gate_proj.weight.data
        state_dict_copy[f'vision_encoder.transformer.layers.{i}.feed_forward.w2.weight'] = finetuned_model.vision_tower.transformer.layers[i].feed_forward.down_proj.weight.data
        state_dict_copy[f'vision_encoder.transformer.layers.{i}.feed_forward.w3.weight'] = finetuned_model.vision_tower.transformer.layers[i].feed_forward.up_proj.weight.data

    # Copy vision language adapter
    state_dict_copy['vision_language_adapter.w_in.weight'] = finetuned_model.multi_modal_projector.linear_1.weight.data
    state_dict_copy['vision_language_adapter.w_out.weight'] = finetuned_model.multi_modal_projector.linear_2.weight.data

    # Copy language model
    for i in range(40):
        state_dict_copy[f'layers.{i}.attention.wq.weight'] = inverse_permute_for_rope(
            finetuned_model.language_model.model.layers[i].self_attn.q_proj.weight.data,
            n_heads=32,
            hidden_size=5120
        )
        state_dict_copy[f'layers.{i}.attention.wk.weight'] = inverse_permute_for_rope(
            finetuned_model.language_model.model.layers[i].self_attn.k_proj.weight.data,
            n_heads=8,
            hidden_size=5120
        )
        state_dict_copy[f'layers.{i}.attention.wv.weight'] = finetuned_model.language_model.model.layers[i].self_attn.v_proj.weight.data
        state_dict_copy[f'layers.{i}.attention.wo.weight'] = finetuned_model.language_model.model.layers[i].self_attn.o_proj.weight.data
        state_dict_copy[f'layers.{i}.feed_forward.w1.weight'] = finetuned_model.language_model.model.layers[i].mlp.gate_proj.weight.data
        state_dict_copy[f'layers.{i}.feed_forward.w2.weight'] = finetuned_model.language_model.model.layers[i].mlp.down_proj.weight.data
        state_dict_copy[f'layers.{i}.feed_forward.w3.weight'] = finetuned_model.language_model.model.layers[i].mlp.up_proj.weight.data
        state_dict_copy[f'layers.{i}.attention_norm.weight'] = finetuned_model.language_model.model.layers[i].input_layernorm.weight.data
        state_dict_copy[f'layers.{i}.ffn_norm.weight'] = finetuned_model.language_model.model.layers[i].post_attention_layernorm.weight.data

    # Copy embedding layer
    state_dict_copy['tok_embeddings.weight'] = finetuned_model.language_model.model.embed_tokens.weight.data

    # Copy output layer
    state_dict_copy['norm.weight'] = finetuned_model.language_model.model.norm.weight.data
    state_dict_copy['output.weight'] = finetuned_model.language_model.lm_head.weight.data
    
    # Replace state dict
    modified_pixtral.load_state_dict(state_dict_copy)
    # Export model
    output_folder = os.path.join(base_folder, finetuned_pixtral_folder + '_vllm_format')
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    save_model(modified_pixtral, os.path.join(output_folder, 'consolidated.safetensors'))
    shutil.copyfile(os.path.join(os.path.join(base_folder, original_pixtral_folder), 'tekken.json'), os.path.join(output_folder, 'tekken.json'))
    shutil.copyfile(os.path.join(os.path.join(base_folder, original_pixtral_folder), 'params.json'), os.path.join(output_folder, 'params.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process settings.')
    parser.add_argument(
        '--model_name',
        help='Name of the model to convert',
        default='pixtral_finetuned_merged'
    )
    args = parser.parse_args()

    transform_hf_pixtral_to_vllm_pixtral(base_folder='/docker_share/models/', original_pixtral_folder='Pixtral-12B-2409', finetuned_pixtral_folder=args.model_name)
