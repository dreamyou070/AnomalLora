import torch
import argparse
import os

def get_noise_noisy_latents_and_timesteps(args, noise_scheduler, latents, noise = None):
    # Sample noise that we'll add to the latents
    if noise is None:
        noise = torch.randn_like(latents, device=latents.device)
    b_size = latents.shape[0]
    min_timestep = 0
    max_timestep = noise_scheduler.config.num_train_timesteps
    timesteps = torch.randint(min_timestep, max_timestep, (b_size,), device=latents.device)
    timesteps = timesteps.long()
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    return noise, noisy_latents, timesteps


def get_input_ids(tokenizer, caption):
    tokenizer_output = tokenizer(caption, padding="max_length", truncation=True,
                                 return_tensors="pt")
    input_ids = tokenizer_output.input_ids
    attention_mask = tokenizer_output.attention_mask
    return input_ids, attention_mask

def get_hidden_states(args: argparse.Namespace, input_ids, tokenizer, text_encoder, weight_dtype=None):
    # with no_token_padding, the length is not max length, return result immediately
    if input_ids.size()[-1] != tokenizer.model_max_length:
        return text_encoder(input_ids)[0]

    # input_ids: b,n=3,77
    b_size = input_ids.size()[0]
    input_ids = input_ids.reshape((-1, tokenizer.model_max_length))  # batch_size*3, 77

    if args.clip_skip is None:
        encoder_hidden_states = text_encoder(input_ids)[0]
    else:
        enc_out = text_encoder(input_ids, output_hidden_states=True, return_dict=True)
        encoder_hidden_states = enc_out["hidden_states"][-args.clip_skip]
        encoder_hidden_states = text_encoder.text_model.final_layer_norm(encoder_hidden_states)

    encoder_hidden_states = encoder_hidden_states.reshape((b_size, -1, encoder_hidden_states.shape[-1]))

    if args.max_token_length is not None:
        states_list = [encoder_hidden_states[:, 0].unsqueeze(1)]  # <BOS>
        for i in range(1, args.max_token_length, tokenizer.model_max_length):
            states_list.append(encoder_hidden_states[:, i : i + tokenizer.model_max_length - 2])  # <BOS> の後から <EOS> の前まで
        states_list.append(encoder_hidden_states[:, -1].unsqueeze(1))  # <EOS>
        encoder_hidden_states = torch.cat(states_list, dim=1)

    if weight_dtype is not None:
        encoder_hidden_states = encoder_hidden_states.to(weight_dtype)
    return encoder_hidden_states

def prepare_scheduler_for_custom_training(noise_scheduler, device):
    if hasattr(noise_scheduler, "all_snr"):
        return
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    alpha = sqrt_alphas_cumprod
    sigma = sqrt_one_minus_alphas_cumprod
    all_snr = (alpha / sigma) ** 2

    noise_scheduler.all_snr = all_snr.to(device)

def pe_model_save(model, save_dtype, save_dir):
    state_dict = model.state_dict()
    for key in list(state_dict.keys()):
        v = state_dict[key]
        v = v.detach().clone().to("cpu").to(save_dtype)
        state_dict[key] = v
    _, file = os.path.split(save_dir)
    if os.path.splitext(file)[1] == ".safetensors":
        from safetensors.torch import save_file
        save_file(state_dict, save_dir)
    else:
        torch.save(state_dict, save_dir)

