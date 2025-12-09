import sys
from typing import Optional, Literal, Union
import numpy as np
import torch
from PIL import Image
import os

from diffusers import StableDiffusionPipeline, DDPMScheduler
import torchvision.transforms as T


module_dir = os.path.abspath('/home/azureuser/code_projects/itay/diffusion_memorization_iclr24')
sys.path.append(module_dir)


# import our p-Laplace functionality (boundary formualtion)
from p_laplace_core import compute_p_laplace_boundary_torch, sample_sphere_normals_nd_torch

def _encode_text_embeds(pipe: StableDiffusionPipeline, prompt: Optional[str], device: str):
    """
    Return (text_emb or None, uncond_emb) on UNet device/dtype with last dim == pipe.unet.config.cross_attention_dim.
    """
    def _enc_one(p: str):
        if hasattr(pipe, "encode_prompt"):
            out = pipe.encode_prompt(
                prompt=p,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
            emb = out[0] if isinstance(out, tuple) else out
        else:
            out = pipe._encode_prompt(p, device, 1, do_classifier_free_guidance=False)
            emb = out[0] if isinstance(out, tuple) else out
        return emb

    text_emb   = _enc_one(prompt) if (prompt is not None and len(prompt) > 0) else None
    uncond_emb = _enc_one("")

    # move to UNet device/dtype
    ud, dt = pipe.unet.device, pipe.unet.dtype
    if text_emb is not None:
        text_emb = text_emb.to(device=ud, dtype=dt)
    uncond_emb = uncond_emb.to(device=ud, dtype=dt)

    # hard checks
    ca_dim = pipe.unet.config.cross_attention_dim
    if text_emb is not None and text_emb.shape[-1] != ca_dim:
        raise ValueError(f"text_emb dim={text_emb.shape[-1]} != UNet cross_attention_dim={ca_dim}")
    if uncond_emb.shape[-1] != ca_dim:
        raise ValueError(f"uncond_emb dim={uncond_emb.shape[-1]} != UNet cross_attention_dim={ca_dim}")

    return text_emb, uncond_emb

def _resolve_t_index(scheduler: DDPMScheduler, timestep: Optional[int], time_frac: Optional[float]) -> int:
    num_train_steps = scheduler.config.num_train_timesteps
    if timestep is not None:
        t_int = int(timestep)
    else:
        t_int = int((time_frac if time_frac is not None else 0.01) * num_train_steps)
    return max(0, min(t_int, num_train_steps - 1))

def predict_noise_factory(
    timesteps: torch.LongTensor,
    text_emb: Optional[torch.FloatTensor],
    unet,
    vae,
    scheduler: DDPMScheduler,
    *,
    uncond_emb: Optional[torch.FloatTensor] = None,
    score_mode: Literal["cond", "uncond", "gap"] = "cond",
):
    """
    Factory returning predict_noise(noisy_latents) -> (N,4,64,64):
      cond   : eps(x_t | y)
      uncond : eps(x_t | ∅)
      gap    : eps(x_t | y) - eps(x_t | ∅)
    """
    device = timesteps.device
    dtype  = unet.dtype
    if score_mode in ("cond", "gap") and text_emb is None:
        raise ValueError("text_emb required for 'cond' or 'gap'")
    if score_mode in ("uncond", "gap") and uncond_emb is None:
        raise ValueError("uncond_emb required for 'uncond' or 'gap'")

    def predict_noise(noisy_latents: torch.FloatTensor) -> torch.FloatTensor:
        N = noisy_latents.shape[0]
        model_in = scheduler.scale_model_input(noisy_latents.to(dtype), timesteps)
        with torch.set_grad_enabled(noisy_latents.requires_grad):
            if score_mode == "cond":
                te = text_emb.expand(N, -1, -1) if (text_emb.dim()==3 and text_emb.shape[0]==1) else text_emb
                return unet(model_in, timesteps, encoder_hidden_states=te).sample
            elif score_mode == "uncond":
                ue = uncond_emb.expand(N, -1, -1) if (uncond_emb.dim()==3 and uncond_emb.shape[0]==1) else uncond_emb
                return unet(model_in, timesteps, encoder_hidden_states=ue).sample
            else:
                te = text_emb.expand(N, -1, -1) if (text_emb.dim()==3 and text_emb.shape[0]==1) else text_emb
                ue = uncond_emb.expand(N, -1, -1) if (uncond_emb.dim()==3 and uncond_emb.shape[0]==1) else uncond_emb
                eps_text = unet(model_in, timesteps, encoder_hidden_states=te).sample
                eps_unco = unet(model_in, timesteps, encoder_hidden_states=ue).sample
                return eps_text - eps_unco

    return predict_noise

def _encode_image_to_latents(pipe: StableDiffusionPipeline, img: Union[str, Image.Image], device: str) -> torch.FloatTensor:
    image = Image.open(img).convert("RGB") if isinstance(img, str) else img.convert("RGB")
    image = pipe.image_processor.preprocess(image).to(device=device, dtype=pipe.unet.dtype)
    latents = pipe.vae.encode(image).latent_dist.sample() * pipe.vae.config.scaling_factor
    return latents[0]  # (4,64,64)

def _get_radius_factor(scheduler: DDPMScheduler, t_index: int) -> float:
    ac = scheduler.alphas_cumprod.to(torch.float32)[t_index].item()
    return float(np.sqrt(max(0.0, 1.0 - ac)))

def prepare_p_laplace_image_diffusion_model_inputs_bckp(pipe, t_idx, mid_gen_path, prompt, score_mode, device, input_mode):
    tokenizer, text_encoder, unet, vae, scheduler = (
        pipe.tokenizer,
        pipe.text_encoder,
        pipe.unet,
        pipe.vae,
        pipe.scheduler,
    )
    
    # 1. resolve step
    timesteps = torch.full((1,), int(t_idx), device=device, dtype=torch.long)

    # 2. text (and no-text) embeddings
    text_inputs = tokenizer(
        [prompt if (prompt and len(prompt) > 0) else ""],
        padding="max_length", max_length=77, truncation=True, return_tensors="pt"
    )
    input_ids = text_inputs.input_ids.to(device)
    text_emb = text_encoder(input_ids).last_hidden_state.to(unet.dtype)

    uncond_inputs = tokenizer([""], padding="max_length", max_length=77, truncation=True, return_tensors="pt")
    uncond_ids = uncond_inputs.input_ids.to(device)
    uncond_emb = text_encoder(uncond_ids).last_hidden_state.to(unet.dtype)

    # 3. center as scaled image latents and radius at step t
    if mid_gen_path is None:
        raise ValueError("mid_gen_path must be provided for eval_stage='post'")
    if input_mode == "image":
        img = Image.open(mid_gen_path).convert("RGB")
        latents_np = _encode_image_to_latents(pipe, img, device=device)
        latents_torch = latents_np.float().to(device).view(1, 4, 64, 64)
        alphas = scheduler.alphas_cumprod.to(device=timesteps.device, dtype=unet.dtype)
        alpha_prod = alphas[timesteps].view(1)
        center = (torch.sqrt(alpha_prod)[0] * latents_torch[0]).to(unet.dtype)
    elif input_mode == "latent":
        latents = torch.load(mid_gen_path, map_location=device)
        if latents.ndim == 3:
            latents_torch = latents.unsqueeze(0).to(device=device, dtype=unet.dtype)
        else:
            latents_torch = latents.to(device=device, dtype=unet.dtype)
        center = latents_torch[0]
    else:
        raise ValueError(f"Unknown input_mode={input_mode!r}")

    radius_factor = _get_radius_factor(scheduler, t_idx)

    # 4. score fn
    predict_noise = predict_noise_factory(
        timesteps=timesteps,
        text_emb=(text_emb if score_mode in ("cond","gap") else None),
        unet=unet,
        vae=vae,
        scheduler=scheduler,
        uncond_emb=(uncond_emb if score_mode in ("uncond","gap") else None),
        score_mode=score_mode,
    )
    return center, radius_factor, predict_noise


def prepare_p_laplace_image_diffusion_model_inputs_debug(pipe, t_idx, mid_gen_path, prompt, score_mode, device, input_mode):
    tokenizer, text_encoder, unet, vae, scheduler = (
        pipe.tokenizer,
        pipe.text_encoder,
        pipe.unet,
        pipe.vae,
        pipe.scheduler,
    )
    
    # 1. resolve step
    timesteps = torch.full((1,), int(t_idx), device=device, dtype=torch.long)

    # 2. text (and no-text) embeddings
    text_inputs = tokenizer(
        [prompt if (prompt and len(prompt) > 0) else ""],
        padding="max_length", max_length=77, truncation=True, return_tensors="pt"
    )
    input_ids = text_inputs.input_ids.to(device)
    text_emb = text_encoder(input_ids).last_hidden_state.to(unet.dtype)

    uncond_inputs = tokenizer([""], padding="max_length", max_length=77, truncation=True, return_tensors="pt")
    uncond_ids = uncond_inputs.input_ids.to(device)
    uncond_emb = text_encoder(uncond_ids).last_hidden_state.to(unet.dtype)

    # 3. center as scaled image latents and radius at step t
    if mid_gen_path is None:
        raise ValueError("mid_gen_path must be provided for eval_stage='post'")
    
    
    if input_mode == "image":
        ###
        latents_path = str(mid_gen_path).replace(".png", "_latents.pt")

        latents_direct_load = torch.load(latents_path, map_location=device)
        if latents_direct_load.ndim == 3:
            latent_direct_load_torch = latents_direct_load.unsqueeze(0).to(device=device, dtype=unet.dtype)
        else:
            latent_direct_load_torch = latents_direct_load.to(device=device, dtype=unet.dtype)

        with torch.no_grad():
            img = pipe.vae.decode(latent_direct_load_torch / 0.18215).sample
        img = ((img.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).cpu().numpy()
        pil_img = Image.fromarray((img[0] * 255).astype(np.uint8))
        # pil_img.save(img_filename)
        # img = Image.open(img_filename).convert("RGB")
        latents_np = _encode_image_to_latents(pipe, pil_img, device=device)
        latents_torch = latents_np.float().to(device).view(1, 4, 64, 64)
        ###
        img_loaded = Image.open(mid_gen_path).convert("RGB")
        latents_loaded_np = _encode_image_to_latents(pipe, img_loaded, device=device)
        latents_loaded_torch = latents_loaded_np.float().to(device).view(1, 4, 64, 64)
        
        # -------------------------------------------------------
        # DEBUG consistency checks
        # -------------------------------------------------------

        # (1) Compare decoded images: pil_img vs img_loaded
        pil_arr    = np.array(pil_img,    dtype=np.float32) / 255.0
        loaded_arr = np.array(img_loaded, dtype=np.float32) / 255.0
        img_abs_diff = np.abs(pil_arr - loaded_arr)

        print("[DEBUG] Image diff stats (pil_img vs img_loaded):")
        print("  [img]   Max abs diff:", float(img_abs_diff.max()))
        print("  [img]   Mean abs diff:", float(img_abs_diff.mean()))
        print("  [img]   L2 norm diff:", float(np.linalg.norm(pil_arr - loaded_arr)))

        # (2) Compare raw latents (pre-batch): latents_np vs latents_loaded_np
        lat_np_diff = (latents_np - latents_loaded_np).abs()
        print("[DEBUG] Latent diff stats (latents_np vs latents_loaded_np):")
        print("  [raw]   Max abs diff:", lat_np_diff.max().item())
        print("  [raw]   Mean abs diff:", lat_np_diff.mean().item())
        print("  [raw]   L2 norm diff:",
            (latents_np - latents_loaded_np).norm().item())

        # (3) Compare batched latents with relaxed tolerance (Option C):
        #     latents_torch vs latents_loaded_torch
        abs_diff = (latents_torch - latents_loaded_torch).abs()
        if abs_diff.max() < 1e-2:
            print("[DEBUG] Batched latents agree within 1e-2 max-abs tolerance.")
        else:
            print("[WARNING] Batched latents differ beyond 1e-2 max-abs tolerance.")
        print("  [batch] Max abs diff:", abs_diff.max().item())
        print("  [batch] Mean abs diff:", abs_diff.mean().item())
        print("  [batch] Std  abs diff:", abs_diff.std().item())
        print("  [batch] L2 norm diff:",
            (latents_torch - latents_loaded_torch).norm().item())

        exit()
        
        alphas = scheduler.alphas_cumprod.to(device=timesteps.device, dtype=unet.dtype)
        alpha_prod = alphas[timesteps].view(1)
        center = (torch.sqrt(alpha_prod)[0] * latents_torch[0]).to(unet.dtype)
    elif input_mode == "latent":
        latents = torch.load(mid_gen_path, map_location=device)
        if latents.ndim == 3:
            latents_torch = latents.unsqueeze(0).to(device=device, dtype=unet.dtype)
        else:
            latents_torch = latents.to(device=device, dtype=unet.dtype)
        center = latents_torch[0]
    else:
        raise ValueError(f"Unknown input_mode={input_mode!r}")

    radius_factor = _get_radius_factor(scheduler, t_idx)

    # 4. score fn
    predict_noise = predict_noise_factory(
        timesteps=timesteps,
        text_emb=(text_emb if score_mode in ("cond","gap") else None),
        unet=unet,
        vae=vae,
        scheduler=scheduler,
        uncond_emb=(uncond_emb if score_mode in ("uncond","gap") else None),
        score_mode=score_mode,
    )
    return center, radius_factor, predict_noise

def prepare_p_laplace_image_diffusion_model_inputs(latents, pipe, t_idx, prompt, score_mode, device, ablate_reencode=False):
    tokenizer, text_encoder, unet, vae, scheduler = (
        pipe.tokenizer,
        pipe.text_encoder,
        pipe.unet,
        pipe.vae,
        pipe.scheduler,
    )
    
    # 1. resolve step
    timesteps = torch.full((1,), int(t_idx), device=device, dtype=torch.long)

    # 2. text (and no-text) embeddings
    text_inputs = tokenizer(
        [prompt if (prompt and len(prompt) > 0) else ""],
        padding="max_length", max_length=77, truncation=True, return_tensors="pt"
    )
    input_ids = text_inputs.input_ids.to(device)
    text_emb = text_encoder(input_ids).last_hidden_state.to(unet.dtype)

    uncond_inputs = tokenizer([""], padding="max_length", max_length=77, truncation=True, return_tensors="pt")
    uncond_ids = uncond_inputs.input_ids.to(device)
    uncond_emb = text_encoder(uncond_ids).last_hidden_state.to(unet.dtype)
    
    # 3. decode->reencode latents
    if not ablate_reencode:
        with torch.no_grad():
            img = pipe.vae.decode(latents / 0.18215).sample
        img = ((img.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).cpu().numpy()
        pil_img = Image.fromarray((img[0] * 255).astype(np.uint8))
        # pil_img.save(img_filename)
        # img = Image.open(img_filename).convert("RGB")
        latents_np = _encode_image_to_latents(pipe, pil_img, device=device)
        latents_torch = latents_np.float().to(device).view(1, 4, 64, 64)
    else:
        if latents.ndim == 3:  # then add a singleton batch dimensions
            latents_torch = latents.unsqueeze(0).to(device=device, dtype=unet.dtype)
        else:
            latents_torch = latents.to(device=device, dtype=unet.dtype)
    
    # 4. get center as scaled latents and radius at step t
    alphas = scheduler.alphas_cumprod.to(device=timesteps.device, dtype=unet.dtype)
    alpha_prod = alphas[timesteps].view(1)
    center = (torch.sqrt(alpha_prod)[0] * latents_torch[0]).to(unet.dtype)  
    radius_factor = _get_radius_factor(scheduler, t_idx)

    # 5. score fn
    predict_noise = predict_noise_factory(
        timesteps=timesteps,
        text_emb=(text_emb if score_mode in ("cond","gap") else None),
        unet=unet,
        vae=vae,
        scheduler=scheduler,
        uncond_emb=(uncond_emb if score_mode in ("uncond","gap") else None),
        score_mode=score_mode,
    )
    return center, radius_factor, predict_noise

    
def compute_k_delta(latents, pipe, t_idx, prompt, n_samples, device):
    ablate_reencode = False  # False  # Hard-code as False to follow best-practice of Brokman et. al, ICLR 24. If set true we skip the decode->reencode step
    score_mode = "gap"  # hard-code as "gap" to follow best-practice of Wen et. al, ICLR 24. "gap" refers to the "delta in k_delta" between cond and uncond
    p = 1.0
    
    center, radius_factor, predict_noise = prepare_p_laplace_image_diffusion_model_inputs(latents, pipe, t_idx, prompt, score_mode, device, ablate_reencode)
    with torch.no_grad():
        k_delta, _ = compute_p_laplace_boundary_torch(
            center=center,
            radius_factor=radius_factor,
            p=float(p),
            get_logp_gradients=predict_noise,
            n_samples=n_samples,
        )
    
    return k_delta
