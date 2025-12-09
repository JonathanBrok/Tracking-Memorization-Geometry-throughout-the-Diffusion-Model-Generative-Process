""""generate.py """

import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np
import urllib.parse
import json

########################################
# Configuration
########################################
num_inference_steps = 500
num_generations = 3
threshold = 100.0  # MSE threshold for deciding memorization

# Toggle: generate non-memorized images from the memorized prompt (MSE > threshold)
ENABLE_NONMEM_FROM_MEM = False

# New JSONL with keys: "memorized" and "non_memorized"
PROMPTS_JSONL = "prompts.jsonl"

prompts_memorized = []
prompts_non_memorized = []

import json
with open(PROMPTS_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        prompts_memorized.append(item["memorized"])
        prompts_non_memorized.append(item["non_memorized"])

########################################
# Paths
########################################

memorized_root = f"./generations/memorized_mse_{threshold}"
non_memorized_root = f"./generations/non_memorized_prompt_mse_{threshold}"
non_memorized_from_mem_root = f"./generations/non_memorized_from_mem_mse_{threshold}"
oracles_root = "oracles"

memorized_mid_gens_root = "./generations/memorized_mid_generations"
non_memorized_mid_gens_root = "./generations/non_memorized_prompt_mid_generations"
non_memorized_from_mem_mid_gens_root = "./generations/non_memorized_from_mem_mid_generations"

# sometimes prompts contain special characters that can't be used as filepaths
import urllib.parse
oracle_paths = [
    os.path.join(oracles_root, f"{urllib.parse.quote(prompt, safe='')}.png")
    for prompt in prompts_memorized
]

memorized_dirs = [os.path.join(memorized_root, prompt) for prompt in prompts_memorized]
non_memorized_dirs = [os.path.join(non_memorized_root, prompt) for prompt in prompts_non_memorized]
non_memorized_from_mem_dirs = [os.path.join(non_memorized_from_mem_root, prompt) for prompt in prompts_memorized]

memorized_mid_gens_dirs = [os.path.join(memorized_mid_gens_root, prompt) for prompt in prompts_memorized]
non_memorized_mid_gens_dirs = [os.path.join(non_memorized_mid_gens_root, prompt) for prompt in prompts_non_memorized]
non_memorized_from_mem_mid_gens_dirs = [os.path.join(non_memorized_from_mem_mid_gens_root, prompt) for prompt in prompts_memorized]

# Create root directories
os.makedirs(memorized_root, exist_ok=True)
os.makedirs(non_memorized_root, exist_ok=True)
os.makedirs(non_memorized_from_mem_root, exist_ok=True)
os.makedirs(oracles_root, exist_ok=True)

# Create dirs for storing mid generation images for future analysis
os.makedirs(memorized_mid_gens_root, exist_ok=True)
os.makedirs(non_memorized_mid_gens_root, exist_ok=True)
os.makedirs(non_memorized_from_mem_mid_gens_root, exist_ok=True)

for i in range(len(memorized_dirs)):
    os.makedirs(memorized_dirs[i], exist_ok=True)
    os.makedirs(non_memorized_dirs[i], exist_ok=True)
    os.makedirs(non_memorized_from_mem_dirs[i], exist_ok=True)

    os.makedirs(memorized_mid_gens_dirs[i], exist_ok=True)
    os.makedirs(non_memorized_mid_gens_dirs[i], exist_ok=True)
    os.makedirs(non_memorized_from_mem_mid_gens_dirs[i], exist_ok=True)

########################################
# Helper Function
########################################

def compute_mse(img1, img2):
    """
    Compute Mean Squared Error (MSE) between two images.
    """
    arr1 = np.array(img1.convert("RGB"))
    arr2 = np.array(img2.convert("RGB"))

    # If sizes differ, resize second image to match the first
    if arr1.shape != arr2.shape:
        arr2 = np.array(img2.resize(img1.size))

    return ((arr1 - arr2) ** 2).mean()

########################################
# Image Generation
########################################


def make_save_callback(pipe, img_name, out_dir):
    def save_intermediate(step, timestep, latents):
        if step < 2 or step % 50 == 0:
            # 1) Save decoded image 
            img_filename = f"{out_dir}/{img_name}_step={step}.png"
            img = pipe.vae.decode(latents / 0.18215).sample
            img = ((img.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).cpu().numpy()
            from PIL import Image
            pil_img = Image.fromarray((img[0] * 255).astype(np.uint8))
            pil_img.save(img_filename)

            # 2) Save raw latents for k_delta (same step)
            latent_filename = f"{out_dir}/{img_name}_step={step}_latents.pt"
            torch.save(latents.detach().cpu(), latent_filename)
    return save_intermediate



def main():
    # Load the Stable Diffusion model
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        safety_checker=None  # disable safety filters that someitmes generate all black images
    ).to("cuda")
    
    
    num_retries = 5 
    skipped_examples = set()

    for i in range(len(prompts_memorized)):
        oracle_path = oracle_paths[i]

        prompt_memorized = prompts_memorized[i]
        prompt_non_memorized_random = prompts_non_memorized[i]

        memorized_dir = memorized_dirs[i]
        non_memorized_random_dir = non_memorized_dirs[i]
        non_memorized_from_mem_dir = non_memorized_from_mem_dirs[i]

        memorized_mid_gens_dir = memorized_mid_gens_dirs[i]
        non_memorized_random_mid_gens_dir = non_memorized_mid_gens_dirs[i]
        non_memorized_from_mem_mid_gens_dir = non_memorized_from_mem_mid_gens_dirs[i]

        
        # Load reference (oracle) image
        try:
            if not os.path.exists(oracle_path):
                print(f"Oracle image not found: {oracle_path}. Generating with prompt: '{prompt_memorized}'")
                model_id = "CompVis/stable-diffusion-v1-4"
                oracle_img = pipe(prompt_memorized, num_inference_steps=num_inference_steps).images[0]
                oracle_img.save(oracle_path)
            else:
                oracle_img = Image.open(oracle_path).convert("RGB")
        except Exception: 
            continue


        # Generate images for both prompts in one run
        for gen_idx in range(num_generations):
            
            img_name = f"memorized_{gen_idx}"
            callback_fn = make_save_callback(pipe, img_name, memorized_mid_gens_dir)

            # Memorized attempt
            memorized_success = False
            non_memorized_from_mem_success = False
            save_path = os.path.join(memorized_dir, f"{img_name}.png")

            if not os.path.exists(save_path):
                for _ in range(num_retries):
                    gen_img = pipe(prompt_memorized, num_inference_steps=num_inference_steps, callback=callback_fn, callback_steps=1).images[0]
                    mse_diff = compute_mse(oracle_img, gen_img)
                    if mse_diff < threshold:
                        
                        gen_img.save(save_path)
                        print(f"[Memorized] #{gen_idx}: MSE={mse_diff:.2f} -> {save_path}")
                        memorized_success = True
                        break
                    else:
                        print(f"[Memorized] #{gen_idx} (retry): MSE={mse_diff:.2f} >= threshold {threshold}")
            else:
                memorized_success = True
                print(f"Image at {save_path} exists. Skipping this index")
                # break
                

            # random non-memorized attempt
            # not looping or comparing mse threshold since random prompts are likely not memorized. this is done to speed things up
            img_name = f"non_memorized_{gen_idx}"
            callback_fn = make_save_callback(pipe, img_name, non_memorized_random_mid_gens_dir)

            save_path = os.path.join(non_memorized_random_dir, f"{img_name}.png")

            if not os.path.exists(save_path):
                gen_img = pipe(prompt_non_memorized_random, num_inference_steps=num_inference_steps, callback=callback_fn, callback_steps=1).images[0]
                mse_diff = compute_mse(oracle_img, gen_img)
                gen_img.save(save_path)
                print(f"[Non-Memorized Random] #{gen_idx}: MSE={mse_diff:.2f} -> {save_path}")
            else:
                print(f"Image at {save_path} exists. Skipping this index")
                # break

            # 3) NON-MEMORIZED FROM MEM PROMPT (same text, MSE > threshold)
            if ENABLE_NONMEM_FROM_MEM:
                img_name = f"non_memorized_from_mem_{gen_idx}"
                callback_fn = make_save_callback(pipe, img_name, non_memorized_from_mem_mid_gens_dir)

                save_path = os.path.join(non_memorized_from_mem_dir, f"{img_name}.png")

                if not os.path.exists(save_path):
                    non_memorized_from_mem_success = False
                    for _ in range(num_retries):
                        gen_img = pipe(
                            prompt_memorized,
                            num_inference_steps=num_inference_steps,
                            callback=callback_fn,
                            callback_steps=1,
                        ).images[0]
                        mse_diff = compute_mse(oracle_img, gen_img)
                        if mse_diff > threshold:
                            gen_img.save(save_path)
                            print(f"[Non-Memorized From Mem] #{gen_idx}: MSE={mse_diff:.2f} -> {save_path}")
                            non_memorized_from_mem_success = True
                            break
                        else:
                            print(
                                f"[Non-Memorized From Mem] #{gen_idx} (retry): "
                                f"MSE={mse_diff:.2f} <= threshold {threshold}"
                            )
                else:
                    non_memorized_from_mem_success = True
                    print(f"[Non-Memorized From Mem] Image at {save_path} exists. Skipping...")
                    # break
            else:
                # If branch disabled, treat as success so the example is not skipped
                non_memorized_from_mem_success = True
            
            # If either failed, skip this example and record the prompts
            if not memorized_success or (ENABLE_NONMEM_FROM_MEM and not non_memorized_from_mem_success):
                skipped_examples.add((prompt_memorized, prompt_non_memorized_random))
                print(f"Skipping example due to exceeding retries: {prompt_memorized} | {prompt_non_memorized_random}")
                break

    # After all generations
    if skipped_examples:
        with open("skipped_examples.json", 'w') as f:
            for mem_prompt, nonmem_prompt_rand in skipped_examples:
                record = {"Memorized": mem_prompt, "Non-Memorized Random": nonmem_prompt_rand}
                f.write(json.dumps(record) + "\n")

if __name__ == "__main__":
    main()
