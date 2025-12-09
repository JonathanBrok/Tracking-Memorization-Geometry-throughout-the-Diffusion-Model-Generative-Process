"""analyze.py

Compute k-delta (p-Laplace flux) for memorized vs. non-memorized mid-generation images.

Inputs:
  - sdv1_500.jsonl in the current working directory
  - Image directories under --base-data-dir:
      <base-data-dir>/memorized_mid_generations/<prompt_mem>/**/*.png
      <base-data-dir>/non_memorized_random_mid_generations/<prompt_non>/**/*.png

Output:
  - A single CSV specified by --out-csv

No figures are produced; only CSV + printed statistics.
"""

import os, glob, math, json
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import pipeline as pipeline_caption
import urllib  # only if needed
import torchvision.transforms as T
import argparse

sys.path.append(os.getcwd())

# from utils.local_pipeline import LocalStableDiffusionPipeline

from core.k_delta import compute_k_delta, _resolve_t_index

PROMPTS_JSONL = "prompts.jsonl"  # "sdv1_500.jsonl"  # "prompts_clean.jsonl"
# Toggle: whether to include the nonmem_from_mem branch in analysis
ENABLE_NONMEM_FROM_MEM_ANALYZE = False


def dprint(*args, **kwargs):
    pass


def _is_crossattn_mismatch(err: Exception) -> bool:
    m = str(err)
    return any(s in m for s in (
        "mat1 and mat2 shapes cannot be multiplied",
        "cross_attention_dim",
    ))


# KEY / METRIC columns (unchanged)
KEY_COLS = ["group", "path", "prompt", "score_mode", "radius_mode", "time_frac", "timestep", "n_samples"]

METRIC_COLS = [
    "flux_p1_new", "grad_mag", "hvp", "ssvm",
    "flux_p1_new_mid", "grad_mag_mid", "hvp_mid", "ssvm_mid",
]


def upsert_rows(rows, force_metrics, csv_path):
    """
    Insert or update rows in the CSV at csv_path based on KEY_COLS.
    """
    import pandas as pd, numpy as np, os

    if not rows:
        return

    force_set = {m.strip().lower() for m in force_metrics}
    force_all = "all" in force_set

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=KEY_COLS + METRIC_COLS)

    # ensure required columns exist
    for c in KEY_COLS + METRIC_COLS:
        if c not in df.columns:
            df[c] = np.nan

    for r in rows:
        r_key = {k: ("" if (k == "timestep" and (r.get(k, "") in (None, ""))) else r.get(k, "")) for k in KEY_COLS}

        mask = pd.Series([True] * len(df))
        for k, v in r_key.items():
            mask &= (df[k].fillna("") == v)
        idx = np.flatnonzero(mask.to_numpy())

        if idx.size == 0:
            df = pd.concat([df, pd.DataFrame([r])], ignore_index=True)
        else:
            i = idx[-1]
            for m in METRIC_COLS:
                if m not in r:
                    continue
                val = r[m]
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    continue
                need = force_all or (m in force_set) or pd.isna(df.at[i, m])
                if need:
                    df.at[i, m] = val

    # ensure directory exists
    out_dir = os.path.dirname(csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tmp = csv_path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, csv_path)

# def load_prompts():
#     """
#     Load (memorized, non_memorized_random, non_memorized_paraphrased)
#     prompt triples from prompts_clean.jsonl
#     """
#     prompts_memorized = []
#     prompts_non_memorized = []
#     prompts_non_memorized_paraphrased = []

#     with open(PROMPTS_JSONL, "r", encoding="utf-8") as f:
#         for line in f:
#             item = json.loads(line)
#             prompts_memorized.append(item["memorized"])
#             prompts_non_memorized.append(item["non_memorized_random"])
#             prompts_non_memorized_paraphrased.append(item["non_memorized_paraphrased"])

#     return list(zip(prompts_memorized, prompts_non_memorized, prompts_non_memorized_paraphrased))


# def load_prompts():
#     """
#     Load (memorized, non_memorized_random) prompt pairs from sdv1_500.jsonl
#     """
#     prompts_memorized = []
#     prompts_non_memorized = []

#     with open("sdv1_500.jsonl", "r", encoding="utf-8") as f:
#         for line in f:
#             item = json.loads(line)
#             prompts_memorized.append(item["memorized"])
#             prompts_non_memorized.append(item["non_memorized_random"])

#     return list(zip(prompts_memorized, prompts_non_memorized))

def load_prompts():
    """
    Load (memorized, non_memorized) prompt pairs from PROMPTS_JSONL
    """
    prompts_memorized = []
    prompts_nonmem = []

    with open(PROMPTS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            prompts_memorized.append(item["memorized"])
            prompts_nonmem.append(item["non_memorized"])

    return list(zip(prompts_memorized, prompts_nonmem))



def traj_id_from_path(p: str) -> str:
    """
    Normalize a path into a trajectory ID by stripping the '_step=NNN' suffix
    and the file extension, keeping the parent directory.
    """
    base = os.path.basename(p)
    if base.endswith(".png"):
        base = base[:-4]
    if "_step=" in base:
        base = base.split("_step=")[0]
    parent = os.path.dirname(p)
    return os.path.join(parent, base)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main(base_data_dir: str, out_csv_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # ensure output directory exists
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)


    # pipe = LocalStableDiffusionPipeline.from_pretrained(
    #     "CompVis/stable-diffusion-v1-4",
    #     torch_dtype=dtype,
    #     safety_checker=None,
    #     requires_safety_checker=False,
    # ).to(device)

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    unet = pipe.unet
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    scheduler = pipe.scheduler

    # ------- experiment knobs (unchanged)
    n_samples = 16
    num_inference_steps = 500
    guidance_scale = 7.5
    seed = 0
    FORCE_METRICS = ["all"]
    time_frac = 0.01
    timestep = 1

    # # # # # #
    PROMPT_PAIRS = load_prompts()

    base_mem_dir = os.path.join(base_data_dir, "memorized_mid_generations")
    base_nonmem_dir = os.path.join(base_data_dir, "non_memorized_prompt_mid_generations")
    base_nonmem_from_mem_dir = os.path.join(base_data_dir, "non_memorized_from_mem_mid_generations")


    FLUSH_EVERY = 2
    rows_buffer = []
    processed_total = 0

    # reuse previously defined config lists
    eval_stages = ["post"]
    score_modes = ["gap"]


    
    # Stats + path aggregation
    prompts_with_mem_images = 0
    prompts_with_nonmem_images = 0
    prompts_with_nonmem_from_mem_images = 0

    all_items = []
    mem_paths_map = {}
    nonmem_paths_map = {}
    nonmem_from_mem_paths_map = {}

    print(f"[INFO] Using base_data_dir={base_data_dir}")
    print(f"[INFO] Memorized mid-gen root:           {base_mem_dir}")
    print(f"[INFO] Non-memorized-prompt mid-gen root: {base_nonmem_dir}")
    print(f"[INFO] Non-memorized-from-mem mid-gen root: {base_nonmem_from_mem_dir}")
    print(f"[INFO] Output CSV:                        {out_csv_path}")

    # ------------------------------------------------------------------
    # Pre-scan all paths and collect dataset-level stats
    # ------------------------------------------------------------------
    pattern = "*_latents.pt"
    from pathlib import Path

    for i, (p_mem, p_non) in enumerate(PROMPT_PAIRS):
        memorized_dir = os.path.join(base_mem_dir, p_mem)
        nonmem_dir = os.path.join(base_nonmem_dir, p_non)
        nonmem_from_mem_dir = os.path.join(base_nonmem_from_mem_dir, p_mem)

        mem_paths = sorted(Path(memorized_dir).rglob(pattern)) if os.path.isdir(memorized_dir) else []
        nonmem_paths = sorted(Path(nonmem_dir).rglob(pattern)) if os.path.isdir(nonmem_dir) else []
        nonmem_from_mem_paths = sorted(Path(nonmem_from_mem_dir).rglob(pattern)) if os.path.isdir(nonmem_from_mem_dir) else []

        if mem_paths:
            prompts_with_mem_images += 1
            mem_paths_map[p_mem] = mem_paths
            for p in mem_paths:
                all_items.append({"group": "mem", "prompt": p_mem, "path": str(p)})

        if nonmem_paths:
            prompts_with_nonmem_images += 1
            nonmem_paths_map[p_non] = nonmem_paths
            for p in nonmem_paths:
                all_items.append({"group": "nonmem", "prompt": p_non, "path": str(p)})
        

        if ENABLE_NONMEM_FROM_MEM_ANALYZE and nonmem_from_mem_paths:
            prompts_with_nonmem_from_mem_images += 1
            nonmem_from_mem_paths_map[p_mem] = nonmem_from_mem_paths
            for p in nonmem_from_mem_paths:
                all_items.append({"group": "nonmem_from_mem", "prompt": p_mem, "path": str(p)})

    total_images = len(all_items)

    # # # # # #

    if total_images == 0:
        print("[WARN] No images found under base_data_dir; exiting before computation.")
        return

    df_paths = pd.DataFrame(all_items)

    # Pre-run stats, purely from file paths
    print("\n================ Pre-run Dataset Statistics ================")
    print(f"Total images to process: {total_images}")
    print(f"Prompt pairs with memorized images:              {prompts_with_mem_images}")
    print(f"Prompt pairs with non-mem PROMPT images:         {prompts_with_nonmem_images}")
    print(f"Prompt pairs with non-mem FROM-MEM images:       {prompts_with_nonmem_from_mem_images}")


    print("\nImages per group:")
    print(df_paths["group"].value_counts())

    print(f"\nUnique prompts (any group): {df_paths['prompt'].nunique()}")
    print("\nUnique prompts per group:")
    print(df_paths.groupby('group')['prompt'].nunique())

    

    # Trajectory statistics: how many timesteps per trajectory (from filename)
    def _extract_timestep_from_path(path: str):
        m = re.search(r"step[=_](\d+)", str(path))
        return int(m.group(1)) if m else None

    df_paths["traj_id"] = df_paths["path"].apply(traj_id_from_path)
    df_paths["timestep_num"] = df_paths["path"].apply(_extract_timestep_from_path)

    df_traj = df_paths.dropna(subset=["timestep_num"]).copy()
    if not df_traj.empty:
        traj_counts = df_traj.groupby("traj_id")["timestep_num"].nunique()
        print(f"\nNumber of trajectories (unique traj_id): {len(traj_counts)}")
        print(f"Mean #timesteps per trajectory:          {traj_counts.mean():.2f}")
        print(f"Std  #timesteps per trajectory:          {traj_counts.std(ddof=0):.2f}")
    else:
        print("\n[INFO] No valid timestep values found in filenames for trajectory statistics.")

    print("============================================================\n")

    # ------------------------------------------------------------------
    # Now run the expensive computation, using the pre-scanned paths
    # ------------------------------------------------------------------
    results = []

    def eval_set(paths, prompt, label, mid_gen_path=None):
        nonlocal processed_total, total_images
        for score_mode in score_modes:
            for eval_stage in eval_stages:
                iter_paths = paths

                for idx, path in enumerate(iter_paths, 1):
                    match = re.search(r"step[=_](\d+)", str(path))
                    if match:
                        timestep = int(match.group(1))
                    else:
                        # raise error(f"Could not resolve timestep from path: {path}")
                        print("[WARNING] could not resolve timestep from path:", path)
                        continue
                    stage_suffix = "" if eval_stage == "post" else "_mid"
                    col_new = "k_delta" + stage_suffix

                    # We no longer rely on existing CSV rows for skipping; always recompute.
                    processed_total += 1
                    if total_images:
                        pct = 100.0 * processed_total / total_images
                        if processed_total <= 10 or processed_total % 50 == 0:
                            print(f"[INFO] Processed images: {processed_total}/{total_images} ({pct:.1f}%)")
                    
                    ###                    
                    # t_idx = _resolve_t_index(scheduler, timestep, time_frac)
                    latents = torch.load(path, map_location=device)
                    k_delta = compute_k_delta(latents=latents, pipe=pipe, t_idx=timestep, prompt=prompt, n_samples=n_samples, device=device)
                    # convert tensor -> plain float
                    if isinstance(k_delta, torch.Tensor):
                        k_delta = k_delta.detach().float().item()
                    ###
                    row = {
                        "group": label,
                        "path": str(path),
                        "prompt": prompt,
                        "score_mode": score_mode,
                        "radius_mode": "",  # not used here
                        "time_frac": time_frac,
                        "timestep": timestep if timestep is not None else "",
                        "n_samples": n_samples,
                        col_new: float(k_delta),
                    }
                    results.append(row)
                    rows_buffer.append(row)

                    if len(rows_buffer) >= FLUSH_EVERY:
                        upsert_rows(rows_buffer, FORCE_METRICS, csv_path=out_csv_path)
                        rows_buffer.clear()

    # ---------------------------
    # Run experiment for each prompt pair using pre-scanned paths
    # ---------------------------
    for i, (p_mem, p_non) in enumerate(PROMPT_PAIRS):
        print(f"[INFO] Handling prompt pair #{i}:")
        print(f"       memorized:            {p_mem}")
        print(f"       non-mem (other prompt): {p_non}")

        mem_paths = mem_paths_map.get(p_mem, [])
        nonmem_paths = nonmem_paths_map.get(p_non, [])
        nonmem_from_mem_paths = nonmem_from_mem_paths_map.get(p_mem, [])

        print(
            f"[INFO] Found {len(mem_paths)} mem mid-step samples, "
            f"{len(nonmem_paths)} non-mem PROMPT mid-step samples, "
            f"{len(nonmem_from_mem_paths)} non-mem FROM-MEM mid-step samples for this pair."
        )

        if mem_paths:
            eval_set(mem_paths, p_mem, label="mem")

        if nonmem_paths:
            eval_set(nonmem_paths, p_non, label="nonmem")

        if ENABLE_NONMEM_FROM_MEM_ANALYZE and nonmem_from_mem_paths:
            eval_set(nonmem_from_mem_paths, p_mem, label="nonmem_from_mem")

    # Flush remaining rows
    upsert_rows(rows_buffer, FORCE_METRICS, csv_path=out_csv_path)
    rows_buffer.clear()
    print(f"[INFO] Upserted results into {out_csv_path}")

    # ------------------------------------------------------------------
    # Post-run summary statistics from the actual CSV
    # ------------------------------------------------------------------
    if not os.path.exists(out_csv_path):
        print("[WARN] Output CSV does not exist; no post-run stats to compute.")
        return

    df_final = pd.read_csv(out_csv_path)
    if df_final.empty:
        print("[WARN] Output CSV is empty; no post-run stats to compute.")
        return

    print("\n================ Post-run Summary Statistics ================")
    print(f"Total rows in CSV: {len(df_final)}")

    if "group" in df_final.columns:
        print("\nRows per group:")
        print(df_final["group"].value_counts())

    if "prompt" in df_final.columns:
        print(f"\nUnique prompts (any group): {df_final['prompt'].nunique()}")
        if "group" in df_final.columns:
            print("\nUnique prompts per group:")
            print(df_final.groupby("group")["prompt"].nunique())

    # Trajectory statistics from CSV
    if "path" in df_final.columns and "timestep" in df_final.columns:
        df_final["traj_id"] = df_final["path"].apply(traj_id_from_path)
        df_final["timestep_num"] = pd.to_numeric(df_final["timestep"], errors="coerce")

        df_traj = df_final.dropna(subset=["timestep_num"]).copy()
        if not df_traj.empty:
            traj_counts = df_traj.groupby("traj_id")["timestep_num"].nunique()
            print(f"\nNumber of trajectories (unique traj_id): {len(traj_counts)}")
            print(f"Mean #timesteps per trajectory:          {traj_counts.mean():.2f}")
            print(f"Std  #timesteps per trajectory:          {traj_counts.std(ddof=0):.2f}")
        else:
            print("\n[INFO] No valid timestep values found for trajectory statistics in CSV.")
    else:
        print("\n[INFO] Missing 'path' or 'timestep' columns in CSV; cannot compute trajectory statistics.")

    print("=============================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute k-delta (p-Laplace flux) over mid-generation images.")

    parser.add_argument(
        "--base-data-dir",
        type=str,
        required=True,
        help=(
            "Base directory containing mid-generation subdirectories:\n"
            "  <base-data-dir>/memorized_mid_generations/<prompt_mem>/...\n"
            "  <base-data-dir>/non_memorized_prompt_mid_generations/<prompt_non>/...\n"
            "  <base-data-dir>/non_memorized_from_mem_mid_generations/<prompt_mem>/...\n"
        ),

    )
    parser.add_argument(
        "--out-csv",
        type=str,
        required=True,
        help="Path to output CSV file (will be created or updated).",
    )

    
    args = parser.parse_args()
    main(
        base_data_dir=args.base_data_dir,
        out_csv_path=args.out_csv,
    )