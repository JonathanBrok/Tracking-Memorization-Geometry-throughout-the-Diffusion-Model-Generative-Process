import os
import re
import textwrap
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# ----------------------------------------------------------
# Global Matplotlib settings (good for trajectory plots)
# ----------------------------------------------------------
plt.rcParams.update({
    "font.size": 30,          # base font size
    "axes.titlesize": 30,
    "axes.labelsize": 30,
    "xtick.labelsize": 27,
    "ytick.labelsize": 27,
    "legend.fontsize": 27,
})

# ----------------------------------------------------------
# Config
# ----------------------------------------------------------
CSV_IN = "./res/k_delta_data.csv"        # adjust if you use a different name
OUT_DIR = "figs/figs_k_delta"       # base output dir for all figures

# Which groups to include in plots
INCLUDE_GROUPS = ["mem", "nonmem"]  # ["mem", "nonmem", "nonmem_from_mem"]


# ----------------------------------------------------------
# Helpers for loading and preprocessing
# ----------------------------------------------------------

def _extract_timestep_from_path(path: str) -> Optional[int]:
    """
    Extract timestep from path using 'step=NNN' or 'step_NNN'.
    Returns int or None if not found.
    """
    m = re.search(r"step[=_](\d+)", str(path))
    if m:
        return int(m.group(1))
    return None


def load_and_prepare_kdelta(csv_path: str) -> pd.DataFrame:
    """
    Load k-delta CSV and prepare:
      - 'time' column (from 'timestep' if present, else parsed from 'path')
      - 'traj_id' (path without '_step=NNN' suffix)
    Expects at least:
      - 'group' ∈ {mem, nonmem, nonmem_from_mem}
      - 'path'
      - 'k_delta'  (post-stage)
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # If k_delta column is missing, but flux_p1_new exists, fall back to that
    if "k_delta" not in df.columns and "flux_p1_new" in df.columns:
        print("[WARNING] 'k_delta' column not found, using 'flux_p1_new' column instead.")
        df = df.rename(columns={"flux_p1_new": "k_delta"})

    def normalize_k_delta_column(df: pd.DataFrame, col: str = "k_delta") -> pd.DataFrame:
        """
        If df[col] contains PyTorch tensor string representations like:
            'tensor(0.0643, device='cuda:0', dtype=torch.float16, ...)'
        convert the entire column to float.
        Otherwise, leave the column as-is.
        """

        # If already numeric, do nothing
        if pd.api.types.is_numeric_dtype(df[col]):
            return df

        # Check if it looks like tensor(...) strings
        sample = df[col].astype(str).head(20)
        looks_like_tensor = sample.str.contains(r"tensor\(", regex=True, na=False).any()

        if not looks_like_tensor:
            return df

        # Extract the first numeric argument inside tensor(...)
        extracted = df[col].astype(str).str.extract(r"tensor\(([^,]+)")[0]
        df[col] = pd.to_numeric(extracted, errors="coerce")

        return df

    df = normalize_k_delta_column(df, col="k_delta")

    required_cols = ["group", "path", "k_delta"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"CSV must contain column '{c}'")

    # Time column: prefer explicit 'timestep' if present and numeric
    if "timestep" in df.columns:
        df["time"] = pd.to_numeric(df["timestep"], errors="coerce")
    else:
        # derive from path if no numeric timestep column
        df["time"] = df["path"].apply(_extract_timestep_from_path)

    if df["time"].isna().all():
        raise ValueError("Could not infer any 'time' values (no 'timestep' and no 'step=...' in path).")

    # Trajectory ID: based on path without step suffix
    def traj_id_from_path(p: str) -> str:
        base = os.path.basename(p)
        if base.endswith(".png"):
            base = base[:-4]
        if "_step=" in base:
            base = base.split("_step=")[0]
        if "_step_" in base:
            base = base.split("_step_")[0]
        parent = os.path.dirname(p)
        return os.path.join(parent, base)

    df["traj_id"] = df["path"].apply(traj_id_from_path)

    # Drop rows with NaNs in required numeric columns
    df = df.dropna(subset=["time", "k_delta"]).copy()
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).copy()

    return df


def agg_mean_std(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Aggregate mean ± std over 'time' for a given value column.
    """
    g = df.groupby("time")[value_col].agg(["mean", "std"]).reset_index()
    return g


def plot_mean_std(ax, stats_df: pd.DataFrame, label: str,
                  linewidth: float = 2.0, color: Optional[str] = None):
    """
    Plot mean and std band, with std in the same color as the mean curve.
    """
    plot_kwargs = {
        "label": label,
        "linewidth": linewidth,
    }
    if color is not None:
        plot_kwargs["color"] = color

    line_list = ax.plot(
        stats_df["time"],
        stats_df["mean"],
        **plot_kwargs,
    )

    # Determine color used
    line_color = color if color is not None else line_list[0].get_color()

    if not stats_df["std"].isna().all():
        ax.fill_between(
            stats_df["time"],
            stats_df["mean"] - stats_df["std"],
            stats_df["mean"] + stats_df["std"],
            alpha=0.3,
            color=line_color,
        )


def compute_pair_auc(df_t: pd.DataFrame, neg_group: str) -> Optional[float]:
    """
    Compute ROC AUC between:
      - positive class: group == 'mem'
      - negative class: group == neg_group (e.g., 'nonmem' or 'nonmem_from_mem')
    for a single time slice df_t (df[df['time'] == t]).
    Returns float AUC or None if not computable.
    """
    pos_group = "mem"

    df_pos = df_t[df_t["group"] == pos_group]
    df_neg = df_t[df_t["group"] == neg_group]

    if df_pos.empty or df_neg.empty:
        return None

    y_true = np.concatenate(
        [
            np.ones(len(df_pos)),
            np.zeros(len(df_neg)),
        ]
    )
    scores = np.concatenate(
        [
            df_pos["k_delta"].to_numpy(),
            df_neg["k_delta"].to_numpy(),
        ]
    )

    # If all scores identical, AUC is undefined
    if np.all(scores == scores[0]):
        return None

    return roc_auc_score(y_true, scores)


def slugify(text: str, maxlen: int = 80) -> str:
    """
    Turn an arbitrary prompt into a filesystem-friendly slug.
    """
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    text = text.strip("_")
    if len(text) > maxlen:
        text = text[:maxlen]
    return text or "prompt"


# ----------------------------------------------------------
# Trajectory plots: per-group
# ----------------------------------------------------------

def plot_group_trajectories(df: pd.DataFrame, out_dir: str):
    """
    Plot one trajectory figure per group (mean±std of k_delta over time).
    Saves:
      - traj_mem.png
      - traj_nonmem.png
      - traj_nonmem_from_mem.png
    """
    os.makedirs(out_dir, exist_ok=True)
    value_col = "k_delta"

    groups = sorted(g for g in df["group"].dropna().unique() if g in INCLUDE_GROUPS)

    for g in groups:
        df_g = df[df["group"] == g].copy()
        ...

    for g in groups:
        df_g = df[df["group"] == g].copy()
        if df_g.empty:
            continue

        stats = agg_mean_std(df_g, value_col=value_col)
        if stats.empty:
            continue

        fig, ax = plt.subplots(figsize=(16, 10))

        if g == "mem":
            label = "memorized"
            color = "tab:blue"
        elif g == "nonmem":
            label = "non-memorized (other prompt)"
            color = "tab:orange"
        elif g == "nonmem_from_mem":
            label = "non-memorized (same prompt)"
            color = "tab:green"
        else:
            label = g
            color = None

        plot_mean_std(
            ax,
            stats,
            label=label,
            linewidth=2.5,
            color=color,
        )

        ax.axhline(0.0, linestyle="--", linewidth=1.5, color="gray")
        ax.set_xlabel("generation steps")
        ax.set_ylabel(r"$\kappa^\Delta$")
        ax.set_title(f"Mean trajectory: {label}")
        ax.legend()

        fig.tight_layout()
        out_path = os.path.join(out_dir, f"traj_{g}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[INFO] Saved trajectories figure: {out_path}")


# ----------------------------------------------------------
# Trajectory plots: overlaid (all groups)
# ----------------------------------------------------------

def plot_overlaid_trajectories(df: pd.DataFrame, out_dir: str):
    """
    Plot a single trajectory figure with all 3 groups overlaid:
    - memorized
    - non-memorized (other prompt)
    - non-memorized (same prompt)
    Uses mean ± std bands per group.
    """
    os.makedirs(out_dir, exist_ok=True)
    value_col = "k_delta"

    fig, ax = plt.subplots(figsize=(16, 10))

    # Group → (label, color)
    group_conf: Dict[str, Tuple[str, str]] = {
        "mem":            ("memorized",                    "tab:blue"),
        "nonmem":         ("non-memorized (other prompt)", "tab:orange"),
        "nonmem_from_mem":("non-memorized (same prompt)",  "tab:green"),
    }

    for g, (label, color) in group_conf.items():
        if g not in INCLUDE_GROUPS:
            continue

        df_g = df[df["group"] == g].copy()
        if df_g.empty:
            continue

        stats = agg_mean_std(df_g, value_col=value_col)
        if stats.empty:
            continue

        plot_mean_std(
            ax,
            stats,
            label=label,
            linewidth=2.5,
            color=color,
        )

    ax.axhline(0.0, linestyle="--", linewidth=1.5, color="gray")
    ax.set_xlabel("generation steps")
    ax.set_ylabel(r"$\kappa^\Delta$")
    ax.set_title("Overlaid mean trajectories per group")
    ax.legend()

    fig.tight_layout()
    out_path = os.path.join(out_dir, "traj_all_groups.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved overlaid trajectories figure: {out_path}")


# ----------------------------------------------------------
# Histograms per timestep, with two AUCs in title
# ----------------------------------------------------------

def plot_histograms_per_timestep(df: pd.DataFrame, out_dir: str, max_panels: int = 8):
    """
    Plot histograms of k_delta per group for a subset of timesteps.
    For each selected timestep t:
      - plot histograms of each group
      - compute:
          AUC(mem vs nonmem)
          AUC(mem vs nonmem_from_mem)
      - put both AUCs in the panel title.
    """
    os.makedirs(out_dir, exist_ok=True)

    value_col = "k_delta"
    groups = INCLUDE_GROUPS

    times_all = sorted(df["time"].unique())
    if not times_all:
        print("[INFO] No time values found; skipping histogram plot.")
        return

    # Subsample timesteps to at most max_panels panels
    if len(times_all) <= max_panels:
        times = times_all
    else:
        step = max(1, len(times_all) // max_panels)
        times = times_all[::step]
        times = times[:max_panels]

    n_panels = len(times)
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    for i, t in enumerate(times):
        ax = axes[i]
        df_t = df[df["time"] == t]
        total_n = 0

        # Per-group histograms
        for g in groups:
            vals = df_t.loc[df_t["group"] == g, value_col].dropna().to_numpy()
            if vals.size == 0:
                continue
            total_n += vals.size
            ax.hist(vals, bins="auto", alpha=0.5, label=g)

        if total_n == 0:
            ax.axis("off")
            continue

        # Two AUCs per timestep
        auc_nonmem = compute_pair_auc(df_t, neg_group="nonmem") if "nonmem" in INCLUDE_GROUPS else None
        auc_nonmem_from_mem = (
            compute_pair_auc(df_t, neg_group="nonmem_from_mem")
            if "nonmem_from_mem" in INCLUDE_GROUPS
            else None
        )


        title = f"time={t}, n={total_n}"

        if auc_nonmem is not None:
            title += f", AUC(mem vs nonmem)={auc_nonmem:.3f}"
        else:
            title += ", AUC(mem vs nonmem)=NA"

        if auc_nonmem_from_mem is not None:
            title += f", AUC(mem vs nonmem_from_mem)={auc_nonmem_from_mem:.3f}"
        else:
            title += ", AUC(mem vs nonmem_from_mem)=NA"

        # Wrap long titles and use smaller fonts specifically for histograms
        wrapped_title = "\n".join(textwrap.wrap(title, 60))
        ax.set_title(wrapped_title, fontsize=9)
        ax.set_ylabel("count", fontsize=9)
        ax.tick_params(axis='both', labelsize=8)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel(r"$\kappa^\Delta$", fontsize=9)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "hist_per_timestep_k_delta.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved per-timestep histogram figure: {out_path}")


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    df_all = load_and_prepare_kdelta(CSV_IN)

    print(f"[INFO] Loaded {len(df_all)} rows from {CSV_IN}")
    print(f"[INFO] Groups present:\n{df_all['group'].value_counts()}")

    plot_group_trajectories(df_all, OUT_DIR)
    plot_overlaid_trajectories(df_all, OUT_DIR)
    plot_histograms_per_timestep(df_all, OUT_DIR)
