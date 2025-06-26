import os
import numpy as np
import pandas as pd

from foldflow.data import utils as du
from metrics import calc_tm_score


def collect_sample_results(results_dir):
    """
    Crawls the results directory structure and collects per-sample self-consistency results.
    Returns a dict: {sample_path: pd.DataFrame of results}
    """

    sample_results = {}
    for length_dir in os.listdir(results_dir):
        length_path = os.path.join(results_dir, length_dir)
        if not os.path.isdir(length_path):
            continue
        for sample_dir in os.listdir(length_path):
            sample_path = os.path.join(length_path, sample_dir, "self_consistency")
            csv_path = os.path.join(sample_path, "sc_results.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                sample_results[(length_dir, sample_path)] = df
    return sample_results


def compute_novelty(sample_results, rmsd_threshold=2.0, novelty_threshold=0.3):
    """
    Computes novelty statistics per protein length.

    Args:
        sample_results: dict of {(length_dir, sample_dir): DataFrame}
        rmsd_threshold: threshold for novelty (default 2.0 Å)
        novelty_threshold: threshold for novelty TM-score (default 0.3)

    Returns:
        novelty_fraction: float, fraction_novel
        novelty_std: float, std of novelty per sample
    """
    novelty_values = []

    for _, df in sample_results.items():
        sample_is_novel = ((df["rmsd"] < rmsd_threshold) & (df["tm_score"] < novelty_threshold)).any()
        novelty_values.append(1.0 if sample_is_novel else 0.0)

    if len(novelty_values) > 0:
        novelty_fraction = np.mean(novelty_values)
        novelty_std = np.std(novelty_values)
    else:
        novelty_fraction = 0.0
        novelty_std = None

    return novelty_fraction, novelty_std


def compute_novelty_per_length(sample_results, rmsd_threshold=2.0, novelty_threshold=0.3):
    """
    Computes novelty statistics per protein length.

    Args:
        sample_results: dict of {(length_dir, sample_dir): DataFrame}
        rmsd_threshold: threshold for designability (default 2.0 Å)
        novelty_threshold: threshold for novelty (default 0.3)

    Returns:
        novelty_fraction: dict {length_dir: fraction_novel}
        novelty_stds: dict {length_dir: std_novel}
    """
    novelty_fraction = {}
    novelty_stds = {}

    # Group by length_dir
    length_groups = {}
    for (length_dir, _), df in sample_results.items():
        if length_dir not in length_groups:
            length_groups[length_dir] = []
        length_groups[length_dir].append(df)

    # Compute metrics per length
    for length_dir, dfs in length_groups.items():
        novelty_values = []

        for df in dfs:
            sample_is_novel = ((df["rmsd"] < rmsd_threshold) & (df["tm_score"] < novelty_threshold)).any()
            novelty_values.append(1.0 if sample_is_novel else 0.0)

        if len(novelty_values) > 0:
            novelty_fraction[length_dir] = np.mean(novelty_values)
            novelty_stds[length_dir] = np.std(novelty_values)
        else:
            novelty_fraction[length_dir] = 0.0
            novelty_stds[length_dir] = None

    return novelty_fraction, novelty_stds


def compute_diversity(sample_results, rmsd_threshold=2.0):
    """
    Computes diversity as average pairwise TM-score of designable generated samples.

    Args:
        sample_results: dict of {(length_dir, sample_dir): DataFrame}
        rmsd_threshold: threshold for designability (default 2.0 Å)
        "rmsd": column name for RMSD in each CSV

    Returns:
        diversity_score: float, average pairwise TM-score of designable samples
    """
    # Collect all designable sample structures
    designable_structures = []

    for (_, sample_path), df in sample_results.items():
        # Check if this sample is designable (has at least one structure with RMSD < threshold)
        if (df["rmsd"] < rmsd_threshold).any():
            # Get all designable structures from this sample
            designable_rows = df[df["rmsd"] < rmsd_threshold]

            for _, row in designable_rows.iterrows():
                esmf_sample_path = sample_path
                esmf_sample_path += f"/esmf/sample_{row.iloc[0]}.pdb"
                # Load PDB and extract coordinates and sequence
                esmf_feats = du.parse_pdb_feats("folded_sample", esmf_sample_path)
                designable_structures.append((esmf_feats["bb_positions"], row["sequence"]))

    if len(designable_structures) < 2:
        # Need at least 2 structures to compute pairwise diversity
        print("Not enough designable structures to compute diversity. Need at least 2.")
        return None, None

    # Compute pairwise TM-scores
    tm_scores = []

    for i in range(len(designable_structures)):
        for j in range(i + 1, len(designable_structures)):
            pos_1, seq_1 = designable_structures[i]
            pos_2, seq_2 = designable_structures[j]

            # Calculate TM-score between structures i and j
            _, tm_score = calc_tm_score(pos_1, pos_2, seq_1, seq_2)
            # Use the normalized TM-score (typically tm_score_1 is normalized by chain 1 length)
            tm_scores.append(tm_score)

    # Return average pairwise TM-score
    return np.mean(tm_scores)


def compute_diversity_per_length(sample_results, rmsd_threshold=2.0, rmsd_col="rmsd"):
    """
    Computes diversity as average pairwise TM-score of designable generated samples per protein length.

    Args:
        sample_results: dict of {(length_dir, sample_dir): DataFrame}
        rmsd_threshold: threshold for designability (default 2.0 Å)
        rmsd_col: column name for RMSD in each CSV

    Returns:
        diversity_scores: dict {length_dir: average pairwise TM-score of designable samples}
    """
    diversity_scores = {}

    # Group by length_dir
    length_groups = {}
    for (length_dir, sample_path), df in sample_results.items():
        if length_dir not in length_groups:
            length_groups[length_dir] = []
        length_groups[length_dir].append((sample_path, df))

    # Compute diversity per length
    for length_dir, samples in length_groups.items():
        # Collect all designable sample structures for this length
        designable_structures = []

        for sample_path, df in samples:
            # Check if this sample is designable (has at least one structure with RMSD < threshold)
            if (df[rmsd_col] < rmsd_threshold).any():
                # Get all designable structures from this sample
                designable_rows = df[df[rmsd_col] < rmsd_threshold]

                for _, row in designable_rows.iterrows():
                    esmf_sample_path = sample_path
                    esmf_sample_path += f"/esmf/sample_{row.iloc[0]}.pdb"
                    try:
                        # Load PDB and extract coordinates and sequence
                        esmf_feats = du.parse_pdb_feats("folded_sample", esmf_sample_path)
                        designable_structures.append((esmf_feats["bb_positions"], row["sequence"]))
                    except Exception as e:
                        # Skip if structure loading fails
                        continue

        if len(designable_structures) < 2:
            # Need at least 2 structures to compute pairwise diversity
            diversity_scores[length_dir] = None
            continue

        # Compute pairwise TM-scores for this length
        tm_scores = []

        for i in range(len(designable_structures)):
            for j in range(i + 1, len(designable_structures)):
                try:
                    pos_1, seq_1 = designable_structures[i]
                    pos_2, seq_2 = designable_structures[j]

                    # Calculate TM-score between structures i and j
                    _, tm_score = calc_tm_score(pos_1, pos_2, seq_1, seq_2)
                    tm_scores.append(tm_score)
                except Exception as e:
                    # Skip if TM-score calculation fails
                    continue

        if len(tm_scores) > 0:
            diversity_scores[length_dir] = np.mean(tm_scores)
        else:
            diversity_scores[length_dir] = None

    return diversity_scores


def compute_designability(sample_results, rmsd_threshold=2.0):
    """
    Computes designability statistics.

    Args:
        sample_results: dict of {(length_dir, sample_dir): DataFrame}
        rmsd_threshold: threshold for designability (default 2.0 Å)

    Returns:
        designability_fraction: float, fraction_designable
        designability_std: float, std of designability per sample
    """
    designability_values = []

    for _, df in sample_results.items():
        sample_is_designable = (df["rmsd"] < rmsd_threshold).any()
        designability_values.append(1.0 if sample_is_designable else 0.0)

    if len(designability_values) > 0:
        designability = np.mean(designability_values)
        designability_std = np.std(designability_values)
    else:
        designability = 0.0
        designability_std = None

    return designability, designability_std


def compute_scrmsd(sample_results):
    """
    Computes scRMSD statistics.

    Args:
        sample_results: dict of {(length_dir, sample_dir): DataFrame}

    Returns:
        mean_scrmsd: float, mean_best_rmsd
        std_scrmsd: float, std_best_rmsd
    """
    best_rmsds = []

    for _, df in sample_results.items():
        min_rmsd = df["rmsd"].min()
        best_rmsds.append(min_rmsd)

    if len(best_rmsds) > 0:
        mean_scrmsd = np.mean(best_rmsds)
        std_scrmsd = np.std(best_rmsds)
    else:
        mean_scrmsd = None
        std_scrmsd = None

    return mean_scrmsd, std_scrmsd


def compute_designability_per_length(sample_results, rmsd_threshold=2.0):
    """
    Computes designability fraction statistics per protein length.

    Args:
        sample_results: dict of {(length_dir, sample_dir): DataFrame}
        rmsd_threshold: threshold for designability (default 2.0 Å)

    Returns:
        designability_fraction: dict {length_dir: fraction_designable}
        designability_stds: dict {length_dir: std_designable}
    """
    designability_fraction = {}
    designability_stds = {}

    # Group by length_dir
    length_groups = {}
    for (length_dir, _), df in sample_results.items():
        if length_dir not in length_groups:
            length_groups[length_dir] = []
        length_groups[length_dir].append(df)

    # Compute metrics per length
    for length_dir, dfs in length_groups.items():
        designability_values = []

        for df in dfs:
            sample_is_designable = (df["rmsd"] < rmsd_threshold).any()
            designability_values.append(1.0 if sample_is_designable else 0.0)

        if len(designability_values) > 0:
            designability_fraction[length_dir] = np.mean(designability_values)
            designability_stds[length_dir] = np.std(designability_values)
        else:
            designability_fraction[length_dir] = 0.0
            designability_stds[length_dir] = None

    return designability_fraction, designability_stds


def compute_scrmsd_per_length(sample_results):
    """
    Computes scRMSD statistics per protein length.

    Args:
        sample_results: dict of {(length_dir, sample_dir): DataFrame}

    Returns:
        mean_scrmsd: dict {length_dir: mean_best_rmsd}
        std_scrmsd: dict {length_dir: std_best_rmsd}
    """
    mean_scrmsd = {}
    std_scrmsd = {}

    # Group by length_dir
    length_groups = {}
    for (length_dir, _), df in sample_results.items():
        if length_dir not in length_groups:
            length_groups[length_dir] = []
        length_groups[length_dir].append(df)

    # Compute metrics per length
    for length_dir, dfs in length_groups.items():
        best_rmsds = []

        for df in dfs:
            min_rmsd = df["rmsd"].min()
            best_rmsds.append(min_rmsd)

        if len(best_rmsds) > 0:
            mean_scrmsd[length_dir] = np.mean(best_rmsds)
            std_scrmsd[length_dir] = np.std(best_rmsds)
        else:
            mean_scrmsd[length_dir] = None
            std_scrmsd[length_dir] = None

    return mean_scrmsd, std_scrmsd


if __name__ == "__main__":
    results_dir = "results/ff2_200k"
    sample_results = collect_sample_results(results_dir)

    # Overall metrics
    designability_frac = compute_designability(sample_results)
    scrmsd = compute_scrmsd(sample_results)
    novelty = compute_novelty(sample_results)
    diversity = compute_diversity(sample_results)

    # Handle None values in overall metrics
    if designability_frac[0] is not None and designability_frac[1] is not None:
        print(f"Designability: {designability_frac[0]:.3f} ± {designability_frac[1]:.3f}")
    else:
        print("Designability: None")

    if scrmsd[0] is not None and scrmsd[1] is not None:
        print(f"scRMSD: {scrmsd[0]:.3f} ± {scrmsd[1]:.3f}")
    else:
        print("scRMSD: None")

    if novelty[0] is not None and novelty[1] is not None:
        print(f"Novelty (designable frac < 0.3): {novelty[0]:.3f} ± {novelty[1]:.3f}")
    else:
        print("Novelty (designable frac < 0.3): None")

    if diversity is not None:
        print(f"Diversity (avg pairwise TM-score): {diversity:.3f}")
    else:
        print("Diversity (avg pairwise TM-score): None")

    # Per-length metrics
    designability_frac_per_length = compute_designability_per_length(sample_results)
    scrmsd_per_length = compute_scrmsd_per_length(sample_results)
    novelty_per_length = compute_novelty_per_length(sample_results)
    diversity_per_length = compute_diversity_per_length(sample_results)

    print("\nDesignability per length:")
    for length_dir, frac in designability_frac_per_length[0].items():
        std_val = designability_frac_per_length[1][length_dir]
        if frac is not None and std_val is not None:
            print(f"  {length_dir}: {frac:.3f} ± {std_val:.3f}")
        elif frac is not None:
            print(f"  {length_dir}: {frac:.3f}")
        else:
            print(f"  {length_dir}: None")

    print("\nscRMSD per length:")
    for length_dir, mean_val in scrmsd_per_length[0].items():
        std_val = scrmsd_per_length[1][length_dir]
        if mean_val is not None and std_val is not None:
            print(f"  {length_dir}: {mean_val:.3f} ± {std_val:.3f}")
        else:
            print(f"  {length_dir}: None")

    print("\nNovelty per length:")
    for length_dir, frac in novelty_per_length[0].items():
        std_val = novelty_per_length[1][length_dir]
        if frac is not None and std_val is not None:
            print(f"  {length_dir}: {frac:.3f} ± {std_val:.3f}")
        elif frac is not None:
            print(f"  {length_dir}: {frac:.3f}")
        else:
            print(f"  {length_dir}: None")

    print("\nDiversity per length:")
    for length_dir, mean_val in diversity_per_length.items():
        if mean_val is not None:
            print(f"  {length_dir}: {mean_val:.3f}")
        else:
            print(f"  {length_dir}: None")
