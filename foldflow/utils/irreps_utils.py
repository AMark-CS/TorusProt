from e3nn import o3
from typing import Dict, Optional


def create_reduced_irreps(
    input_irreps: o3.Irreps,
    target_dim: int,
    degree_weights: Optional[Dict[int, float]] = None,
    min_multiplicity: int = 1,
    preserve_degrees: bool = True,
) -> o3.Irreps:
    """
    Create reduced irreps with specified target dimension while controlling
    the proportion of multiplicities for different degrees.

    Args:
        input_irreps: Input irreducible representations
        target_dim: Target total dimension
        degree_weights: Dictionary mapping degree -> weight for that degree.
                       Higher weights mean more multiplicities for that degree.
                       If None, uses uniform weighting.
        min_multiplicity: Minimum multiplicity for each degree (default: 1)
        preserve_degrees: If True, preserves all degrees from input_irreps

    Returns:
        o3.Irreps with reduced multiplicities and target dimension
    """
    # Extract unique degrees from input irreps
    degrees = sorted(set(mul_ir.ir.l for mul_ir in input_irreps))

    if not preserve_degrees:
        # Optionally filter to only keep lower degrees if target_dim is very small
        max_affordable_degree = max(degrees)
        while sum(2 * l + 1 for l in degrees if l <= max_affordable_degree) * min_multiplicity > target_dim:
            max_affordable_degree -= 1
        degrees = [l for l in degrees if l <= max_affordable_degree]

    # Default uniform weights if not provided
    if degree_weights is None:
        degree_weights = {l: 1.0 for l in degrees}
    else:
        # Ensure all degrees in input_irreps have weights defined
        for l in degrees:
            if l not in degree_weights:
                raise ValueError(f"Degree {l} is missing from degree_weights")

    # Calculate dimension used by minimum multiplicities
    min_dim = sum(min_multiplicity * (2 * l + 1) for l in degrees)

    # Remaining dimension to distribute
    remaining_dim = target_dim - min_dim

    # Calculate weighted distribution of remaining dimension
    total_weight = sum(degree_weights[l] * (2 * l + 1) for l in degrees)

    # Initialize multiplicities with minimum values
    multiplicities = {l: min_multiplicity for l in degrees}

    # Distribute remaining dimension proportionally
    for l in degrees:
        weight_contribution = degree_weights[l] * (2 * l + 1) / total_weight
        additional_mults = int(remaining_dim * weight_contribution / (2 * l + 1))
        multiplicities[l] += additional_mults

    # Handle any remaining dimension due to rounding
    remaining_after_distribution = target_dim - sum(multiplicities[l] * (2 * l + 1) for l in degrees)

    # Add remaining dimensions to the 0th degree
    if remaining_after_distribution:
        multiplicities[0] += remaining_after_distribution

    # Construct the final irreps string
    irreps_terms = []
    for l in sorted(degrees):
        if multiplicities[l] > 0:
            parity = "e" if l % 2 == 0 else "o"
            irreps_terms.append(f"{multiplicities[l]}x{l}{parity}")

    return o3.Irreps("+".join(irreps_terms))
