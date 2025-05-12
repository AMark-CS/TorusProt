"""
The structure of this file is greatly influenced by SE3 Diffusion by Yim et. al 2023
Link: https://github.com/jasonkyuyim/se3_diffusion
"""

import logging
import math
from typing import Union

import numpy as np
import torch
from einops import rearrange
from torch import vmap
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from scipy.spatial.transform import Rotation
from torch import Tensor

from foldflow.utils.so3_condflowmatcher import SO3ConditionalFlowMatcher
from foldflow.utils.so3_helpers import (
    exp,
    expmap,
    hat,
    log,
)

from foldflow.utils.igso3 import _batch_sample


def _flat_vec(vec, return_batch=False):
    if return_batch:
        batch = vec.shape[0]
    vec = rearrange(vec, "t n c -> (t n) c", c=3)
    if return_batch:
        return batch, vec
    return vec


class SO3FM:
    def __init__(self, so3_conf, stochastic_paths):
        self._log = logging.getLogger(__name__)
        self.so3_group = SpecialOrthogonal(n=3, point_type="matrix")
        self.so3_cfm = SO3ConditionalFlowMatcher(manifold=self.so3_group)
        self.stochastic_paths = stochastic_paths
        self.g = so3_conf.g
        self.min_sigma = so3_conf.min_sigma
        self.inference_scaling = so3_conf.inference_scaling

    def sample(self, n_samples: float = 1):
        return Rotation.random(n_samples).as_matrix()

    def sample_ref(self, n_samples: float = 1):
        return self.sample(n_samples=n_samples)

    def compute_sigma_t(self, t):
        if isinstance(t, float):
            t = torch.tensor(t)
        return torch.sqrt(self.g**2 * t * (1 - t) + self.min_sigma**2)

    def forward_marginal(self, rot_0: np.ndarray, t: float, rot_1: Union[torch.Tensor, None] = None):
        """Samples from the forward diffusion process at time index t.

        Args:
            rot_0: [..., 3] initial rotations.
            t: continuous time in [0, 1].
            rot_1: [..., 3] noise rotations.

        Returns:
            rot_t: [..., 3] noised rotation vectors.
            rot_vectorfield: [..., 3] vectorfield of rot_t as a rotation vector.
        """
        n_samples = len(rot_0)

        # Sample Unif w.r.t Haar Measure on SO(3)
        # This corresponds IGSO(3) with high concentration param
        rot_1 = self.sample_ref(n_samples) if rot_1 is None else rot_1
        t = torch.tensor(t).repeat(rot_0.shape[0])
        rot_0 = torch.from_numpy(rot_0).double()  # data distribution
        rot_1 = torch.from_numpy(rot_1).double()  # uniform prior
        rot_t = self.so3_cfm.sample_xt(rot_0, rot_1, t)
        if self.stochastic_paths:
            epsilon_t = self.compute_sigma_t(t)
            rot_t = _batch_sample(rot_t, epsilon_t, 1)
        return rot_t, rot_0

    def reverse(
        self,
        rot_t: np.ndarray,
        v_t: np.ndarray,
        t: float,
        dt: float,
        flow_mask: np.ndarray = None,
        noise_scale: float = 1.0,
    ):
        """Simulates the reverse ODE/SDE for 1 step using the Euler integration step or the Geodesic random walk.

        Args:
            rot_t: [..., 3] current rotations at time t.
            v_t: [..., 3] rotation vectorfield at time t.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            flow_mask: True indicates which residues to flow.
            noise_scale: scale of the noise to be added.

        Returns:
            [..., 3] rotation vector at next step.
        """
        if not np.isscalar(t):
            raise ValueError(f"{t} must be a scalar.")

        perturb = -v_t * dt  # scale the velocity by the time step for performing the Euler step

        if flow_mask is not None:
            perturb *= flow_mask[..., None]

        # Euler step in the direction of the reversed vector field v_t along the geodesic from rot_t towards
        # the data distribution at rot_0
        rot_t_1 = expmap(R0=torch.tensor(rot_t).double(), tangent=torch.tensor(perturb).double())
        if self.stochastic_paths:
            z = noise_scale * torch.randn(size=v_t.shape[:-1], device=rot_t_1.device, dtype=torch.float64)
            z = _flat_vec(z)
            dB_skew_sym = hat(self.g * np.sqrt(dt) * z)
            dB_skew_sym = dB_skew_sym.reshape(rot_t.shape)
            rot_t_1 = rot_t_1 @ exp(dB_skew_sym)
        rot_t_1 = rot_t_1.reshape(rot_t.shape)
        return rot_t_1.detach().cpu().numpy()

    def vectorfield(self, rot_0, rot_t, t):
        """uses rot_0 and rot_t and t to calculate ut"""
        batch_size = t.shape[0]
        t = torch.clamp(t, min=1e-4, max=1 - 1e-4).repeat_interleave(rot_0.shape[1]).double()
        rot_0 = rearrange(rot_0, "t n c d -> (t n) c d", c=3, d=3).double()
        rot_t = rearrange(rot_t, "t n c d -> (t n) c d", c=3, d=3).double()

        # Move rot_t to rot_0, multiplying it by the inverse of rot_0 (that is equivalent to rot_t - rot_0 on SO(3))
        # which is the relative rotation between the rot_0 and rot_t
        rot_t_minus_0 = rot_0.transpose(-1, -2) @ rot_t

        if self.inference_scaling < 0:
            # If no inference scaling is used compute the vector field following the formula [3] in the FoldFlow1 paper
            # u_t = log_rt(r0) / t
            # logmap is computed with the hat operator that produces a skew-symmetric matrix ("velocity vector")
            # on so(3).
            # The left multiplication with rot_t transports the "velocity" to the tangent space at rot_t.
            u_t = rot_t @ (log(rot_t_minus_0) / torch.clamp(t[:, None, None], min=-self.inference_scaling))
        else:
            # If inference scaling is used, take it into account. The scaling is time dependent and is equal to c*t
            # Therefore, time t cancels out in the denominator -->
            # u_t = log_rt(r0) * c*t / t = log_rt(r0) * c
            # The left multiplication with rot_t transports the "velocity" to the tangent space at rot_t.
            u_t = rot_t @ (log(rot_t_minus_0) * self.inference_scaling)
        rot_t = rearrange(rot_t, "(t n) c d -> t n c d", t=batch_size, c=3, d=3)
        u_t = rearrange(u_t, "(t n) c d -> t n c d", t=batch_size, c=3, d=3)
        return None, u_t

    def vectorfield_scaling(self, t: np.ndarray):
        return 1
