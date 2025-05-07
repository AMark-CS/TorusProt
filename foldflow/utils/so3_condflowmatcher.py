import torch
from einops import rearrange
from torch import vmap
from geomstats.geometry.special_orthogonal import SpecialOrthogonal

from foldflow.utils.so3_helpers import rotmat_to_rotvec


class SO3ConditionalFlowMatcher:
    def __init__(self, manifold):
        self.sigma = None
        self.manifold = manifold
        self.vec_manifold = SpecialOrthogonal(n=3, point_type="vector")

    def vec_log_map(self, x0, x1):
        # get logmap of x_1 from x_0 (direction of the geodesic that shoots from x_0 and lands on x_1)
        # convert to axis angle to compute logmap efficiently
        rot_x0 = rotmat_to_rotvec(x0)  # rotation vectors for the uniform prior
        rot_x1 = rotmat_to_rotvec(x1)  # rotation vectors for the data distribution

        torch.set_default_dtype(torch.float64)
        log_x1 = self.vec_manifold.log_not_from_identity(point=rot_x1, base_point=rot_x0)
        torch.set_default_dtype(torch.float32)
        return log_x1, rot_x0

    def sample_xt(self, x0, x1, t):
        log_x1, rot_x0 = self.vec_log_map(x0.double(), x1.double())
        # Get a point along the geodesic from x0 to x1, using the direction computed by the logmap
        # t is a point in [0, 1] that determines the location along the geodesic
        torch.set_default_dtype(torch.float64)
        xt = self.vec_manifold.exp_not_from_identity(tangent_vec=t.reshape(-1, 1) * log_x1, base_point=rot_x0)
        xt = self.vec_manifold.matrix_from_rotation_vector(xt)
        torch.set_default_dtype(torch.float32)
        return xt

    def compute_conditional_flow_simple(self, t, xt):
        xt = rearrange(xt, "b c d -> b (c d)", c=3, d=3)

        def index_time_der(i):
            return torch.autograd.grad(outputs=xt, inputs=t, grad_outputs=i, create_graph=True, retain_graph=True)[0]

        # A batch of identity matrices used to separately compute the time derivative for for each component of xt
        identity_matrices = torch.eye(9).to(xt.device).repeat(xt.shape[0], 1, 1)
        xt_dot = vmap(index_time_der, in_dims=1)(identity_matrices)
        return rearrange(xt_dot, "(c d) b -> b c d", c=3, d=3)

    def sample_location_and_conditional_flow_simple(self, x0, x1):
        t = torch.rand(x0.shape[0]).type_as(x0).to(x0.device)  # point in time in [0, 1] to sample along the geodesic
        t.requires_grad = True
        xt = self.sample_xt(x0, x1, t)  # point along the geodesic from x0 to x1
        ut = self.compute_conditional_flow_simple(t, xt)  # conditional flow given by xt_dot
        return t, xt, ut
