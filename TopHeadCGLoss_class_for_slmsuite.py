import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from functions import build_target, build_weighting_mask, phase_gradient


class TopHeadCGLoss(nn.Module):
    """
    Slmsuite-compatible loss module using the same overlap objective as cg_optimize.
    """

    def __init__(self, cfg, use_passed_target=False):
        super().__init__()
        self.cfg = cfg.clone()
        self.cfg.update_derived()
        self.use_passed_target = use_passed_target

        target_amplitude = build_target(self.cfg)
        phase_reference = phase_gradient(self.cfg.NTx, self.cfg.NTy, self.cfg.kx, self.cfg.ky)
        weighting_mask = build_weighting_mask(self.cfg)

        target_amplitude = target_amplitude * weighting_mask

        self.register_buffer(
            "target_amplitude",
            torch.as_tensor(target_amplitude, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "phase_reference",
            torch.as_tensor(phase_reference, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "weighting_mask",
            torch.as_tensor(weighting_mask, dtype=torch.float64),
            persistent=False,
        )

    def _prepare_buffer(self, tensor, farfield):
        return tensor.to(device=farfield.device, dtype=torch.float64)

    def _resize_real(self, tensor, shape):
        if tuple(tensor.shape) == tuple(shape):
            return tensor
        return F.interpolate(tensor[None, None, :, :], size=shape, mode="area")[0, 0]

    def _resize_complex(self, tensor, shape):
        if tuple(tensor.shape) == tuple(shape):
            return tensor
        real = self._resize_real(torch.real(tensor), shape)
        imag = self._resize_real(torch.imag(tensor), shape)
        return torch.complex(real, imag)

    def _target_from_argument(self, target, farfield):
        target_tensor = torch.as_tensor(target, device=farfield.device)
        if torch.is_complex(target_tensor):
            target_tensor = torch.abs(target_tensor)
        return target_tensor.to(dtype=torch.float64)

    def forward(self, farfield, target=None):
        farfield = torch.as_tensor(farfield)
        target_shape = tuple(self.target_amplitude.shape)
        farfield = self._resize_complex(farfield, target_shape)

        amp = torch.abs(farfield).to(dtype=torch.float64)
        phase = torch.angle(farfield).to(dtype=torch.float64)

        if self.use_passed_target and target is not None:
            target_amplitude = self._target_from_argument(target, farfield)
            target_amplitude = self._resize_real(target_amplitude, target_shape)
        else:
            target_amplitude = self._prepare_buffer(self.target_amplitude, farfield)

        phase_reference = self._prepare_buffer(self.phase_reference, farfield)
        weighting_mask = self._prepare_buffer(self.weighting_mask, farfield)

        if amp.shape != target_amplitude.shape:
            raise ValueError(
                f"farfield shape {tuple(amp.shape)} does not match target shape "
                f"{tuple(target_amplitude.shape)}."
            )

        overlap = torch.sum(target_amplitude * amp * weighting_mask * torch.cos(phase - phase_reference))
        norm = torch.sqrt(torch.sum(target_amplitude**2) * torch.sum((amp * weighting_mask) ** 2))
        overlap = overlap / torch.clamp(norm, min=np.finfo(float).eps)

        return (10**self.cfg.C1) * (1 - overlap) ** 2
