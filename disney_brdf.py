# disney_brdf.py
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from pathlib import Path
from typing import Dict, Optional, List

from PIL import Image, ImageDraw, ImageFont


def hdr_to_ldr(
    hdr,
    percentile=99.5,
    gamma=1.0,
    method="reinhard",          # "linear" | "reinhard" | "filmic" | "aces"
    exposure=1.0,             # extra multiplier after percentile normalization
    white=11.2,               # used by filmic (and optionally reinhard_white)
    reinhard_white=True,     # if True, use Reinhard w/ white point
    return_int8=False,
    return_max_val=False,
    eps=1e-8,
):
    """
    HDR (linear) -> LDR (display-encoded).
    - percentile normalization gives a stable scene-dependent scale (max_val).
    - method selects tone mapping curve:
        * linear   : clip(hdr/max_val)
        * reinhard : x/(1+x)  (optionally white-point variant)
        * filmic   : Hable/Uncharted2 curve normalized by 'white'
        * aces     : ACES fitted (Narkowicz) curve
    Note: Only 'linear' is meaningfully invertible with a stored max_val.
    ldr_lin, s = ToneMap.hdr_to_ldr(hdr, method="linear",  percentile=99.5, gamma=2.2, return_max_val=True)
    ldr_rei    = ToneMap.hdr_to_ldr(hdr, method="reinhard", percentile=99.5, gamma=2.2)
    ldr_fil    = ToneMap.hdr_to_ldr(hdr, method="filmic",   percentile=99.5, gamma=2.2, white=11.2)
    ldr_aces   = ToneMap.hdr_to_ldr(hdr, method="aces",     percentile=99.5, gamma=2.2)

    """
    hdr = np.asarray(hdr, dtype=np.float32)

    # Robust percentile (ignore NaN/Inf)
    finite = np.isfinite(hdr)
    if np.any(finite):
        max_val = float(np.percentile(hdr[finite], percentile))
    else:
        max_val = 1.0
    max_val = max(max_val, eps)

    # Scene-normalized linear values
    x = (hdr / max_val) * float(exposure)
    x = np.maximum(x, 0.0)  # optional: clamp negatives

    method = method.lower()

    if method == "linear":
        y = np.clip(x, 0.0, 1.0)

    elif method == "reinhard":
        if reinhard_white:
            w2 = float(white) * float(white) + eps
            y = (x * (1.0 + x / w2)) / (1.0 + x + eps)
        else:
            y = x / (1.0 + x + eps)
        y = np.clip(y, 0.0, 1.0)

    elif method == "filmic":
        # Hable / Uncharted 2 filmic curve
        A, B, C, D, E, F = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30

        def uncharted2_curve(t):
            return ((t * (A * t + C * B) + D * E) / (t * (A * t + B) + D * F)) - E / F

        y = uncharted2_curve(x)
        w = uncharted2_curve(np.array(float(white), dtype=np.float32))
        y = y / (w + eps)
        y = np.clip(y, 0.0, 1.0)

    elif method == "aces":
        # ACES fitted (Narkowicz 2015)
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        y = (x * (a * x + b)) / (x * (c * x + d) + e)
        y = np.clip(y, 0.0, 1.0)

    else:
        raise ValueError(f"Unknown method='{method}'. Use: linear|reinhard|filmic|aces")

    # Display encoding (simple gamma; set gamma=1.0 to keep linear)
    if gamma is not None and gamma != 1.0:
        y = y ** (1.0 / float(gamma))

    if return_int8:
        y = (np.clip(y, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    return (y, max_val) if return_max_val else y

def hdr_to_ldr_torch(hdr,
        percentile=99.5,
        gamma=1.0,
        method="reinhard",          # "linear" | "reinhard" | "filmic" | "aces"
        exposure=1.0,
        white=11.2,             # filmic white (and optional Reinhard white-point)
        reinhard_white=True,
        return_int8=False,
        return_max_val=False,
        eps=1e-8):
    """
    One function that works for BOTH NumPy arrays and Torch tensors.

    HDR (linear) -> LDR (tone-mapped + gamma encoded).
    - percentile: scene scale estimator (computed over all elements)
    - method:
        * linear   : clip(hdr/max_val)
        * reinhard : x/(1+x) (or white-point variant)
        * filmic   : Hable/Uncharted2 normalized by `white`
        * aces     : ACES fitted (Narkowicz)
    - gamma: simple gamma encoding; set gamma=1.0 to keep linear output
    - return_max_val: also returns the computed max_val (scalar/tensor)

    NOTE: Only 'linear' is meaningfully invertible using max_val. Others are not.
    """
    import numpy as np
    try:
        import torch
    except ImportError:
        torch = None

    is_torch = (torch is not None) and isinstance(hdr, torch.Tensor)

    # ---- helpers ----
    def clip01(x):
        return x.clamp(0.0, 1.0) if is_torch else np.clip(x, 0.0, 1.0)

    def isfinite(x):
        return torch.isfinite(x) if is_torch else np.isfinite(x)

    def percentile_val(x, q):
        if is_torch:
            f = torch.isfinite(x)
            if not torch.any(f):
                return x.new_tensor(1.0)
            v = x[f].float()
            return torch.quantile(v, float(q) / 100.0)
        else:
            f = np.isfinite(x)
            if not np.any(f):
                return 1.0
            return float(np.percentile(x[f], q))

    def gamma_encode(x):
        if gamma is None or gamma == 1.0:
            return x
        return x ** (1.0 / float(gamma))

    # ---- cast ----
    x = hdr.float() if is_torch else np.asarray(hdr, dtype=np.float32)

    # ---- compute scale ----
    max_val = percentile_val(x, percentile)
    if is_torch:
        max_val = torch.clamp(max_val, min=float(eps))
    else:
        max_val = max(float(max_val), float(eps))

    # ---- normalize + exposure ----
    x = (x / max_val) * float(exposure)
    x = torch.clamp(x, min=0.0) if is_torch else np.maximum(x, 0.0)

    m = method.lower()

    # ---- tone map ----
    if m == "linear":
        y = clip01(x)

    elif m == "reinhard":
        if reinhard_white:
            if is_torch:
                w = x.new_tensor(float(white))
                w2 = w * w + float(eps)
                y = (x * (1.0 + x / w2)) / (1.0 + x + float(eps))
            else:
                w2 = float(white) * float(white) + float(eps)
                y = (x * (1.0 + x / w2)) / (1.0 + x + float(eps))
        else:
            y = x / (1.0 + x + float(eps))
        y = clip01(y)

    elif m == "filmic":
        # Hable / Uncharted 2
        A, B, C, D, E, F = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30

        def uncharted2_curve(t):
            return ((t * (A * t + C * B) + D * E) / (t * (A * t + B) + D * F)) - E / F

        y = uncharted2_curve(x)
        if is_torch:
            w = uncharted2_curve(x.new_tensor(float(white)))
            y = y / (w + float(eps))
        else:
            w = uncharted2_curve(np.array(float(white), dtype=np.float32))
            y = y / (w + float(eps))
        y = clip01(y)

    elif m == "aces":
        # ACES fitted (Narkowicz)
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        y = (x * (a * x + b)) / (x * (c * x + d) + e)
        y = clip01(y)

    else:
        raise ValueError(f"Unknown method='{method}'. Use: linear|reinhard|filmic|aces")

    # ---- gamma encode + optional uint8 ----
    y = gamma_encode(y)

    if return_int8:
        if is_torch:
            y = (clip01(y) * 255.0 + 0.5).to(torch.uint8)
        else:
            y = (np.clip(y, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    return (y, max_val) if return_max_val else y


def _safe_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True).clamp_min(eps))


def _luminance(rgb: torch.Tensor) -> torch.Tensor:
    # rgb: (..., 3)
    return 0.3 * rgb[..., 0] + 0.6 * rgb[..., 1] + 0.1 * rgb[..., 2]


def _schlick_fresnel(u: torch.Tensor) -> torch.Tensor:
    # (1-u)^5
    m = (1.0 - u).clamp(0.0, 1.0)
    return m * m * m * m * m


def _build_tangent_frame(n: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build a stable tangent/bitangent given only normals.
    n: (..., 3)
    returns t, b: (..., 3), (..., 3)
    """
    # pick a helper axis not parallel to n
    up = torch.tensor([0.0, 0.0, 1.0], device=n.device, dtype=n.dtype).expand_as(n)
    alt = torch.tensor([0.0, 1.0, 0.0], device=n.device, dtype=n.dtype).expand_as(n)

    use_alt = (n[..., 2].abs() > 0.999).unsqueeze(-1)  # near +/-Z
    a = torch.where(use_alt, alt, up)
    t = _safe_normalize(torch.cross(a, n, dim=-1))
    b = torch.cross(n, t, dim=-1)
    return t, b


def _to_hwc3(x: torch.Tensor) -> torch.Tensor:
    """
    Accept (H,W,3) or (3,H,W) and return (H,W,3).
    """
    if x.dim() != 3:
        raise ValueError(f"Expected 3D tensor, got {x.shape}")
    if x.shape[-1] == 3:
        return x
    if x.shape[0] == 3:
        return x.permute(1, 2, 0).contiguous()
    raise ValueError(f"Unrecognized V/normal layout: {x.shape}")


def _logit(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.clamp(eps, 1 - eps)
    return torch.log(x) - torch.log(1 - x)


def _to_chw3(x: torch.Tensor) -> torch.Tensor:
    """
    Accept (H,W,3) or (3,H,W) and return (3,H,W).
    """
    if x.dim() != 3:
        raise ValueError(f"Expected 3D tensor, got {x.shape}")
    if x.shape[0] == 3:
        return x
    if x.shape[-1] == 3:
        return x.permute(2, 0, 1).contiguous()
    raise ValueError(f"Unrecognized image layout: {x.shape}")


@dataclass
class DisneyParamConfig:
    per_pixel: bool = True
    init_baseColor: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    init_metallic: float = 0.0
    init_subsurface: float = 0.0
    init_specular: float = 0.5
    init_roughness: float = 0.5
    init_specularTint: float = 0.0
    init_anisotropic: float = 0.0
    init_sheen: float = 0.0
    init_sheenTint: float = 0.5
    init_clearcoat: float = 0.0
    init_clearcoatGloss: float = 0.5


class DisneyBRDFPrinciple(nn.Module):
    """
    Differentiable Disney BRDF (no transmission).
    Learnable parameters:
      normal, baseColor, metallic, subsurface, specular, roughness,
      specularTint, anisotropic, sheen, sheenTint, clearcoat, clearcoatGloss
    Parameter semantics follow Disney BRDF notes / BRDF Explorer definitions. :contentReference[oaicite:1]{index=1}
    """

    def __init__(
        self,
        H: int,
        W: int,
        device: torch.device,
        cfg: DisneyParamConfig = DisneyParamConfig(),
        eps: float = 1e-6,
    ):
        super().__init__()
        self.H, self.W = H, W
        self.eps = eps
        self.per_pixel = cfg.per_pixel

        def make_map(ch: int, init_val: torch.Tensor) -> nn.Parameter:
            if self.per_pixel:
                x = init_val.view(ch, 1, 1).expand(ch, H, W).clone()
            else:
                x = init_val.view(ch, 1, 1).clone()  # broadcast later
            return nn.Parameter(x.to(device))

        # normals: unconstrained 3-vector map, normalized in forward
        n0 = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        self.normal_un = make_map(3, n0)

        # color in [0,1] -> store as logits-like, map via sigmoid
        bc0 = torch.tensor(cfg.init_baseColor, dtype=torch.float32)
        self.baseColor_un = make_map(3, bc0)

        # scalar params in [0,1] -> store unconstrained, map via sigmoid
        def make_scalar(init: float) -> nn.Parameter:
            t = torch.tensor([init], dtype=torch.float32)
            return make_map(1, t)

        self.metallic_un = make_scalar(cfg.init_metallic)
        self.subsurface_un = make_scalar(cfg.init_subsurface)
        self.specular_un = make_scalar(cfg.init_specular)
        self.roughness_un = make_scalar(cfg.init_roughness)
        self.specularTint_un = make_scalar(cfg.init_specularTint)
        self.anisotropic_un = make_scalar(cfg.init_anisotropic)
        self.sheen_un = make_scalar(cfg.init_sheen)
        self.sheenTint_un = make_scalar(cfg.init_sheenTint)
        self.clearcoat_un = make_scalar(cfg.init_clearcoat)
        self.clearcoatGloss_un = make_scalar(cfg.init_clearcoatGloss)

    def _expand(self, x_chw: torch.Tensor) -> torch.Tensor:
        # x_chw: (C, H, W) if per_pixel else (C,1,1) -> expand to (C,H,W)
        if x_chw.shape[-2:] == (self.H, self.W):
            return x_chw
        return x_chw.expand(x_chw.shape[0], self.H, self.W)

    def _param_maps(self) -> dict:
        # Return all parameters as (H,W,*) in their constrained forms
        n = _to_hwc3(self._expand(self.normal_un)).contiguous()
        n = _safe_normalize(n, eps=self.eps)

        baseColor = torch.sigmoid(_to_hwc3(self._expand(self.baseColor_un)))
        metallic = torch.sigmoid(self._expand(self.metallic_un)).squeeze(0)  # (H,W)
        subsurface = torch.sigmoid(self._expand(self.subsurface_un)).squeeze(0)
        specular = torch.sigmoid(self._expand(self.specular_un)).squeeze(0)
        roughness = torch.sigmoid(self._expand(self.roughness_un)).squeeze(0)
        specularTint = torch.sigmoid(self._expand(self.specularTint_un)).squeeze(0)
        anisotropic = torch.sigmoid(self._expand(self.anisotropic_un)).squeeze(0)
        sheen = torch.sigmoid(self._expand(self.sheen_un)).squeeze(0)
        sheenTint = torch.sigmoid(self._expand(self.sheenTint_un)).squeeze(0)
        clearcoat = torch.sigmoid(self._expand(self.clearcoat_un)).squeeze(0)
        clearcoatGloss = torch.sigmoid(self._expand(self.clearcoatGloss_un)).squeeze(0)

        # avoid exact zeros
        roughness = roughness.clamp_min(0.02)

        return dict(
            normal=n,
            baseColor=baseColor,
            metallic=metallic,
            subsurface=subsurface,
            specular=specular,
            roughness=roughness,
            specularTint=specularTint,
            anisotropic=anisotropic,
            sheen=sheen,
            sheenTint=sheenTint,
            clearcoat=clearcoat,
            clearcoatGloss=clearcoatGloss,
        )
        
    def _print_param_stats(self):
        P = self._param_maps()
        for k, v in P.items():
            print(f"{k}: min={v.min().item():.3f}, max={v.max().item():.3f}, mean={v.mean().item():.3f}, requires_grad={v.requires_grad}")
        
        # total number of parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params}")

    def evaluate_brdf(
        self,
        V: torch.Tensor,      # (H,W,3) view direction (surface -> camera)
        L: torch.Tensor,      # (N,3) light directions (surface -> light)
        P: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Returns:
          brdf: (N,H,W,3) RGB BRDF value (not multiplied by n·l).
        """
        P = self._param_maps() if P is None else P
        n = P["normal"]                   # (H,W,3)
        baseColor = P["baseColor"]        # (H,W,3)
        metallic = P["metallic"]          # (H,W)
        subsurface = P["subsurface"]      # (H,W)
        specular = P["specular"]          # (H,W)
        roughness = P["roughness"]        # (H,W)
        specularTint = P["specularTint"]  # (H,W)
        anisotropic = P["anisotropic"]    # (H,W)
        sheen = P["sheen"]                # (H,W)
        sheenTint = P["sheenTint"]        # (H,W)
        clearcoat = P["clearcoat"]        # (H,W)
        clearcoatGloss = P["clearcoatGloss"]  # (H,W)

        V = _to_hwc3(V).to(n.dtype).to(n.device)
        V = _safe_normalize(V, eps=self.eps)

        L = L.to(n.dtype).to(n.device)
        L = _safe_normalize(L, eps=self.eps)  # (N,3)

        # dot products: (N,H,W)
        nDotV = (n * V).sum(dim=-1).clamp_min(0.0)
        # expand L to (N,1,1,3)
        L4 = L.view(-1, 1, 1, 3).expand(-1, self.H, self.W, 3)
        n4 = n.unsqueeze(0)
        V4 = V.unsqueeze(0)

        nDotL = (n4 * L4).sum(dim=-1).clamp_min(0.0)
        H = _safe_normalize(L4 + V4, eps=self.eps)  # (N,H,W,3)
        nDotH = (n4 * H).sum(dim=-1).clamp_min(0.0)
        lDotH = (L4 * H).sum(dim=-1).clamp_min(0.0)

        # Base tint
        lum = _luminance(baseColor).clamp_min(self.eps)  # (H,W)
        Ctint = baseColor / lum.unsqueeze(-1)            # (H,W,3)

        # Specular F0 color (Disney-style)
        # dielectricSpec = 0.08 * specular
        dielectricSpec = 0.08 * specular  # (H,W)
        Cspec0 = dielectricSpec.unsqueeze(-1) * (
            (1.0 - specularTint).unsqueeze(-1) * 1.0 + specularTint.unsqueeze(-1) * Ctint
        )
        # metals use baseColor as F0
        Cspec0 = (1.0 - metallic).unsqueeze(-1) * Cspec0 + metallic.unsqueeze(-1) * baseColor  # (H,W,3)

        # --- Diffuse (Burley) + Subsurface approx ---
        FL = _schlick_fresnel(nDotL)
        FV = _schlick_fresnel(nDotV).unsqueeze(0)  # (1,H,W)
        FD90 = 0.5 + 2.0 * (lDotH ** 2) * roughness.unsqueeze(0)
        Fd = (1.0 + (FD90 - 1.0) * FL) * (1.0 + (FD90 - 1.0) * FV)  # (N,H,W)
        diffuse = (baseColor / math.pi).unsqueeze(0) * Fd.unsqueeze(-1)  # (N,H,W,3)

        # subsurface term (Hanrahan-Krueger-ish approximation used in Disney notes)
        FSS90 = (lDotH ** 2) * roughness.unsqueeze(0)
        Fss = (1.0 + (FSS90 - 1.0) * FL) * (1.0 + (FSS90 - 1.0) * FV)  # (N,H,W)
        ss = 1.25 * (Fss * (1.0 / (nDotL + nDotV.unsqueeze(0) + self.eps) - 0.5) + 0.5)  # (N,H,W)
        subsurface_term = (baseColor / math.pi).unsqueeze(0) * ss.unsqueeze(-1)

        diffuse = (1.0 - subsurface).unsqueeze(0).unsqueeze(-1) * diffuse + subsurface.unsqueeze(0).unsqueeze(-1) * subsurface_term
        diffuse = diffuse * (1.0 - metallic).unsqueeze(0).unsqueeze(-1)

        # --- Sheen ---
        Fsheen = _schlick_fresnel(lDotH)  # (N,H,W)
        sheenColor = (1.0 - sheenTint).unsqueeze(-1) * 1.0 + sheenTint.unsqueeze(-1) * Ctint  # (H,W,3)
        sheen_term = sheen.unsqueeze(0).unsqueeze(-1) * Fsheen.unsqueeze(-1) * sheenColor.unsqueeze(0)
        sheen_term = sheen_term * (1.0 - metallic).unsqueeze(0).unsqueeze(-1)

        # --- Specular microfacet (GGX / GTR2 anisotropic) ---
        # Build tangent frame for anisotropy
        t, b = _build_tangent_frame(n)  # (H,W,3), (H,W,3)
        t4 = t.unsqueeze(0)
        b4 = b.unsqueeze(0)

        # transform vectors into local frame
        def to_local(w4):
            wx = (w4 * t4).sum(dim=-1)
            wy = (w4 * b4).sum(dim=-1)
            wz = (w4 * n4).sum(dim=-1)
            return wx, wy, wz

        Hx, Hy, Hz = to_local(H)
        Lx, Ly, Lz = to_local(L4)
        Vx, Vy, Vz = to_local(V4)

        a = (roughness ** 2).unsqueeze(0)  # (1,H,W)
        # aspect ratio from anisotropic
        aspect = torch.sqrt((1.0 - 0.9 * anisotropic).clamp_min(0.1)).unsqueeze(0)  # (1,H,W)
        ax = (a / aspect).clamp_min(0.001)
        ay = (a * aspect).clamp_min(0.001)

        # D: anisotropic GTR2
        denom = (Hx * Hx / (ax * ax) + Hy * Hy / (ay * ay) + Hz * Hz)
        D = 1.0 / (math.pi * ax * ay * (denom * denom).clamp_min(self.eps))  # (N,H,W)

        # G: Smith GGX anisotropic (Heitz-style lambda)
        def lambda_ggx_aniso(wx, wy, wz, ax, ay):
            wz2 = (wz * wz).clamp_min(self.eps)
            t2 = (wx * wx) * (ax * ax) + (wy * wy) * (ay * ay)
            return (-1.0 + torch.sqrt(1.0 + t2 / wz2)) * 0.5

        lamL = lambda_ggx_aniso(Lx, Ly, Lz.clamp_min(self.eps), ax, ay)
        lamV = lambda_ggx_aniso(Vx, Vy, Vz.clamp_min(self.eps), ax, ay)
        G = 1.0 / (1.0 + lamL + lamV).clamp_min(self.eps)

        # Fresnel
        Fspec = Cspec0.unsqueeze(0) + (1.0 - Cspec0.unsqueeze(0)) * _schlick_fresnel(lDotH).unsqueeze(-1)

        spec = (D * G).unsqueeze(-1) * Fspec / (4.0 * nDotL * nDotV.unsqueeze(0) + self.eps).unsqueeze(-1)

        # --- Clearcoat lobe (GTR1) ---
        # Clearcoat is special-purpose specular lobe with fixed F0 ~ 0.04
        a_cc = (0.1 * (1.0 - clearcoatGloss) + 0.001 * clearcoatGloss).unsqueeze(0)  # (1,H,W)
        # GTR1 (from Disney notes): D = (a^2 - 1)/(pi * log(a^2) * (1 + (a^2 - 1) * (n·h)^2))
        a2 = (a_cc * a_cc).clamp_min(1e-6)
        denom_cc = (1.0 + (a2 - 1.0) * (nDotH ** 2)).clamp_min(self.eps)
        Dcc = (a2 - 1.0) / (math.pi * torch.log(a2) * denom_cc)

        # clearcoat G: use isotropic GGX with fixed alpha=0.25 (common Disney impl)
        alpha_g = 0.25
        def smith_g1(nDotW, alpha):
            a2 = alpha * alpha
            return 1.0 / (nDotW + torch.sqrt(a2 + (1.0 - a2) * nDotW * nDotW)).clamp_min(self.eps)

        Gcc = smith_g1(nDotL, alpha_g) * smith_g1(nDotV.unsqueeze(0), alpha_g)
        Fcc = 0.04 + (1.0 - 0.04) * _schlick_fresnel(lDotH)  # scalar
        clearcoat_term = (0.25 * clearcoat).unsqueeze(0) * (Dcc * Gcc * Fcc) / (4.0 * nDotL * nDotV.unsqueeze(0) + self.eps)
        clearcoat_term = clearcoat_term.unsqueeze(-1)  # (N,H,W,1) -> broadcast RGB

        brdf = diffuse + sheen_term + spec + clearcoat_term.expand(-1, -1, -1, 3)
        return brdf  # (N,H,W,3)

    def render(
        self,
        V: torch.Tensor,            # (H,W,3) or (3,H,W)
        L_dir: torch.Tensor,        # (N,3)
        L_rgb: torch.Tensor,        # (N,3)
        irradiance_scale: float = 1.0,
        variant_cls=None,
    ) -> torch.Tensor:
        """
        Simple direct lighting: sum_j ( brdf(v,l_j) * (n·l_j) * L_rgb_j )
        Returns: (3,H,W)
        """
        V = _to_hwc3(V)
        L_rgb = L_rgb.to(V.device).to(V.dtype)  # (N,3)

        P_full = self._param_maps()
        P = P_full if variant_cls is None else variant_cls.constrain(P_full)
        brdf = self.evaluate_brdf(V, L_dir, P=P)  # (N,H,W,3)

        # n·l
        n = P["normal"]  # (H,W,3)
        L4 = _safe_normalize(L_dir.to(n.device).to(n.dtype)).view(-1, 1, 1, 3)
        nDotL = (n.unsqueeze(0) * L4).sum(dim=-1).clamp_min(0.0)  # (N,H,W)

        # accumulate
        Li = L_rgb.view(-1, 1, 1, 3)  # (N,1,1,3)
        weight = (4.0 * math.pi) / max(L_dir.shape[0], 1)   # scalar
        out = (brdf * nDotL.unsqueeze(-1) * Li).sum(dim=0) * weight  # (H,W,3)
        out = out * float(irradiance_scale)
        return out.permute(2, 0, 1).contiguous()  # (3,H,W)

    def forward(
        self,
        V: torch.Tensor,
        L_dir: torch.Tensor,
        L_rgb: torch.Tensor,
        irradiance_scale: float = 1.0,
        variant_cls=None,
    ) -> torch.Tensor:
        """
        Supports:
          - L_rgb: (N,3) -> returns (3,H,W)
          - L_rgb: (B,N,3) -> returns (B,3,H,W)
        """
        if L_rgb.dim() == 2:
            return self.render(V, L_dir, L_rgb, irradiance_scale=irradiance_scale, variant_cls=variant_cls)
        elif L_rgb.dim() == 3:
            outs = []
            assert L_rgb.shape[0] == L_dir.shape[0], f"Expected L_rgb shape (B,N,3) to match L_dir shape (B,N,3), got {L_rgb.shape} and {L_dir.shape}"
            for b in range(L_rgb.shape[0]):
                outs.append(self.render(V, L_dir[b], L_rgb[b], irradiance_scale=irradiance_scale, variant_cls=variant_cls))
            return torch.stack(outs, dim=0)  # (B,3,H,W)
        else:
            raise ValueError(f"Unexpected L_rgb shape: {L_rgb.shape}")
        
    @torch.no_grad()
    def to_display_maps(self, gamma: Optional[float] = 2.2) -> Dict[str, torch.Tensor]:
        """
        Returns uint8 HWC images ready to be saved as PNG:
          - normal, baseColor: (H,W,3) uint8
          - scalars: (H,W,3) uint8 grayscale triplets
        """
        P = self._param_maps()

        def to_uint8_chw(x_chw: torch.Tensor, gamma: Optional[float]) -> torch.Tensor:
            x = x_chw.float().clamp(0, 1)
            if gamma is not None:
                x = x.clamp_min(1e-8) ** (1.0 / gamma)
            x = (x * 255.0 + 0.5).clamp(0, 255).to(torch.uint8)
            return x.permute(1, 2, 0).contiguous().cpu()  # HWC uint8

        out: Dict[str, torch.Tensor] = {}

        # normal: [-1,1] -> [0,1]
        normal = P["normal"].permute(2, 0, 1)  # CHW
        normal_vis = normal * 0.5 + 0.5
        out["normal"] = to_uint8_chw(normal_vis, gamma=None)

        baseColor = P["baseColor"].permute(2, 0, 1)
        out["baseColor"] = to_uint8_chw(baseColor, gamma=gamma)

        def scalar_to_rgb(name: str):
            if name not in P:
                return
            s = P[name].unsqueeze(0).repeat(3, 1, 1)  # (3,H,W)
            out[name] = to_uint8_chw(s, gamma=None)

        for k in ["metallic", "roughness", "specular", "specularTint",
                  "subsurface", "anisotropic", "sheen", "sheenTint",
                  "clearcoat", "clearcoatGloss"]:
            scalar_to_rgb(k)

        return out

    @staticmethod
    def _pil_add_text(im: Image.Image, text: str, xy=(10, 10)) -> Image.Image:
        draw = ImageDraw.Draw(im)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        x, y = xy
        # bbox: (left, top, right, bottom)
        bbox = draw.textbbox((x, y), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = 4
        draw.rectangle([x - pad, y - pad, x + tw + pad, y + th + pad], fill=(0, 0, 0))
        draw.text((x, y), text, fill=(255, 255, 255), font=font)
        return im

    @staticmethod
    def _pil_grid(images_hwc_uint8: List[torch.Tensor], rows: int, cols: int) -> Image.Image:
        assert len(images_hwc_uint8) <= rows * cols
        H, W = images_hwc_uint8[0].shape[0], images_hwc_uint8[0].shape[1]
        canvas = Image.new("RGB", (cols * W, rows * H), (0, 0, 0))
        for k, img in enumerate(images_hwc_uint8):
            r, c = k // cols, k % cols
            canvas.paste(Image.fromarray(img.numpy()), (c * W, r * H))
        return canvas

    @staticmethod
    def _save_png(img_hwc_uint8: torch.Tensor, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img_hwc_uint8.numpy()).save(str(path))

    @torch.no_grad()
    def save_visuals(
        self,
        save_dir: str,
        V: torch.Tensor,             # (H,W,3) or (3,H,W)
        L_dir_all: torch.Tensor,        # (N,3)
        L_rgb_all: torch.Tensor,     # (M,N,3)
        tgt_all: torch.Tensor,       # (M,3,H,W)
        epoch: int,
        loss_value: float,
        save_every: int = 5,
        max_vis: int = 8,
        gamma: Optional[float] = 2.2,
        save_triplets: bool = True,
        err_gain: float = 4.0,
    ):
        """
        Saves:
          params/epoch_xxxx/*.png   (normal/baseColor/scalars)
          mosaics/mosaic_epoch_xxxx.png   (condensed grid with text)
          renders/epoch_xxxx/*_tgt|pred|err.png (optional)

        Uses only a subset (max_vis) of lighting conditions to keep it fast.
        """
        if (epoch % save_every) != 0:
            return

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # ---- parameter maps
        disp = self.to_display_maps(gamma=gamma)
        params_dir = save_dir / "params" / f"epoch_{epoch:04d}"
        for k, img in disp.items():
            self._save_png(img, params_dir / f"{k}.png")

        # ---- choose subset indices
        M = tgt_all.shape[0]
        vis = min(max_vis, M)
        if vis == M:
            idx = torch.arange(M, device=tgt_all.device)
        else:
            idx = torch.linspace(0, M - 1, steps=vis, device=tgt_all.device).round().long()

        # ---- render preds batched
        pred = self(V=V, L_dir=L_dir_all.index_select(0, idx), L_rgb=L_rgb_all.index_select(0, idx))  # (vis,3,H,W)
        pred = torch.stack([hdr_to_ldr_torch(p) for p in pred]) # TODO: may need to adjust hdr_to_ldr_torch for batched input
        tgt = tgt_all.index_select(0, idx)                                     # (vis,3,H,W)
        err = (pred - tgt).abs()
        err_vis = (err * float(err_gain)).clamp(0, 1)

        # ---- to uint8 HWC for mosaic
        def to_uint8_hwc(img_chw: torch.Tensor, gamma: Optional[float]) -> torch.Tensor:
            x = img_chw.detach().float().clamp(0, 1)
            if gamma is not None:
                x = x.clamp_min(1e-8) ** (1.0 / gamma)
            x = (x * 255.0).clamp(0, 255).to(torch.uint8)
            return x.permute(1, 2, 0).contiguous().cpu()

        tiles: List[torch.Tensor] = []
        for i in range(vis):
            tiles.append(to_uint8_hwc(tgt[i], gamma=gamma))
            tiles.append(to_uint8_hwc(pred[i].clamp(0, 1), gamma=gamma))
            tiles.append(to_uint8_hwc(err_vis[i], gamma=None))

        mosaic = self._pil_grid(tiles, rows=vis, cols=3)
        header = f"epoch={epoch:04d}  loss={loss_value:.6f}   rows: tgt | pred | abs(err)*{err_gain:g}"
        mosaic = self._pil_add_text(mosaic, header, xy=(10, 10))

        mosaics_dir = save_dir / "mosaics"
        mosaics_dir.mkdir(parents=True, exist_ok=True)
        mosaic.save(str(mosaics_dir / f"mosaic_epoch_{epoch:04d}.png"))

        # ---- optional per-image triplets
        if save_triplets:
            renders_dir = save_dir / "renders" / f"epoch_{epoch:04d}"
            renders_dir.mkdir(parents=True, exist_ok=True)
            for j in range(vis):
                self._save_png(to_uint8_hwc(tgt[j], gamma=gamma), renders_dir / f"{j:02d}_tgt.png")
                self._save_png(to_uint8_hwc(pred[j].clamp(0, 1), gamma=gamma), renders_dir / f"{j:02d}_pred.png")
                self._save_png(to_uint8_hwc(err_vis[j], gamma=None), renders_dir / f"{j:02d}_err.png")

    @torch.no_grad()
    def init_basecolor_from_image(self, base0: torch.Tensor, eps: float = 1e-6):
        """
        base0: (3,H,W) or (H,W,3), values in [0,1]
        Initializes baseColor_un so that sigmoid(baseColor_un) ~= base0.
        """
        if base0.dim() != 3:
            raise ValueError(base0.shape)

        if base0.shape[0] == 3:
            base_chw = base0
        elif base0.shape[-1] == 3:
            base_chw = base0.permute(2, 0, 1).contiguous()
        else:
            raise ValueError(base0.shape)

        base_chw = base_chw.to(self.baseColor_un.device).to(self.baseColor_un.dtype)

        # if model is not per-pixel, collapse to mean color
        if not self.per_pixel:
            base_chw = base_chw.mean(dim=(1, 2), keepdim=True)  # (3,1,1)

        # write into the *unconstrained* parameter
        self.baseColor_un.copy_(_logit(base_chw, eps=eps))

    @staticmethod
    def _img(p: Path, size=None, label=None, label_size=32):
        """Load RGB image or return a labeled black tile."""
        if p is not None and p.exists():
            im = Image.open(str(p)).convert("RGB")
            return im if (size is None or im.size == size) else im.resize(size, Image.BILINEAR)
        # fallback tile
        W, H = size if size is not None else (256, 256)
        im = Image.new("RGB", (W, H), (0, 0, 0))
        if label:
            draw = ImageDraw.Draw(im)
            try:
                font = ImageFont.load_default(label_size)
            except Exception:
                font = None
            draw.text((6, 6), label, fill=(255, 255, 255), font=font)
        return im

    @staticmethod
    def _grid(rows_imgs, out_path: Path, pad=2):
        """rows_imgs: List[List[PIL.Image]]"""
        rows, cols = len(rows_imgs), max(len(r) for r in rows_imgs)
        tw, th = rows_imgs[0][0].size
        canvas = Image.new("RGB", (cols * tw + (cols - 1) * pad, rows * th + (rows - 1) * pad), (0, 0, 0))
        for r in range(rows):
            for c in range(cols):
                canvas.paste(rows_imgs[r][c], (c * (tw + pad), r * (th + pad)))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(str(out_path))
        return str(out_path)

    @staticmethod
    def _epochs(save_dir: Path):
        out = []
        for sub in ["params", "renders"]:
            p = save_dir / sub
            if not p.exists():
                continue
            for d in p.iterdir():
                if d.is_dir() and d.name.startswith("epoch_"):
                    try:
                        out.append(int(d.name.split("_")[-1]))
                    except Exception:
                        pass
        return sorted(set(out))

    @torch.no_grad()
    def _compare(
        self,
        row_items,              # list of (row_label, save_dir, epoch_int)
        out_dir,                # path
        params,                 # list[str]
        render_indices,         # list[int]
        gt_index=0,
        tile_size=None,
    ):
        out_dir = Path(out_dir)
        # infer tile size from first available param
        if tile_size is None:
            for _, d, e in row_items:
                for k in params:
                    p = Path(d) / "params" / f"epoch_{e:04d}" / f"{k}.png"
                    if p.exists():
                        tile_size = Image.open(str(p)).convert("RGB").size
                        break
                if tile_size is not None:
                    break
            tile_size = tile_size or (256, 256)

        # ---- params.png
        header = [self._img(None, tile_size, "row\\param")] + [self._img(None, tile_size, k) for k in params]
        rows = [header]
        for lab, d, e in row_items:
            r = [self._img(None, tile_size, f"{lab}\n{e:04d}")]
            for k in params:
                p = Path(d) / "params" / f"epoch_{e:04d}" / f"{k}.png"
                r.append(self._img(p, tile_size, f"missing\n{k}"))
            rows.append(r)
        params_png = self._grid(rows, out_dir / "params.png")

        # ---- renders.png (GT last col; use GT from first row if exists)
        header = [self._img(None, tile_size, "row\\render")] + \
                 [self._img(None, tile_size, f"pred {j:02d}") for j in render_indices]
        rows = [header]

        # add GT row (use first available GT among settings)
        gt_tile = None
        for _, d, e in row_items:
            gp = Path(d) / "renders" / f"epoch_{e:04d}" / f"{gt_index:02d}_tgt.png"
            if gp.exists():
                gt_tile = self._img(gp, tile_size)
                break
        gt_tile = gt_tile or self._img(None, tile_size, "missing\nGT")

        gt_row = [self._img(None, tile_size, f"GT\n{gt_index:02d}")]
        gt_row += [gt_tile for _ in render_indices]   # repeat so grid stays rectangular
        # rows.append(gt_row)

        # then one row per setting
        for lab, d, e in row_items:
            r = [self._img(None, tile_size, f"{lab}\n{e:04d}")]
            for j in render_indices:
                p = Path(d) / "renders" / f"epoch_{e:04d}" / f"{j:02d}_pred.png"
                r.append(self._img(p, tile_size, f"missing\npred {j:02d}"))
            rows.append(r)

        renders_png = self._grid(rows, out_dir / "renders.png")

        return {"params": params_png, "renders": renders_png}

    # -------- public APIs --------

    @torch.no_grad()
    def compare_epochs(
        self,
        save_dir,
        epochs=None,
        out_dir=None,
        params=None,
        render_indices=None,
        gt_index=0,
        tile_size=None,
    ):
        save_dir = Path(save_dir)
        epochs = list(epochs) if epochs is not None else self._epochs(save_dir)
        assert len(epochs) > 0, f"No epochs found under {save_dir}"
        out_dir = Path(out_dir) if out_dir is not None else (save_dir / "compare_epochs")
        params = list(params) if params is not None else [
            "normal", "baseColor",
            "metallic", "roughness", "specular", "specularTint",
            "subsurface", "anisotropic", "sheen", "sheenTint",
            "clearcoat", "clearcoatGloss",
        ]
        render_indices = list(render_indices) if render_indices is not None else [0, 1, 2, 3]
        row_items = [(f"epoch", save_dir, e) for e in epochs]
        return self._compare(row_items, out_dir, params, render_indices, gt_index=gt_index, tile_size=tile_size)

    @torch.no_grad()
    def compare_settings(
        self,
        save_dirs,
        labels=None,
        epoch="latest",
        out_dir=None,
        params=None,
        render_indices=None,
        gt_index=0,
        tile_size=None,
    ):
        save_dirs = [Path(d) for d in save_dirs]
        labels = list(labels) if labels is not None else [d.name for d in save_dirs]
        assert len(labels) == len(save_dirs)

        # pick epoch per setting
        row_items = []
        for lab, d in zip(labels, save_dirs):
            if isinstance(epoch, int):
                e = epoch
            else:
                es = self._epochs(d)
                e = es[-1] if len(es) else None
            if e is None:
                continue
            row_items.append((lab, d, e))
        assert len(row_items) > 0, "No valid epochs found in provided save_dirs"

        out_dir = Path(out_dir) if out_dir is not None else (save_dirs[0] / "compare_settings")
        params = list(params) if params is not None else [
            "normal", "baseColor",
            "metallic", "roughness", "specular", "specularTint",
            "subsurface", "anisotropic", "sheen", "sheenTint",
            "clearcoat", "clearcoatGloss",
        ]
        render_indices = list(render_indices) if render_indices is not None else [0, 1, 2, 3]
        return self._compare(row_items, out_dir, params, render_indices, gt_index=gt_index, tile_size=tile_size)

class DisneyBRDFConstrained(DisneyBRDFPrinciple):
    """
    Variant base class: subclasses override `constrain(P)` only.
    `_param_maps()` is shared and applies the constraint on top of DisneyBRDF's maps.
    """
    @staticmethod
    def constrain(P: dict) -> dict:
        return P  # identity by default

    def _param_maps(self) -> dict:
        # IMPORTANT: constrain should return a new dict (or we copy here)
        return self.constrain(dict(super()._param_maps()))


class DisneyBRDFDiffuse(DisneyBRDFConstrained):
    # surface diffuse only
    @staticmethod
    def constrain(P: dict) -> dict:
        zero3 = torch.zeros_like(P["baseColor"])
        zero1 = zero3[..., 0]
        
        # P["normal"] = zero3
        # P["baseColor"] = zero3
        P["metallic"] = zero1
        P["subsurface"] = zero1
        P["specular"] = zero1
        # P["roughness"] = zero1
        P["specularTint"] = zero1
        P["anisotropic"] = zero1
        P["sheen"] = zero1
        P["sheenTint"] = zero1
        P["clearcoat"] = zero1
        P["clearcoatGloss"] = zero1
        return P
    
    
class DisneyBRDFDiffuseSubsurface(DisneyBRDFConstrained):
    # surface diffuse + subsurface
    @staticmethod
    def constrain(P: dict) -> dict:
        zero3 = torch.zeros_like(P["baseColor"])
        zero1 = zero3[..., 0]
        # P["normal"] = zero3
        # P["baseColor"] = zero3
        P["metallic"] = zero1
        # P["subsurface"] = zero1
        P["specular"] = zero1
        # P["roughness"] = zero1
        P["specularTint"] = zero1
        P["anisotropic"] = zero1
        P["sheen"] = zero1
        P["sheenTint"] = zero1
        P["clearcoat"] = zero1
        P["clearcoatGloss"] = zero1
        return P

class DisneyBRDFSpecular(DisneyBRDFConstrained):
    # surface specular only
    @staticmethod
    def constrain(P: dict) -> dict:
        one3 = torch.ones_like(P["baseColor"])
        one1 = one3[..., 0]
        zero3 = torch.zeros_like(P["baseColor"])
        zero1 = zero3[..., 0]

        # P["normal"] = zero3
        # P["baseColor"] = zero3
        P["metallic"] = one1
        P["subsurface"] = zero1
        # P["specular"] = zero1
        # P["roughness"] = zero1
        # P["specularTint"] = zero1
        # P["anisotropic"] = zero1
        P["sheen"] = zero1
        P["sheenTint"] = zero1
        P["clearcoat"] = zero1
        P["clearcoatGloss"] = zero1
        return P

class DisneyBRDFSpecularClearcoat(DisneyBRDFConstrained):
    # surface specular + clearcoat
    @staticmethod
    def constrain(P: dict) -> dict:
        one3 = torch.ones_like(P["baseColor"])
        one1 = one3[..., 0]
        zero3 = torch.zeros_like(P["baseColor"])
        zero1 = zero3[..., 0]

        # P["normal"] = zero3
        # P["baseColor"] = zero3
        P["metallic"] = one1
        P["subsurface"] = zero1
        # P["specular"] = zero1
        # P["roughness"] = zero1
        # P["specularTint"] = zero1
        # P["anisotropic"] = zero1
        P["sheen"] = zero1
        P["sheenTint"] = zero1
        # P["clearcoat"] = zero1
        # P["clearcoatGloss"] = zero1
        return P


class DisneyBRDFSimplified(DisneyBRDFConstrained):
    # surface diffuse + specular, no layers
    @staticmethod
    def constrain(P: dict) -> dict:
        zero3 = torch.zeros_like(P["baseColor"])
        zero1 = zero3[..., 0]
        
        # P["normal"] = zero3
        # P["baseColor"] = zero3
        # P["metallic"] = zero1
        P["subsurface"] = zero1
        # P["specular"] = zero1
        # P["roughness"] = zero1
        # P["specularTint"] = zero1
        # P["anisotropic"] = zero1
        P["sheen"] = zero1
        P["sheenTint"] = zero1
        P["clearcoat"] = zero1
        P["clearcoatGloss"] = zero1
        return P
    
class DisneyBRDFSimplifiedMultiLayer(DisneyBRDFConstrained):
    # surface diffuse + subsurface + surface specular + clearcoat
    @staticmethod
    def constrain(P: dict) -> dict:
        zero3 = torch.zeros_like(P["baseColor"])
        zero1 = zero3[..., 0]
        
        # P["normal"] = zero3
        # P["baseColor"] = zero3
        # P["metallic"] = zero1
        # P["subsurface"] = zero1
        # P["specular"] = zero1
        # P["roughness"] = zero1
        # P["specularTint"] = zero1
        # P["anisotropic"] = zero1
        P["sheen"] = zero1
        P["sheenTint"] = zero1
        # P["clearcoat"] = zero1
        # P["clearcoatGloss"] = zero1
        return P