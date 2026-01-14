# disney_brdf.py
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from typing import Dict, Optional, List

from PIL import Image, ImageDraw, ImageFont



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


class DisneyBRDF(nn.Module):
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
    ) -> torch.Tensor:
        """
        Returns:
          brdf: (N,H,W,3) RGB BRDF value (not multiplied by n·l).
        """
        P = self._param_maps()
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
    ) -> torch.Tensor:
        """
        Simple direct lighting: sum_j ( brdf(v,l_j) * (n·l_j) * L_rgb_j )
        Returns: (3,H,W)
        """
        V = _to_hwc3(V)
        L_rgb = L_rgb.to(V.device).to(V.dtype)  # (N,3)

        brdf = self.evaluate_brdf(V, L_dir)  # (N,H,W,3)

        # n·l
        n = self._param_maps()["normal"]  # (H,W,3)
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
    ) -> torch.Tensor:
        """
        Supports:
          - L_rgb: (N,3) -> returns (3,H,W)
          - L_rgb: (B,N,3) -> returns (B,3,H,W)
        """
        if L_rgb.dim() == 2:
            return self.render(V, L_dir, L_rgb, irradiance_scale=irradiance_scale)
        elif L_rgb.dim() == 3:
            outs = []
            assert L_rgb.shape[0] == L_dir.shape[0], f"Expected L_rgb shape (B,N,3) to match L_dir shape (B,N,3), got {L_rgb.shape} and {L_dir.shape}"
            for b in range(L_rgb.shape[0]):
                outs.append(self.render(V, L_dir[b], L_rgb[b], irradiance_scale=irradiance_scale))
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



class DisneyBRDFSimplified(DisneyBRDF):
    """
    Simplified variant:
      subsurface = 0, anisotropic = 0, sheen = 0, clearcoat = 0
    So we only fit: normal, baseColor, metallic, specular, roughness, specularTint, sheenTint(irrelevant but kept)
    """
    def _param_maps(self) -> dict:
        P = super()._param_maps()
        zero3 = torch.zeros_like(P["baseColor"])
        zero1 = zero3[..., 0]
        
        # P["normal"] = zero3
        # P["baseColor"] = zero3
        # P["metallic"] = zero1
        P["subsurface"] = zero1
        # P["specular"] = zero1
        # P["roughness"] = zero1
        # P["specularTint"] = zero1
        P["anisotropic"] = zero1
        P["sheen"] = zero1
        P["sheenTint"] = zero1
        P["clearcoat"] = zero1
        P["clearcoatGloss"] = zero1
        return P

class DisneyBRDFDiffuse(DisneyBRDF):
    def _param_maps(self) -> dict:
        P = super()._param_maps()
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

class DisneyBRDFSpecular(DisneyBRDF):
    def _param_maps(self) -> dict:
        P = super()._param_maps()
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
        # P["sheen"] = zero1
        # P["sheenTint"] = zero1
        # P["clearcoat"] = zero1
        # P["clearcoatGloss"] = zero1
        return P
