import torch
import torch.nn as nn
import timm

if not hasattr(torch.amp, "custom_fwd"):
    from torch.cuda.amp import custom_fwd, custom_bwd
    torch.amp.custom_fwd = custom_fwd
    torch.amp.custom_bwd = custom_bwd

orig_custom_fwd = torch.amp.custom_fwd
orig_custom_bwd = torch.amp.custom_bwd


def custom_fwd(*args, **kwargs):
    def wrapper(fn):
        return orig_custom_fwd(fn)
    return wrapper


def custom_bwd(*args, **kwargs):
    def wrapper(fn):
        return orig_custom_bwd(fn)
    return wrapper


torch.amp.custom_fwd = custom_fwd
torch.amp.custom_bwd = custom_bwd


# --- Block wrapper: pass through **kwargs (e.g., rope) ---
class NPBeforeBlock(nn.Module):
    def __init__(self, block: nn.Module, np_layer: nn.Module):
        super().__init__()
        self.block = block
        self.np = np_layer

    def forward(self, x, *args, **kwargs):
        x = self.np(x)
        return self.block(x, *args, **kwargs)  # Pass kwargs through as-is


class MixStyleTokens(nn.Module):
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p, self.alpha, self.eps = p, alpha, eps

    def forward(self, x, **kwargs):  # x: [B, N, D]
        if (not self.training) or torch.rand(1).item() > self.p:
            return x
        B, N, D = x.size()
        mu = x.mean(1, keepdim=True)          # [B,1,D]
        var = x.var(1, unbiased=False, keepdim=True)
        sig = (var + self.eps).sqrt()

        x_norm = (x - mu) / sig

        perm = torch.randperm(B, device=x.device)
        mu2, sig2 = mu[perm], sig[perm]

        lam = torch.distributions.Beta(
            self.alpha, self.alpha).sample((B, 1, 1)).to(x.device)
        mu_mix = lam * mu + (1-lam) * mu2
        sig_mix = lam * sig + (1-lam) * sig2

        return x_norm * sig_mix + mu_mix


class MixStyleTokens_dinov3(nn.Module):
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p, self.alpha, self.eps = p, alpha, eps

    def forward(self, x, **kwargs):  # x: [B, N, D]
        if (not self.training) or torch.rand(1).item() > self.p:
            return x
        B, H, W, D = x.size()
        mu = x.mean(dim=(2, 3), keepdim=True)          # [B,1,D]
        var = x.var(dim=(2, 3), unbiased=False, keepdim=True)
        sig = (var + self.eps).sqrt()

        x_norm = (x - mu) / sig

        perm = torch.randperm(B, device=x.device)
        mu2, sig2 = mu[perm], sig[perm]

        lam = torch.distributions.Beta(
            self.alpha, self.alpha).sample((B, 1, 1, 1)).to(x.device)
        mu_mix = lam * mu + (1-lam) * mu2
        sig_mix = lam * sig + (1-lam) * sig2

        return x_norm * sig_mix + mu_mix

# --------- Normalization Perturbation ----------


class NormalizationPerturbationTokens(nn.Module):
    def __init__(self, p=0.5, std=0.75, use_np_plus=True, eps=1e-6):
        super().__init__()
        self.p, self.std, self.use_np_plus, self.eps = p, std, use_np_plus, eps

    def Normalization_Perturbation(self, x):  # [B,N,D]
        if (not self.training) or torch.rand(1).item() > self.p:
            return x
        mu = x.mean(dim=(2, 3), keepdim=True)  # [B,1,D]
        ones_mat = torch.ones_like(mu)

        alpha = torch.normal(ones_mat, self.std * ones_mat)
        beta = torch.normal(ones_mat, self.std * ones_mat)

        return alpha * x - alpha * mu + beta * mu

    def Normalization_Perturbation_Plus(self, x):  # [B,N,D]
        if (not self.training) or torch.rand(1).item() > self.p:
            return x
        mu = x.mean(dim=(2, 3), keepdim=True)  # [B,1,D]
        ones_mat = torch.ones_like(mu)
        zeros_mat = torch.zeros_like(mu)

        mean_diff = torch.std(mu, 0, keepdim=True)
        mean_scale = mean_diff / mean_diff.max() * 1.5

        alpha = torch.normal(ones_mat, self.std * ones_mat)
        beta = 1 + torch.normal(zeros_mat, self.std * ones_mat) * mean_scale

        return alpha * x - alpha * mu + beta * mu

    def forward(self, x, **kwargs):
        if self.use_np_plus:
            return self.Normalization_Perturbation_Plus(x)
        else:
            return self.Normalization_Perturbation(x)


def add_np_to_dinov3(model: nn.Module,
                     np_p: float = 0.5, np_std: float = 0.75, np_plus: bool = True,
                     apply_at=("blocks",), shallow_blocks=(0, 1)) -> nn.Module:
    np_layer = NormalizationPerturbationTokens(
        p=np_p, std=np_std, use_np_plus=np_plus, eps=1e-6)

    for i in shallow_blocks:
        model.downsample_layers[i] = NPBeforeBlock(
            model.downsample_layers[i], np_layer)

    return model


class CustomModel(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1, pretrained: bool = True, model_name: str = 'convnextv2_large', mixstyle: bool = False):
        super().__init__()
        # 1) Build backbone
        #    - Pretrained weights are IN-1K (finetuned from IN-22K → IN-1K)
        if model_name == 'convnextv2_base':
            self.model = timm.create_model(
                'convnextv2_base.fcmae_ft_in22k_in1k_384',
                pretrained=pretrained,
                num_classes=num_classes  # Reinitialize starting from the head layer
            )
        elif model_name == 'resnext':
            self.model = timm.create_model(
                'seresnextaa201d_32x8d.sw_in12k_ft_in1k_384',
                pretrained=pretrained,
                num_classes=num_classes  # Reinitialize starting from the head layer
            )
        elif model_name == 'convnextv2_large':
            self.model = timm.create_model(
                'convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384',
                pretrained=pretrained,
                num_classes=num_classes  # Reinitialize starting from the head layer
            )
        elif model_name == 'eva':
            self.model = timm.create_model(
                'eva02_large_patch14_448.mim_m38m_ft_in22k_in1k	',
                pretrained=pretrained,
                num_classes=num_classes  # Reinitialize starting from the head layer
            )

            if mixstyle:
                self.model.patch_embed = nn.Sequential(
                    self.model.patch_embed,
                    MixStyleTokens(p=0.3, alpha=0.1)
                )

        # 2) Replace first conv stem (patch embed) if in_channels != 3
        if in_channels != 3:
            # convnextv2_base.stem: nn.Sequential(conv, norm)
            orig_conv = self.model.stem[0]
            new_conv = nn.Conv2d(
                in_channels,
                orig_conv.out_channels,
                kernel_size=orig_conv.kernel_size,
                stride=orig_conv.stride,
                padding=orig_conv.padding,
                bias=orig_conv.bias is not None
            )
            # Copy averaged weights (RGB → new in_channels)
            with torch.no_grad():
                # orig_conv.weight: [out, 3, k, k] → average over channel dim
                w = orig_conv.weight
                # New weight shape: [out, in_channels, k, k]
                new_conv.weight[:] = w.mean(
                    dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
            self.model.stem[0] = new_conv

        # 3) Classifier (head) is already set by create_model(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
