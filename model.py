import inspect
import math
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@dataclass
class DDPDConfig:
    model_type: str = "ddpd"
    block_size: int = 1024  # 32x32 image tokens
    vocab_size: int = int(2**16 + 1)
    n_layer: int = 6
    n_head: int = 4
    n_embd: int = 512
    qk_layernorm: bool = True
    timesteps: int = 1000
    max_t: float = 0.98
    num_classes: int = 1000  # Number of ImageNet classes


class Rotary(nn.Module):
    def __init__(self, dim, base=100, h=64, w=64, var_like_order=False):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / (dim)))
        self.h = h
        self.w = w

        t_h = torch.arange(h).type_as(self.inv_freq)
        t_w = torch.arange(w).type_as(self.inv_freq)
        freqs_h = torch.outer(t_h, self.inv_freq).unsqueeze(1)
        freqs_w = torch.outer(t_w, self.inv_freq).unsqueeze(0)
        freqs_h = freqs_h.repeat(1, w, 1)
        freqs_w = freqs_w.repeat(h, 1, 1)
        freqs_hw = torch.cat([freqs_h, freqs_w], 2)

        self.register_buffer("freqs_hw_cos", freqs_hw.cos())
        self.register_buffer("freqs_hw_sin", freqs_hw.sin())
        self.cache_cos = None
        self.cache_sin = None

    def forward(
        self, x, height_width=None, extend_with_register_tokens=0, augment=False
    ):
        if self.cache_cos is not None and self.cache_sin is not None:
            return self.cache_cos, self.cache_sin

        if height_width is not None:
            this_h, this_w = height_width
        else:
            this_hw = x.shape[1]
            this_h, this_w = int(this_hw**0.5), int(this_hw**0.5)

        if augment:
            start_h = torch.randint(0, self.h - this_h + 1, (1,)).item()
            start_w = torch.randint(0, self.w - this_w + 1, (1,)).item()
        else:
            start_h = 0
            start_w = 0

        cos = self.freqs_hw_cos[start_h : start_h + this_h, start_w : start_w + this_w]
        sin = self.freqs_hw_sin[start_h : start_h + this_h, start_w : start_w + this_w]

        cos = cos.clone().reshape(this_h * this_w, -1)
        sin = sin.clone().reshape(this_h * this_w, -1)

        if extend_with_register_tokens > 0:
            cos = torch.cat(
                [
                    torch.ones(extend_with_register_tokens, cos.shape[1]).to(
                        cos.device
                    ),
                    cos,
                ],
                0,
            )
            sin = torch.cat(
                [
                    torch.zeros(extend_with_register_tokens, sin.shape[1]).to(
                        sin.device
                    ),
                    sin,
                ],
                0,
            )

        self.cache_cos = cos[None, :, None, :]
        self.cache_sin = sin[None, :, None, :]

        return self.cache_cos, self.cache_sin  # 1, T, 1, D


def apply_rotary_emb(x, cos, sin):
    cos, sin = cos[:, : x.shape[1]], sin[:, : x.shape[1]]
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Combined projections for self-attention and MLP
        self.chunked_fc = nn.Linear(config.n_embd, 8 * config.n_embd, bias=False)
        self.attn_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # Cross attention
        self.cross_k = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.cross_v = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.cross_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # MLP output projection
        self.mlp_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd

        # init proj to zeros
        torch.nn.init.zeros_(self.attn_proj.weight)
        torch.nn.init.zeros_(self.cross_proj.weight)
        torch.nn.init.zeros_(self.mlp_proj.weight)

        # make initialization bit smaller than typical
        torch.nn.init.normal_(
            self.chunked_fc.weight, mean=0.0, std=0.02 / math.sqrt(config.n_embd)
        )

    def forward(self, x, freq=None, context=None):
        B, T, C = x.size()
        H = self.n_head

        # Combined self-attention + MLP input projection
        qkv_mlp = F.rms_norm(x, (x.size(-1),))
        chunks = self.chunked_fc(qkv_mlp).split([C, C, C, 4 * C, C], dim=-1)
        q, k, v, mlp_intermediate, cross_q = chunks

        # Self attention
        q = q.view(B, T, H, self.head_dim)
        k = k.view(B, T, H, self.head_dim)
        v = v.view(B, T, H, self.head_dim)

        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = freq
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

        attn = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False
        )
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.attn_proj(attn)

        # Cross attention
        if context is not None:
            _, S, _ = context.size()

            q = cross_q.view(B, T, H, self.head_dim)
            k = self.cross_k(context).view(B, S, H, self.head_dim)
            v = self.cross_v(context).view(B, S, H, self.head_dim)

            q = F.rms_norm(q, (q.size(-1),))
            k = F.rms_norm(k, (k.size(-1),))

            cross_attn = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False
            )
            cross_attn = cross_attn.transpose(1, 2).contiguous().view(B, T, C)
            x = x + self.cross_proj(cross_attn)

        mlp = self.mlp_proj(F.relu(mlp_intermediate).square())
        x = x + mlp

        return x


class DDPDModel(nn.Module):
    def __init__(self, config, model_type="planner"):
        super().__init__()
        self.config = config
        self.model_type = model_type

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wce=nn.Embedding(
                    config.num_classes, 4 * config.n_embd
                ),  # Class embedding
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            )
        )

        if model_type == "planner":
            self.head = nn.Linear(config.n_embd, 1, bias=True)
        else:  # denoiser
            self.head = nn.Linear(config.n_embd, config.vocab_size - 1, bias=True)

        dim = config.n_embd // (2 * config.n_head)
        print(f"Rotary half of head dim: {dim}")
        self.rotary = Rotary(dim, base=100, h=64, w=64)
        self.time_embed = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.SiLU(),
            nn.Linear(config.n_embd, config.n_embd),
        )

        for pn, p in self.named_parameters():
            if pn.endswith("proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.002 / math.sqrt(2 * config.n_layer)
                )

        # init head zero

        torch.nn.init.zeros_(self.head.weight)

    def forward(self, idx, time, class_labels, targets=None, mask=None):
        b, t = idx.size()

        x = self.transformer.wte(idx)
        class_emb = self.transformer.wce(class_labels).reshape(b, 4, -1)

        if self.model_type == "denoiser":
            time_emb = self._get_time_embedding(time * 1000, self.config.n_embd)
            cond = torch.cat([class_emb, time_emb.unsqueeze(1)], dim=1)
        else:
            # planner does not need time embedding
            cond = class_emb

        freq = self.rotary(None, height_width=(32, 32))

        for block in self.transformer.h:
            x = block(x, freq, context=cond)

        x = F.rms_norm(x, (x.size(-1),))

        logits = self.head(x).float()

        if self.model_type == "planner":
            logits = logits.squeeze(-1)

        if targets is not None:
            if self.model_type == "planner":
                loss = F.binary_cross_entropy_with_logits(logits, targets.float())
            else:  # denoiser
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none"
                )
                loss = (loss * mask.view(-1)).sum() / (mask.sum() + 1e-5)
            return logits, loss
        else:
            return logits

    def _get_time_embedding(self, timesteps, dim):
        half_dim = dim // 2
        emb = math.log(1000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        return self.time_embed(emb)

    @torch.no_grad()
    def sample(
        self,
        denoiser,
        class_labels=None,
        batch_size=1,
        sequence_length=128,
        temperature=1.0,
        top_k=None,
        device="cuda",
        dynamic=False,
        num_samples=10,
        infer_time_from_planner=True,
    ):
        if self.model_type != "planner":
            raise ValueError("Sampling can only be done with planner model")

        x = torch.randint(
            0, self.config.vocab_size - 1, (batch_size, sequence_length), device=device
        )

        if class_labels is None:
            class_labels = torch.randint(
                0, self.config.num_classes, (batch_size,), device=device
            )

        time_steps = torch.linspace(1.0, 0.00, 300, device=device)
        last_t = len(time_steps) - 1

        for idx, t in enumerate(time_steps):
            current_t = torch.full((batch_size,), t, device=device)

            planner_logits = self(x, current_t, class_labels)
            planner_probs = torch.sigmoid(planner_logits / 0.1)
            # infer time
            if infer_time_from_planner:
                t = planner_probs.mean().item()
                current_t = torch.full((batch_size,), t, device=device)

            if idx == last_t:
                # change everywhere with > 0.01 prob
                mask = planner_probs > 0.01

            else:
                change_dim = torch.multinomial(planner_probs, num_samples=num_samples)
                mask = torch.zeros_like(planner_probs.squeeze(-1), dtype=torch.bool)
                mask.scatter_(1, change_dim, True)

            if mask.sum() > 0:
                x[mask] = self.config.vocab_size - 1
                denoiser_logits = denoiser(x, current_t, class_labels)
                masked_logits = denoiser_logits[mask]

                if idx == len(time_steps) - 1:
                    masked_logits = masked_logits / 0.01
                else:
                    masked_logits = masked_logits / 1.0

                probs = F.softmax(masked_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

                x[mask] = next_tokens

            if infer_time_from_planner and t < 0.01:
                break

        return x


def print0(s):
    if dist.get_rank() == 0:
        print(s)


def set_seed(seed: int):
    """Set random seed for reproducibility across torch, numpy, and Python random."""
    import random

    import numpy as np

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def configure_optimizers(
    model, weight_decay, learning_rate, betas, device_type, custom_lrs={}
):
    param_groups = []
    param_dict = {pn: p for pn, p in model.named_parameters()}
    for name, param in model.named_parameters():

        if "weight" in name:
            fan_in = torch.nn.init._calculate_fan_in_and_fan_out(param)[0]
            lr = learning_rate / max(fan_in, 1)  # Avoid division by zero
            param_groups.append(
                {"params": [param], "weight_decay": weight_decay, "lr": lr}
            )
        else:
            if name in custom_lrs:
                lr = custom_lrs[name]
            else:
                lr = learning_rate * 0.1
            param_groups.append(
                {"params": [param], "weight_decay": weight_decay, "lr": lr}
            )

        print0(f"name: {name}, lr: {lr}")

    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(param_groups, betas=betas, **extra_args)

    return optimizer
