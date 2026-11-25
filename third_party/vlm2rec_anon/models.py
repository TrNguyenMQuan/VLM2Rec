import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, List, Optional, Union, Dict, Any
from peft import LoraConfig, get_peft_model

from transformers import (
    AutoModelForCausalLM,
    Qwen2VLConfig,
    Qwen2_5_VLConfig,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)

AutoModelForCausalLM.register(config_class=Qwen2VLConfig, model_class=Qwen2VLForConditionalGeneration)
AutoModelForCausalLM.register(config_class=Qwen2_5_VLConfig, model_class=Qwen2_5_VLForConditionalGeneration)


_DEFAULT_LORA_PATTERNS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
_DEFAULT_XMODAL_PATTERNS = [
    "multi_modal_projector", "mm_projector", "vision_projector", "visual_projector",
    "image_projector", "connector", "perceiver_resampler", "image_to_text",
]


def _collect_linear_module_names(model: nn.Module) -> List[str]:
    return [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]


def _auto_pick_targets(
    names: List[str],
    include_patterns: Iterable[str],
    exclude_patterns: Iterable[str] = (),
) -> List[str]:
    inc = [re.compile(p) for p in include_patterns]
    exc = [re.compile(p) for p in exclude_patterns]
    picked = []
    for n in names:
        if any(r.search(n) for r in inc) and not any(r.search(n) for r in exc):
            picked.append(n.split(".")[-1])
    return sorted(list(set(picked)))


class VLEmbedder(nn.Module):
    """Qwen2.5-VL backbone with LoRA, used as a sequence/item encoder."""

    def __init__(
        self,
        model_path: str,
        pooling_type: str = "last",
        use_lora: bool = True,
        padding_side: str = "right",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.2,
        target_modules: Union[str, List[str], None] = "auto",
        include_xmodal: bool = False,
        use_gc: bool = True,
        cache_dir: Optional[str] = "./pretrained_vlm",
        from_pretrained_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        if from_pretrained_kwargs is None:
            from_pretrained_kwargs = {}

        base_kwargs = {
            "device_map": "auto",
            "cache_dir": cache_dir,
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
        }
        base_kwargs.update(from_pretrained_kwargs)

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **base_kwargs)
        self.padding_side = padding_side

        if hasattr(self.model, "config"):
            self.model.config.use_cache = False

        self.pooling_type = pooling_type

        if use_lora:
            if target_modules is None:
                tm_list = []
            elif target_modules == "auto":
                linear_names = _collect_linear_module_names(self.model)
                exclude_patterns = list(_DEFAULT_XMODAL_PATTERNS) if not include_xmodal else []
                exclude_patterns += [
                    "vision_tower", "visual", "vision", "image_encoder", "vit", "clip",
                    "perceiver_resampler",
                ]
                patterns = list(_DEFAULT_LORA_PATTERNS)
                tm_list = _auto_pick_targets(
                    names=linear_names,
                    include_patterns=[rf"(?:^|\.)(?:{'|'.join(map(re.escape, patterns))})(?:$|\.)"],
                    exclude_patterns=exclude_patterns,
                )
                if not tm_list:
                    tm_list = ["q_proj", "k_proj", "v_proj", "o_proj"]
            else:
                tm_list = list(target_modules)

            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=tm_list,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_cfg)
            print("LoRA training is enabled.")

        if use_gc:
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
            try:
                self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                self.model.gradient_checkpointing_enable()

    def _pooling(self, last_hidden: torch.Tensor, last_token_idx: torch.Tensor) -> torch.Tensor:
        if self.pooling_type == "last":
            if self.padding_side == "right":
                reps = last_hidden[
                    torch.arange(last_hidden.size(0), device=last_hidden.device),
                    last_token_idx,
                ]
            else:
                reps = last_hidden[:, -1, :]
        else:
            B, L, H = last_hidden.size()
            rng = torch.arange(L, device=last_hidden.device).unsqueeze(0).expand(B, -1)
            if self.padding_side == "right":
                mask = rng <= last_token_idx.unsqueeze(1)
            else:
                valid_len = last_token_idx + 1
                mask = rng >= (L - valid_len.unsqueeze(1))
            masked = last_hidden * mask.unsqueeze(2)
            reps = masked.sum(dim=1) / mask.sum(dim=1).unsqueeze(1).clamp(min=1)

        return F.normalize(reps, dim=-1)

    def _get_last_hidden(self, outputs) -> torch.Tensor:
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            return outputs.hidden_states[-1]
        raise RuntimeError(
            "Could not find last hidden states. Make sure output_hidden_states=True is supported."
        )

    def forward(self, inputs: Dict[str, torch.Tensor]):
        entry_device = next(self.model.parameters()).device
        inputs = {k: v.to(entry_device) for k, v in inputs.items()}

        outputs = self.model(**inputs, output_hidden_states=True)

        last_hidden = self._get_last_hidden(outputs)
        attn = inputs.get("attention_mask", None)
        if attn is None:
            last_idx = torch.full(
                (last_hidden.size(0),),
                fill_value=last_hidden.size(1) - 1,
                dtype=torch.long,
                device=last_hidden.device,
            )
        else:
            last_idx = attn.sum(dim=1) - 1
        return self._pooling(last_hidden, last_idx)
