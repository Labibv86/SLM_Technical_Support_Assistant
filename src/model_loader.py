"""
Load tokenizer and base model in 4-bit, then wrap with LoRA adapters.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from config import (
    MODEL_NAME, LOAD_IN_4BIT, USE_GRADIENT_CHECKPOINTING,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, TARGET_MODULES,
)


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_base_model():
    bnb_config = None
    if LOAD_IN_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": 0},       # force everything onto cuda:0
        trust_remote_code=True,
    )
    model.config.use_cache = False
    return model


def load_lora_model():
    """Full pipeline: base (4-bit) + prepare + LoRA wrap."""
    tok = load_tokenizer()
    model = load_base_model()

    if LOAD_IN_4BIT:
        model = prepare_model_for_kbit_training(model)

    if USE_GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tok