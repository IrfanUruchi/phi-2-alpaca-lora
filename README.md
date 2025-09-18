# Phi-2-Alpaca-LoRA

[![GitHub Repo](https://img.shields.io/badge/GitHub-phi--2--alpaca--lora-181717?style=for-the-badge&logo=github)](https://github.com/IrfanUruchi/phi-2-alpaca-lora)
[![Model Weights](https://img.shields.io/badge/🤗-Model_Weights-FFD21F?style=for-the-badge)](https://huggingface.co/Irfanuruchi/phi-2-alpaca-lora)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

---

## Overview

This repository provides code and documentation for a fine‑tuned version of [microsoft/phi-2](https://huggingface.co/microsoft/phi-2) (2.7B parameters).  
Fine‑tuning was performed with **LoRA (Low-Rank Adaptation)** on:

- [yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) (subset of ~5k samples)  
- Custom instruction datasets (collected separately)  

After training, all LoRA adapters were merged with the base weights. The final model can be loaded directly with Hugging Face Transformers.

---

## Model Configuration

- **Base model:** microsoft/phi-2  
- **Fine-tuning method:** LoRA  
- **Target layers:** q_proj, k_proj, v_proj, dense  
- **LoRA parameters:** r=16, alpha=32, dropout=0.05  
- **Sequence length:** 256

---

## Usage Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "Irfanuruchi/phi-2-alpaca-lora"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

def generate(prompt: str, max_new_tokens: int = 200) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    prompt = "### Instruction: Explain the OSI model in networking.\n### Response:"
    print(generate(prompt))
```


---
