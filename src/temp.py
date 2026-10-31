import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model_loader import load_tokenizer, load_base_model

tok = load_tokenizer()
m = load_base_model().eval()

prompt = "<|im_start|>user\nHow do I reverse a list in Python?<|im_end|>\n<|im_start|>assistant\n"
inputs = tok(prompt, return_tensors="pt").to(m.device)
out = m.generate(**inputs, max_new_tokens=100, do_sample=False, pad_token_id=tok.pad_token_id)
print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))