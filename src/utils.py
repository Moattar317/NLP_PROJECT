# src/utils.py
import re
import math
import torch
from transformers import AutoTokenizer as NTok, AutoModelForSeq2SeqLM

# ---------- Answer extraction ----------
def extract_answer(text: str) -> str:
    """Extract numeric answer from model output (from your notebook)."""
    if not text:
        return None
    patterns = [
        r'[Tt]he answer is[:\\s]+([\\-]?[\\d,\\.]+)',
        r'####\\s*([\\-]?[\\d,\\.]+)',
        r'=\\s*([\\-]?[\\d,\\.]+)\\s*$',
        r'answer:\\s*([\\-]?[\\d,\\.]+)',
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1).replace(',', '')
    numbers = re.findall(r'[\\-]?\\d+(?:[\\.,]\\d+)?', text.replace(',', ''))
    return numbers[-1] if numbers else None

# ---------- Alignment score (PPL) using NLLB ----------
def compute_alignment_score(non_en_text, en_text, src_lang, nllb_tokenizer, nllb_model, cfg_lang_codes):
    """
    Compute PPL of English given non‑English source.
    Lower PPL = better alignment.
    cfg_lang_codes should map language keys like 'zh' to NLLB codes like 'zho_Hans'.
    """
    src_code = cfg_lang_codes.get(src_lang, src_lang)
    nllb_tokenizer.src_lang = src_code
    inputs = nllb_tokenizer(non_en_text, return_tensors='pt', truncation=True, max_length=512).to(nllb_model.device)
    targets = nllb_tokenizer(text_target=en_text, return_tensors='pt', truncation=True, max_length=512).to(nllb_model.device)
    outputs = nllb_model(input_ids=inputs['input_ids'], labels=targets['input_ids'])
    loss = outputs.loss.item()
    ppl = math.exp(min(loss, 20))
    return {'ppl': ppl, 'loss': loss}

# ---------- Translation using NLLB ----------
def translate_question(question, target_lang_code, nllb_model, nllb_tokenizer):
    """Translate English question to target language using NLLB."""
    nllb_tokenizer.src_lang = 'eng_Latn'
    enc = nllb_tokenizer(question, return_tensors='pt', truncation=True, max_length=256).to(nllb_model.device)
    forced_id = nllb_tokenizer.convert_tokens_to_ids(target_lang_code)
    with torch.no_grad():
        out = nllb_model.generate(**enc, forced_bos_token_id=forced_id, max_new_tokens=256)
    return nllb_tokenizer.decode(out[0], skip_special_tokens=True)

# ---------- Sampling (for preference estimation) ----------
def sample_outputs(model, tokenizer, question, n=8, temperature=0.7, max_new_tokens=512, prompt_template=None):
    """Generate n reasoning traces for a given question."""
    if prompt_template is None:
        prompt_template = (
            'Below is an instruction that describes a task. '
            'Write a response that appropriately completes the request.\\n\\n'
            '### Instruction:\\n{question}\\n\\n'
            '### Response: Let\'s think step by step.\\n'
        )
    prompt = prompt_template.format(question=question)
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024).to(model.device)
    outputs = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        num_return_sequences=n,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    pl = inputs['input_ids'].shape[1]
    return [tokenizer.decode(o[pl:], skip_special_tokens=True) for o in outputs]
