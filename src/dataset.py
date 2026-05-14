# src/dataset.py
import json
import random
import re
from datasets import Dataset

# ---------------- Synthetic MNumGLUESub ----------------
def build_synthetic_numglue(n_train=1700, n_test=530):
    """Synthetic arithmetic problems matching NumGLUE's style (from your notebook)."""
    random.seed(42)
    templates = [
        ('There are {a} students in a class. {b} more join. How many students are there now?',
         lambda a,b: a+b, 1),
        ('A factory makes {a} widgets per day. How many does it make in {b} days?',
         lambda a,b: a*b, 1),
        ('A store has {a} items. They sell {b}. How many remain?',
         lambda a,b: a-b, 1),
        ('What is the sum of {a} and {b}?',
         lambda a,b: a+b, 1),
        ('Multiply {a} by {b}.',
         lambda a,b: a*b, 1),
        ('{a} divided by {b} equals what number?',
         lambda a,b: a//b if b!=0 else 0, 4),
        ('What number added to {a} gives {c}?',
         lambda a,c: c-a, 4),
        ('{a} minus {b} equals?',
         lambda a,b: a-b, 4),
        ('{a} students each buy {b} pencils and {c} pens. How many items total?',
         lambda a,b,c: a*(b+c), 8),
        ('A car travels {a} mph for {b} hours, then {c} mph for {d} hours. Total distance?',
         lambda a,b,c,d: a*b + c*d, 8),
        ('A rectangle has length {a} and width {b}. Perimeter?',
         lambda a,b: 2*(a+b), 8),
        ('If a number {a} is increased by {b} and then doubled, what is the result?',
         lambda a,b: 2*(a+b), 8),
    ]
    def make_item(idx, task_override=None):
        tmpl, fn, task = templates[idx % len(templates)]
        if task_override is not None:
            task = task_override
        placeholder_names = sorted(set(re.findall(r'\{([a-z])\}', tmpl)))
        values_for_placeholders = [random.randint(2, 20) for _ in range(len(placeholder_names))]
        fmt_dict = dict(zip(placeholder_names, values_for_placeholders))
        if 'remain' in tmpl or 'sell' in tmpl or 'minus' in tmpl:
            if 'a' in fmt_dict and 'b' in fmt_dict:
                fmt_dict['b'] = min(fmt_dict['b'], fmt_dict['a'])
        if 'divided' in tmpl or '÷' in tmpl:
            if 'a' in fmt_dict and 'b' in fmt_dict and fmt_dict['b'] != 0:
                fmt_dict['a'] = fmt_dict['a'] * fmt_dict['b']
        q = tmpl.format(**fmt_dict)
        fn_arg_names = fn.__code__.co_varnames[:fn.__code__.co_argcount]
        fn_args_to_pass = [fmt_dict[arg_name] for arg_name in fn_arg_names]
        ans = str(fn(*fn_args_to_pass))
        return {'question': q, 'answer': ans, 'task': task, 'synthetic': True}
    train = [make_item(i) for i in range(n_train)]
    test  = [make_item(i) for i in range(n_test)]
    return train, test

# ---------------- Preference data loading ----------------
def load_preference_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    # Convert to HF Dataset
    return Dataset.from_list(data)

def to_hf_dataset(pref_data):
    """Convert list of preference dicts to HuggingFace Dataset."""
    return Dataset.from_list([
        {"prompt": item["prompt"], "chosen": item["chosen"], "rejected": item["rejected"]}
        for item in pref_data
    ])

def format_sft_instruction(example):
    """Convert instruction-response pair to a single text string for SFT."""
    return {"text": f"### Instruction:\n{example['prompt']}\n### Response:\n{example['chosen']}"}
