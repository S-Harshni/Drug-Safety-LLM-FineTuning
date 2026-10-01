"""LoRA fine-tuning of a small open-source chat model, and the two ways it is read out afterwards.

LoRA freezes the model and learns a low-rank update (two small matrices) next to chosen weight matrices,
so well under 1% of the parameters are trained and the result is an adapter of a few megabytes.
"""
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
MAX_TRAIN_TOKENS = 192      # back-propagation keeps about 1.7 MB of activations per token for this model


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def load(name: str = BASE_MODEL):
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.padding_side = "left"            # so the last position of every row is where the answer starts
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(device())
    return tokenizer, model


def add_lora(model, rank: int = 8, alpha: int | None = None, dropout: float = 0.05):
    config = LoraConfig(r=rank, lora_alpha=alpha or 2 * rank, lora_dropout=dropout, target_modules=TARGETS, task_type="CAUSAL_LM")
    return get_peft_model(model, config)


def free_cache() -> None:
    """The MPS allocator keeps a block for every new batch shape; without this, memory grows until the laptop swaps."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def count_parameters(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "trainable_share": round(trainable / total, 6)}


def hidden_states(model, ids, mask):
    """The transformer's output before the vocabulary projection.

    Projecting every position onto a 150,000-token vocabulary costs about 2 GB per batch, so callers pick
    the positions they need first and project only those.
    """
    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    return core, core.model(input_ids=ids, attention_mask=mask).last_hidden_state


def answer_loss(model, ids, mask, labels):
    """Next-token cross-entropy on the answer tokens only (labels are -100 everywhere else)."""
    core, hidden = hidden_states(model, ids, mask)
    keep = labels[:, 1:] != -100                       # position t predicts token t + 1
    logits = core.lm_head(hidden[:, :-1][keep])
    return torch.nn.functional.cross_entropy(logits.float(), labels[:, 1:][keep])


def encode(tokenizer, prompt_messages: list[dict], answer: str | None = None, max_length: int = 256):
    """Token ids for the chat prompt, and (for training) labels that are -100 everywhere except the answer."""
    prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"][-max_length:]
    if answer is None:
        return prompt_ids, None
    answer_ids = tokenizer(answer + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    return prompt_ids + answer_ids, [-100] * len(prompt_ids) + answer_ids


def collate(tokenizer, rows: list[tuple[list[int], list[int] | None]]):
    width = max(len(ids) for ids, _ in rows)
    ids = torch.full((len(rows), width), tokenizer.pad_token_id)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    labels = torch.full((len(rows), width), -100)
    for i, (row, target) in enumerate(rows):                 # left padding
        ids[i, width - len(row):] = torch.tensor(row)
        mask[i, width - len(row):] = 1
        if target is not None:
            labels[i, width - len(row):] = torch.tensor(target)
    return ids, mask, labels


def train(model, tokenizer, examples: list[tuple[list[dict], str]], epochs: int = 2, batch_size: int = 16,
          learning_rate: float = 2e-4, seed: int = 0, log=print, micro_batch: int = 4) -> dict:
    """Next-token loss on the answer only; AdamW with linear warm-up and decay.

    Each optimiser step sees `batch_size` examples, processed `micro_batch` at a time with the gradients
    added up, so the activations kept for back-propagation fit in a laptop's memory.
    """
    torch.manual_seed(seed)
    rows = [encode(tokenizer, m, a) for m, a in examples]
    rows = [r for r in rows if len(r[0]) <= MAX_TRAIN_TOKENS]         # a few very long sentences would not fit in memory
    steps = epochs * ((len(rows) + batch_size - 1) // batch_size)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate, weight_decay=0.0)
    warmup = max(1, steps // 20)
    schedule = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: min((s + 1) / warmup, max(0.0, (steps - s) / max(steps - warmup, 1))))
    generator = torch.Generator().manual_seed(seed)
    model.train()
    step, t0, losses = 0, time.time(), []
    for _ in range(epochs):
        order = torch.randperm(len(rows), generator=generator).tolist()
        for start in range(0, len(order), batch_size):
            picked = sorted(order[start:start + batch_size], key=lambda i: len(rows[i][0]))
            answer_tokens = sum(sum(t != -100 for t in rows[i][1]) for i in picked)
            optim.zero_grad(set_to_none=True)
            total = 0.0
            for at in range(0, len(picked), micro_batch):
                part = picked[at:at + micro_batch]
                ids, mask, labels = (t.to(model.device) for t in collate(tokenizer, [rows[i] for i in part]))
                weight = sum(sum(t != -100 for t in rows[i][1]) for i in part) / answer_tokens      # so the sum is the batch mean
                loss = answer_loss(model, ids, mask, labels) * weight
                loss.backward()
                total += loss.item()
                del ids, mask, labels, loss
                free_cache()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optim.step()
            schedule.step()
            losses.append(total)
            step += 1
            if step % 50 == 0:
                log(f"step {step}/{steps} loss {sum(losses[-50:]) / 50:.4f} {time.time() - t0:.0f}s", flush=True)
    model.eval()
    return {"steps": steps, "seconds": round(time.time() - t0), "final_loss": round(sum(losses[-20:]) / len(losses[-20:]), 4),
            "examples_used": len(rows), "examples_too_long": len(examples) - len(rows)}


@torch.no_grad()
def yes_no_scores(model, tokenizer, prompts: list[list[dict]], batch_size: int = 16) -> list[float]:
    """logit('yes') - logit('no') for the first answer token: one forward pass, no text generation."""
    yes, no = (tokenizer(w, add_special_tokens=False)["input_ids"][0] for w in ("yes", "no"))
    model.eval()
    encoded = [encode(tokenizer, m) for m in prompts]
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i][0]))        # similar lengths together: little padding
    scores = [0.0] * len(encoded)
    for start in range(0, len(order), batch_size):
        batch = order[start:start + batch_size]
        ids, mask, _ = (t.to(model.device) for t in collate(tokenizer, [encoded[i] for i in batch]))
        core, hidden = hidden_states(model, ids, mask)
        logits = core.lm_head(hidden[:, -1])
        for i, value in zip(batch, (logits[:, yes] - logits[:, no]).float().cpu().tolist(), strict=True):
            scores[i] = value
        free_cache()
    return scores


@torch.no_grad()
def generate(model, tokenizer, prompts: list[list[dict]], max_new_tokens: int = 64, batch_size: int = 8) -> list[str]:
    """Greedy decoding, batched."""
    model.eval()
    encoded = [encode(tokenizer, m) for m in prompts]
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i][0]))
    out = [""] * len(encoded)
    for start in range(0, len(order), batch_size):
        batch = order[start:start + batch_size]
        ids, mask, _ = (t.to(model.device) for t in collate(tokenizer, [encoded[i] for i in batch]))
        got = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
        for i, text in zip(batch, tokenizer.batch_decode(got[:, ids.size(1):], skip_special_tokens=True), strict=True):
            out[i] = text.strip()
        free_cache()
    return out
