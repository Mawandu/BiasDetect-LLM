"""B4: fine-tune a transformer classifier on StereoSet-train (run on a GPU, e.g. Colab).

    python -m experiments.train_transformer --model roberta-base --seed 13
    python -m experiments.train_transformer --model distilroberta-base --seed 13

The model is saved to results/finetuned/<model>/ and then evaluated by
    python -m experiments.run --finetuned roberta-base --only B4_ft_roberta-base
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from biasdetect.data.loaders import load_processed

ROOT = Path(__file__).resolve().parents[1]


def batches(tok, df, bs, shuffle, seed):
    idx = np.random.default_rng(seed).permutation(len(df)) if shuffle else np.arange(len(df))
    texts, labels = df.text.values[idx], df.label.values[idx]
    for i in range(0, len(df), bs):
        enc = tok(list(texts[i:i + bs]), padding=True, truncation=True, max_length=128, return_tensors="pt")
        enc["labels"] = torch.tensor(labels[i:i + bs])
        yield enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev_ = "cuda" if torch.cuda.is_available() else "cpu"
    ss = load_processed("stereoset", args.seed)
    train, dev = ss[ss.split == "train"], ss[ss.split == "dev"]

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(dev_)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * (len(train) // args.bs + 1)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    best, out = -1, ROOT / "results" / "finetuned" / f"{args.model.replace('/', '_')}_s{args.seed}"
    for ep in range(args.epochs):
        model.train()
        for b in batches(tok, train, args.bs, True, args.seed + ep):
            loss = model(**{k: v.to(dev_) for k, v in b.items()}).loss
            loss.backward()
            opt.step(); sched.step(); opt.zero_grad()
        model.eval()
        correct = 0
        with torch.no_grad():
            for b in batches(tok, dev, 64, False, 0):
                logits = model(**{k: v.to(dev_) for k, v in b.items() if k != "labels"}).logits
                correct += (logits.argmax(-1).cpu() == b["labels"]).sum().item()
        acc = correct / len(dev)
        print(f"epoch {ep + 1}: dev acc = {acc:.4f}")
        if acc > best:  # model selection on dev only
            best = acc
            out.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(out); tok.save_pretrained(out)
    print(f"best dev acc {best:.4f} -> {out}")


if __name__ == "__main__":
    main()
