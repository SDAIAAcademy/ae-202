"""Write the supplied logged runs of Day 4 (data/runs/*.jsonl) by running the two small
open-weight models in this process. The course author runs this once per edition; it loads
both models (about 1 GB each) and takes about half an hour on a laptop processor.

    AE202_LLM=runtime uv run python _tools/make_runs.py            # everything
    AE202_LLM=runtime uv run python _tools/make_runs.py d4_2       # one lab
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("AE202_LLM", "runtime")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nawl as U  # noqa: E402
import llm_tasks as T  # noqa: E402

OUT = U.DATA / "runs"
OUT.mkdir(parents=True, exist_ok=True)
SOURCE = "run in process with the Transformers library on a laptop processor, " + time.strftime("%Y-%m-%d")


def write(lab, rows):
    with open(OUT / f"{lab}.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(lab, len(rows), "records", flush=True)


def record(lab, case, setting, kind, messages, res, **extra):
    model = U.LLM_BASE if kind == "base" else U.LLM_INSTRUCT
    return [{"lab": lab, "case": case, "setting": setting, "sample": k, "kind": kind, "model": model, **extra,
             "text": r["text"], "prompt_tokens": r["prompt_tokens"], "new_tokens": r["new_tokens"],
             "truncated": r["truncated"], "seconds": r["seconds"], "source": SOURCE} for k, r in enumerate(res)]


def d4_1():
    rows = []
    for name, s in T.SETTINGS.items():
        for case, product in enumerate(T.products()):
            msgs = T.description_messages(product, strict=s["strict"])
            res = U.generate(msgs, temperature=s["temperature"], top_p=s["top_p"], max_new_tokens=48, n=5,
                             seed=1000 + case)
            rows += record("d4_1", product["id"], name, "instruct", msgs, res, temperature=s["temperature"],
                           top_p=s["top_p"], strict=s["strict"])
        print(" ", name, flush=True)
    write("d4_1", rows)
    # the distribution over the next token after the first prompt: the 200 most likely tokens
    import torch
    tok, model = U._load(U.LLM_INSTRUCT)
    text = U.render(T.description_messages(T.products()[0], strict=False))
    with torch.no_grad():
        logits = model(tok(text, return_tensors="pt").input_ids).logits[0, -1]
    top = torch.topk(logits, 200)
    p = torch.softmax(logits, 0)
    out = {"prompt": T.description_messages(T.products()[0], strict=False)[-1]["content"],
           "tokens": [tok.decode([i]) for i in top.indices.tolist()],
           "logits": [round(v, 4) for v in top.values.tolist()],
           "mass_of_these_at_temperature_1": round(float(p[top.indices].sum()), 6),
           "vocabulary_size": int(logits.shape[0]), "model": U.LLM_INSTRUCT, "source": SOURCE}
    (OUT / "d4_1_next_token.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def d4_2():
    rows = []
    for task in T.TASKS:
        for setting, kind, prompt in (("instruct", "instruct", task["prompt"]), ("base", "base", task["prompt"]),
                                      ("base few-shot", "base", task["few_shot"])):
            res = U.generate(prompt, kind=kind, temperature=0.0, max_new_tokens=40)
            rows += record("d4_2", task["id"], setting, kind, prompt, res[:1], prompt=prompt, rendered=U.render(prompt, kind))
    write("d4_2", rows)


def d4_3():
    rows = []
    for q in T.UNANSWERABLE + T.CUTOFF:
        res = U.generate(q["question"], temperature=0.0, max_new_tokens=70)
        rows += record("d4_3", q["id"], q["group"], "instruct", None, res[:1], prompt=q["question"])
    for q in T.POLICY_QUESTIONS:
        for setting, prompt in T.context_prompts(q).items():
            res = U.generate(prompt, temperature=0.0, max_new_tokens=50)
            rows += record("d4_3", q["id"], setting, "instruct", None, res[:1], question=q["en"] if "arabic" not in setting else q["ar"])
            print(" ", q["id"], setting, res[0]["prompt_tokens"], res[0]["truncated"], res[0]["text"][:70].replace("\n", " "), flush=True)
    write("d4_3", rows)


if __name__ == "__main__":
    which = sys.argv[1:] or ["d4_1", "d4_2", "d4_3"]
    for name in which:
        t = time.time()
        globals()[name]()
        print(name, "took", round((time.time() - t) / 60, 1), "min", flush=True)
