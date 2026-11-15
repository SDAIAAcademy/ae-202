"""Assemble data/measurements.json for Lab 4.4 from the reference results of the labs and
from latencies measured here, one request at a time, on a laptop processor (the course
author runs this once per edition, after every lab solution has run).

    AE202_MODELS=load AE202_LLM=runtime uv run python _tools/make_measurements.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("AE202_MODELS", "load")
os.environ.setdefault("AE202_LLM", "runtime")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nawl as U  # noqa: E402
import llm_tasks as T  # noqa: E402

ref = U.reference
torch.set_grad_enabled(False)
X, _ = U.images("test", 8)

cnn = U.small_cnn().eval()
ms_cnn = U.timeit(lambda: cnn(X[:1] - 0.5), 20)
res = U.backbone().eval()
ms_res = U.timeit(lambda: res(U.to_backbone_input(X[:1])), 20)

docs = U.descriptions()
D = U.embed([d["text"] for d in docs], prefix="passage: ", live=False)
q = "a red striped rug"
ms_embed = U.timeit(lambda: U.embed([q], live=True), 10)
qv = U.embed([q], live=True)[0]
ms_search = U.timeit(lambda: np.argsort(-(D @ qv))[:5], 20)
query_tokens = round(float(np.mean([len(U.tokeniser("e5").encode(x["text"]).ids) for x in U.queries()])))
index = U.keyword_index([d["text"] for d in docs])
ms_keyword = U.timeit(lambda: U.keyword_top_k(q, index), 5)

torch.set_grad_enabled(True)
U.generate(T.description_messages(T.products()[0], strict=True), max_new_tokens=8)
t = time.time()
out = [U.generate(T.description_messages(p, strict=True), max_new_tokens=48)[0] for p in T.products()[:5]]
elapsed = time.time() - t
s_llm = elapsed / 5
tok_per_s = sum(o["new_tokens"] for o in out) / elapsed

d2 = ref("d2_3_results_full")["rows"]
d24 = ref("d2_4_report")
d33, d34 = ref("d3_3_recall"), ref("d3_4_cases")["recall_at_5"]
d41 = ref("d4_1_table")["table"]
clean = lambda name: round(1 - d41[name]["any_violation"], 3)
langs = lambda d: {k: round(v, 3) for k, v in d.items() if k != "all"}

m = {
    "measured_on": "a laptop processor, one request at a time, " + time.strftime("%Y-%m-%d"),
    "note": "quality figures are the reference results of the labs; latencies are medians of repeated single requests",
    "use_cases": {
        "naqsh": {
            "task": "name the pattern of a textile photograph on the shop's upload page",
            "quality_measure": "test accuracy, 1,000 photographs, trained on the 800 labelled ones",
            "options": [
                {"id": "naqsh-scratch", "model": "small-cnn", "what": "small convolutional network trained from scratch",
                 "quality": round(d2[0]["test_accuracy"], 3), "latency_ms": round(ms_cnn, 1)},
                {"id": "naqsh-frozen", "model": "resnet18-imagenet", "what": "frozen ResNet-18 features and a linear layer",
                 "quality": round(d2[1]["test_accuracy"], 3), "latency_ms": round(ms_res, 1)},
                {"id": "naqsh-finetuned", "model": "resnet18-imagenet", "what": "ResNet-18 with its last block fine-tuned",
                 "quality": round(d2[2]["test_accuracy"], 3), "latency_ms": round(ms_res, 1),
                 "quality_without_near_duplicates": round(d24["accuracy_without"], 3)},
                {"id": "naqsh-hosted", "model": "hosted-vision-language", "what": "a hosted vision-language model asked to name the pattern",
                 "quality": None, "latency_ms": None},
            ],
        },
        "khayt": {
            "task": "find products in the bilingual catalogue from a customer's words",
            "quality_measure": "recall at 5 on the 50 labelled queries",
            "options": [
                {"id": "khayt-keywords", "model": "no-model", "what": "keyword search", "quality": round(d33["keywords"]["all"], 3),
                 "latency_ms": round(ms_keyword, 1), "quality_by_language": langs(d33["keywords"])},
                {"id": "khayt-embeddings", "model": "multilingual-e5-small", "what": "sentence embeddings, brute-force search",
                 "quality": round(d34["plain"]["all"], 3), "latency_ms": round(ms_embed + ms_search, 1), "quality_by_language": langs(d34["plain"])},
                {"id": "khayt-embeddings-paired", "model": "multilingual-e5-small", "what": "sentence embeddings, results paired by product",
                 "quality": round(d34["pair_by_product"]["all"], 3), "latency_ms": round(ms_embed + ms_search, 1),
                 "quality_by_language": langs(d34["pair_by_product"])},
                {"id": "khayt-hosted", "model": "hosted-embedding", "what": "a hosted multilingual embedding service", "quality": None, "latency_ms": None},
            ],
            "tokens_per_request": {"prompt": query_tokens, "output": 0},
        },
        "wasf": {
            "task": "draft one product line of at most 20 words from a product's attributes",
            "quality_measure": "share of 100 drafts (20 products, 5 samples) that break none of the four rules",
            "options": [
                {"id": "wasf-template", "model": "no-model", "what": "a sentence template filled with the attributes", "quality": 1.0,
                 "latency_ms": 0.1, "note": "every product line has the same shape"},
                {"id": "wasf-small", "model": "qwen2.5-0.5b-instruct", "what": "small model, plain instruction, temperature 0.7",
                 "quality": clean("T=0.7"), "latency_ms": round(1000 * s_llm)},
                {"id": "wasf-small-strict", "model": "qwen2.5-0.5b-instruct", "what": "small model, strict instruction, temperature 0.7",
                 "quality": clean("T=0.7 strict"), "latency_ms": round(1000 * s_llm)},
                {"id": "wasf-3b", "model": "qwen2.5-3b-instruct", "what": "a model six times larger of the same family", "quality": None, "latency_ms": None},
                {"id": "wasf-7b", "model": "qwen2.5-7b-instruct", "what": "a model fifteen times larger of the same family", "quality": None, "latency_ms": None},
                {"id": "wasf-hosted", "model": "hosted-language-model", "what": "a hosted large language model", "quality": None, "latency_ms": None},
            ],
            "tokens_per_second": round(tok_per_s, 1),
            "tokens_per_request": {"prompt": out[0]["prompt_tokens"], "output": round(sum(o["new_tokens"] for o in out) / 5)},
        },
    },
}
(U.DATA / "measurements.json").write_text(json.dumps(m, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps(m, indent=1))
