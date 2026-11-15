"""The first hour of the capstone: load the data of one task and print what you need to start.

    uv run python capstone/starter.py naqsh      # image classifier
    uv run python capstone/starter.py khayt      # bilingual search
    uv run python capstone/starter.py wasf       # constrained text generation
    uv run python capstone/starter.py naqsh --second   # the second dataset of the follow-up task

Run it from labs/. Copy what you need into your own repository; `nawl.py`, `llm_tasks.py` and
`prepare_data.py` may be copied with it (say so in your README).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nawl as U  # noqa: E402

task = next((a for a in sys.argv[1:] if not a.startswith("-")), "naqsh")
second = "--second" in sys.argv

if task == "naqsh":
    if second:
        import prepare_data
        f = U.CACHE / "textiles_second.npz"
        if not f.exists():
            print("drawing the second image set (once) ...")
            prepare_data.write_images(style=1)
        with np.load(f) as d:
            pack = {k: d[k] for k in d.files}
        print("the second set: photographs taken at another workshop, with another camera and light")
    else:
        pack = U.textiles()
    print(f"photographs: {pack['X'].shape}, classes: {U.CLASSES}")
    for split in ("small", "val", "test"):
        y = pack["y"][pack[split]]
        print(f"  {split:5s} {len(y):5d} photographs, per class: {np.bincount(y, minlength=8).tolist()}")
    print("train on `small` (the 800 labelled photographs), choose on `val`, report on `test` once")
    print("the simpler alternative: U.small_cnn() trained from scratch with U.fit (Lab 2.1)")
    print("before you report: near-duplicates between `small` and `test`, and wrong labels (Lab 2.4)")
elif task == "khayt":
    docs, queries = U.descriptions(), U.queries()
    print(f"{len(docs):,} descriptions of {len({d['product'] for d in docs}):,} products; "
          f"{len(queries)} labelled queries: " + str({k: sum(q['lang'] == k for q in queries) for k in ('en', 'ar', 'mixed')}))
    print("a description:", docs[0]["text"])
    print("a query:", queries[0]["text"], "->", queries[0]["relevant"])
    print("the simpler alternative: U.keyword_index and U.keyword_top_k (Lab 3.3)")
    print("supplied vectors: U.embed(texts, prefix, live=False); new text needs the model: Colab, or AE202_MODELS=load")
    print("write and label 20 queries of your own before you run them: the 50 were used to design the fixes of Lab 3.4")
    if second:
        print("the second dataset of the follow-up: 30 new queries, 10 per language group, written by a colleague")
elif task == "wasf":
    import llm_tasks as T
    products = [d for d in U.descriptions() if d["lang"] == "en"]
    chosen = products[20:40] if second else T.products()
    print(f"{len(chosen)} products" + (" (the second set: products 21 to 40)" if second else " (the fixed products of Lab 4.1)"))
    print("the prompt of the first one:\n" + T.description_messages(chosen[0], strict=True)[-1]["content"])
    print("the four rules, checked by code: T.violations(text) ->", list(T.violations("x")))
    print("the simpler alternative: a sentence template filled with the attributes")
    print("a live model here:", U.llm_backend() or "none: open the work in Colab, or set LLM_API_KEY, LLM_BASE_URL and LLM_MODEL")
else:
    raise SystemExit("choose one of: naqsh, khayt, wasf")
