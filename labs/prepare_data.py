"""The data generator of the Al-Nawl Crafts Cooperative case study.

Everything is synthetic and drawn from fixed seeds: the same bytes on every run.

    uv run python prepare_data.py            # the image pack into cache/ (about 20 seconds)
    uv run python prepare_data.py --text     # rewrite data/descriptions.jsonl and data/queries.jsonl
    uv run python prepare_data.py --second   # the second image set of the follow-up task

The image pack is drawn the first time a lab asks for it (nawl.ensure_data). The text files
are small and are committed.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

LABS = Path(__file__).resolve().parent
DATA = LABS / "data"
CACHE = LABS / "cache"

CLASSES = ["stripe", "chevron", "diamond", "check", "triangle", "dot", "wave", "star"]
SIZE = 48
N_IMAGES = 8000
SPLITS = {"train": 6000, "val": 1000, "test": 1000}
N_SMALL = 800          # the labelled set of Day 2: 100 per class, taken from train
N_DUPLICATES = 40      # test images that are near-copies of images in the small set
N_MISLABELLED = 12     # test images whose label is wrong
SEED = 202

# Ten yarn colours (RGB); every image picks two of them, whatever its class
PALETTE = np.array([
    [176, 46, 38], [28, 62, 122], [44, 110, 73], [214, 168, 52], [34, 30, 32],
    [236, 228, 208], [204, 108, 44], [112, 72, 48], [120, 48, 96], [70, 140, 150],
], dtype=np.float32)


def _tri(t):
    """A triangle wave with period 1 and values in [-1, 1]."""
    return 4.0 * np.abs(t - np.floor(t + 0.5)) - 1.0


def _mask(cls: int, u, v, p: float, r: np.random.Generator):
    """The 0/1 pattern of one class on rotated coordinates u, v with period p."""
    fu = u / p - np.floor(u / p + 0.5)   # position inside a cell, in [-0.5, 0.5)
    fv = v / p - np.floor(v / p + 0.5)
    if cls == 0:   # stripe
        return np.sin(2 * np.pi * u / p) > 0
    if cls == 1:   # chevron: stripes pushed sideways by a zigzag
        a, q = r.uniform(0.10, 0.22), r.uniform(1.3, 2.2) * p
        return np.sin(2 * np.pi * (u + a * _tri(v / q)) / p) > 0
    if cls == 6:   # wave: stripes pushed sideways by a sine
        a, q = r.uniform(0.07, 0.16), r.uniform(1.5, 2.6) * p
        return np.sin(2 * np.pi * (u + a * np.sin(2 * np.pi * v / q)) / p) > 0
    if cls == 3:   # check
        return np.sin(2 * np.pi * u / p) * np.sin(2 * np.pi * v / p) > 0
    if cls == 2:   # diamond: one diamond per cell
        return (np.abs(fu) + np.abs(fv)) < r.uniform(0.30, 0.40)
    if cls == 5:   # dot: one disc per cell
        return (fu ** 2 + fv ** 2) < r.uniform(0.26, 0.36) ** 2
    if cls == 4:   # triangle: one upward triangle per cell
        return (fv + 0.5) > 2.0 * np.abs(fu) * r.uniform(0.95, 1.15)
    if cls == 7:   # star: a four-point star per cell
        return (np.sqrt(np.abs(fu)) + np.sqrt(np.abs(fv))) < r.uniform(0.86, 0.98)
    raise ValueError(cls)


def draw(cls: int, params_seed: int, noise_seed: int, shift=(0.0, 0.0), style: int = 0) -> np.ndarray:
    """One 48 by 48 RGB image (uint8). params_seed fixes the cloth, noise_seed the photograph.

    style 0 is the cooperative's catalogue; style 1 is the second set of the follow-up task:
    coarser weave, stronger lighting and larger motifs.
    """
    r = np.random.default_rng([SEED, 1, params_seed])
    n = np.random.default_rng([SEED, 2, noise_seed])
    lin = np.linspace(-1, 1, SIZE, dtype=np.float32)
    x, y = np.meshgrid(lin + shift[0], lin + shift[1])
    if style == 2:   # a swatch on the sorting table: laid straight, cut at the same place
        theta = r.uniform(-0.12, 0.12)
        off = r.uniform(0, 0.12, 2)
        p = r.uniform(0.50, 0.62)
    else:
        theta = r.uniform(0, np.pi) if cls in (0, 1, 6) else r.uniform(-0.35, 0.35)
        off = r.uniform(0, 1, 2)
        p = r.uniform(0.34, 0.62) if style == 0 else r.uniform(0.50, 0.80)
    u = np.cos(theta) * x + np.sin(theta) * y + off[0]
    v = -np.sin(theta) * x + np.cos(theta) * y + off[1]
    m = _mask(cls, u, v, p, r).astype(np.float32)
    c0, c1 = PALETTE[r.choice(len(PALETTE), 2, replace=False)]
    jit = r.normal(0, 10, (2, 3)).astype(np.float32)
    img = m[..., None] * (c1 + jit[1]) + (1 - m[..., None]) * (c0 + jit[0])
    # the weave: fine threads in both directions
    wf = r.uniform(9, 14) if style == 0 else r.uniform(5, 8)
    weave = 1.0 - (0.10 if style == 0 else 0.20) * (np.sin(np.pi * wf * x) ** 2 + np.sin(np.pi * wf * y) ** 2)
    img = img * weave[..., None]
    # the photograph: a soft blur, uneven light, sensor noise
    img = (img + np.roll(img, 1, 0) + np.roll(img, 1, 1) + np.roll(img, (1, 1), (0, 1))) / 4.0
    g = n.uniform(-0.25, 0.25, 2) * (1.0 if style == 0 else 1.8)
    img = img * (1.0 + g[0] * x + g[1] * y)[..., None] * n.uniform(0.75, 1.1)
    img = img + n.normal(0, n.uniform(6, 20), img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def make_images(style: int = 0) -> dict:
    """The whole pack: images, labels, the true classes, the splits and what was planted."""
    r = np.random.default_rng([SEED, 9, style])
    true = np.arange(N_IMAGES) % len(CLASSES)
    r.shuffle(true)
    X = np.stack([draw(int(c), i, i, style=style) for i, c in enumerate(true)])
    idx = r.permutation(N_IMAGES)
    train, val, test = idx[:6000], idx[6000:7000], idx[7000:]
    # the small labelled set: 100 per class from train
    small = np.concatenate([train[true[train] == c][:N_SMALL // len(CLASSES)] for c in range(len(CLASSES))])
    small.sort()
    labels = true.copy()
    planted = {"duplicates": [], "mislabelled": []}
    if style == 0:
        # planted problem 1: near-duplicates. The same cloth photographed again (new noise,
        # a shift of one pixel) sits in the test set.
        src = r.choice(small, N_DUPLICATES, replace=False)
        slots = r.choice(test, N_DUPLICATES + N_MISLABELLED, replace=False)
        for s, t in zip(src, slots[:N_DUPLICATES]):
            X[t] = draw(int(true[s]), int(s), int(t), shift=(2.0 / SIZE, 0.0))
            true[t] = true[s]
            labels[t] = true[s]
            planted["duplicates"].append([int(t), int(s)])
        # planted problem 2: wrong labels in the test set (the label says the confusable class)
        swap = {0: 6, 6: 1, 1: 6, 3: 2, 2: 7, 7: 2, 5: 2, 4: 7}
        for t in slots[N_DUPLICATES:]:
            labels[t] = swap[int(true[t])]
            planted["mislabelled"].append([int(t), int(true[t]), int(labels[t])])
    return {"X": X, "y": labels.astype(np.int64), "y_true": true.astype(np.int64), "train": np.sort(train),
            "val": np.sort(val), "test": np.sort(test), "small": small, "planted": planted}


def write_images(style: int = 0) -> Path:
    CACHE.mkdir(exist_ok=True)
    pack = make_images(style)
    out = CACHE / ("textiles.npz" if style == 0 else "textiles_second.npz")
    np.savez_compressed(out, X=pack["X"], y=pack["y"], train=pack["train"], val=pack["val"],
                        test=pack["test"], small=pack["small"])
    if style == 0:
        key = {"classes": CLASSES, **pack["planted"]}
        (CACHE / "planted_key.json").write_text(json.dumps(key), encoding="utf-8")
    return out


# --------------------------------------------------------------------------- the catalogue text

TYPES = {  # key: (English, English synonyms used by queries, Arabic, Arabic synonyms used by queries)
    "rug": ("rug", ["carpet", "floor covering"], "سجادة", ["زولية", "بساط"]),
    "cushion": ("cushion cover", ["pillow case", "pillow cover"], "غطاء وسادة", ["كيس مخدة", "غطاء مخدة"]),
    "bag": ("tote bag", ["shoulder bag", "carry bag"], "حقيبة كتف", ["شنطة", "شنطة كتف"]),
    "hanging": ("wall hanging", ["wall decoration", "piece to hang on the wall"], "معلقة جدارية", ["لوحة قماش للجدار", "زينة جدار"]),
    "runner": ("table runner", ["table cloth strip", "runner for a dining table"], "مفرش طاولة", ["شرشف طاولة", "غطاء طاولة طويل"]),
    "shawl": ("shawl", ["wrap", "scarf"], "شال", ["وشاح", "لفحة"]),
    "blanket": ("blanket", ["throw", "bed cover"], "بطانية", ["لحاف", "غطاء سرير"]),
    "basket": ("basket", ["storage bin", "woven container"], "سلة", ["قفة", "زنبيل"]),
    "coasters": ("set of coasters", ["cup mats", "drink mats"], "طقم قواعد أكواب", ["قواعد فناجين", "قواعد للأكواب"]),
    "pouch": ("pouch", ["small purse", "zip case"], "محفظة قماش", ["جزدان", "كيس صغير بسحاب"]),
    "placemat": ("placemat", ["plate mat", "dining mat"], "مفرش صحن", ["قاعدة صحن", "مفرش سفرة صغير"]),
    "apron": ("apron", ["kitchen smock", "cooking cover"], "مريلة", ["مريول مطبخ", "صدرية مطبخ"]),
    "bookmark": ("bookmark", ["page marker", "book ribbon"], "فاصل كتاب", ["علامة صفحة", "شريط كتاب"]),
    "curtain": ("curtain", ["window drape", "window hanging"], "ستارة", ["برداية", "ستار نافذة"]),
    "belt": ("belt", ["waist band", "sash"], "حزام", ["حزام خصر", "زنار"]),
    "headband": ("headband", ["hair band", "head wrap"], "ربطة رأس", ["طوق شعر", "عصبة رأس"]),
}
PATTERNS = {
    "stripe": ("striped", ["with lines", "banded"], "مخطط", ["بخطوط", "مقلم"]),
    "chevron": ("chevron", ["zigzag", "with a zigzag design"], "بنقشة متعرجة", ["زجزاج", "بخطوط متكسرة"]),
    "diamond": ("diamond-patterned", ["with rhombus shapes", "with lozenges"], "بنقشة المعينات", ["بأشكال معينات", "بنقشة الألماس"]),
    "check": ("checked", ["chequered", "with squares"], "بنقشة المربعات", ["كاروهات", "مربعات مثل رقعة الشطرنج"]),
    "triangle": ("triangle-patterned", ["with triangles", "with a row of peaks"], "بنقشة المثلثات", ["بمثلثات", "بأشكال مثلثة"]),
    "dot": ("dotted", ["polka dot", "spotted"], "منقط", ["بنقاط", "بدوائر صغيرة"]),
    "wave": ("wavy", ["with ripples", "with curved lines"], "مموج", ["بتموجات", "بخطوط منحنية"]),
    "star": ("star-patterned", ["with stars", "with four-point stars"], "بنقشة النجوم", ["بنجوم", "بأشكال نجمية"]),
}
COLOURS = {
    "red": ("red", ["crimson", "scarlet"], "أحمر", ["قرمزي", "احمر"]),
    "blue": ("blue", ["indigo", "navy"], "أزرق", ["نيلي", "ازرق"]),
    "green": ("green", ["olive", "emerald"], "أخضر", ["زيتي", "اخضر"]),
    "yellow": ("yellow", ["golden", "mustard"], "أصفر", ["ذهبي", "خردلي"]),
    "black": ("black", ["charcoal", "jet"], "أسود", ["فحمي", "اسود"]),
    "white": ("white", ["cream", "ivory"], "أبيض", ["كريمي", "عاجي"]),
    "orange": ("orange", ["rust", "terracotta"], "برتقالي", ["بلون الصدأ", "طوبي"]),
    "brown": ("brown", ["chestnut", "coffee"], "بني", ["بلون القهوة", "كستنائي"]),
}
MATERIALS = {
    "wool": ("wool", ["sheep's wool"], "صوف", ["صوف الغنم"]),
    "cotton": ("cotton", ["cotton yarn"], "قطن", ["خيوط قطنية"]),
    "palm": ("palm fibre", ["date-palm leaves"], "خوص النخيل", ["سعف النخل"]),
    "camel": ("camel hair", ["camel wool"], "وبر الجمل", ["شعر الإبل"]),
    "linen": ("linen", ["flax"], "كتان", ["خيوط الكتان"]),
}
SIZES = ["30 x 30", "40 x 40", "45 x 45", "60 x 90", "80 x 120", "120 x 180", "160 x 230", "35 x 150", "50 x 200", "20 x 25"]
EN_TEMPLATES = [
    "Handwoven {type} in {mat}, {pat}, {c1}. Size {size} cm.",
    "{Pat} {type} made of {mat}. Colour: {c1}. {size} cm.",
    "A {c1} {type}, {pat}, woven from {mat} by the cooperative's makers. {size} cm.",
    "{Type}: {mat}, {pat} design, {c1}. Dimensions {size} cm.",
    "This {type} is {pat} and {c1}. Material: {mat}. {size} cm.",
    "{Mat} {type} with a {pat} design in {c1}. Size {size} cm.",
]
AR_TEMPLATES = [
    "{type} منسوجة يدويا من {mat}، {pat}، باللون ال{c1}. المقاس {size} سم.",
    "{type} {pat} من {mat}. اللون: {c1}. {size} سم.",
    "{type} باللون ال{c1}، {pat}، من {mat} بأيدي حرفيات الجمعية. {size} سم.",
    "{type}: {mat}، {pat}، باللون ال{c1}. الأبعاد {size} سم.",
    "هذه القطعة {type} {pat} باللون ال{c1}. الخامة: {mat}. {size} سم.",
    "{type} من {mat} {pat} باللون ال{c1}. المقاس {size} سم.",
]


def make_descriptions() -> list[dict]:
    """2,000 products, two for each of 1,000 combinations of type, pattern and colour (they differ
    in material and size). Every product has an English and an Arabic description, written
    from different templates: 4,000 descriptions."""
    r = random.Random(SEED)
    combos = [(t, p, c) for t in TYPES for p in PATTERNS for c in COLOURS]
    r.shuffle(combos)
    rows, n = [], 0
    for t, p, c in combos[:1000]:
        mats = r.sample(list(MATERIALS), 2)
        sizes = r.sample(SIZES, 2)
        for m, size in zip(mats, sizes):
            n += 1
            en = r.choice(EN_TEMPLATES).format(
                type=TYPES[t][0], Type=TYPES[t][0].capitalize(), pat=PATTERNS[p][0], Pat=PATTERNS[p][0].capitalize(),
                mat=MATERIALS[m][0], Mat=MATERIALS[m][0].capitalize(), c1=COLOURS[c][0], size=size)
            ar = r.choice(AR_TEMPLATES).format(type=TYPES[t][2], pat=PATTERNS[p][2], mat=MATERIALS[m][2],
                                               c1=COLOURS[c][2], size=size)
            for lang, text in (("en", en), ("ar", ar)):
                rows.append({"product": f"P{n:04d}", "lang": lang, "text": text, "type": t, "pattern": p,
                             "colour": c, "material": m, "size": size})
    r.shuffle(rows)
    return [{"id": f"D{k + 1:04d}", **row} for k, row in enumerate(rows)]


# The 50 labelled queries. Each names a type, a pattern and a colour; its relevant
# descriptions are the four with the three attributes: two products, each in both languages.
# "how" says how the query words them, which is what the failure analysis of Lab 3.4 looks
# for (it is in the answer key only).
QUERY_PLAN = (
    [("en", "plain")] * 8 + [("en", "synonym")] * 8 + [("en", "negation")] * 2 + [("en", "size")] * 2
    + [("ar", "plain")] * 7 + [("ar", "synonym")] * 6 + [("ar", "dialect")] * 4 + [("ar", "spelling")] * 3
    + [("mixed", "codeswitch")] * 6 + [("mixed", "arabizi")] * 4
)
ARABIZI = {
    "rug": "sajjada", "cushion": "ghita wsada", "bag": "shanta", "hanging": "mu3allaqa lil jidar", "runner": "mafrash tawla",
    "shawl": "shal", "blanket": "battaniya", "basket": "salla", "coasters": "qawa3id akwab", "pouch": "mahfaza qmash",
    "placemat": "mafrash sa7n", "apron": "maryala", "bookmark": "fasil kitab", "curtain": "sitara", "belt": "7izam", "headband": "rabtat ras",
    "red": "a7mar", "blue": "azraq", "green": "akhdar", "yellow": "asfar", "black": "aswad", "white": "abyad",
    "orange": "burtuqali", "brown": "bunni", "stripe": "mukhattat", "chevron": "zigzag", "diamond": "mu3ayyanat",
    "check": "murabba3at", "triangle": "muthallathat", "dot": "munaqqat", "wave": "mumawwaj", "star": "nujoom",
}
DIALECT = {"rug": "زولية", "cushion": "تلبيسة مخدة", "bag": "شنطة", "hanging": "شي ينعلق على الجدار", "runner": "شرشف طاولة",
           "shawl": "لفحة", "blanket": "لحاف", "basket": "زبيل", "coasters": "قواعد بيالات", "pouch": "جزدان",
           "placemat": "مفرش صحون", "apron": "مريول", "bookmark": "علامة كتاب", "curtain": "برداية", "belt": "حزام", "headband": "ربطة شعر"}


def make_queries(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    r = random.Random(SEED + 1)
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["type"], row["pattern"], row["colour"]), []).append(row)
    usable = sorted(groups)
    r.shuffle(usable)
    queries, key = [], []
    for n, (lang, how) in enumerate(QUERY_PLAN):
        combo = usable[n]
        g = groups[combo]
        t, p, c = combo
        relevant = [x["id"] for x in g]
        ten, tsyn, tar, tarsyn = TYPES[t]
        pen, psyn, par, parsyn = PATTERNS[p]
        cen, csyn, car, carsyn = COLOURS[c]
        if (lang, how) == ("en", "plain"):
            text = r.choice([f"{cen} {pen} {ten}", f"a {pen} {ten} in {cen}", f"{ten}, {pen}, {cen}"])
        elif (lang, how) == ("en", "synonym"):
            text = r.choice([f"I am looking for a {r.choice(csyn)} {r.choice(tsyn)} {r.choice(psyn)}",
                             f"{r.choice(tsyn)} {r.choice(psyn)}, mostly {r.choice(csyn)}",
                             f"do you have a {r.choice(tsyn)} {r.choice(psyn)} in {r.choice(csyn)}?"])
        elif how == "negation":
            unwanted = g[0]["material"]
            text = f"{cen} {pen} {ten}, but not made of {MATERIALS[unwanted][0]}"
            relevant = [x["id"] for x in g if x["material"] != unwanted]
        elif how == "size":
            size = g[0]["size"]
            text = f"{cen} {pen} {ten} {size.replace(' x ', ' by ')} cm"
            relevant = [x["id"] for x in g if x["size"] == size]
        elif (lang, how) == ("ar", "plain"):
            text = r.choice([f"{tar} {par} باللون ال{car}", f"{tar} {car} {par}", f"أبحث عن {tar} {par} لونها {car}"])
        elif (lang, how) == ("ar", "synonym"):
            text = r.choice([f"أبحث عن {r.choice(tarsyn)} {r.choice(parsyn)} لونها {carsyn[0]}",
                             f"هل عندكم {r.choice(tarsyn)} {r.choice(parsyn)} باللون ال{car}؟",
                             f"{r.choice(tarsyn)} {r.choice(parsyn)} {carsyn[0]}"])
        elif how == "dialect":
            text = r.choice([f"أبي {DIALECT[t]} {parsyn[0]} لونها {car}", f"عندكم {DIALECT[t]} {car} {parsyn[0]}؟",
                             f"ودي ب{DIALECT[t]} {parsyn[0]} {car} زينة"])
        elif how == "spelling":
            bare = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ة": "ه", "ى": "ي"})
            text = f"{tar} {par} {car}".translate(bare)
        elif how == "codeswitch":
            text = r.choice([f"أبغى {ten} {pen} لونه {cen}", f"عندكم {ten} باللون ال{car} {pen}؟",
                             f"{tar} {pen} {cen} للهدية"])
        elif how == "arabizi":
            text = f"abgha {ARABIZI[t]} {ARABIZI[p]} {ARABIZI[c]}"
        else:
            raise ValueError((lang, how))
        qid = f"Q{n + 1:02d}"
        queries.append({"id": qid, "lang": lang, "text": text, "relevant": sorted(relevant)})
        key.append({"id": qid, "lang": lang, "how": how, "type": t, "pattern": p, "colour": c})
    return queries, key


def write_text() -> None:
    DATA.mkdir(exist_ok=True)
    rows = make_descriptions()
    queries, key = make_queries(rows)
    with open(DATA / "descriptions.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(DATA / "queries.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    (LABS / "solutions").mkdir(exist_ok=True)
    with open(LABS / "solutions" / "queries_key.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(key, f, ensure_ascii=False, indent=1)
    print(f"wrote {len(rows)} descriptions and {len(queries)} queries")


if __name__ == "__main__":
    if "--text" in sys.argv:
        write_text()
    elif "--second" in sys.argv:
        print("wrote", write_images(style=1).name)
    else:
        print("wrote", write_images().name)
