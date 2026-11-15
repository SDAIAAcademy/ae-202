"""The fixed prompts, tasks and checks of the Day 4 labs (import llm_tasks as T).

Wasf is the cooperative's description drafter: a small open-weight language model that turns
the attributes of a product into one line for the shop page. The prompts are fixed so that
your live runs can be compared with the supplied logged runs.
"""

from __future__ import annotations

import re

import nawl as U

# --------------------------------------------------------------------------- Lab 4.1: sampling

SETTINGS = {  # name: temperature, nucleus (top_p), and whether the strict instruction is used
    "T=0": {"temperature": 0.0, "top_p": 1.0, "strict": False},
    "T=0.7": {"temperature": 0.7, "top_p": 1.0, "strict": False},
    "T=1.0": {"temperature": 1.0, "top_p": 1.0, "strict": False},
    "T=1.5": {"temperature": 1.5, "top_p": 1.0, "strict": False},
    "T=1.5 p=0.9": {"temperature": 1.5, "top_p": 0.9, "strict": False},
    "T=1.5 p=0.5": {"temperature": 1.5, "top_p": 0.5, "strict": False},
    "T=0.7 strict": {"temperature": 0.7, "top_p": 1.0, "strict": True},
    "T=1.5 strict": {"temperature": 1.5, "top_p": 1.0, "strict": True},
}
STRICT = ("You write product lines for the online shop of a crafts cooperative. Rules: exactly one sentence; "
          "at most 20 words; English only; never a price, a currency or a discount; use only the facts given.")
ASK = "Write one sentence in English, at most 20 words, that describes this product for the shop page. Do not mention a price."


def products() -> list[dict]:
    """The 20 fixed products of Lab 4.1: the first 20 English descriptions of the catalogue."""
    return [d for d in U.descriptions() if d["lang"] == "en"][:20]


def description_messages(product: dict, strict: bool = False) -> list[dict]:
    facts = f"{product['type']}, {product['pattern']}, {product['colour']}, {product['material']}, {product['size']} cm"
    user = {"role": "user", "content": f"{ASK}\nProduct: {facts}"}
    return ([{"role": "system", "content": STRICT}] if strict else []) + [user]


PRICE = re.compile(r"(\$|€|£|\bSAR\b|\bUSD\b|riyal|dollar|\bprice[ds]?\b|discount|\bcheap|\baffordable)", re.I)


def violations(text: str) -> dict:
    """The four rules of the task, checked by code. True means the rule is broken."""
    body = text.strip()
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", body) if s.strip()]
    return {
        "more than one sentence": len(sentences) > 1 or "\n" in body,
        "more than 20 words": len(body.split()) > 20,
        "not English only": any(ord(ch) > 0x24F and not 0x2000 <= ord(ch) <= 0x206F for ch in body),
        "mentions a price": bool(PRICE.search(body)),
    }


# --------------------------------------------------------------------------- Lab 4.2: base and instruction-tuned

def _task(i, prompt, expect, few_shot):
    return {"id": f"T{i:02d}", "prompt": prompt, "expect": expect, "few_shot": few_shot}


TASKS = [
    _task(1, "Which colour is this product? Answer with one word.\nProduct: Handwoven rug in wool, striped, red.", "red",
          "Product: Cotton shawl with a dotted design in blue.\nColour: blue\n\nProduct: Linen basket, checked, green.\nColour: green\n\nProduct: Handwoven rug in wool, striped, red.\nColour:"),
    _task(2, "Translate to English: سجادة حمراء", "red",
          "Arabic: شال أزرق\nEnglish: blue shawl\n\nArabic: سلة خضراء\nEnglish: green basket\n\nArabic: سجادة حمراء\nEnglish:"),
    _task(3, "In which language is this text written? Answer with one word.\nText: هذه القطعة منسوجة يدويا", "arabic",
          "Text: This piece is woven by hand\nLanguage: English\n\nText: Cette pièce est tissée à la main\nLanguage: French\n\nText: هذه القطعة منسوجة يدويا\nLanguage:"),
    _task(4, 'Return only a JSON object with the keys "type" and "colour".\nProduct: A yellow belt, wavy, woven from linen.', '"colour": "yellow"',
          'Product: A red rug, striped, woven from wool.\nJSON: {"type": "rug", "colour": "red"}\n\nProduct: A blue shawl, dotted, woven from cotton.\nJSON: {"type": "shawl", "colour": "blue"}\n\nProduct: A yellow belt, wavy, woven from linen.\nJSON:'),
    _task(5, "Is wool a plant fibre? Answer yes or no.", "no",
          "Question: Is cotton a plant fibre?\nAnswer: yes\n\nQuestion: Is silk a plant fibre?\nAnswer: no\n\nQuestion: Is wool a plant fibre?\nAnswer:"),
    _task(6, "How many items are in this list? Answer with a number.\nList: rug, shawl, basket", "3",
          "List: belt, apron\nCount: 2\n\nList: rug, curtain, pouch, shawl\nCount: 4\n\nList: rug, shawl, basket\nCount:"),
    _task(7, "Write this in capital letters: wool rug", "WOOL RUG",
          "Text: blue shawl\nCapitals: BLUE SHAWL\n\nText: linen belt\nCapitals: LINEN BELT\n\nText: wool rug\nCapitals:"),
    _task(8, "What is the capital of Saudi Arabia? Answer with one word.", "riyadh",
          "Country: France\nCapital: Paris\n\nCountry: Japan\nCapital: Tokyo\n\nCountry: Saudi Arabia\nCapital:"),
    _task(9, "Is this review positive or negative? Answer with one word.\nReview: The shawl arrived torn and late.", "negative",
          "Review: Beautiful rug, fast delivery.\nSentiment: positive\n\nReview: The colours faded after one wash.\nSentiment: negative\n\nReview: The shawl arrived torn and late.\nSentiment:"),
    _task(10, "ما خامة هذا المنتج؟ أجب بكلمة واحدة.\nالمنتج: شال مخطط من صوف باللون الأزرق", "صوف",
          "المنتج: سلة من خوص النخيل باللون البني\nالخامة: خوص النخيل\n\nالمنتج: حزام من قطن باللون الأحمر\nالخامة: قطن\n\nالمنتج: شال مخطط من صوف باللون الأزرق\nالخامة:"),
]


def first_line(text: str) -> str:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[0] if lines else ""


def task_score(task: dict, row: dict) -> dict:
    """right: the expected answer is in the first line. stopped: the model ended by itself
    before the 40-token limit. clean: it is right, stopped, and wrote one line only."""
    text = row["text"]
    right = task["expect"].lower() in first_line(text).lower()
    stopped = row["new_tokens"] < 40
    return {"right": right, "stopped": stopped, "clean": right and stopped and len(text.strip().splitlines()) == 1}


# --------------------------------------------------------------------------- Lab 4.3: limits

def _q(i, group, question):
    return {"id": f"{group[0].upper()}{i:02d}", "group": group, "question": question}


UNANSWERABLE = [  # nothing about the fictional cooperative can be in any model's training data
    _q(1, "unanswerable", "How many days do customers have to return an order at the online shop of Al-Nawl Crafts Cooperative?"),
    _q(2, "unanswerable", "In which year was Al-Nawl Crafts Cooperative founded, and by whom?"),
    _q(3, "unanswerable", "What is the customer service phone number of Al-Nawl Crafts Cooperative?"),
    _q(4, "unanswerable", "How much does product P0417 cost at Al-Nawl Crafts Cooperative?"),
    _q(5, "unanswerable", "How many weavers are members of Al-Nawl Crafts Cooperative?"),
    _q(6, "unanswerable", "Which award did Al-Nawl Crafts Cooperative win last year?"),
    _q(7, "unanswerable", "كم يوما مهلة إرجاع الطلب في المتجر الإلكتروني لجمعية النول التعاونية للحرف؟"),
    _q(8, "unanswerable", "في أي سنة تأسست جمعية النول التعاونية للحرف ومن أسسها؟"),
]
CUTOFF = [
    _q(1, "cutoff", "What is today's date?"),
    _q(2, "cutoff", "What is the most recent year you have information about?"),
    _q(3, "cutoff", "Which team won the football World Cup held in 2026?"),
    _q(4, "cutoff", "Name the newest language model you know of and the year it was released."),
]
ABSTAIN = re.compile(r"(i (do not|don't|cannot|can't|am not|'m not)|unable|no (specific )?information|not (aware|able|sure|have)|"
                     r"as an ai|sorry|unknown|لا أملك|لا أستطيع|لا يمكنني|ليس لدي|لا أعرف|آسف|عذرا)", re.I)


def abstains(text: str) -> bool:
    """A reference check for 'the model said it does not know'. Read the answers too: a
    pattern misses polite fabrications and counts hedged ones."""
    return bool(ABSTAIN.search(text))


POLICY_EN = """Al-Nawl Crafts Cooperative: shop policy (a synthetic document for this course)

1. Returns. An order may be returned within 14 days of delivery if the piece is unused. Made-to-order pieces cannot be returned.
2. Made-to-order. A made-to-order piece is ready in 10 working days.
3. Delivery. Delivery is free for orders above 300 riyals. Below that, delivery costs 25 riyals.
4. Questions. The shop team answers messages within 2 working days.
"""
POLICY_AR = """جمعية النول التعاونية للحرف: سياسة المتجر (وثيقة مصطنعة لهذا المقرر)

1. الإرجاع. يمكن إرجاع الطلب خلال 14 يوما من الاستلام إذا لم تستخدم القطعة. القطع المصنوعة عند الطلب لا ترجع.
2. الصنع عند الطلب. تجهز القطعة المصنوعة عند الطلب خلال 10 أيام عمل.
3. التوصيل. التوصيل مجاني للطلبات التي تزيد على 300 ريال. وما دون ذلك يكلف التوصيل 25 ريالا.
4. الاستفسارات. يرد فريق المتجر على الرسائل خلال 2 يوم عمل.
"""
POLICY_QUESTIONS = [
    {"id": "P01", "en": "Within how many days can an unused order be returned? Answer with the number.",
     "ar": "خلال كم يوما يمكن إرجاع طلب لم يستخدم؟ أجب بالرقم.", "expect": "14"},
    {"id": "P02", "en": "In how many working days is a made-to-order piece ready? Answer with the number.",
     "ar": "خلال كم يوم عمل تجهز القطعة المصنوعة عند الطلب؟ أجب بالرقم.", "expect": "10"},
    {"id": "P03", "en": "Above which order value in riyals is delivery free? Answer with the number.",
     "ar": "ما قيمة الطلب بالريال التي يصبح التوصيل فوقها مجانيا؟ أجب بالرقم.", "expect": "300"},
    {"id": "P04", "en": "Within how many working days does the shop team answer a message? Answer with the number.",
     "ar": "خلال كم يوم عمل يرد فريق المتجر على الرسالة؟ أجب بالرقم.", "expect": "2"},
]


def catalogue_listing(n: int = 110) -> str:
    """A long attachment: the first n English descriptions, one per line."""
    rows = [d for d in U.descriptions() if d["lang"] == "en"][:n]
    return "Catalogue extract\n" + "\n".join(f"{d['id']}: {d['text']}" for d in rows) + "\n"


def context_prompts(q: dict) -> dict:
    """Four arrangements of the same question. 'fits': the policy and the question.
    'policy first, too long': the policy, then a long catalogue extract, then the question:
    more than the window holds, so the runtime drops the start. 'policy last, too long': the
    same texts with the policy after the extract. 'fits, arabic': the Arabic policy and question."""
    ask = "Answer from the document above.\nQuestion: " + q["en"]
    return {
        "fits": POLICY_EN + "\n" + ask,
        "policy first, too long": POLICY_EN + "\n" + catalogue_listing() + "\n" + ask,
        "policy last, too long": catalogue_listing() + "\n" + POLICY_EN + "\n" + ask,
        "fits, arabic": POLICY_AR + "\nأجب من الوثيقة أعلاه.\nالسؤال: " + q["ar"],
    }


def has_number(text: str, expect: str) -> bool:
    """True when the expected number appears in the answer as a whole number."""
    western = text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    return expect in re.findall(r"\d+", western)
