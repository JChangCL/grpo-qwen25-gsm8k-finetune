"""Shared MATH helpers: \\boxed{} extraction + math-equivalence checking.

Uses the `math_verify` library (LaTeX/sympy aware) when available so that
1/2 == 0.5 == \\frac{1}{2}; falls back to a normalized boxed-string compare.
"""

try:
    from math_verify import parse as _mv_parse, verify as _mv_verify
    _HAS_MV = True
except Exception:  # pragma: no cover
    _HAS_MV = False


MATH_SYSTEM_PROMPT = (
    "You are a careful math problem solver. Reason step by step, then give the "
    "final answer on the last line as \\boxed{your_answer}."
)


def last_boxed(text: str) -> str:
    """Return the content of the last \\boxed{...} (brace-balanced)."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return ""
    i = idx + len("\\boxed")
    while i < len(text) and text[i] != "{":
        i += 1
    if i >= len(text):
        return ""
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
    return ""


def is_correct(pred_text: str, gold: str) -> bool:
    """True if the model's answer matches the reference (math-equivalent).

    The gold is a bare answer string (e.g. ``\\dfrac{7}{20}``); math_verify only
    parses it as LaTeX math when it is delimited, so wrap it in \\boxed{} first.
    """
    if _HAS_MV:
        try:
            g = str(gold)
            gold_expr = g if ("\\boxed" in g or "$" in g) else f"\\boxed{{{g}}}"
            return bool(_mv_verify(_mv_parse(gold_expr), _mv_parse(pred_text)))
        except Exception:
            return False
    pred = last_boxed(pred_text).strip().replace(" ", "")
    return pred != "" and pred == str(gold).strip().replace(" ", "")
