"""
common.py
---------
Shared prompt + message construction used by BOTH data preparation and inference.

Keeping this in one place guarantees the system prompt the model is TRAINED with is
byte-for-byte identical to the one used at TRANSLATION time. For a small fine-tuned
model that consistency matters a lot -- a mismatched prompt noticeably hurts quality.
"""

# The instruction that shapes the model's behavior. The proper-noun rule is what makes
# it keep names, cities, brands, events, and holidays (with no Cuyonon term) unchanged.
SYSTEM_PROMPT = (
    "You are a precise translation engine specialized in the Tagalog and Cuyonon "
    "languages. Translate exactly what the user gives you.\n"
    "Keep proper nouns unchanged: people's names, cities, provinces, countries, "
    "brands, organizations, and the names of events or holidays that have no Cuyonon "
    "equivalent must be copied over exactly as written - never translated or altered.\n"
    "Output only the translation - no notes, no explanations, no quotation marks."
)


def build_user_turn(text, src_lang, tgt_lang, keep_terms=None):
    """Compose the user message. `keep_terms` optionally lists terms to preserve verbatim."""
    msg = f"Translate from {src_lang} to {tgt_lang}:\n\n{text}"
    if keep_terms:
        terms = ", ".join(keep_terms)
        msg = f"Keep these terms unchanged: {terms}\n\n{msg}"
    return msg


def build_messages(text, src_lang, tgt_lang, keep_terms=None, assistant=None):
    """Build the chat message list. If `assistant` is given, it's a training example;
    otherwise it's an inference prompt (assistant turn to be generated)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_turn(text, src_lang, tgt_lang, keep_terms)},
    ]
    if assistant is not None:
        messages.append({"role": "assistant", "content": assistant})
    return messages
