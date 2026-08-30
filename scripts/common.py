"""
common.py
---------
Shared configuration and prompt construction, used by BOTH data preparation and
inference.

Keeping the prompt here guarantees the text the model is TRAINED with is
byte-for-byte identical to the one used at TRANSLATION time. For a small
fine-tuned model that consistency matters a lot -- a mismatched prompt
noticeably hurts quality.

The language set lives in `languages.json` at the repo root, so adding a
language is a config change rather than a code edit.
"""

import json
import os

# The instruction that shapes the model's behavior. The proper-noun rule is what
# makes it keep names, cities, brands, events, and holidays (with no equivalent
# in the target language) unchanged.
#
# Deliberately names no languages. The user turn always says "Translate from X
# to Y", so the model is told the pair on every single example, and a fixed
# prompt means adding a language to languages.json cannot silently invalidate an
# adapter that was trained under the old list.
SYSTEM_PROMPT = (
    "You are a precise translation engine. Translate exactly what the user "
    "gives you.\n"
    "Keep proper nouns unchanged: people's names, cities, provinces, countries, "
    "brands, organizations, and the names of events or holidays that have no "
    "equivalent in the target language must be copied over exactly as written - "
    "never translated or altered.\n"
    "Output only the translation - no notes, no explanations, no quotation marks."
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(_REPO_ROOT, "languages.json")


def load_config(path=None):
    """Read languages.json and return it with a few conveniences filled in.

    Adds:
      code_to_name / name_to_code -- for the HTTP layer, which speaks codes
      language_names              -- declaration order, used for stable output
    Corpus paths are resolved relative to the repo root, so scripts work no
    matter which directory they are launched from.
    """
    path = path or DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)

    languages = config.get("languages", [])
    if len(languages) < 2:
        raise ValueError(f"{path}: need at least two languages to translate between")

    config["code_to_name"] = {lang["code"]: lang["name"] for lang in languages}
    config["name_to_code"] = {lang["name"]: lang["code"] for lang in languages}
    config["language_names"] = [lang["name"] for lang in languages]

    known = set(config["language_names"])
    for corpus in config.get("corpora", []):
        corpus["path"] = os.path.join(_REPO_ROOT, corpus["file"])
        unknown = [c for c in corpus["columns"] if c not in known]
        if unknown:
            raise ValueError(
                f"{path}: corpus {corpus['file']} names {unknown}, which is not in "
                f"`languages`. Add it there first."
            )

    if config.get("entity_dir"):
        config["entity_path"] = os.path.join(_REPO_ROOT, config["entity_dir"])

    return config


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


def parse_user_turn(user_content):
    """Recover (src_lang, tgt_lang, text) from a user message built above.

    Used by evaluate.py to segment results by direction, and kept next to
    build_user_turn so the two cannot drift apart.
    """
    header = user_content.split(":", 1)[0]        # "Translate from X to Y"
    words = header.split()
    src_lang = words[words.index("from") + 1]
    tgt_lang = words[words.index("to") + 1]
    text = user_content.split("\n\n", 1)[1]
    return src_lang, tgt_lang, text
