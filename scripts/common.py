"""
common.py
---------
Shared configuration, file reading and prompt construction, used by data
preparation, training, inference and the data tools.

Keeping the prompt here guarantees the text the model is TRAINED with is
byte-for-byte identical to the one used at TRANSLATION time. For a small
fine-tuned model that consistency matters a lot -- a mismatched prompt
noticeably hurts quality.

The language set lives in `languages.json` at the repo root, so adding a
language is a config change rather than a code edit.
"""

import collections
import json
import os
import re

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

# The two fixed pieces of the user turn. parse_user_turn() relies on them.
KEEP_PREFIX = "Keep these terms unchanged: "
TRANSLATE_HEADER = "Translate from "

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(_REPO_ROOT, "languages.json")


def load_config(path=None):
    """Read languages.json and return it with a few conveniences filled in.

    Adds:
      code_to_name / name_to_code -- code <-> prompt name
      language_lookup             -- lowercased code OR name -> name; use
                                     resolve_language() rather than reading it
      language_names              -- declaration order, used for stable output
    and, for every entry in `corpora` and `test`:
      path        -- resolved against the repo root, so scripts work no matter
                     which directory they are launched from
      format      -- "sep" for a <sep> file, "moses" for an OPUS-style folder
      train_only  -- never used for validation/test. Defaults to True for Moses
                     folders: big downloaded corpora are there for fluency, not
                     to be scored on.
      synthetic   -- machine-made on one side (e.g. back-translation). Implies
                     train_only; prepare_data.py emits it one way only.
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
    # What the HTTP API and the Supabase log accept for a language: its code or
    # its name, in any case. Learnut sends the name ("Tagalog", "Cuyonon").
    config["language_lookup"] = {}
    for lang in languages:
        config["language_lookup"][lang["code"].lower()] = lang["name"]
        config["language_lookup"][lang["name"].lower()] = lang["name"]

    known = set(config["language_names"])
    config.setdefault("corpora", [])
    config.setdefault("test", [])
    for section in ("corpora", "test"):
        for corpus in config[section]:
            # `file` = one <sep>-delimited file; `dir` = a Moses folder holding
            # one plain-text file per language, which is what OPUS downloads
            # unpack to.
            source = corpus.get("file") or corpus.get("dir")
            if not source:
                raise ValueError(f"{path}: every entry in `{section}` needs a `file` or a `dir`")
            corpus["path"] = os.path.join(_REPO_ROOT, source)
            corpus.setdefault("format", "moses" if corpus.get("dir") else "sep")
            corpus.setdefault("label", source)
            corpus.setdefault("train_only", corpus["format"] == "moses")
            corpus.setdefault("synthetic", False)
            if corpus["synthetic"]:
                corpus["train_only"] = True
            columns = corpus.get("columns") or []
            if len(columns) != 2:
                raise ValueError(f"{path}: {corpus['label']} needs exactly two `columns`")
            unknown = [c for c in columns if c not in known]
            if unknown:
                raise ValueError(
                    f"{path}: {corpus['label']} names {unknown}, which is not in "
                    f"`languages`. Add it there first."
                )

    if config.get("entity_dir"):
        config["entity_path"] = os.path.join(_REPO_ROOT, config["entity_dir"])

    return config


def resolve_language(config, value):
    """The prompt name for a code or name from the app ('cyo', 'Cuyonon',
    'cuyonon' -> 'Cuyonon'), or None when it is not a configured language."""
    if not isinstance(value, str):
        return None
    return config["language_lookup"].get(value.strip().lower())


def read_sep_file(path):
    """Return the (left, right) pairs of a `<sep>` file.

    Blank lines and lines starting with '#' are skipped. That matters: the
    generated English file starts with a '# Tagalog<sep>English -- ...' header,
    which used to be parsed -- and trained on -- as a translation pair. Lines
    without exactly one `<sep>` are skipped too. A missing file gives no pairs.
    """
    pairs = []
    if not os.path.isfile(path):
        return pairs
    # utf-8-sig: tolerate the byte-order mark some Windows editors add.
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("<sep>")
            if len(parts) != 2:
                continue
            left, right = parts[0].strip(), parts[1].strip()
            if left and right:
                pairs.append((left, right))
    return pairs


_EDGE_PUNCTUATION = " \t.!?,;:'\"()[]{}<>«»…·-–—"
_CURLY_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def loose_key(text):
    """Case-, spacing- and edge-punctuation-insensitive form of `text`.

    'Paalam po!' and 'paalam po' give the same key. Used wherever two spellings
    of one sentence must count as the same sentence: keeping both on one side
    of the train/validation split, and spotting duplicates.
    """
    text = " ".join(text.translate(_CURLY_QUOTES).lower().split())
    return text.strip(_EDGE_PUNCTUATION)


def term_pattern(term):
    """Regex matching `term` as a whole word: 'Cuyo' but not inside 'Cuyonon'."""
    # [^\W\d_] is "a letter" in Python's Unicode-aware regex syntax.
    return re.compile(r"(?<![^\W\d_])" + re.escape(term) + r"(?![^\W\d_])")


def contains_term(text, term):
    return bool(term_pattern(term).search(text))


def load_entities(entity_dir):
    """{category: [terms]} from every .txt file in entity_dir.

    The category is the file name without .txt (names, places, events). Lines
    starting with '#' are comments.
    """
    entities = collections.OrderedDict()
    if not entity_dir or not os.path.isdir(entity_dir):
        return entities
    for fname in sorted(os.listdir(entity_dir)):
        if not fname.endswith(".txt"):
            continue
        terms = []
        with open(os.path.join(entity_dir, fname), "r", encoding="utf-8-sig") as f:
            for line in f:
                term = line.strip()
                if term and not term.startswith("#") and term not in terms:
                    terms.append(term)
        entities[fname[:-len(".txt")]] = terms
    return entities


def build_user_turn(text, src_lang, tgt_lang, keep_terms=None):
    """Compose the user message. `keep_terms` optionally lists terms to preserve verbatim."""
    msg = f"{TRANSLATE_HEADER}{src_lang} to {tgt_lang}:\n\n{text}"
    if keep_terms:
        msg = f"{KEEP_PREFIX}{', '.join(keep_terms)}\n\n{msg}"
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

    Skips the optional "Keep these terms unchanged" line, so it works on every
    example prepare_data.py writes. Kept next to build_user_turn so the two
    cannot drift apart.
    """
    start = user_content.find(TRANSLATE_HEADER)
    body = user_content[start:] if start >= 0 else user_content
    header, text = body.split("\n\n", 1)
    words = header.rstrip(":").split()
    src_lang = words[words.index("from") + 1]
    tgt_lang = words[words.index("to") + 1]
    return src_lang, tgt_lang, text
