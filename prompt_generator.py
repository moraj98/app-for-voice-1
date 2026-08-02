"""Turn loose words into a structured, refined prompt.

Everything here is plain Python with no network calls: the input words are
parsed, sorted into buckets (subject, style, lighting, tone, format, ...), and
assembled into a prompt template chosen by the detected intent. A refinement
pass then strips filler, removes duplicates and reports what is still missing.
"""
import argparse
import json
import re

STOPWORDS = {
    "a", "an", "the", "of", "and", "or", "with", "in", "on", "at", "for", "to",
    "is", "are", "was", "be", "it", "its", "that", "this", "these", "those",
    "please", "kindly", "just", "very", "really", "basically", "actually",
    "some", "thing", "things", "stuff", "i", "me", "my", "we", "our", "you",
    "want", "wants", "need", "needs", "like", "about", "into", "as", "by",
}

# Filler phrases removed during the refinement pass.
FILLER_PATTERNS = [
    r"\bi (?:just )?(?:want|need|would like) you to\b",
    r"\bplease\b", r"\bkindly\b", r"\bbasically\b", r"\bactually\b",
    r"\bvery\b", r"\breally\b", r"\bsort of\b", r"\bkind of\b",
]

# bucket -> vocabulary. Multi-word entries are matched first, as whole phrases.
VOCAB = {
    "style": {
        "photorealistic", "photoreal", "cinematic", "anime", "manga", "cartoon",
        "watercolor", "oil painting", "gouache", "charcoal", "sketch", "doodle",
        "3d render", "octane render", "pixel art", "voxel", "low poly",
        "isometric", "flat design", "line art", "vector art", "collage",
        "minimalist", "maximalist", "brutalist", "art deco", "art nouveau",
        "baroque", "impressionist", "surreal", "abstract", "vaporwave",
        "cyberpunk", "steampunk", "solarpunk", "film noir", "noir", "retro",
        "vintage", "futuristic", "gothic", "ukiyo-e", "claymation", "stop motion",
        "documentary", "storybook illustration", "comic book", "concept art",
    },
    "mood": {
        "moody", "dreamy", "eerie", "creepy", "cozy", "epic", "serene", "calm",
        "peaceful", "melancholic", "somber", "joyful", "cheerful", "tense",
        "suspenseful", "whimsical", "playful", "dramatic", "chaotic",
        "nostalgic", "ominous", "hopeful", "lonely", "romantic", "energetic",
    },
    "lighting": {
        "golden hour", "blue hour", "sunset", "sunrise", "backlit", "rim light",
        "neon", "neon lights", "candlelit", "candlelight", "firelight",
        "overcast", "harsh shadows", "soft light", "soft lighting", "hard light",
        "volumetric light", "god rays", "moonlight", "starlight", "studio lighting",
        "silhouette", "high key", "low key", "dappled light", "foggy", "misty",
    },
    "camera": {
        "close-up", "closeup", "extreme close-up", "wide shot", "wide angle",
        "medium shot", "macro", "aerial", "drone shot", "bird's eye view",
        "top-down", "low angle", "high angle", "dutch angle", "over the shoulder",
        "portrait", "bokeh", "shallow depth of field", "deep focus", "tilt shift",
        "35mm", "50mm", "85mm", "telephoto", "fisheye", "long exposure",
        "tracking shot", "dolly zoom", "handheld", "slow motion", "time-lapse",
        "first person", "pan", "zoom in", "zoom out",
    },
    "color": {
        "monochrome", "black and white", "grayscale", "pastel", "muted",
        "desaturated", "vibrant", "saturated", "high contrast", "low contrast",
        "sepia", "earth tones", "teal and orange", "warm tones", "cool tones",
        "duotone", "colorful", "dark palette", "bright palette",
    },
    "tone": {
        "formal", "informal", "casual", "friendly", "professional", "witty",
        "humorous", "funny", "sarcastic", "serious", "persuasive", "technical",
        "academic", "authoritative", "empathetic", "encouraging", "blunt",
        "direct", "conversational", "neutral", "enthusiastic", "poetic",
        # delivery adjectives, common in voice and copy briefs
        "warm", "upbeat", "soothing", "punchy", "snappy", "crisp", "deadpan",
        "reassuring", "urgent", "confident", "matter-of-fact",
    },
    "format": {
        "blog post", "article", "essay", "email", "cold email", "newsletter",
        "tweet", "thread", "linkedin post", "instagram caption", "caption",
        "script", "video script", "voiceover script", "podcast script",
        "outline", "summary", "tldr", "bullet points", "bullets", "table",
        "checklist", "listicle", "report", "memo", "press release", "faq",
        "readme", "documentation", "changelog", "commit message", "pull request",
        "poem", "haiku", "song lyrics", "story", "short story", "dialogue",
        "json", "yaml", "markdown", "csv", "slide deck", "presentation",
    },
    "audience": {
        "beginners", "beginner", "experts", "developers", "engineers",
        "designers", "marketers", "executives", "founders", "students",
        "children", "kids", "teenagers", "customers", "recruiters",
        "non-technical readers", "general audience",
    },
    "length": {
        "short", "brief", "concise", "long", "detailed", "in-depth",
        "one paragraph", "two paragraphs", "one page", "one sentence",
        "one liner", "bite-sized", "comprehensive", "exhaustive",
    },
    "aspect": {
        "16:9", "9:16", "1:1", "4:3", "3:2", "21:9", "portrait orientation",
        "landscape", "square", "vertical", "horizontal", "widescreen",
        "4k", "8k", "hd", "ultra hd", "high resolution",
    },
    "tech": {
        "python", "javascript", "typescript", "react", "vue", "svelte", "node",
        "flask", "django", "fastapi", "rails", "go", "golang", "rust", "java",
        "kotlin", "swift", "c++", "c#", "php", "sql", "postgres", "mysql",
        "sqlite", "mongodb", "redis", "docker", "kubernetes", "terraform",
        "aws", "gcp", "azure", "graphql", "rest api", "api", "cli", "regex",
        "webhook", "websocket", "pandas", "numpy", "pytorch", "tensorflow",
    },
}

# Intent detection vocabulary: intent -> trigger words.
INTENT_TRIGGERS = {
    "image": {
        "image", "photo", "photograph", "picture", "illustration", "painting",
        "drawing", "art", "artwork", "poster", "logo", "icon", "wallpaper",
        "portrait", "render", "thumbnail", "sticker", "avatar", "mockup",
    },
    "video": {
        "video", "clip", "reel", "short", "film", "movie", "animation",
        "animated", "trailer", "shot", "scene", "b-roll", "montage", "cinematic",
        "slow motion", "time-lapse", "tracking shot",
    },
    "audio": {
        "audio", "voice", "voiceover", "narration", "speech", "podcast", "song",
        "music", "sound", "soundtrack", "jingle", "tts", "speak", "spoken",
    },
    "code": {
        "code", "function", "class", "script", "app", "bug", "fix", "refactor",
        "unit test", "tests", "endpoint", "api", "cli", "database", "query",
        "algorithm", "debug", "migration", "component", "library", "regex",
    },
    "writing": {
        "write", "writing", "blog", "article", "essay", "email", "copy",
        "copywriting", "story", "poem", "script", "caption", "post", "newsletter",
        "headline", "tagline", "summary", "rewrite", "edit", "translate",
    },
    "analysis": {
        "analyze", "analysis", "review", "compare", "evaluate", "research",
        "explain", "breakdown", "insights", "data", "metrics", "report",
        "audit", "critique", "pros and cons", "recommend", "strategy", "plan",
    },
}

INTENT_ROLES = {
    "image": "a visual art director writing prompts for an image model",
    "video": "a director and storyboard artist writing prompts for a video model",
    "audio": "an audio producer writing a voice and sound brief",
    "code": "a senior software engineer",
    "writing": "an experienced writer and editor",
    "analysis": "a domain analyst who reasons from evidence",
    "general": "a helpful expert assistant",
}

# Buckets whose words describe *how* something should look or sound rather than
# what it is about.
VISUAL_BUCKETS = ("style", "lighting", "camera", "color", "mood", "aspect")
TEXT_BUCKETS = ("tone", "format", "audience", "length")

NEGATION_PREFIX = re.compile(r"^(?:no|not|without|avoid|exclude|minus)\s+(.+)$", re.IGNORECASE)

# Scaffolding the generator writes itself, recognised when reading a prompt
# back in so that refining the same prompt twice is a no-op.
SKIP_PREFIXES = ("you are", "if anything", "constraints:", "motion:", "duration:")
AUTO_TERMS = {"high detail", "sharp focus on the subject"}
AUTO_CONSTRAINTS = {
    "keep the change minimal and match the surrounding code style",
    "cover the edge cases with tests",
    "state the evidence behind each conclusion",
    "flag anything you are uncertain about instead of guessing",
}
OPENER_RE = re.compile(
    r"^an?\s+(?:(?P<style>.+?)\s+)?(?:image|video)\s+of\s+(?P<subject>.+?)\.?$",
    re.IGNORECASE,
)

# "30 seconds", "500 words", "3 paragraphs" -- always a length constraint.
MEASURE_RE = re.compile(
    r"\b\d+\s*(?:s|m|sec|secs|second|seconds|min|mins|minute|minutes|hour|hours|"
    r"word|words|char|chars|character|characters|page|pages|para|paras|paragraph|"
    r"paragraphs|sentence|sentences|line|lines|bullet|bullets|slide|slides)\b",
    re.IGNORECASE,
)

_PHRASES = sorted(
    ((phrase, bucket) for bucket, words in VOCAB.items() for phrase in words if " " in phrase or "-" in phrase),
    key=lambda pair: -len(pair[0]),
)
_SINGLE_WORDS = {
    phrase: bucket
    for bucket, words in VOCAB.items()
    for phrase in words
    if " " not in phrase and "-" not in phrase
}
_INTENT_PHRASES = sorted(
    ((phrase, intent) for intent, words in INTENT_TRIGGERS.items() for phrase in words),
    key=lambda pair: -len(pair[0]),
)


def _strip_filler(text):
    for pattern in FILLER_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", text).strip()


def parse_terms(raw):
    """Split raw input into classified terms.

    Returns (buckets, negatives). ``buckets`` maps a bucket name to an ordered
    list of unique terms; unrecognised wording lands in ``subject``.

    A chunk written as a sentence (five words or more) is kept verbatim as the
    subject, so "why my deploy is slow" survives intact; shorter keyword-style
    chunks have their stopwords and recognised vocabulary stripped out.
    """
    buckets = {}
    negatives = []
    chunks = [c for c in re.split(r"[,\n;/|]+", raw or "") if c.strip()]

    for chunk in chunks:
        text = _strip_filler(chunk.strip())
        text = re.sub(r"^[-–—]\s*", "no ", text)
        negated = False
        match = NEGATION_PREFIX.match(text)
        if match:
            negated = True
            text = match.group(1)

        verbatim = len(text.split()) >= 5
        found = []
        for measure in MEASURE_RE.finditer(text):
            found.append((measure.group(0).lower().strip(), "length"))
        if not verbatim:
            text = MEASURE_RE.sub(" ", text)

        for phrase, bucket in _PHRASES:
            pattern = r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])"
            if re.search(pattern, text, flags=re.IGNORECASE):
                found.append((phrase, bucket))
                if not verbatim:
                    text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

        leftover = []
        for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9'&+.#:-]*", text):
            word = word.strip(".:-'")
            lowered = word.lower()
            if not word:
                continue
            bucket = _SINGLE_WORDS.get(lowered)
            if bucket:
                found.append((lowered, bucket))
                if verbatim:
                    leftover.append(word)
            elif verbatim or lowered not in STOPWORDS:
                leftover.append(word)

        if leftover:
            found.insert(0, (" ".join(leftover), "subject"))

        for term, bucket in found:
            if negated:
                if term not in negatives:
                    negatives.append(term)
                continue
            values = buckets.setdefault(bucket, [])
            if term not in values:
                values.append(term)

    return buckets, negatives


def detect_intent(raw, buckets):
    """Pick the intent with the most trigger hits, defaulting to ``general``."""
    text = " " + re.sub(r"[^a-z0-9\s'-]+", " ", (raw or "").lower()) + " "
    scores = {}
    for phrase, intent in _INTENT_PHRASES:
        if re.search(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", text):
            scores[intent] = scores.get(intent, 0) + 1

    # Descriptive buckets are strong evidence even without an explicit noun.
    if any(buckets.get(b) for b in ("lighting", "camera", "color")):
        scores["image"] = scores.get("image", 0) + 1
    if buckets.get("tech"):
        scores["code"] = scores.get("code", 0) + 1
    if buckets.get("format") or buckets.get("tone") or buckets.get("audience"):
        scores["writing"] = scores.get("writing", 0) + 1

    if not scores:
        return "general", scores

    best = max(scores.values())
    # Stable tie-break: the more specific intents win over the broader ones.
    for intent in ("code", "image", "video", "audio", "writing", "analysis"):
        if scores.get(intent) == best:
            return intent, scores
    return "general", scores


def _join(values, conjunction="and"):
    values = [v for v in values if v]
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    return ", ".join(values[:-1]) + f" {conjunction} " + values[-1]


def _article(noun):
    """Prefix a bare noun phrase with a/an unless it already has a determiner."""
    if not noun or noun.split()[0].lower() in {"a", "an", "the", "some", "my", "your"}:
        return noun
    if noun.split()[-1].lower().endswith("s") and not noun.lower().endswith("ss"):
        return noun
    return ("an " if noun[0].lower() in "aeiou" else "a ") + noun


def _subject_text(buckets):
    subjects = buckets.get("subject") or []
    return _join(subjects) if subjects else "the subject described below"


def _line(label, values, conjunction="and"):
    text = _join(values, conjunction) if isinstance(values, (list, tuple)) else values
    return f"{label}: {text}" if text else None


def _visual_prompt(buckets, negatives, intent, options):
    subject = _subject_text(buckets)
    style = _join(buckets.get("style", []))
    medium = "video" if intent == "video" else "image"
    opener = f"A {style} {medium} of {subject}." if style else f"A {medium} of {subject}."

    lines = [opener, ""]
    for label, bucket in (
        ("Composition", "camera"),
        ("Lighting", "lighting"),
        ("Color", "color"),
        ("Mood", "mood"),
    ):
        line = _line(label, buckets.get(bucket, []))
        if line:
            lines.append(line)

    if intent == "video":
        lines.append("Motion: deliberate camera movement; the subject stays readable throughout.")
        lines.append(f"Duration: {options.get('duration', '5-8 seconds')}, single continuous shot.")

    extras = []
    for bucket in ("tech", "audience", "format", "tone", "length"):
        extras.extend(buckets.get(bucket, []))
    if extras:
        lines.append(_line("Also include", extras))

    lines.append(_line("Output", buckets.get("aspect", []) + ["high detail", "sharp focus on the subject"]))
    if negatives:
        lines.append(_line("Avoid", negatives))
    return "\n".join(line for line in lines if line is not None)


def _text_prompt(buckets, negatives, intent, options):
    subject = _subject_text(buckets)
    role = INTENT_ROLES[intent]
    # Skip formats the subject already names, or the task line stutters
    # ("Write a blog post about blog post about remote work").
    fresh = [f for f in buckets.get("format", []) if f.lower() not in subject.lower()]
    fmt = _join([_article(f) for f in fresh])
    task = {
        "code": f"Implement {subject}",
        "writing": f"Write {fmt} about {subject}" if fmt else f"Write {subject}",
        "analysis": f"Analyse {subject}",
        "audio": f"Produce {fmt} of {subject}" if fmt else f"Produce {subject}",
        "general": subject[0].upper() + subject[1:],
    }[intent]

    lines = [
        f"You are {role}.",
        "",
        f"Task: {task}.",
    ]

    if buckets.get("tech"):
        lines.append(_line("Stack", buckets["tech"]))
    for label, bucket in (
        ("Audience", "audience"),
        ("Tone", "tone"),
        ("Format", "format"),
        ("Length", "length"),
    ):
        line = _line(label, buckets.get(bucket, []))
        if line:
            lines.append(line)

    descriptive = []
    for bucket in VISUAL_BUCKETS:
        if bucket == "mood":
            continue  # folded into Tone for text prompts
        descriptive.extend(buckets.get(bucket, []))
    if descriptive:
        lines.append(_line("Style notes", descriptive))

    constraints = list(options.get("constraints") or [])
    if negatives:
        constraints.append("do not include " + _join(negatives, "or"))
    if intent == "code":
        constraints.append("keep the change minimal and match the surrounding code style")
        constraints.append("cover the edge cases with tests")
    if intent == "analysis":
        constraints.append("state the evidence behind each conclusion")
        constraints.append("flag anything you are uncertain about instead of guessing")
    if constraints:
        lines.append("")
        lines.append("Constraints:")
        lines.extend(f"- {c}" for c in constraints)

    lines.append("")
    lines.append("If anything essential is ambiguous, ask before starting.")
    return "\n".join(line for line in lines if line is not None)


def _effective_buckets(buckets, intent):
    """Mood words describe delivery, not visuals, once the prompt is text."""
    if intent in ("image", "video") or not buckets.get("mood"):
        return buckets
    merged = {key: list(values) for key, values in buckets.items()}
    merged["tone"] = merged.get("tone", []) + merged.pop("mood")
    return merged


def _missing_slots(buckets, intent):
    """What the prompt still lacks, phrased as a nudge to the user."""
    checks = {
        "image": [
            ("subject", "what the image is actually of"),
            ("style", "an art style or medium (photoreal, watercolor, 3d render...)"),
            ("lighting", "lighting (golden hour, neon, soft light...)"),
            ("camera", "a shot type (close-up, wide shot, aerial...)"),
            ("aspect", "an aspect ratio or resolution (16:9, square, 4k...)"),
        ],
        "video": [
            ("subject", "what happens in the shot"),
            ("style", "a visual style"),
            ("camera", "camera movement (tracking shot, slow pan...)"),
            ("aspect", "an aspect ratio (9:16 for reels, 16:9 for widescreen)"),
        ],
        "audio": [
            ("subject", "what should be said or played"),
            ("tone", "a delivery tone (warm, energetic, neutral...)"),
            ("length", "a target length"),
        ],
        "code": [
            ("subject", "what the code should do"),
            ("tech", "a language or framework"),
            ("format", "the expected output (a function, a CLI, a patch...)"),
        ],
        "writing": [
            ("subject", "the topic"),
            ("audience", "who is reading it"),
            ("tone", "a tone of voice"),
            ("format", "a format (blog post, email, thread...)"),
            ("length", "a target length"),
        ],
        "analysis": [
            ("subject", "what is being analysed"),
            ("audience", "who the analysis is for"),
            ("format", "how to present the findings"),
        ],
        "general": [
            ("subject", "the topic or task"),
            ("format", "the output format you want"),
            ("audience", "who the answer is for"),
        ],
    }[intent]
    return [hint for bucket, hint in checks if not buckets.get(bucket)]


def refine_text(prompt):
    """Tidy an assembled prompt: drop filler, collapse blank runs and repeats."""
    seen = set()
    lines = []
    for line in prompt.splitlines():
        line = _strip_filler(line).rstrip()
        key = line.lower()
        if line and key in seen:
            continue
        if line:
            seen.add(key)
        if not line and lines and not lines[-1]:
            continue
        lines.append(line)
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def generate_prompt(words, intent=None, **options):
    """Build a refined prompt from loose words.

    ``words`` may be comma separated, newline separated or a plain phrase.
    Prefixing a term with ``no``/``without``/``-`` moves it to the avoid list.
    Returns a dict with the prompt, the detected intent, the parsed buckets,
    and suggestions for what is still missing.
    """
    raw = (words or "").strip()
    if not raw:
        raise ValueError("Enter at least one word to build a prompt from.")

    buckets, negatives = parse_terms(raw)
    detected, scores = detect_intent(raw, buckets)
    if intent and intent != "auto":
        if intent not in INTENT_ROLES:
            raise ValueError(f"Unknown intent: {intent}")
        chosen = intent
    else:
        chosen = detected

    buckets = _effective_buckets(buckets, chosen)
    if chosen in ("image", "video"):
        prompt = _visual_prompt(buckets, negatives, chosen, options)
    else:
        prompt = _text_prompt(buckets, negatives, chosen, options)

    return {
        "prompt": refine_text(prompt),
        "intent": chosen,
        "detected_intent": detected,
        "scores": scores,
        "buckets": buckets,
        "avoid": negatives,
        "suggestions": _missing_slots(buckets, chosen),
    }


def _split_values(value):
    parts = re.split(r",|\band\b", value)
    return [p.strip().strip(".") for p in parts if p.strip().strip(".")]


def deconstruct(prompt):
    """Pull the meaningful words back out of a prompt this module generated.

    Scaffolding the generator adds itself (the role line, boilerplate
    constraints, default output notes) is dropped, so a prompt can be fed back
    in repeatedly without growing.
    """
    terms = []
    for raw_line in (prompt or "").splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith(SKIP_PREFIXES):
            continue

        if line[0] in "-*":
            line = line.lstrip("-* ").strip()
            lowered = line.lower()
            if lowered in AUTO_CONSTRAINTS:
                continue
            match = re.match(r"do not include (.+)", lowered)
            if match:
                terms.extend("no " + value for value in _split_values(match.group(1)))
            else:
                terms.append(line)
            continue

        match = OPENER_RE.match(line)
        if match:
            terms.extend(filter(None, [match.group("style"), match.group("subject")]))
            continue

        label, separator, value = line.partition(":")
        if not separator or len(label.split()) > 3:
            terms.append(line.rstrip("."))
            continue

        label = label.strip().lower()
        if label == "task":
            terms.append(re.sub(r"^(?:write|implement|analyse|analyze|produce)\s+", "", value.strip(), flags=re.IGNORECASE).rstrip("."))
        elif label == "avoid":
            terms.extend("no " + value for value in _split_values(value))
        else:
            terms.extend(v for v in _split_values(value) if v.lower() not in AUTO_TERMS)
    return terms


def refine_prompt(prompt, extra_words="", intent=None, **options):
    """Re-run generation over an existing prompt plus any new words.

    Earlier choices ("Tone: friendly") are read back in, so refining
    repeatedly sharpens the same prompt instead of restarting it.
    """
    combined = ", ".join(filter(None, deconstruct(prompt) + [(extra_words or "").strip()]))
    return generate_prompt(combined, intent=intent, **options)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Turn loose words into a refined prompt.")
    parser.add_argument("words", help="comma or space separated words to build from")
    parser.add_argument(
        "--intent",
        default="auto",
        choices=["auto"] + sorted(INTENT_ROLES),
        help="force a prompt style instead of detecting one",
    )
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = parser.parse_args(argv)

    result = generate_prompt(args.words, intent=args.intent)
    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print(result["prompt"])
    if result["suggestions"]:
        print("\n--- still missing ---")
        for suggestion in result["suggestions"]:
            print(f"- {suggestion}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
