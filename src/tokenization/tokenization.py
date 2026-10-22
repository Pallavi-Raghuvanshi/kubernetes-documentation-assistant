import re


def tokenize(text: str) -> list[str]:
    """Convert text to lowercase word tokens."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    return re.findall(r"\b\w+\b", text.lower())