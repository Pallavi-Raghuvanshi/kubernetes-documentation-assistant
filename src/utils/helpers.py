

import re
from urllib.parse import urlparse

def filename_from_url(url: str) -> str:
    """Generate HTML filename from a documentation URL."""
    path = urlparse(url).path.rstrip("/") # https://example.com/docs/setup@first/ -> /docs/setup
    name = path.split("/")[-1] # /docs/setup@first -> docs, setup@first -> setup@first

    # Replace any character that is not alphanumeric(Cap, small), dot, underscore, or hyphen with an underscore
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name) # e.g. setup@first -> setup_first.html

    return f"{name or 'index'}.html"