"""Downloads the NLTK corpora that tools/build_tables.py uses (WordNet and SentiWordNet)."""
import nltk

for pkg in ("wordnet", "omw-1.4", "sentiwordnet"):
    print(pkg, "->", nltk.download(pkg, quiet=True))
