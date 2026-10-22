"""Language Coherence Model (LCM) — access to the Subjects, Concepts and Connectors tables.

    from lcm import load
    t = load()                       # reads data/LCM_tables.json
    t.find(\"happy\")                  # nodes from any table
    t[\"subjects\"].near(\"dog\", 1)      # neighbors in the grid

This does NOT generate text yet: it loads, positions and walks the tables.
"""
from .geometry import position, cell, axis_of, N, AXES
from .tables import load, Tables, Table

__all__ = ["load", "Tables", "Table", "position", "cell", "axis_of", "N", "AXES"]
__version__ = "0.5.0"
