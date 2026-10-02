from pathlib import Path


def read_shotlist(shotlist_file: Path | str) -> list[int]:
    """Shots of a shotlist file, one per line, in file order. Lines that are not a shot number are skipped."""
    with open(shotlist_file) as f:
        lines = [line.strip() for line in f]
    return [int(line) for line in lines if line.isdigit()]
