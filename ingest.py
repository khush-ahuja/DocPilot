"""Index files or folders from the command line.

    python ingest.py sample_docs/ my_notes.pdf
"""
import argparse

from docpilot import create_pilot
from docpilot.ingest import chunk_file, iter_files


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+")
    args = p.parse_args()

    pilot, settings = create_pilot()
    files = iter_files(args.paths)
    if not files:
        raise SystemExit("no .pdf, .md or .txt files found")
    for f in files:
        pilot.store.delete_source(f.name)
        n = pilot.store.add(chunk_file(f, settings.chunk_size, settings.chunk_overlap))
        print(f"{f.name}: {n} chunks")
    print(f"total chunks in '{settings.collection}': {pilot.store.count()}")


if __name__ == "__main__":
    main()
