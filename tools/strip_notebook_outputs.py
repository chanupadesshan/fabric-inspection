"""Remove the saved outputs (pictures, tables, printed text, tracebacks) from the notebooks.

Why: the experiment notebooks carry their pictures inside the file (about 140 MB together), which is too
big for a clean GitHub repository, and the outputs also contain local paths (user name, folders).
Results are not lost: every run also writes results/<method>/scores_fabricXX.csv and split_fabricXX.json,
and rerunning a notebook draws the pictures again.

    python tools/strip_notebook_outputs.py            # strip every notebook in experiments/
    python tools/strip_notebook_outputs.py --check    # only list size and whether outputs are present

Cell code is never changed.
"""
import argparse
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def has_outputs(nb):
    return any(c.get("cell_type") == "code" and (c.get("outputs") or c.get("execution_count")) for c in nb["cells"])


def strip(nb):
    for c in nb["cells"]:
        if c.get("cell_type") == "code":
            c["outputs"] = []
            c["execution_count"] = None
    nb.get("metadata", {}).pop("widgets", None)
    return nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only report, change nothing")
    ap.add_argument("folder", nargs="?", default=os.path.join(ROOT, "experiments"))
    args = ap.parse_args()
    total_before = total_after = 0
    for path in sorted(glob.glob(os.path.join(args.folder, "*.ipynb"))):
        before = os.path.getsize(path)
        with open(path, encoding="utf-8") as f:
            nb = json.load(f)
        dirty = has_outputs(nb)
        if not args.check and dirty:
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(strip(nb), f, indent=1, ensure_ascii=False)
                f.write("\n")
        after = before if args.check or not dirty else os.path.getsize(path)
        total_before += before
        total_after += after
        print(f"{os.path.basename(path):34s} {before / 1e6:7.2f} MB  {'outputs' if dirty else 'clean'}"
              + ("" if args.check or not dirty else f"  ->  {after / 1e6:.2f} MB"))
    print(f"total {total_before / 1e6:.1f} MB" + ("" if args.check else f"  ->  {total_after / 1e6:.1f} MB"))


if __name__ == "__main__":
    main()
