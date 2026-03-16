import os
from pathlib import Path
from typing import List


def generate_repo_tree(root_dir: str, max_depth: int = 10, prefix: str = "") -> str:
    """
    Recursively generate a textual tree of the repository.
    """
    if max_depth < 0:
        return ""

    tree_str = ""
    entries = sorted(Path(root_dir).iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
    for i, entry in enumerate(entries):
        if entry.is_dir() and entry.name == "__pycache__":
            continue  # skip __pycache__ dirs
        connector = "└── " if i == len(entries) - 1 else "├── "
        tree_str += f"{prefix}{connector}{entry.name}\n"
        if entry.is_dir():
            extension = "    " if i == len(entries) - 1 else "│   "
            tree_str += generate_repo_tree(entry, max_depth - 1, prefix + extension)
    return tree_str


def collect_py_files(
    root_dir: str,
    ignore_files: List[str] = None,
    ignore_init: bool = True,
) -> List[Path]:
    """
    Collect all .py files recursively, excluding ignored files and optionally __init__.py.
    """
    ignore_files = ignore_files or []
    py_files = []
    for path in Path(root_dir).rglob("*.py"):
        if ignore_init and path.name == "__init__.py":
            continue
        if "__pycache__" in path.parts:
            continue
        rel_path = str(path.relative_to(root_dir))
        if rel_path in ignore_files:
            continue
        py_files.append(path)
    return py_files


def aggregate_repo_content(
    root_dir: str,
    output_file: str,
    ignore_files: List[str] = None,
    ignore_init: bool = True,
    max_tree_depth: int = 10,
):
    """
    Crawl a repository, aggregate .py files and tree structure into a single text file.
    """
    root_dir = Path(root_dir).resolve()

    # 1. Generate tree
    tree_str = f"Repository structure for: {root_dir}\n\n"
    tree_str += generate_repo_tree(root_dir, max_depth=max_tree_depth)
    tree_str += "\n\n"

    # 2. Collect python files
    py_files = collect_py_files(root_dir, ignore_files, ignore_init)

    # 3. Aggregate content
    content_str = tree_str
    for py_file in py_files:
        rel_path = py_file.relative_to(root_dir)
        content_str += f"# === File: {rel_path} ===\n"
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                file_content = f.read()
            content_str += file_content + "\n\n"
        except Exception as e:
            content_str += f"# Could not read file: {e}\n\n"

    # 4. Save to output
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content_str)

    print(f"Aggregated repository content saved to {output_file}")


if __name__ == "__main__":
    # Example usage
    aggregate_repo_content(
        root_dir="time_series_dl",  # path to your repo
        output_file="repo_aggregated.txt",
        ignore_files=[
            "scripts/plot_example_forecast_window_baseline.py",
            "scripts/plot_horizon_metrics_baseline.py",
        ],  # optional
        ignore_init=True,
        max_tree_depth=10,
    )
