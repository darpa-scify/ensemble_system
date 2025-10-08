import fnmatch
import glob
import os.path
from typing import List


def resolve_input_paths(paths: List[str]) -> List[str]:
    """
    Given a list of inputs, split each on commas, resolve globs, and then validate that every file exists

    >>> resolve_input_paths(["foo.json", "bar.json,bletch.json", "asdf/*.json", "*/*.json,foo.txt"])
    ["foo.json", "bar.json", "bletch.json", "asdf/foo.json", "asdf/foo.json", "asdf/foo.json", "foo.txt"]
    """
    split_paths = []
    for path in paths:
        split_paths.extend(path.split(","))

    globbed_paths = []
    for path in split_paths:
        if glob.escape(path) == path:
            globbed_paths.append(path)
        else:
            globbed_paths.extend(glob.glob(path, recursive=True))

    for path in globbed_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")

    return globbed_paths


def resolve_exclusion_ids(ids: List[str], problem_ids: List[str]):
    """
    Given a list of inputs, split each on commas, resolve globs, and then return the list of matching ids

    >>> resolve_exclusion_ids(["alloys_0001", "semiconductors_*"], ...)
    ["alloys_0001", "semiconductors_0001", "semiconductors_0002", ...]
    """
    split_ids = []
    for pattern in ids:
        split_ids.extend(pattern.split(","))

    globbed_ids = []
    for pattern in split_ids:
        globbed_ids.extend(fnmatch.filter(problem_ids, pattern))

    return globbed_ids
