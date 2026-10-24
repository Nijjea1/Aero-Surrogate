"""Publication must not leak local paths when grading runs on a different OS."""
import shlex

import pytest

from src.eval.harness import portable_command, portable_paths


@pytest.mark.parametrize("root,value,expected", [
    ("/home/runner/repo", "C:/Users/someone/AppData/Local/Temp/best.pt", "best.pt"),
    ("/home/runner/repo", r"C:\Users\someone\Coding Projects\best.pt", "best.pt"),
    ("C:/work/repo", "/home/someone/Coding Projects/table.csv", "table.csv"),
    ("/home/runner/Coding Projects/repo", "/home/runner/Coding Projects/repo/runs/best.pt", "runs/best.pt"),
    ("C:/Coding Projects/repo", r"c:\Coding Projects\repo\runs\best.pt", "runs/best.pt"),
    ("/home/runner/repo", "/home/runner/repo/../private/best.pt", "best.pt"),
    ("C:/work/repo", "C:/work/repo/../private/best.pt", "best.pt"),
    ("/home/runner/repo", r"\\server\private\someone\best.pt", "best.pt"),
    (r"\\server\share\repo", r"\\server\share\repo\runs\best.pt", "runs/best.pt"),
])
def test_path_flavour_is_independent_of_host_platform(root, value, expected):
    assert portable_paths(value, root) == expected


def test_quoted_commands_keep_boundaries_and_do_not_leak_paths_with_spaces():
    raw = 'harness.py --checkpoint "C:\\Users\\someone\\Coding Projects\\best.pt" --data \'/Users/someone/Coding Projects/table.csv\' --out runs/m.json'
    assert portable_paths(raw, "/home/runner/repo") == 'harness.py --checkpoint "best.pt" --data \'table.csv\' --out runs/m.json'


def test_urls_ratios_and_relative_paths_are_preserved():
    values = ["https://example.com/source?q=/home/public/data", "cd 0.2/0.3 over runs/x",
              "source https://example.com/C:/published/file?q=/home/public", "runs/a file.pt"]
    assert portable_paths(values, "/home/runner/repo") == values


def test_command_metadata_sanitizes_argv_before_joining():
    argv = ["harness.py", "--checkpoint", "/Users/someone/Coding Projects/best.pt",
            "--label", "Release scalar report", "--out", "/home/runner/repo/runs/my metrics.json"]
    result = portable_command(argv, "/home/runner/repo")
    assert shlex.split(result) == ["harness.py", "--checkpoint", "best.pt", "--label",
                                  "Release scalar report", "--out", "runs/my metrics.json"]
    assert "someone" not in result and "Coding Projects" not in result


def test_nested_path_values_are_sanitized():
    values = {"nested": [{"path": "C:/Users/someone/best.pt"}], "args": ("/home/private/a.csv",)}
    assert portable_paths(values, "/home/runner/repo") == {"nested": [{"path": "best.pt"}], "args": ["a.csv"]}
