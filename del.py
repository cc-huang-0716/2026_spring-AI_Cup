from pathlib import Path
import ast
import io
import shutil
import tokenize
import argparse


SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "env", ".mypy_cache", ".pytest_cache"}


def get_standalone_string_lines(source):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()

    remove_lines = set()

    for node in ast.walk(tree):
        body = getattr(node, "body", None)

        if not isinstance(body, list):
            continue

        for stmt in body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                if isinstance(stmt.value.value, str):
                    start = getattr(stmt, "lineno", None)
                    end = getattr(stmt, "end_lineno", None)

                    if start is not None and end is not None:
                        for line_no in range(start, end + 1):
                            remove_lines.add(line_no)

    return remove_lines


def remove_standalone_strings(source):
    remove_lines = get_standalone_string_lines(source)
    lines = source.splitlines(keepends=True)

    for line_no in remove_lines:
        index = line_no - 1
        if 0 <= index < len(lines):
            lines[index] = "\n"

    return "".join(lines)


def remove_hash_comments(source):
    result = []
    reader = io.StringIO(source).readline

    for token in tokenize.generate_tokens(reader):
        if token.type == tokenize.COMMENT:
            continue
        result.append(token)

    return tokenize.untokenize(result)


def clean_blank_lines(source):
    lines = source.splitlines()
    cleaned = []
    blank_count = 0

    for line in lines:
        if line.strip() == "":
            blank_count += 1
            if blank_count <= 2:
                cleaned.append(line)
        else:
            blank_count = 0
            cleaned.append(line)

    return "\n".join(cleaned) + "\n"


def clean_file(path, backup=True):
    source = path.read_text(encoding="utf-8")

    cleaned = remove_standalone_strings(source)
    cleaned = remove_hash_comments(cleaned)
    cleaned = clean_blank_lines(cleaned)

    if backup:
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))

    path.write_text(cleaned, encoding="utf-8")
    print(f"完成：{path}")


def should_skip(path):
    return any(part in SKIP_DIRS for part in path.parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    target = Path(args.target)
    backup = not args.no_backup

    if target.is_file():
        if target.suffix == ".py":
            clean_file(target, backup=backup)
        return

    for file in target.rglob("*.py"):
        if file.name == "remove_py_comments.py":
            continue

        if should_skip(file):
            continue

        clean_file(file, backup=backup)


if __name__ == "__main__":
    main()
