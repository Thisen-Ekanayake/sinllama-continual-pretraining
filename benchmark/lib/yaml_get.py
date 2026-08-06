#!/usr/bin/env python3
"""Read a dotted key out of one or more merged YAML config files.

Usage:
  yaml_get.py file1.yml [file2.yml ...] <dotted.key> [--field NAME]

Files are deep-merged in order (later files override earlier keys), so a
task's config.yml can sit on top of main.yml. The resolved value is printed:
  - scalar             -> printed as-is
  - list of scalars    -> one item per line (read into a bash array with
                           `mapfile -t arr < <(yaml_get.py ...)`)
  - list of dicts       -> requires --field; prints that field per item, one
                           per line (e.g. `models --field path`)
  - dict                -> requires --field; prints that single field
"""
import sys

import yaml


def deep_merge(base, override):
    if isinstance(base, dict) and isinstance(override, dict):
        out = dict(base)
        for k, v in override.items():
            out[k] = deep_merge(base.get(k), v) if k in base else v
        return out
    return override


def main():
    args = sys.argv[1:]
    field = None
    if "--field" in args:
        i = args.index("--field")
        field = args[i + 1]
        del args[i : i + 2]

    *files, key = args
    merged = {}
    for f in files:
        with open(f) as fh:
            data = yaml.safe_load(fh) or {}
        merged = deep_merge(merged, data)

    node = merged
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            print(f"error: key '{key}' not found (missing '{part}')", file=sys.stderr)
            sys.exit(1)
        node = node[part]

    def render(v):
        # PyYAML gives Python bools (True/False); shell scripts compare
        # against lowercase "true"/"false", so normalize here.
        if isinstance(v, bool):
            return "true" if v else "false"
        return v

    if isinstance(node, list):
        for item in node:
            if field is not None and isinstance(item, dict):
                print(render(item[field]))
            else:
                print(render(item))
    elif isinstance(node, dict):
        if field is None:
            print(f"error: key '{key}' is a mapping, pass --field", file=sys.stderr)
            sys.exit(1)
        print(render(node[field]))
    else:
        print(render(node))


if __name__ == "__main__":
    main()
