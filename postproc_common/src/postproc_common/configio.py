import os
from pathlib import Path

import yaml


CONFIG_FILENAME = "postproc_config.yaml"


def _normalize_config(data):
    if data is None:
        data = {}

    if not isinstance(data, dict):
        raise ValueError("postproc_config.yaml must contain a top-level mapping")

    aliases = data.get("aliases", {})
    defaults = data.get("defaults", {})

    if aliases is None:
        aliases = {}
    if defaults is None:
        defaults = {}

    if not isinstance(aliases, dict):
        raise ValueError("'aliases' must be a mapping")
    if not isinstance(defaults, dict):
        raise ValueError("'defaults' must be a mapping")

    return {
        "aliases": {str(k): str(v) for k, v in aliases.items()},
        "defaults": defaults,
    }


def load_postproc_config(path):
    path = Path(path).expanduser()
    data = yaml.safe_load(path.read_text())
    config = _normalize_config(data)
    config["_config_path"] = str(path)
    return config


def _candidate_dirs(paths=None, cwd=None):
    seen = set()
    candidates = []

    if cwd is None:
        cwd = Path.cwd()
    cwd = Path(cwd).expanduser().resolve()

    candidates.append(cwd)

    if paths:
        path_objs = [Path(p).expanduser().resolve() for p in paths]
        common_root = Path(
            os.path.commonpath([str(p.parent if p.is_file() else p) for p in path_objs])
        )
        if common_root not in candidates:
            candidates.append(common_root)

    out = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)

    return out


def find_postproc_config(paths=None, cwd=None):
    for directory in _candidate_dirs(paths=paths, cwd=cwd):
        candidate = directory / CONFIG_FILENAME
        if candidate.is_file():
            return candidate
    return None


def load_config_for_paths(paths=None, cwd=None):
    config_path = find_postproc_config(paths=paths, cwd=cwd)
    if config_path is None:
        return {
            "aliases": {},
            "defaults": {},
            "_config_path": None,
        }
    return load_postproc_config(config_path)


def file_key_for_path(path):
    path = Path(path).expanduser()
    return path.parent.name if path.suffix else path.name


def alias_for_key(file_key, config):
    if config is None:
        return None
    return config.get("aliases", {}).get(str(file_key))


def alias_for_path(path, config):
    return alias_for_key(file_key_for_path(path), config)


def label_for_path(path, config):
    key = file_key_for_path(path)
    alias = alias_for_key(key, config)
    return alias if alias is not None else key


def default_from_config(config, *keys, fallback=None):
    node = config.get("defaults", {}) if config else {}
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return fallback
        node = node[key]
    return node
