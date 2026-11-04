"""Environment-based configuration loader for entry-point scripts.

All runtime-tunable values -- data paths, hyperparameters, sampling
thresholds and feature toggles -- are read from environment variables,
optionally populated from a local ``.env`` file. This keeps the working
configuration matrix out of the public repository while leaving every knob
overridable from the command line.

Usage::

    from src.utils.env_config import env_str, env_int, env_float, env_bool

    epochs = env_int('TRAIN_EPOCHS', description='Training epochs')
    lr = env_float('TRAIN_LEARNING_RATE', description='Peak learning rate')

Semantics
---------
* ``.env`` files are plain ``KEY=VALUE`` text: blank lines and ``#`` comments
  are skipped, optional ``export `` prefixes are supported, matching
  single/double quotes are stripped, and unquoted inline comments
  (``KEY=value  # comment``) are removed.
* The loader searches the current working directory and then the repository
  root, and loads at most one file, exactly once (lazy, idempotent).
* Variables already set in the real environment always take precedence over
  ``.env`` values (12-factor precedence).
* Getters are strict by default: a missing, empty, or unparsable variable
  raises :class:`MissingConfigurationError` with an actionable message.
  Pass ``required=False, default=...`` for genuinely optional settings.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Optional

__all__ = [
    'MissingConfigurationError',
    'load_env',
    'env_str',
    'env_int',
    'env_float',
    'env_bool',
]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE_CANDIDATES = (Path.cwd() / '.env', _REPO_ROOT / '.env')

_TRUE_VALUES = {'1', 'true', 'yes', 'on'}
_FALSE_VALUES = {'0', 'false', 'no', 'off'}

_loaded = False


class MissingConfigurationError(RuntimeError):
    """Raised when a required environment variable is absent or invalid."""


def load_env(force: bool = False) -> None:
    """Load the first ``.env`` file found (cwd first, then repo root).

    Idempotent unless ``force=True``. Missing files are not an error.
    """
    global _loaded
    if _loaded and not force:
        return
    for candidate in _ENV_FILE_CANDIDATES:
        if candidate.is_file():
            _parse_env_file(candidate)
            break
    _loaded = True


def _parse_env_file(path: Path) -> None:
    """Parse a ``.env`` file into ``os.environ`` without overwriting."""
    for raw_line in path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        if line.lower().startswith('export '):
            line = line[len('export '):].lstrip()
        if '=' not in line:
            continue
        key, _, value = line.partition('=')
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        else:
            value = value.split(' #', 1)[0].rstrip()
        if key and key not in os.environ:
            os.environ[key] = value


def _resolve(
    name: str,
    converter: Callable[[str], Any],
    type_label: str,
    description: str,
    required: bool,
    default: Any,
) -> Any:
    """Read, validate, and convert one environment variable."""
    load_env()
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        if required:
            hint = f' ({description})' if description else ''
            raise MissingConfigurationError(
                f"Missing required environment variable '{name}'{hint}.\n"
                '  This project reads its runtime configuration from environment\n'
                '  variables, loaded from a local .env file that is deliberately\n'
                '  excluded from version control.\n'
                '  Fix: copy .env.example to .env and fill in the values, e.g.:\n'
                '      cp .env.example .env'
            )
        return default
    value = raw.strip()
    try:
        return converter(value)
    except ValueError:
        hint = f' ({description})' if description else ''
        raise MissingConfigurationError(
            f"Invalid value for environment variable '{name}'{hint}: "
            f'expected {type_label}, got {value!r}. '
            'Check your .env file.'
        ) from None


def env_str(
    name: str,
    description: str = '',
    *,
    required: bool = True,
    default: Optional[str] = None,
) -> Optional[str]:
    """Read a string environment variable."""
    return _resolve(name, str, 'a string', description, required, default)


def env_int(
    name: str,
    description: str = '',
    *,
    required: bool = True,
    default: Optional[int] = None,
) -> Optional[int]:
    """Read an integer environment variable."""
    return _resolve(name, int, 'an integer', description, required, default)


def env_float(
    name: str,
    description: str = '',
    *,
    required: bool = True,
    default: Optional[float] = None,
) -> Optional[float]:
    """Read a float environment variable."""
    return _resolve(name, float, 'a number', description, required, default)


def env_bool(
    name: str,
    description: str = '',
    *,
    required: bool = True,
    default: Optional[bool] = None,
) -> Optional[bool]:
    """Read a boolean environment variable (1/true/yes/on vs 0/false/no/off)."""

    def to_bool(raw: str) -> bool:
        lowered = raw.strip().lower()
        if lowered in _TRUE_VALUES:
            return True
        if lowered in _FALSE_VALUES:
            return False
        raise ValueError(f'not a boolean: {raw!r}')

    return _resolve(name, to_bool, 'a boolean (true/false)', description, required, default)
