"""Core argument coercion implementation."""

from __future__ import annotations

import ast
import inspect
import json
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Union, get_args, get_origin


class CoercionError(Exception):
    """Raised in strict mode when a coercion fails.

    Attributes:
        name: argument name being coerced
        expected: expected type representation
        got: original value
        original_error: underlying exception, if any
    """

    def __init__(
        self,
        name: str,
        expected: str,
        got: Any,
        original_error: Exception | None = None,
    ):
        self.name = name
        self.expected = expected
        self.got = got
        self.original_error = original_error
        msg = f"failed to coerce arg {name!r} to {expected}: got {type(got).__name__}={got!r}"
        if original_error is not None:
            msg += f" ({type(original_error).__name__}: {original_error})"
        super().__init__(msg)


@dataclass(frozen=True)
class Coercion:
    """Record of a single successful coercion applied to an arg."""

    name: str
    from_type: str
    to_type: str
    original: Any
    coerced: Any


@dataclass(frozen=True)
class CoercionFailure:
    """Record of a coercion attempt that failed (value kept as raw)."""

    name: str
    expected: str
    got: Any
    error_message: str


@dataclass
class CoercionResult:
    """Output of `coerce_args`. `.args` has the coerced/passed-through dict."""

    args: dict[str, Any]
    coercions: list[Coercion] = field(default_factory=list)
    failures: list[CoercionFailure] = field(default_factory=list)


_BOOL_TRUE = {"true", "yes", "1", "on"}
_BOOL_FALSE = {"false", "no", "0", "off"}


def _type_name(t: Any) -> str:
    """Best-effort short name for a type, generic alias, or python value."""
    if t is None or t is type(None):
        return "None"
    if isinstance(t, type):
        return t.__name__
    return str(t)


def _str_to_bool(s: str) -> bool:
    lowered = s.strip().lower()
    if lowered in _BOOL_TRUE:
        return True
    if lowered in _BOOL_FALSE:
        return False
    raise ValueError(f"not a recognized bool string: {s!r}")


def _parse_collection(s: str) -> Any:
    """Try JSON first, then python literal."""
    s = s.strip()
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"not valid JSON or python literal: {s!r}") from e


def _normalize_target_type(t: Any) -> Any:
    """Normalize a string target like 'int' to the actual type."""
    if isinstance(t, str):
        lookup = {
            "int": int,
            "float": float,
            "bool": bool,
            "str": str,
            "list": list,
            "dict": dict,
        }
        if t in lookup:
            return lookup[t]
    return t


def _is_optional(t: Any) -> tuple[bool, Any]:
    """If t is Optional[X] (Union[X, None] or X | None), return (True, X).
    Otherwise (False, t)."""
    origin = get_origin(t)
    # typing.Union and types.UnionType (PEP 604 `X | None`) both work via get_args
    if origin is Union or (
        hasattr(types, "UnionType") and isinstance(t, types.UnionType)
    ):
        args = [a for a in get_args(t) if a is not type(None)]
        if len(args) == 1 and type(None) in get_args(t):
            return True, args[0]
    return False, t


def _coerce_value(
    name: str,
    value: Any,
    target: Any,
    custom_coercers: dict[tuple[type, type], Callable[[Any], Any]] | None,
) -> tuple[Any, Coercion | None, CoercionFailure | None]:
    """Coerce a single value to target type.

    Returns (coerced_value, coercion_record, failure_record). Either
    coercion_record or failure_record is None (both may be None if no
    coercion was needed). On failure, returns the original value plus a
    CoercionFailure record."""

    target = _normalize_target_type(target)

    # None handling for Optional[T]
    optional, inner = _is_optional(target)
    if optional:
        if value is None:
            return None, None, None
        target = inner
        target = _normalize_target_type(target)

    # If no target type, pass through unchanged
    if target is None or target is Any:
        return value, None, None

    origin = get_origin(target)
    type_args = get_args(target)
    # base type for instanceof checks (List[int] -> list, Dict[str, int] -> dict)
    base_type = origin if origin is not None else target

    original_type = type(value)
    expected_name = _type_name(target)

    # custom coercer wins over built-in
    if custom_coercers and isinstance(base_type, type):
        key = (original_type, base_type)
        if key in custom_coercers:
            try:
                coerced = custom_coercers[key](value)
                return (
                    coerced,
                    Coercion(
                        name=name,
                        from_type=original_type.__name__,
                        to_type=expected_name,
                        original=value,
                        coerced=coerced,
                    ),
                    None,
                )
            except Exception as e:
                return value, None, CoercionFailure(
                    name=name,
                    expected=expected_name,
                    got=value,
                    error_message=f"{type(e).__name__}: {e}",
                )

    # Already the right base type - may still need elementwise coercion for generics
    if isinstance(base_type, type) and isinstance(value, base_type):
        # bool is a subclass of int in python, so guard against accepting True as int
        if base_type is int and isinstance(value, bool):
            pass  # fall through and try to coerce (bool -> int is fine but record it)
        elif base_type is list and type_args:
            return _coerce_list_elements(name, value, type_args[0], custom_coercers)
        elif base_type is dict and len(type_args) == 2:
            return _coerce_dict_elements(
                name, value, type_args[0], type_args[1], custom_coercers
            )
        else:
            return value, None, None

    # str -> X
    if isinstance(value, str):
        if base_type is int:
            try:
                coerced = int(value.strip())
            except ValueError as e:
                return value, None, CoercionFailure(
                    name=name,
                    expected=expected_name,
                    got=value,
                    error_message=str(e),
                )
            return (
                coerced,
                Coercion(
                    name=name,
                    from_type="str",
                    to_type=expected_name,
                    original=value,
                    coerced=coerced,
                ),
                None,
            )
        if base_type is float:
            try:
                coerced = float(value.strip())
            except ValueError as e:
                return value, None, CoercionFailure(
                    name=name,
                    expected=expected_name,
                    got=value,
                    error_message=str(e),
                )
            return (
                coerced,
                Coercion(
                    name=name,
                    from_type="str",
                    to_type=expected_name,
                    original=value,
                    coerced=coerced,
                ),
                None,
            )
        if base_type is bool:
            try:
                coerced = _str_to_bool(value)
            except ValueError as e:
                return value, None, CoercionFailure(
                    name=name,
                    expected=expected_name,
                    got=value,
                    error_message=str(e),
                )
            return (
                coerced,
                Coercion(
                    name=name,
                    from_type="str",
                    to_type=expected_name,
                    original=value,
                    coerced=coerced,
                ),
                None,
            )
        if base_type is list:
            try:
                parsed = _parse_collection(value)
            except ValueError as e:
                return value, None, CoercionFailure(
                    name=name,
                    expected=expected_name,
                    got=value,
                    error_message=str(e),
                )
            if not isinstance(parsed, list):
                return value, None, CoercionFailure(
                    name=name,
                    expected=expected_name,
                    got=value,
                    error_message=f"parsed value is not a list: {type(parsed).__name__}",
                )
            # elementwise coerce if generic
            if type_args:
                inner_target = type_args[0]
                coerced_list, _, _ = _coerce_list_elements(
                    name, parsed, inner_target, custom_coercers
                )
                # one outer coercion record (str -> list); elements may already be right
                return (
                    coerced_list,
                    Coercion(
                        name=name,
                        from_type="str",
                        to_type=expected_name,
                        original=value,
                        coerced=coerced_list,
                    ),
                    None,
                )
            return (
                parsed,
                Coercion(
                    name=name,
                    from_type="str",
                    to_type=expected_name,
                    original=value,
                    coerced=parsed,
                ),
                None,
            )
        if base_type is dict:
            try:
                parsed = _parse_collection(value)
            except ValueError as e:
                return value, None, CoercionFailure(
                    name=name,
                    expected=expected_name,
                    got=value,
                    error_message=str(e),
                )
            if not isinstance(parsed, dict):
                return value, None, CoercionFailure(
                    name=name,
                    expected=expected_name,
                    got=value,
                    error_message=f"parsed value is not a dict: {type(parsed).__name__}",
                )
            if len(type_args) == 2:
                coerced_dict, _, _ = _coerce_dict_elements(
                    name, parsed, type_args[0], type_args[1], custom_coercers
                )
                return (
                    coerced_dict,
                    Coercion(
                        name=name,
                        from_type="str",
                        to_type=expected_name,
                        original=value,
                        coerced=coerced_dict,
                    ),
                    None,
                )
            return (
                parsed,
                Coercion(
                    name=name,
                    from_type="str",
                    to_type=expected_name,
                    original=value,
                    coerced=parsed,
                ),
                None,
            )

    # int -> float
    if isinstance(value, int) and not isinstance(value, bool) and base_type is float:
        coerced = float(value)
        return (
            coerced,
            Coercion(
                name=name,
                from_type="int",
                to_type=expected_name,
                original=value,
                coerced=coerced,
            ),
            None,
        )

    # int / float -> str
    if isinstance(value, (int, float)) and not isinstance(value, bool) and base_type is str:
        coerced = str(value)
        return (
            coerced,
            Coercion(
                name=name,
                from_type=original_type.__name__,
                to_type=expected_name,
                original=value,
                coerced=coerced,
            ),
            None,
        )

    # bool -> int (record it explicitly)
    if isinstance(value, bool) and base_type is int:
        coerced = int(value)
        return (
            coerced,
            Coercion(
                name=name,
                from_type="bool",
                to_type=expected_name,
                original=value,
                coerced=coerced,
            ),
            None,
        )

    # bool -> str
    if isinstance(value, bool) and base_type is str:
        coerced = str(value)
        return (
            coerced,
            Coercion(
                name=name,
                from_type="bool",
                to_type=expected_name,
                original=value,
                coerced=coerced,
            ),
            None,
        )

    # Nothing matched
    return value, None, CoercionFailure(
        name=name,
        expected=expected_name,
        got=value,
        error_message=f"no coercion path from {original_type.__name__} to {expected_name}",
    )


def _coerce_list_elements(
    name: str,
    value: list,
    inner_target: Any,
    custom_coercers: dict[tuple[type, type], Callable[[Any], Any]] | None,
) -> tuple[list, Coercion | None, CoercionFailure | None]:
    """Coerce every element of a list to the inner target type."""
    out: list[Any] = []
    any_failure: CoercionFailure | None = None
    for i, elem in enumerate(value):
        coerced, _, failure = _coerce_value(
            f"{name}[{i}]", elem, inner_target, custom_coercers
        )
        out.append(coerced)
        if failure is not None and any_failure is None:
            any_failure = failure
    if any_failure is not None:
        return value, None, any_failure
    return out, None, None


def _coerce_dict_elements(
    name: str,
    value: dict,
    key_target: Any,
    val_target: Any,
    custom_coercers: dict[tuple[type, type], Callable[[Any], Any]] | None,
) -> tuple[dict, Coercion | None, CoercionFailure | None]:
    """Coerce keys and values of a dict to the given target types."""
    out: dict[Any, Any] = {}
    any_failure: CoercionFailure | None = None
    for k, v in value.items():
        coerced_k, _, k_fail = _coerce_value(
            f"{name}.<key>", k, key_target, custom_coercers
        )
        coerced_v, _, v_fail = _coerce_value(
            f"{name}.{k}", v, val_target, custom_coercers
        )
        if k_fail is not None and any_failure is None:
            any_failure = k_fail
        if v_fail is not None and any_failure is None:
            any_failure = v_fail
        out[coerced_k] = coerced_v
    if any_failure is not None:
        return value, None, any_failure
    return out, None, None


def _schema_from_callable(target: Callable[..., Any]) -> dict[str, Any]:
    """Build a name -> type-hint dict from a callable's signature."""
    sig = inspect.signature(target)
    hints: dict[str, Any] = {}
    try:
        # works for normal functions and bound methods
        from typing import get_type_hints

        resolved = get_type_hints(target)
    except Exception:
        resolved = {}
    for pname, param in sig.parameters.items():
        if pname == "self" or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if pname in resolved:
            hints[pname] = resolved[pname]
        elif param.annotation is not inspect.Parameter.empty:
            hints[pname] = param.annotation
    return hints


def coerce_args(
    raw: dict[str, Any],
    target: dict[str, Any] | Callable[..., Any],
    *,
    strict: bool = False,
    custom_coercers: dict[tuple[type, type], Callable[[Any], Any]] | None = None,
) -> CoercionResult:
    """Coerce a dict of raw LLM-supplied args to the expected types.

    `target` may be a schema dict mapping arg name -> python type (or the
    string aliases "int", "float", "bool", "str", "list", "dict"), or a
    callable whose signature/type hints are inspected.

    Extra args in `raw` not present in the schema are passed through unchanged.
    Missing args are simply absent from the output (no defaulting).

    `strict=True` raises CoercionError on the first failure instead of
    recording it. `custom_coercers` is a mapping of (source_type, target_type)
    tuples to callables that produce the coerced value; they take precedence
    over built-in paths.
    """
    if not isinstance(raw, dict):
        raise TypeError("raw args must be a dict")

    if callable(target) and not isinstance(target, dict):
        schema = _schema_from_callable(target)
    elif isinstance(target, dict):
        schema = dict(target)
    else:
        raise TypeError("target must be a dict schema or a callable")

    coercions: list[Coercion] = []
    failures: list[CoercionFailure] = []
    out: dict[str, Any] = {}

    for name, value in raw.items():
        if name not in schema:
            # extra arg: pass through unchanged
            out[name] = value
            continue
        target_type = schema[name]
        coerced, record, failure = _coerce_value(
            name, value, target_type, custom_coercers
        )
        if failure is not None:
            if strict:
                raise CoercionError(
                    name=name,
                    expected=_type_name(_normalize_target_type(target_type)),
                    got=value,
                )
            failures.append(failure)
            out[name] = value  # keep raw value
        else:
            out[name] = coerced
            if record is not None:
                coercions.append(record)

    return CoercionResult(args=out, coercions=coercions, failures=failures)
