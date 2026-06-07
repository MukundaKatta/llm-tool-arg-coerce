"""Tests for llm_tool_arg_coerce.coerce_args."""

from __future__ import annotations

from typing import Optional

import pytest

from llm_tool_arg_coerce import (
    Coercion,
    CoercionError,
    CoercionFailure,
    CoercionResult,
    coerce_args,
)

# ---------- str -> primitive coercions ----------


def test_str_to_int_basic():
    r = coerce_args({"n": "5"}, {"n": int})
    assert r.args == {"n": 5}
    assert len(r.coercions) == 1
    c = r.coercions[0]
    assert c.name == "n"
    assert c.from_type == "str"
    assert c.to_type == "int"
    assert c.original == "5"
    assert c.coerced == 5
    assert r.failures == []


def test_str_to_int_with_whitespace_and_negative():
    r = coerce_args({"a": " 5 ", "b": "-3"}, {"a": int, "b": int})
    assert r.args == {"a": 5, "b": -3}


def test_str_to_float_basic_and_scientific():
    r = coerce_args({"pi": "3.14", "big": "1e10"}, {"pi": float, "big": float})
    assert r.args["pi"] == 3.14
    assert r.args["big"] == 1e10
    assert len(r.coercions) == 2


def test_str_to_bool_all_aliases():
    schema = {"a": bool, "b": bool, "c": bool, "d": bool, "e": bool, "f": bool}
    raw = {"a": "True", "b": "yes", "c": "ON", "d": "false", "e": "no", "f": "OFF"}
    r = coerce_args(raw, schema)
    assert r.args == {
        "a": True,
        "b": True,
        "c": True,
        "d": False,
        "e": False,
        "f": False,
    }


def test_str_to_bool_with_1_and_0():
    r = coerce_args({"a": "1", "b": "0"}, {"a": bool, "b": bool})
    assert r.args == {"a": True, "b": False}


def test_str_to_list_from_json():
    r = coerce_args({"tags": '["x", "y"]'}, {"tags": list})
    assert r.args == {"tags": ["x", "y"]}
    assert r.coercions[0].to_type == "list"


def test_str_to_list_from_python_literal():
    # Single-quoted strings are not valid JSON but are valid python literal
    r = coerce_args({"tags": "['x', 'y']"}, {"tags": list})
    assert r.args == {"tags": ["x", "y"]}


def test_str_to_dict_from_json():
    r = coerce_args({"meta": '{"a": 1}'}, {"meta": dict})
    assert r.args == {"meta": {"a": 1}}


def test_str_to_dict_from_python_literal():
    r = coerce_args({"meta": "{'a': 1}"}, {"meta": dict})
    assert r.args == {"meta": {"a": 1}}


# ---------- numeric promotions ----------


def test_int_to_float_promotion():
    r = coerce_args({"x": 5}, {"x": float})
    assert r.args == {"x": 5.0}
    assert isinstance(r.args["x"], float)
    assert r.coercions[0].from_type == "int"
    assert r.coercions[0].to_type == "float"


def test_int_to_str():
    r = coerce_args({"id": 42}, {"id": str})
    assert r.args == {"id": "42"}


def test_float_to_str():
    r = coerce_args({"x": 1.5}, {"x": str})
    assert r.args == {"x": "1.5"}


def test_bool_to_int_is_recorded():
    r = coerce_args({"x": True}, {"x": int})
    assert r.args == {"x": 1}
    assert isinstance(r.args["x"], int)
    assert len(r.coercions) == 1
    assert r.coercions[0].from_type == "bool"
    assert r.coercions[0].to_type == "int"


def test_bool_to_str():
    r = coerce_args({"x": False}, {"x": str})
    assert r.args == {"x": "False"}
    assert r.coercions[0].from_type == "bool"
    assert r.coercions[0].to_type == "str"


# ---------- pass-through / no coercion needed ----------


def test_already_correct_type_no_coercion_recorded():
    r = coerce_args({"n": 5, "s": "hi"}, {"n": int, "s": str})
    assert r.args == {"n": 5, "s": "hi"}
    assert r.coercions == []
    assert r.failures == []


def test_extra_args_pass_through_unchanged():
    r = coerce_args({"n": "5", "extra": object()}, {"n": int})
    assert r.args["n"] == 5
    # extra arg present and untouched
    assert "extra" in r.args


def test_missing_args_are_simply_absent():
    r = coerce_args({}, {"n": int, "s": str})
    assert r.args == {}
    assert r.coercions == []
    assert r.failures == []


# ---------- string-alias schema ----------


def test_string_alias_schema():
    r = coerce_args(
        {"n": "5", "b": "true", "lst": "[1, 2]"},
        {"n": "int", "b": "bool", "lst": "list"},
    )
    assert r.args == {"n": 5, "b": True, "lst": [1, 2]}


# ---------- function-signature mode ----------


def test_function_signature_mode():
    def my_tool(count: int, active: bool, tags: list) -> None:
        pass

    raw = {"count": "5", "active": "true", "tags": '["a","b"]'}
    r = coerce_args(raw, my_tool)
    assert r.args == {"count": 5, "active": True, "tags": ["a", "b"]}


def test_function_signature_skips_self_and_varargs():
    class C:
        def m(self, *args: int, x: int = 0, **kw: int) -> None:
            pass

    raw = {"x": "7"}
    r = coerce_args(raw, C().m)
    assert r.args == {"x": 7}


# ---------- Optional[T] / generics ----------


def test_optional_none_passes_through():
    # intentionally exercising typing.Optional API (not the PEP 604 alias)
    r = coerce_args({"x": None}, {"x": Optional[int]})  # noqa: UP045
    assert r.args == {"x": None}
    assert r.coercions == []


def test_optional_inner_is_still_coerced():
    # intentionally exercising typing.Optional API (not the PEP 604 alias)
    r = coerce_args({"x": "5"}, {"x": Optional[int]})  # noqa: UP045
    assert r.args == {"x": 5}


def test_pep604_optional_none_passes_through():
    r = coerce_args({"x": None}, {"x": int | None})
    assert r.args == {"x": None}


def test_list_of_int_coerces_elements():
    r = coerce_args({"xs": [1, "2", "3"]}, {"xs": list[int]})
    assert r.args == {"xs": [1, 2, 3]}


def test_list_of_int_from_json_string_coerces_elements():
    r = coerce_args({"xs": '["1", "2"]'}, {"xs": list[int]})
    assert r.args == {"xs": [1, 2]}


def test_dict_str_int_coerces_values():
    r = coerce_args({"d": {"a": "1", "b": "2"}}, {"d": dict[str, int]})
    assert r.args == {"d": {"a": 1, "b": 2}}


def test_dict_str_int_from_json_string_coerces_values():
    r = coerce_args({"d": '{"a": "1", "b": "2"}'}, {"d": dict[str, int]})
    assert r.args == {"d": {"a": 1, "b": 2}}
    assert r.coercions[0].from_type == "str"
    assert r.coercions[0].to_type == "dict[str, int]"


def test_nested_list_of_list_of_int():
    r = coerce_args({"m": [["1", "2"], ["3"]]}, {"m": list[list[int]]})
    assert r.args == {"m": [[1, 2], [3]]}


def test_pep604_optional_inner_is_still_coerced():
    r = coerce_args({"x": "5"}, {"x": int | None})
    assert r.args == {"x": 5}


def test_optional_generic_from_json_string():
    # intentionally exercising typing.Optional API (not the PEP 604 alias)
    r = coerce_args({"x": '["1", "2"]'}, {"x": Optional[list[int]]})  # noqa: UP045
    assert r.args == {"x": [1, 2]}


def test_none_to_plain_type_is_failure():
    r = coerce_args({"x": None}, {"x": int})
    assert r.args == {"x": None}
    assert r.failures and r.failures[0].name == "x"


# ---------- failures ----------


def test_invalid_int_string_recorded_as_failure():
    r = coerce_args({"n": "not-a-number"}, {"n": int})
    assert r.args == {"n": "not-a-number"}  # raw kept
    assert r.coercions == []
    assert len(r.failures) == 1
    f = r.failures[0]
    assert isinstance(f, CoercionFailure)
    assert f.name == "n"
    assert f.expected == "int"
    assert f.got == "not-a-number"
    assert f.error_message  # non-empty


def test_invalid_bool_string_recorded_as_failure():
    r = coerce_args({"b": "maybe"}, {"b": bool})
    assert r.failures and r.failures[0].name == "b"
    assert r.args == {"b": "maybe"}


def test_garbage_list_string_recorded_as_failure():
    r = coerce_args({"xs": "not a list"}, {"xs": list})
    assert r.failures and r.failures[0].name == "xs"
    assert r.args == {"xs": "not a list"}


def test_list_string_that_parses_to_dict_is_failure_for_list():
    r = coerce_args({"xs": '{"a": 1}'}, {"xs": list})
    assert r.failures
    assert r.args == {"xs": '{"a": 1}'}


def test_no_path_from_object_to_int_is_failure():
    r = coerce_args({"n": object()}, {"n": int})
    assert r.failures


def test_list_element_failure_keeps_raw_list():
    r = coerce_args({"xs": ["1", "x"]}, {"xs": list[int]})
    assert r.args == {"xs": ["1", "x"]}  # raw kept on element failure
    assert r.failures and r.failures[0].name == "xs[1]"
    assert r.coercions == []


# ---------- strict mode ----------


def test_strict_mode_raises_on_failure():
    with pytest.raises(CoercionError) as ei:
        coerce_args({"n": "nope"}, {"n": int}, strict=True)
    assert ei.value.name == "n"
    assert ei.value.expected == "int"
    assert ei.value.got == "nope"


def test_strict_mode_passes_when_all_coercions_succeed():
    r = coerce_args({"n": "5"}, {"n": int}, strict=True)
    assert r.args == {"n": 5}


# ---------- custom coercers ----------


def test_custom_coercer_wins_over_builtin():
    # Build a custom (str, int) coercer that returns -999 for any input
    r = coerce_args(
        {"n": "5"},
        {"n": int},
        custom_coercers={(str, int): lambda s: -999},
    )
    assert r.args == {"n": -999}
    # Coercion record should reflect the custom path
    assert r.coercions and r.coercions[0].to_type == "int"


def test_custom_coercer_for_new_type():
    class Tag:
        def __init__(self, name: str) -> None:
            self.name = name

        def __eq__(self, other: object) -> bool:
            return isinstance(other, Tag) and self.name == other.name

    r = coerce_args(
        {"t": "alpha"},
        {"t": Tag},
        custom_coercers={(str, Tag): Tag},
    )
    assert r.args == {"t": Tag("alpha")}


def test_custom_coercer_failure_recorded():
    def explode(_s: str) -> int:
        raise RuntimeError("nope")

    r = coerce_args(
        {"n": "5"},
        {"n": int},
        custom_coercers={(str, int): explode},
    )
    assert r.failures and "nope" in r.failures[0].error_message
    assert r.args == {"n": "5"}  # raw kept


def test_custom_coercer_overrides_already_correct_type():
    # value is already an int but the custom (int, int) coercer must still win
    r = coerce_args(
        {"n": 5},
        {"n": int},
        custom_coercers={(int, int): lambda x: x * 10},
    )
    assert r.args == {"n": 50}
    assert r.coercions and r.coercions[0].to_type == "int"


def test_custom_coercer_does_not_apply_on_type_mismatch():
    # value is an int, custom key is (str, int) -> custom should not fire,
    # and no built-in coercion is needed, so value passes through unchanged
    r = coerce_args(
        {"n": 5},
        {"n": int},
        custom_coercers={(str, int): lambda s: -999},
    )
    assert r.args == {"n": 5}
    assert r.coercions == []
    assert r.failures == []


def test_strict_mode_raises_on_custom_coercer_failure():
    def explode(_s: str) -> int:
        raise RuntimeError("boom")

    with pytest.raises(CoercionError) as ei:
        coerce_args(
            {"n": "5"},
            {"n": int},
            strict=True,
            custom_coercers={(str, int): explode},
        )
    assert ei.value.name == "n"


# ---------- result shape ----------


def test_result_is_coercion_result_dataclass():
    r = coerce_args({}, {})
    assert isinstance(r, CoercionResult)
    assert r.args == {}
    assert r.coercions == []
    assert r.failures == []


def test_coercion_record_fields_populated():
    r = coerce_args({"n": "5"}, {"n": int})
    c = r.coercions[0]
    assert isinstance(c, Coercion)
    assert c.name == "n"
    assert c.original == "5"
    assert c.coerced == 5


# ---------- input validation ----------


def test_raw_must_be_dict():
    with pytest.raises(TypeError):
        coerce_args("not a dict", {})  # type: ignore[arg-type]


def test_target_must_be_dict_or_callable():
    with pytest.raises(TypeError):
        coerce_args({}, 123)  # type: ignore[arg-type]
