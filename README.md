# llm-tool-arg-coerce

[![PyPI](https://img.shields.io/pypi/v/llm-tool-arg-coerce.svg)](https://pypi.org/project/llm-tool-arg-coerce/)
[![Python](https://img.shields.io/pypi/pyversions/llm-tool-arg-coerce.svg)](https://pypi.org/project/llm-tool-arg-coerce/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Coerce LLM-generated tool args to expected types.**

Models often hand you tool args with the wrong type: `"5"` instead of `5`,
`"true"` instead of `True`, `'["a","b"]'` instead of `["a","b"]`. This is
a small zero-dependency library that walks a raw args dict against a target
schema (or a function signature) and returns the coerced values, plus a
record of every coercion that fired.

You decide what to do with that record: log it, warn the model, ignore it,
or escalate. The library never rewrites silently.

## Install

```bash
pip install llm-tool-arg-coerce
```

## Use

Schema-based:

```python
from llm_tool_arg_coerce import coerce_args

raw = {"count": "5", "active": "true", "tags": '["a","b"]'}
schema = {"count": int, "active": bool, "tags": list}

result = coerce_args(raw, schema)
result.args        # {"count": 5, "active": True, "tags": ["a", "b"]}
result.coercions   # 3 Coercion records (one per arg coerced)
result.failures    # []
```

Function-signature based (type hints + `inspect.signature`):

```python
def my_tool(count: int, active: bool, tags: list) -> None: ...

result = coerce_args(raw, my_tool)
```

Strict mode (raise on the first failure instead of recording it):

```python
from llm_tool_arg_coerce import CoercionError

try:
    coerce_args({"count": "not-a-number"}, {"count": int}, strict=True)
except CoercionError as e:
    print(e.name, e.expected, e.got)
```

Custom coercer (overrides the built-in path for that `(from_type, to_type)`):

```python
from datetime import date

def parse_date(s: str) -> date:
    return date.fromisoformat(s)

result = coerce_args(
    {"due": "2026-05-24"},
    {"due": date},
    custom_coercers={(str, date): parse_date},
)
result.args["due"]   # datetime.date(2026, 5, 24)
```

## What it handles

- `str` -> `int`, `float`, `bool`, `list`, `dict`
- `int` -> `float`, `str`
- `float` -> `str`
- `bool` -> `int`, `str`
- `Optional[T]` (None passes through)
- `List[T]` and `Dict[K, V]` (elementwise coercion)
- JSON strings for `list` / `dict`; Python literal fallback (`ast.literal_eval`)
- Schema as a dict, or as any callable whose signature you can inspect
- String aliases in the schema: `"int"`, `"float"`, `"bool"`, `"str"`,
  `"list"`, `"dict"`
- Extra args (not in schema) pass through unchanged

## What it does NOT do

- It does not validate value ranges, regexes, or business rules. For that,
  pair it with [`agentvet`](https://github.com/MukundaKatta/agentvet)
  which validates tool args after coercion and returns an LLM-friendly
  retry hint on rejection.
- It does not enforce structured output on the raw model response. For
  that, use [`agentcast`](https://github.com/MukundaKatta/agentcast)
  which loops repair-validate-retry on JSON output before the args dict
  even reaches this library.
- It does not call any LLM, and has no async or HTTP dependencies.

## License

MIT
