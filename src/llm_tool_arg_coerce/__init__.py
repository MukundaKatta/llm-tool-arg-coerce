"""llm-tool-arg-coerce - coerce LLM-generated tool args to expected types.

LLMs frequently return tool args as strings ("5" instead of int 5, "true"
instead of bool True, JSON-string lists instead of arrays). This library
coerces them based on a target schema or function signature, returning the
coerced args plus a list of coercions applied so callers can decide whether
to log or warn.

    from llm_tool_arg_coerce import coerce_args

    raw = {"count": "5", "active": "true", "tags": '["a","b"]'}
    schema = {"count": int, "active": bool, "tags": list}
    result = coerce_args(raw, schema)
    result.args        # {"count": 5, "active": True, "tags": ["a", "b"]}
    result.coercions   # 3 Coercion records
    result.failures    # []

Sibling to `agentvet` (validate the coerced args) and `agentcast`
(structured-output enforcer for the model JSON itself).
"""

from llm_tool_arg_coerce.coerce import (
    Coercion,
    CoercionError,
    CoercionFailure,
    CoercionResult,
    coerce_args,
)

__version__ = "0.1.0"

__all__ = [
    "Coercion",
    "CoercionError",
    "CoercionFailure",
    "CoercionResult",
    "__version__",
    "coerce_args",
]
