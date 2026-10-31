"""Structured, JSON-native MCP tool replies, shared by the main and the finishing MCP servers.

FastMCP (mcp 1.30) sends a tool's return value twice: as text, through ``pydantic_core.to_json(..., fallback=str)``,
and, when the tool declares an output schema, as ``structuredContent`` through ``model_dump(mode="json")``. Two things
went wrong with that. A bare ``-> dict`` annotation silently declares no schema, so those tools sent text only. And
the structured path raises on values the text path quietly turned into strings (a NumPy integer or bool), so such a
value turned a successful call into a tool error.

``structured_tools(server)`` gives a ``@tool()`` decorator that fixes both: the tool is registered with
``structured_output=True`` (an annotation FastMCP cannot describe fails when the server is built, not at call time),
and its result is converted once by ``json_native`` before FastMCP serialises it, so the text and the structured reply
carry the same JSON.

The conversion runs after the tool has finished, side effects included (a workflow submitted, a review applied). So
it refuses as little as it can: a small NumPy array becomes a list, and only a large one is refused, with an error
that says the tool has finished and carries the result's top-level scalar fields (``workflow_dir``, ``project_path``,
``session_path`` ...), so an agent can continue instead of repeating the call.

No SDK import at module level: path checks import the servers without the SDK.
"""
import functools
import inspect
import json
from pathlib import PurePath
from typing import Any

# A NumPy array of at most this many elements is returned as a (nested) list; a larger one is refused.
ARRAY_LIST_LIMIT = 64


class _ArrayInResult(TypeError):
    def __init__(self, array):
        super().__init__(array)
        self.array = array


def _array_location(value, where="result"):
    """Where the first NumPy array too large to return sits in a result (``result['a'][0]``), for the error."""
    import numpy as np
    if isinstance(value, np.ndarray):
        return where if value.size > ARRAY_LIST_LIMIT else None
    if isinstance(value, dict):
        entries = ((f"{where}[{key!r}]", item) for key, item in value.items())
    elif isinstance(value, (list, tuple)):
        entries = ((f"{where}[{index}]", item) for index, item in enumerate(value))
    else:
        return None
    return next((found for path, item in entries if (found := _array_location(item, path))), None)


def _scalar_fields(value) -> str:
    """The top-level scalar fields of a dict result as JSON (identifiers such as ``workflow_dir``), else ``{}``."""
    import numpy as np
    import pydantic_core
    if not isinstance(value, dict):
        return "{}"
    scalars = {key: item for key, item in value.items()
               if isinstance(item, (str, int, float, type(None), np.generic, PurePath))}
    native = pydantic_core.to_jsonable_python(
        scalars, fallback=lambda item: item.item() if isinstance(item, np.generic) else str(item), inf_nan_mode="null")
    return json.dumps(native, ensure_ascii=False, skipkeys=True, default=str)


def json_native(value: Any) -> Any:
    """Return ``value`` as JSON-native data: the same JSON the text reply has always carried for ordinary values.

    Conversion is pydantic's own (``to_jsonable_python``, the counterpart of the text path's ``to_json``): paths become
    strings, tuples and sets lists, models and dataclasses dicts. NumPy scalars become ``int`` / ``float`` / ``bool``
    (the text path wrote them as strings, and the structured path refused them). NaN and infinities become ``None``,
    as in the structured reply (the text path wrote non-standard ``NaN``). Any other unknown object is shown as its
    ``str()``, exactly as the text path did. A NumPy array of at most ``ARRAY_LIST_LIMIT`` elements becomes a nested
    list of native values (the text path wrote its ``str()``). A larger one is refused: returning it is a bug in the
    tool, and ``str()`` would truncate it. Called only on a finished tool's result, so the refusal says the tool has
    finished and lists the result's top-level scalar fields.
    """
    import numpy as np
    import pydantic_core

    def fallback(item):
        if isinstance(item, np.ndarray):
            if item.size > ARRAY_LIST_LIMIT:
                raise _ArrayInResult(item)
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        return str(item)

    try:
        return pydantic_core.to_jsonable_python(value, fallback=fallback, inf_nan_mode="null")
    except _ArrayInResult as error:
        where = _array_location(value) or "result"
        shape = tuple(error.array.shape)
        limit = ARRAY_LIST_LIMIT
        fields = _scalar_fields(value)
        raise TypeError(f"工具已执行完毕，但结果的 {where} 是 NumPy 数组（形状 {shape}，超过 {limit} 个元素），"
                        f"不能作为 JSON 返回；结果的顶层标量字段：{fields}") from None


def json_reply(result: Any) -> Any:
    """A tool's reply for FastMCP: a ``CallToolResult`` keeps its content blocks and gets JSON-native structured
    content; anything else is converted by ``json_native``."""
    from mcp.types import CallToolResult
    if isinstance(result, CallToolResult):
        if result.structuredContent is None:
            return result
        return result.model_copy(update={"structuredContent": json_native(result.structuredContent)})
    return json_native(result)


def structured_tools(server):
    """``@tool()`` for ``server``: register a tool whose every reply is structured and JSON-native.

    The registered callable wraps the tool function (``functools.wraps`` keeps its name, docstring and signature, so
    the input and output schemas are the function's own); the decorator returns the function itself unchanged.
    """
    def tool():
        def register(function):
            @functools.wraps(function)
            async def replying(*args, **kwargs):
                result = function(*args, **kwargs)
                if inspect.isawaitable(result):
                    result = await result
                return json_reply(result)
            replying.json_native_reply = True
            server.tool(structured_output=True)(replying)
            return function
        return register
    return tool
