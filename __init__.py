"""forget-context plugin — agent-controlled relief for context-flooding tool results.

Shape: a general user plugin (``~/.hermes/plugins/forget-context/``) that registers
a context engine via ``ctx.register_context_engine()``. The user-plugins dir lives
outside the Hermes checkout, so ``hermes update`` (a git pull of the repo) cannot
touch this file.

Update-proofing rules applied here (keep them on future edits):
1. SUBCLASS, don't copy: :class:`ForgetCompressor` inherits everything from the
   built-in ``ContextCompressor`` and overrides only ``name``,
   ``get_tool_schemas`` and ``handle_tool_call`` (plus ``__deepcopy__``, which the
   host explicitly recommends for plugin engines). Upstream changes to
   thresholds, pruning, summarization and future ABC methods flow through.
2. Depend only on stable seams: ``agent.context_compressor.ContextCompressor``,
   the ``ContextEngine`` tool contract (``get_tool_schemas`` /
   ``handle_tool_call(..., messages=<live transcript>)``) and
   ``ctx.register_context_engine``. No private ``agent.*`` helpers.
3. Fail open: ``register()`` and ``handle_tool_call()`` never raise; every
   failure is a JSON error string, so a bug here degrades to "tool reported an
   error", never a broken session.
4. Transcript surgery is replace-only: tool messages keep their ``role`` and
   ``tool_call_id`` (the assistant ``tool_calls`` pairing and role alternation
   stay valid); only the oversized ``content`` is swapped for a short marker.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Tuple

from agent.context_compressor import ContextCompressor

logger = logging.getLogger(__name__)

ENGINE_NAME = "forget"

# Marker prefix for replaced results. list/forget both key off it, so keep it stable.
FORGOTTEN_PREFIX = "[forgotten by forget_tool_result"

# Safety floor: results smaller than this are left alone unless the caller passes
# an explicit min_chars=0. Prevents nuking small but critical outputs by accident.
DEFAULT_MIN_CHARS = 2000


def _content_chars(content: Any) -> int:
    """Best-effort char count for a tool message body (str or content-block list)."""
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, dict):
                text = block.get("text", "")
                total += len(text) if isinstance(text, str) else len(str(text))
            elif isinstance(block, str):
                total += len(block)
        return total
    return len(str(content))


def _is_forgotten(content: Any) -> bool:
    if isinstance(content, str):
        return content.startswith(FORGOTTEN_PREFIX)
    if isinstance(content, list) and content and isinstance(content[0], dict):
        text = content[0].get("text", "")
        return isinstance(text, str) and text.startswith(FORGOTTEN_PREFIX)
    return False


def _index_tool_results(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Scan the transcript; return one entry per tool result message.

    Each entry: {index, tool_call_id, tool_name (or ""), chars, forgotten}.
    tool_name is resolved via the assistant message's tool_calls pairing.
    """
    id_to_name: Dict[str, str] = {}
    for m in messages:
        if not isinstance(m, dict):
            continue
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                cid, fname = tc.get("id"), fn.get("name") if isinstance(fn, dict) else None
                if cid and fname:
                    id_to_name[cid] = fname
    entries = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or m.get("role") != "tool":
            continue
        cid = m.get("tool_call_id") or m.get("id") or ""
        content = m.get("content", "")
        entries.append({
            "index": i,
            "tool_call_id": cid,
            "tool_name": id_to_name.get(cid, ""),
            "chars": _content_chars(content),
            "forgotten": _is_forgotten(content),
        })
    return entries


def _replace_content(message: Dict[str, Any], placeholder: str) -> None:
    """In-place, shape-preserving content swap (str stays str, block-list stays list)."""
    if isinstance(message.get("content"), list):
        message["content"] = [{"type": "text", "text": placeholder}]
    else:
        message["content"] = placeholder


class ForgetCompressor(ContextCompressor):
    """Built-in compressor + agent-callable forgetting. Nothing else overridden."""

    @property
    def name(self) -> str:  # must match `context.engine` in config.yaml
        return ENGINE_NAME

    def __deepcopy__(self, memo: Dict[int, Any]) -> "ForgetCompressor":
        # The host deep-copies the registered singleton per agent; rebuild fresh
        # base state instead of copying (locks/DB handles must never cross
        # agents). The host calls update_model() right after selection, so dummy
        # construction values here are immediately corrected.
        fresh = type(self).__new__(type(self))
        memo[id(self)] = fresh
        ContextCompressor.__init__(
            fresh, model=getattr(self, "model", "") or "", quiet_mode=True)
        return fresh

    # -- engine tools ------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "forget_tool_result",
                "description": (
                    "Drop large tool result(s) from context after you have extracted "
                    "what you need (e.g. a 100k-char Snowflake dump). The result body "
                    "is replaced with a short note; the conversation stays valid. "
                    "One cache-prefix break now saves tokens on every future turn. "
                    "Prefer dry_run=true first for bulk forgets."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tool_call_id": {
                            "type": "string",
                            "description": "Forget this one result (its tool_call_id).",
                        },
                        "tool_name": {
                            "type": "string",
                            "description": "Forget results from this tool (e.g. 'snowflake_query'). Resolved via the transcript pairing.",
                        },
                        "keep_last_n": {
                            "type": "integer",
                            "description": "With tool_name: keep the newest N matching results, forget the rest. Default 0.",
                        },
                        "min_chars": {
                            "type": "integer",
                            "description": f"Only touch results at/above this size. Default {DEFAULT_MIN_CHARS}; pass 0 to disable.",
                        },
                        "summary": {
                            "type": "string",
                            "description": "One-line note kept in place of the data (e.g. 'Q3 revenue by region, totals used in analysis above').",
                        },
                        "dry_run": {
                            "type": "boolean",
                            "description": "Report what would be forgotten (with sizes) without changing anything. Default false.",
                        },
                    },
                },
            },
            {
                "name": "list_tool_results",
                "description": (
                    "List tool results currently in context, biggest first, with "
                    "their tool_call_id, tool name and size. Use to find what to forget."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "top_n": {
                            "type": "integer",
                            "description": "How many of the largest to show. Default 10.",
                        },
                        "min_chars": {
                            "type": "integer",
                            "description": "Only show results at/above this size. Default 1000.",
                        },
                    },
                },
            },
        ]

    def handle_tool_call(self, name: str, args: Dict[str, Any] = None, **kwargs: Any) -> str:
        try:
            args = args or {}
            messages = kwargs.get("messages") or []
            if name == "forget_tool_result":
                return json.dumps(self._forget(args, messages))
            if name == "list_tool_results":
                return json.dumps(self._list(args, messages))
            return json.dumps({"success": False, "error": f"Unknown tool: {name}"})
        except Exception as exc:  # fail open: error string, never raise
            logger.debug("forget-context %s failed: %s", name, exc)
            return json.dumps({"success": False, "error": f"{name} failed: {exc}"})

    # -- implementation (plain transcript helpers; no upstream internals) ----

    def _list(self, args: Dict[str, Any], messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            top_n = max(1, int(args.get("top_n", 10)))
        except (TypeError, ValueError):
            top_n = 10
        try:
            min_chars = max(0, int(args.get("min_chars", 1000)))
        except (TypeError, ValueError):
            min_chars = 1000
        entries = [e for e in _index_tool_results(messages) if e["chars"] >= min_chars]
        entries.sort(key=lambda e: e["chars"], reverse=True)
        return {"success": True, "count": len(entries), "results": entries[:top_n]}

    def _forget(self, args: Dict[str, Any], messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        tool_call_id = args.get("tool_call_id") or ""
        tool_name = args.get("tool_name") or ""
        if not tool_call_id and not tool_name:
            return {"success": False,
                    "error": "Pass tool_call_id and/or tool_name (use list_tool_results to find them)."}
        try:
            keep_last_n = max(0, int(args.get("keep_last_n", 0)))
        except (TypeError, ValueError):
            keep_last_n = 0
        try:
            min_chars = max(0, int(args.get("min_chars", DEFAULT_MIN_CHARS)))
        except (TypeError, ValueError):
            min_chars = DEFAULT_MIN_CHARS
        summary = args.get("summary") or "no longer needed"
        dry_run = bool(args.get("dry_run", False))

        entries = _index_tool_results(messages)
        if tool_call_id:
            targets = [e for e in entries if e["tool_call_id"] == tool_call_id]
            if not targets:
                return {"success": False, "error": f"tool_call_id not in context: {tool_call_id}"}
        else:
            matching = [e for e in entries if e["tool_name"] == tool_name]
            if not matching:
                return {"success": False, "error": f"no results from tool in context: {tool_name}"}
            # Newest = highest transcript index; keep the newest keep_last_n.
            matching.sort(key=lambda e: e["index"], reverse=True)
            targets = matching[keep_last_n:] if keep_last_n else matching

        forgotten, skipped, freed = [], [], 0
        for e in targets:
            if e["forgotten"]:
                skipped.append({**e, "reason": "already forgotten"})
                continue
            if e["chars"] < min_chars:
                skipped.append({**e, "reason": f"below min_chars={min_chars}"})
                continue
            detail = {"tool_call_id": e["tool_call_id"], "tool_name": e["tool_name"],
                      "freed_chars": e["chars"]}
            if dry_run:
                forgotten.append({**detail, "dry_run": True})
                freed += e["chars"]
                continue
            placeholder = (f"{FORGOTTEN_PREFIX}, was ~{e['chars']} chars "
                           f"from {e['tool_name'] or 'unknown tool'}: {summary}]")
            _replace_content(messages[e["index"]], placeholder)
            forgotten.append(detail)
            freed += e["chars"]

        return {"success": True, "dry_run": dry_run,
                "forgotten": forgotten, "skipped": skipped,
                "freed_chars_total": freed}


def register(ctx) -> None:
    """General-plugin entry point: register our engine (single slot; a second
    engine plugin would be rejected by the host with a warning)."""
    try:
        ctx.register_context_engine(ForgetCompressor(model="", quiet_mode=True))
    except Exception as exc:  # never break plugin discovery
        logger.warning("forget-context: engine registration failed: %s", exc)
