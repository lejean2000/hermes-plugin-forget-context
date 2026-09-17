# forget-context

Agent-controlled context relief for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

Large tool outputs — database dumps, 100k-char query results, accidental binary
`cat`s — get re-sent on every turn and flood the context window. Hermes already
prunes old tool results automatically, but only the model knows the moment a
result it just received is no longer needed. This plugin gives it a handle:
forget the result now, keep a one-line note in its place, stop paying for it on
every future turn.

## How it works

A context engine that subclasses the built-in `ContextCompressor` and adds two
agent-callable tools. Everything else — thresholds, pruning, summarization — is
inherited untouched.

- `forget_tool_result` — replace a tool result body with a short marker.
  Select by `tool_call_id`, or by `tool_name` (with `keep_last_n` to spare the
  newest). A `min_chars` floor (default 2000) protects small results, `dry_run`
  previews without mutating, and `summary` leaves a note where the data was.
- `list_tool_results` — list results in context, biggest first, with IDs and
  sizes, so the model can find what to forget.

Surgery is replace-only: messages keep their `role` and `tool_call_id`, so the
assistant `tool_calls` pairing and role alternation stay valid. One
prompt-cache break at forget time, savings on every turn after.

## Install

```bash
hermes plugins install <you>/hermes-plugin-forget-context --enable
hermes config set context.engine forget
```

Start a new session — engine selection happens once at session start. Revert
any time with `hermes config set context.engine compressor`.

## Tools reference

`forget_tool_result`:

| arg | meaning |
| --- | ------- |
| `tool_call_id` | forget this one result |
| `tool_name` | forget results from this tool (e.g. `snowflake_query`) |
| `keep_last_n` | with `tool_name`: keep the newest N, forget the rest (default 0) |
| `min_chars` | only touch results at/above this size (default 2000, 0 disables) |
| `summary` | one-line note kept in place of the data |
| `dry_run` | report what would be forgotten, change nothing |

At least one of `tool_call_id` / `tool_name` is required. Re-forgetting an
already-forgotten result is a safe no-op skip, and every failure returns a JSON
error string — the tool never raises.

## Layout

```
plugin.yaml    manifest (name, version, description)
__init__.py    ForgetCompressor + register(ctx); the whole plugin
```

## Update-proofing

- Lives in the user plugins dir (`~/.hermes/plugins/`), outside the Hermes
  checkout, so `hermes update` cannot touch it.
- Subclasses `ContextCompressor`; only `name`, `get_tool_schemas` and
  `handle_tool_call` are overridden (plus `__deepcopy__`, which the host
  recommends for plugin engines). Upstream compression changes flow through.
- Depends only on stable seams: `agent.context_compressor.ContextCompressor`,
  the engine tool contract, `ctx.register_context_engine`. No private imports.
