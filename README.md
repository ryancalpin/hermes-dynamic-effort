# dynamic-effort

Per-turn reasoning-effort routing for Hermes Agent, decided locally by a small
decision model (Jev-style "System One" model) before each main LLM call.

## What it does

Before every turn, a local decision model (default `tev1:0.8b` via Ollama
`/v1/systemone`) reads the current conversation and picks the reasoning effort
the next reply needs. The plugin applies that effort to the outgoing call.

The option list is derived from the efforts the active provider/model actually
supports (deepseek → low/medium/high/max, kimi-k3 → low/high/max, glm-5.2 →
high/max, openai-compat → low..max), so the decider never proposes a level the
provider rejects. Hermes clamps downstream anyway, so an unsupported level can
never reach the wire.

A change is applied only when the recommended level's top probability clears a
per-direction threshold: **up-shift ≥ 0.45, down-shift ≥ 0.60**. Down-shifts
need more confidence because a wrong down costs quality; a wrong up only costs
a little latency/tokens. Same level → no-op.

## Requirements

- **Hermes ≥ 0.21** (uses the `pre_llm_call` hook and `register_command`).
- **Ollama ≥ 0.35** with a decision model pulled (`ollama pull tev1:0.8b`).
  The `/v1/systemone` endpoint is a TypeSafe Jev-style API.
- **Adequate output cap on the main model.** This is the important one: the
  plugin can raise effort to `high`, which increases reasoning tokens. If the
  model's `max_tokens` output cap is small, reasoning can consume the whole
  budget and the model produces **no visible answer**. Set `model.max_tokens`
  high enough (e.g. 16384–65536) for the models you route, and enable the
  `max-output-budget` plugin so the cap reaches the wire. Without this, raising
  effort can cause silent empty replies.

## Install

```bash
# copy the plugin dir into the profile's plugins/ and enable it
hermes -p <profile> plugins enable dynamic-effort
```

## Usage

The `/effort` slash command controls it from the chat (Telegram/TUI/CLI):

| Command | Effect |
|---------|--------|
| `/effort` | Status: on/off, mode, gates, last decision |
| `/effort on` / `/effort off` | Enable / disable the decider |
| `/effort auto` | Back to per-turn decisions |
| `/effort level <L>` | Pin every turn to a level (low/medium/high/xhigh/max/...) |
| `/effort gate up|down <0-1>` | Tune the confidence thresholds |
| `/effort last` | Most recent decision |

When the effort level changes, a status line is emitted (e.g. `⚙️ reasoning
effort: medium → high`). Every decision is logged to
`<profile>/plugin-data/dynamic-effort/decisions.jsonl`.

## Settings

All settings live under `plugins.entries.dynamic-effort.settings` in the
profile's `config.yaml` (schema declared in `plugin.yaml`). Key ones:

- `enabled` (bool, default true) — master switch.
- `mode` (`auto`/`manual`) + `manual_effort` — pin a level.
- `up_gate` / `down_gate` (float) — confidence thresholds.
- `endpoint` / `decider_model` — the local decision service.
- `timeout_s` (default 8.0) — must cover cold model load (~3s on a GTX 1660).
- `allow_none` (bool, default false) — allow the decider to hard-disable
  reasoning. Off by default so the decider can never turn thinking off.

## Safety

- **Fail-open.** Any error leaves the turn untouched at its configured effort.
- **Never injects text into prompts** (the hook returns `None`).
- **Never enables reasoning when the user disabled it.**
- **Respects manual overrides** — a built-in `/reasoning` session override
  wins over the decider.

## Notes

- The plugin reaches into a couple of Hermes gateway internals
  (`_agent_cache`, `_session_reasoning_overrides`) to resolve the live agent
  and detect manual overrides. This is the same pattern the bundled
  `max-output-budget` plugin uses; it may need updating if those internals move.
- The `_supported_efforts` vocabulary mirrors `agent/reasoning_effort.py`; if
  the core adds a new provider tier, the local fallback may go stale (the core
  import is tried first, so it stays current while the import path is stable).
