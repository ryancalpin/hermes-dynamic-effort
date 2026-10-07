"""dynamic-effort — per-turn local reasoning-effort routing.

Fires on ``pre_llm_call`` (once per user turn, before the first LLM call;
context returned is injected into the user message, so this callback always
returns None and only mutates the agent's reasoning_config):

  1. resolve the live agent for this session (gateway cache, else CLI ref)
  2. build a short decision state from the current user message + recent
     history (+ tool-activity markers)
  3. ask a local Jev-style decision model (default tev1:0.8b on Ollama
     /v1/systemone) which reasoning effort the next reply needs, with the
     choice list constrained to the efforts the ACTIVE provider/model
     supports (deepseek low/medium/high/max, kimi-k3 low/high/max, glm-5.2
     high/max, openai-compat low..max, ...)
  4. apply the change only when the recommended level's top probability
     clears a per-direction threshold (down-shifts need more confidence than
     up-shifts: a wrong down costs quality, a wrong up only costs a little
     latency/tokens). Keep the current effort otherwise.
  5. log every decision (JSONL + agent log) for calibration tuning.

Hard rules:
  - NEVER return context text (would pollute every turn's prompt).
  - NEVER block, raise, or take longer than ``timeout_s`` (+ small slack):
    every step is guarded; the hook is not timeout-bounded by core, so the
    plugin enforces its own budget. Hermes' downstream transport clamps any
    level onto the wire anyway, so worst case this plugin only degrades to
    the configured default.
  - NEVER enable reasoning when the user explicitly disabled it
    (reasoning_config.enabled is False) and never touch sessions whose
    current effort is not part of the provider's supported set.
  - ``none`` is excluded by default so the decider can never hard-disable
    thinking (set settings.allow_none to include it).
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request

logger = logging.getLogger(__name__)

__all__ = ["register"]

_gateway = None
_ctx = None
_last = {}   # last decision (for /effort status)

# ---------------------------------------------------------------------------
# Fallback vocabulary mirroring agent/reasoning_effort.py (import wins when
# available; the wide set is always SAFE downstream because Hermes clamps).
# ---------------------------------------------------------------------------
WIDE = ("low", "medium", "high", "xhigh", "max")
_FALLBACK = {
    "deepseek": ("low", "medium", "high", "max"),
    "k3": ("low", "high", "max"),
    "glm": ("high", "max"),
}

DEFAULT_SETTINGS = {
    "endpoint": "http://127.0.0.1:11434/v1/systemone",
    "decider_model": "tev1:0.8b",
    "enabled": True,
    "mode": "auto",            # "auto" = decider per turn; "manual" = pinned level
    "manual_effort": "",       # level used while mode == "manual" ("" = fall back to auto)
    "notify_changes": True,    # emit a status line on Telegram/TUI when effort changes
    "up_gate": 0.45,      # top-prob needed to raise effort
    "down_gate": 0.60,    # top-prob needed to lower effort
    "timeout_s": 8.0,     # must cover cold model load (~3s) plus warm decision
    "max_state_chars": 1500,
    "skip_platforms": [],     # e.g. ["cron"]
    "allow_none": False,
    "default_effort": "medium",
    "min_message_chars": 4,
}


def _settings():
    out = dict(DEFAULT_SETTINGS)
    try:
        if _ctx is not None:
            for k in DEFAULT_SETTINGS:
                out[k] = _ctx.get_config(k, DEFAULT_SETTINGS[k])
    except Exception:
        pass
    return out


def _supported_efforts(provider, model, api_mode):
    """Provider-supported reasoning efforts for (provider, model, api_mode).

    Tries the canonical core module first; falls back to local literals.
    ``none`` (hard disable) is removed unless settings.allow_none.
    """
    effort_sets = None
    try:
        from agent import reasoning_effort as re
        m = (model or "").strip().lower()
        p = (provider or "").strip().lower()
        if p == "openai-codex":
            effort_sets = re.codex_supported_efforts(model)
            if api_mode == "codex_app_server" and "max" in effort_sets:
                effort_sets = (*effort_sets, "ultra")
        elif "deepseek" in m:
            effort_sets = re.DEEPSEEK_V4_EFFORTS
        elif p == "moonshot" or "k3" in m or "kimi" in m:
            effort_sets = re.kimi_supported_efforts(model)
        elif "glm" in m:
            effort_sets = re.GLM52_EFFORTS if "glm-5.2" in m else re.GLM53_EFFORTS
        elif "grok" in m:
            effort_sets = re.XAI_GROK46_EFFORTS if "4.6" in m else re.XAI_LEGACY_EFFORTS
        elif p in ("tencent", "nebius", "upstage"):
            effort_sets = {"tencent": re.TOKENHUB_EFFORTS, "nebius": re.NEBIUS_EFFORTS,
                           "upstage": re.SOLAR_EFFORTS}[p]
        elif p == "meta" or "muse" in m:
            effort_sets = re.META_AI_EFFORTS
        elif p == "ollama":
            effort_sets = re.OLLAMA_CLOUD_EFFORTS if "cloud" in (m + api_mode or "") else WIDE
        elif m == "x-preview-f-free":
            effort_sets = re.OX_ALPHA_EFFORTS
        else:
            effort_sets = re.OPENAI_COMPAT_WIRE_EFFORTS
    except Exception:
        m = (model or "").strip().lower()
        if "deepseek" in m:
            effort_sets = _FALLBACK["deepseek"]
        elif "k3" in m or "kimi" in m:
            effort_sets = _FALLBACK["k3"]
        elif "glm" in m:
            effort_sets = _FALLBACK["glm"]
        else:
            effort_sets = WIDE
    if not effort_sets:
        effort_sets = WIDE
    out = [str(e).strip().lower() for e in effort_sets if str(e).strip().lower()]
    if not _settings()["allow_none"]:
        out = [e for e in out if e != "none"]
    return tuple(out)


def _resolve_agent(session_id, model):
    """Return ``(agent, cache_key)``; ``cache_key`` is the gateway agent-cache
    key when resolved via the gateway (used to check per-session manual overrides)."""
    # gateway path (mirrors max-output-budget)
    agent = None
    cache_key = None
    try:
        cache = getattr(_gateway, "_agent_cache", None)
        if isinstance(cache, dict) and session_id:
            for _key, entry in list(cache.items()):
                a = entry[0] if isinstance(entry, tuple) and entry else entry
                if getattr(a, "session_id", None) == session_id:
                    agent = a
                    cache_key = _key
                    break
    except Exception:
        agent = None
    if agent is not None:
        return agent, cache_key
    # CLI / TUI path
    try:
        cli_ref = getattr(getattr(_ctx, "_manager", None), "_cli_ref", None)
        a = getattr(cli_ref, "agent", None)
        if a is not None and (not session_id or getattr(a, "session_id", None) == session_id):
            return a, None
    except Exception:
        pass
    return None, None


def _decide(state, choices, timeout_s):
    settings = _settings()
    payload = {
        "model": settings["decider_model"],
        "state": state,
        "keep_alive": "30m",   # refresh residency: decisions keep the model warm
        "questions": {
            "effort": {
                "type": "choice",
                "instructions": (
                    "Decide the reasoning effort the AI assistant needs for its next reply. "
                    "Error handling, debugging, code review, architecture or security analysis "
                    "REQUIRE the highest offered level. Summaries, explanations and short code "
                    "are medium. Pure factual lookups, greetings and acknowledgements are the "
                    "lowest offered level. Never pick the lowest level for debugging or analysis."
                ),
                "criteria": {
                    ch: {
                        "low": "factual answer, greeting, thank-you, or simple lookup",
                        "medium": "summary, explanation, multi-part answer, small code",
                        "high": "debugging, code review, architecture, security, or long-context analysis",
                        "xhigh": "very complex multi-step analysis over long context",
                        "max": "hardest tasks: deep multi-step debugging/design over long context",
                        "minimal": "near-trivial single-word answer",
                        "none": "no reasoning needed at all",
                    }.get(ch, "standard effort level")
                    for ch in choices
                },
            }
        },
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(settings["endpoint"], data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        data = json.loads(r.read().decode())
    a = data["answers"]["effort"]
    return str(a.get("choice", "")).strip().lower(), a.get("probabilities", {}) or {}


def _build_state(user_message, conversation_history):
    settings = _settings()
    lines = []
    hist = conversation_history or []
    for msg in hist[-6:]:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "?")
        content = msg.get("content")
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("text"))
        text = str(content or "").strip()
        if not text:
            if role == "assistant" and msg.get("tool_calls"):
                lines.append("[assistant called tools]")
            elif role == "tool":
                lines.append("[tool result returned]")
            continue
        lines.append(f"{role}: {text[:250]}")
    lines.append(f"user: {str(user_message or '')[:400]}")
    state = "\n".join(lines[-12:])
    if len(state) > settings["max_state_chars"]:
        state = state[: settings["max_state_chars"]]
    return state


def _log_decision(entry):
    global _last
    _last = dict(entry)
    try:
        logger.info("dynamic-effort decision: %s", json.dumps(entry, sort_keys=True))
        from hermes_constants import get_hermes_home
        log_dir = get_hermes_home() / "plugin-data" / "dynamic-effort"
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "decisions.jsonl", "a") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")
    except Exception as exc:
        logger.warning("dynamic-effort: decision log write failed (non-blocking): %s", exc)


def _last_from_log():
    """Most recent decision from decisions.jsonl (survives process restarts)."""
    try:
        from hermes_constants import get_hermes_home
        log_file = get_hermes_home() / "plugin-data" / "dynamic-effort" / "decisions.jsonl"
        if not log_file.exists():
            return None
        with open(log_file, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            if size == 0:
                return None
            f.seek(max(0, size - 4096))
            tail = f.read().decode("utf-8", "replace").strip().splitlines()
        if not tail:
            return None
        return json.loads(tail[-1])
    except Exception:
        return None


def _on_pre_llm_call(session_id=None, user_message=None, conversation_history=None,
                     model=None, platform=None, **kwargs):
    settings = _settings()
    if not settings["enabled"]:
        return None
    t0 = time.time()
    try:
        platform = str(platform or "")
        if platform in [str(p).strip().lower() for p in settings["skip_platforms"]]:
            return None
        um = str(user_message or "")
        if len(um.strip()) < settings["min_message_chars"]:
            return None
        agent, cache_key = _resolve_agent(session_id, model)
        if agent is None:
            return None
        rc = getattr(agent, "reasoning_config", None)

        # A built-in /reasoning session override wins over the decider.
        try:
            ovr = getattr(_gateway, "_session_reasoning_overrides", None) or {}
            if isinstance(ovr, dict) and cache_key and cache_key in ovr and ovr.get(cache_key):
                _log_decision({
                    "event": "manual_override_respected", "session": session_id,
                    "model": getattr(agent, "model", model),
                })
                return None
        except Exception:
            pass

        # Manual mode (`/effort level <X>`): pin the requested level every turn.
        if settings["mode"] == "manual" and settings.get("manual_effort"):
            level = str(settings["manual_effort"]).strip().lower()
            if level:
                new_rc = dict(rc) if isinstance(rc, dict) else {}
                new_rc["enabled"] = True
                new_rc["effort"] = level
                agent.reasoning_config = new_rc
                _log_decision({
                    "event": "manual_applied", "session": session_id,
                    "model": getattr(agent, "model", model),
                    "provider": getattr(agent, "provider", None),
                    "manual_effort": level,
                })
                return None

        # Never override an explicit disable; never touch unset reasoning when
        # the configured default would not exist (unset stays unset).
        if isinstance(rc, dict) and rc.get("enabled") is False:
            return None
        choices = _supported_efforts(getattr(agent, "provider", None),
                                     getattr(agent, "model", model),
                                     getattr(agent, "api_mode", None))
        if len(choices) < 2:
            return None
        current = ""
        if isinstance(rc, dict):
            current = str(rc.get("effort") or "").strip().lower()
        if not current:
            current = settings["default_effort"]
        if current not in choices:
            return None  # manual/session effort outside our ladder: respect it
        state = _build_state(um, conversation_history)
        recommended, probs = _decide(state, choices, settings["timeout_s"])
        if recommended not in choices or recommended == current:
            return None
        top = float(probs.get(recommended, 0.0) or 0.0)
        idx_r = choices.index(recommended)
        idx_c = choices.index(current)
        gate = settings["up_gate"] if idx_r > idx_c else settings["down_gate"]
        if top < gate:
            _log_decision({
                "event": "recommendation_below_gate", "session": session_id,
                "model": getattr(agent, "model", model), "provider": getattr(agent, "provider", None),
                "current": current, "recommended": recommended, "top_prob": round(top, 3),
                "gate": gate, "ms": round((time.time() - t0) * 1000),
            })
            return None
        # Apply: keep enabled semantics, swap only the effort level.
        new_rc = dict(rc) if isinstance(rc, dict) else {}
        new_rc["enabled"] = True
        new_rc["effort"] = recommended
        agent.reasoning_config = new_rc
        _log_decision({
            "event": "applied", "session": session_id,
            "model": getattr(agent, "model", model), "provider": getattr(agent, "provider", None),
            "platform": platform,
            "current": current, "recommended": recommended, "top_prob": round(top, 3),
            "gate": gate, "choices": list(choices), "ms": round((time.time() - t0) * 1000),
        })
        if settings.get("notify_changes"):
            try:
                emit = getattr(agent, "_emit_status", None)
                if callable(emit):
                    emit(f"⚙️ reasoning effort: {current} → {recommended}")
            except Exception:
                pass
    except Exception:
        logger.warning("dynamic-effort: non-blocking failure", exc_info=True)
    return None


def _capture_gateway(event=None, gateway=None, **kwargs):
    global _gateway
    if gateway is not None:
        _gateway = gateway
    return None


_LADDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra")


def _fmt_last():
    if not _last:
        return "none yet"
    e = _last
    target = e.get("recommended") or e.get("manual_effort") or e.get("current") or "-"
    return (f"{e.get('event', '?')} · {e.get('model', '?')} · "
            f"{e.get('current', '')} → {target} "
            f"(prob {e.get('top_prob', '-')}, {e.get('ms', '-')}ms)")


def _cmd_effort(raw_args):
    """/effort [status|on|off|auto|level <X>|gate up/down <f>|last] — control
    dynamic reasoning effort from the chat (Telegram/TUI/CLI)."""
    args = (raw_args or "").strip().split()
    cmd = args[0].lower() if args else "status"
    s = _settings()
    if cmd == "on":
        if _ctx is not None:
            _ctx.set_config("enabled", True)
        return "⚙️ Dynamic effort is now *ON*. The local decider picks the level each turn. Send */effort status* anytime."
    if cmd == "off":
        if _ctx is not None:
            _ctx.set_config("enabled", False)
        return "⚙️ Dynamic effort is now *OFF*. Effort stays at your configured level. */effort on* re-enables it."
    if cmd == "auto":
        if _ctx is not None:
            _ctx.set_config("mode", "auto")
            _ctx.set_config("manual_effort", "")
        return "⚙️ Back to *auto* — the decider controls the level each turn."
    if cmd in ("level", "set"):
        if len(args) < 2:
            return "Usage: /effort level low|medium|high|xhigh|max|minimal|none|ultra"
        lvl = args[1].lower()
        if lvl not in _LADDER:
            return f"❓ Unknown level '{lvl}'. Choose: {' / '.join(_LADDER)}"
        if _ctx is not None:
            _ctx.set_config("mode", "manual")
            _ctx.set_config("manual_effort", lvl)
        return f"📌 Effort pinned to *{lvl}* for every turn. Send */effort auto* to release."
    if cmd == "gate":
        if len(args) < 3 or args[1] not in ("up", "down"):
            return "Usage: /effort gate up|down <0.0-1.0>"
        try:
            val = round(float(args[2]), 3)
        except ValueError:
            return "Gate must be a number between 0 and 1."
        if not 0 <= val <= 1:
            return "Gate must be between 0 and 1."
        key = "up_gate" if args[1] == "up" else "down_gate"
        if _ctx is not None:
            _ctx.set_config(key, val)
        return f"🎚️ {'Up' if key == 'up_gate' else 'Down'} gate = *{val}*."
    if cmd in ("last", "log"):
            entry = _last_from_log() or _last
            if not entry:
                return "⚙️ Last decision: none yet"
            target = entry.get("recommended") or entry.get("manual_effort") or entry.get("current") or "-"
            return (f"⚙️ Last decision: *{entry.get('event', '?')}* · {entry.get('model', '?')} · "
                    f"{entry.get('current', '')} → {target} "
                    f"(prob {entry.get('top_prob', '-')}, {entry.get('ms', '-')}ms)")
    # status (default)
    on = bool(s.get("enabled"))
    if not on:
        head = "Dynamic effort: *OFF* · stays at your configured level"
    elif s.get("mode") == "manual" and s.get("manual_effort"):
        head = f"Dynamic effort: *ON* · pinned to *{s.get('manual_effort')}*"
    else:
        head = "Dynamic effort: *ON* · auto mode"
    return (
        f"⚙️ {head}\n"
        f"Gates: up ≥ {s.get('up_gate')} · down ≥ {s.get('down_gate')}\n"
        f"Decider: {s.get('decider_model')}\n\n"
        f"_Last decision:_ {_fmt_last()}"
    )


def register(ctx):
    global _ctx
    _ctx = ctx
    try:
        ctx.register_hook("pre_gateway_dispatch", _capture_gateway)
    except Exception:
        logger.warning("dynamic-effort: pre_gateway_dispatch registration failed", exc_info=True)
    try:
        ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    except Exception:
        logger.warning("dynamic-effort: pre_llm_call registration failed", exc_info=True)
    try:
        ctx.register_command(
            "effort", handler=_cmd_effort,
            description="Control dynamic reasoning effort (status/on/off/auto/level/gate/last).",
        )
    except Exception:
        logger.warning("dynamic-effort: /effort command registration failed", exc_info=True)
