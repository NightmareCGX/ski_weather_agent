#!/usr/bin/env python3
"""
Ski/Weather Agent Orchestrator V0

Default behavior (hints-on):
- Always pass USER_QUESTION to the LLM and critic.
- Extract lightweight USER_HINTS (target_resorts + date_hints) from USER_QUESTION.
- If target_resorts is non-empty, trim CONTEXT.resorts to only those resorts (reduces tokens + reduces drift).
- Do NOT create or pass any "task" field. The model should infer what to do from USER_QUESTION.

Ultra-simple mode (--no-hints):
- Pass USER_QUESTION + full CONTEXT, and do not provide USER_HINTS.
- Useful as a baseline to compare against hints-on behavior.

Also:
- Runs a critic pass; optionally runs one repair pass.
- Uses OpenAI Python SDK, but can target any OpenAI-compatible server via OPENAI_BASE_URL (e.g., Ollama OpenAI endpoint).

Expected prompt files:
- core/prompt/system.txt
- core/prompt/critic.txt
(Override via --system / --critic)

Notes:
- This script intentionally avoids keyword-based "task routing".
- It performs only safe, deterministic extraction of resort name mentions + date strings.
"""

import argparse
import copy
import json
import os
import re
from typing import Any, Dict, Callable, List, Optional, Tuple

from openai import OpenAI


# ----------------------------
# IO helpers
# ----------------------------

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_json(text: str) -> Dict[str, Any]:
    """Extract a JSON object from model text."""
    text = (text or "").strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    # Try to recover first top-level JSON object
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("No JSON object found in model output.")
    return json.loads(m.group(0))

UNIT_CONVERSIONS: Dict[str, Tuple[str, Callable[[float], float]]] = {
    "_f":   ("_c",  lambda x: (x - 32.0) * 5.0 / 9.0),
    "_mph": ("_ms", lambda x: x * 0.44704),
    "_in":  ("_cm", lambda x: x * 2.54),
    "_ft":  ("_m",  lambda x: x * 0.3048),
    "_kts": ("_ms", lambda x: x * 0.514444),
}

def normalize_to_si(context: dict) -> dict:
    """
    Recursively walk a nested dict/list structure and:
    - Convert values whose keys end with known non-SI suffixes
    - Rename keys to SI suffixes
    - Leave unknown fields untouched
    """
    if isinstance(context, dict):
        new_context = {}
        for key, value in context.items():
            converted = False

            if isinstance(value, (int, float)):
                for suffix, (si_suffix, fn) in UNIT_CONVERSIONS.items():
                    if key.endswith(suffix):
                        new_key = key[: -len(suffix)] + si_suffix
                        try:
                            new_context[new_key] = fn(float(value))
                        except Exception:
                            new_context[new_key] = value
                        converted = True
                        break

            if not converted:
                new_context[key] = normalize_to_si(value)

        return new_context

    elif isinstance(context, list):
        return [normalize_to_si(x) for x in context]

    else:
        return context

# ----------------------------
# OpenAI client + chat
# ----------------------------

def make_client() -> OpenAI:
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Set it to your OpenAI key, or 'EMPTY' for local OpenAI-compatible servers."
        )
    if base_url:
        return OpenAI(base_url=base_url, api_key=api_key)
    return OpenAI(api_key=api_key)


def chat_json(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_payload: Dict[str, Any],
    temperature: float = 0.2,
) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    text = resp.choices[0].message.content or ""
    return extract_json(text)


# ----------------------------
# Hints extraction (no "task")
# ----------------------------

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def extract_date_hints(query: str) -> List[str]:
    """
    Extract common date strings from query WITHOUT interpreting them.

    Captures:
      - M/D or MM/DD (e.g., 12/3, 12/13)
      - YYYY-MM-DD (e.g., 2025-12-13)

    We intentionally do not parse ranges; the raw strings are sufficient hints.
    """
    q = query or ""
    md = re.findall(r"\b(\d{1,2}/\d{1,2})\b", q)
    ymd = re.findall(r"\b(\d{4}-\d{2}-\d{2})\b", q)

    # Stable order, unique
    seen = set()
    out: List[str] = []
    for x in md + ymd:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def extract_target_resorts(query: str, resort_names: List[str]) -> List[str]:
    """
    Find resort name mentions inside query (case-insensitive) with a conservative match.

    Strategy:
    - Normalize both query and resort name (lowercase + collapse whitespace).
    - Match if normalized resort name appears as a substring in normalized query.

    This is simple but effective for V0. For ambiguous matches, keep all hits.
    """
    ql = _normalize(query or "")
    hits: List[str] = []
    for name in resort_names:
        nl = _normalize(name or "")
        if not nl:
            continue
        if nl in ql:
            hits.append(name)
    return hits


def build_user_hints(query: str, context: Dict[str, Any]) -> Dict[str, Any]:
    resorts = context.get("resorts", [])
    resort_names = [r.get("name", "") for r in resorts if isinstance(r, dict) and r.get("name")]
    target_resorts = extract_target_resorts(query, resort_names)
    date_hints = extract_date_hints(query)

    return {
        "target_resorts": target_resorts,  # list[str]
        "date_hints": date_hints,          # list[str]
    }


def trim_context_to_targets(context: Dict[str, Any], target_resorts: List[str]) -> Dict[str, Any]:
    """
    Return a shallow-trimmed copy of context where CONTEXT.resorts includes only target resorts.
    If target_resorts is empty, return context unchanged.
    """
    if not target_resorts:
        return context

    ctx = copy.deepcopy(context)
    resorts = ctx.get("resorts", [])
    if not isinstance(resorts, list):
        return ctx

    keep = set(target_resorts)
    ctx["resorts"] = [r for r in resorts if isinstance(r, dict) and r.get("name") in keep]
    return ctx


# ----------------------------
# Orchestration
# ----------------------------

def run_agent(
    client: OpenAI,
    model: str,
    system_prompt: str,
    critic_prompt: str,
    user_question: str,
    context: Dict[str, Any],
    user_hints: Optional[Dict[str, Any]] = None,
    repair: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # 1) Main generation
    payload: Dict[str, Any] = {
        "USER_QUESTION": user_question,
        "CONTEXT": context,
    }
    if user_hints is not None:
        payload["USER_HINTS"] = user_hints

    output = chat_json(client, model, system_prompt, payload, temperature=0.2)

    # 2) Critic
    critic_payload: Dict[str, Any] = {
        "USER_QUESTION": user_question,
        "CONTEXT": context,
        "CANDIDATE_OUTPUT": output,
    }
    if user_hints is not None:
        critic_payload["USER_HINTS"] = user_hints

    critique = chat_json(client, model, critic_prompt, critic_payload, temperature=0.0)

    # 3) Optional one-pass repair
    if repair and (not critique.get("pass", False)):
        repair_instructions = {
            "issues": critique.get("issues", []),
            "suggested_fixes": critique.get("suggested_fixes", []),
        }
        repair_payload: Dict[str, Any] = {
            "USER_QUESTION": user_question,
            "CONTEXT": context,
            "REPAIR_INSTRUCTIONS": repair_instructions,
            "CANDIDATE_OUTPUT": output,
            "REQUIREMENTS": "Return ONLY a corrected JSON object matching OUTPUT_SCHEMA. No extra keys.",
        }
        if user_hints is not None:
            repair_payload["USER_HINTS"] = user_hints

        output = chat_json(client, model, system_prompt, repair_payload, temperature=0.1)

        # Re-critic (no further repair)
        critic_payload["CANDIDATE_OUTPUT"] = output
        critique = chat_json(client, model, critic_prompt, critic_payload, temperature=0.0)

    return output, critique


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", required=True, help="Path to context JSON (resorts + forecasts + heuristics)")
    ap.add_argument("--query", default=None, help="Natural language user question")
    ap.add_argument("--model", required=True, help="Model name (OpenAI or local OpenAI-compatible server)")
    ap.add_argument("--system", default=None, help="Path to system prompt text file (optional)")
    ap.add_argument("--critic", default=None, help="Path to critic prompt text file (optional)")
    ap.add_argument("--no-hints", action="store_true", help="Ultra-simple mode: do not pass USER_HINTS and do not trim context.")
    ap.add_argument("--no-trim", action="store_true", help="Do not trim CONTEXT.resorts even if target resorts are detected.")
    ap.add_argument("--no-repair", action="store_true", help="Disable one-pass repair")
    ap.add_argument("--debug", action="store_true", help="Print debug info (prompt heads and extracted hints)")
    args = ap.parse_args()

    if not args.query:
        raise SystemExit("Provide --query")

    user_question = args.query
    context_full = normalize_to_si(load_json(args.context))
    context_full["units"] = "SI"

    core_dir = os.path.dirname(__file__)
    system_path = args.system or os.path.join(core_dir, "prompt", "system.txt")
    critic_path = args.critic or os.path.join(core_dir, "prompt", "critic.txt")

    system_prompt = load_text(system_path)
    critic_prompt = load_text(critic_path)

    user_hints: Optional[Dict[str, Any]] = None
    context_for_model = context_full

    if not args.no_hints:
        user_hints = build_user_hints(user_question, context_full)
        if not args.no_trim and user_hints.get("target_resorts"):
            context_for_model = trim_context_to_targets(context_full, user_hints["target_resorts"])

    if args.debug:
        print("=== SYSTEM PROMPT HEAD ===")
        print(system_prompt[:250])
        print("=== CRITIC PROMPT HEAD ===")
        print(critic_prompt[:250])
        print("=== USER_HINTS ===")
        print(json.dumps(user_hints, ensure_ascii=False, indent=2))
        if context_for_model is not context_full:
            kept = [r.get("name") for r in context_for_model.get("resorts", []) if isinstance(r, dict)]
            print(f"=== CONTEXT TRIMMED: resorts kept ({len(kept)}) ===")
            print(", ".join([k for k in kept if k]))

    client = make_client()

    output, critique = run_agent(
        client=client,
        model=args.model,
        system_prompt=system_prompt,
        critic_prompt=critic_prompt,
        user_question=user_question,
        context=context_for_model,
        user_hints=user_hints,
        repair=not args.no_repair,
    )

    print(json.dumps({"output": output, "critique": critique}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

