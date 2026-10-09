from __future__ import annotations

from dataclasses import dataclass
import json
import re


ISSUE_RE = re.compile(r"<(?:issue_description|pr_description)>(.*?)</(?:issue_description|pr_description)>", re.S)


@dataclass(frozen=True)
class SerializedExample:
    text: str
    last_included_message: int
    landmark_token_count: int
    monitor_token_count: int
    included_message_indices: tuple[int, ...]


def _body(message: dict) -> str:
    body = str(message.get("content") or "")
    reasoning = message.get("reasoning_content")
    if reasoning:
        body += "\n" + str(reasoning)
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        body += f"\nTOOL {function.get('name', 'unknown')} {function.get('arguments', '')}"
    return body.strip()


def _issue(row: dict) -> str:
    explicit = row.get("issue")
    if explicit:
        return str(explicit)
    for message in row.get("messages") or row.get("trajectory") or []:
        if message.get("role") == "user":
            text = str(message.get("content") or "")
            match = ISSUE_RE.search(text)
            return (match.group(1) if match else text).strip()
    return ""


def _complete_prefix(messages: list[dict], landmark: int, tokenizer) -> tuple[list[tuple[int, dict]], int]:
    kept: list[tuple[int, dict]] = []
    used = 0
    for index, message in enumerate(messages):
        count = len(tokenizer.encode(_body(message), add_special_tokens=False))
        if used + count > landmark:
            break
        kept.append((index, message))
        used += count
    return kept, used


def _step_text(index: int, message: dict) -> str:
    action = message.get("action") or ""
    if not action and message.get("tool_calls"):
        action = ((message["tool_calls"][0].get("function") or {}).get("name") or "tool")
    role = message.get("role") or "unknown"
    return f"[message {index}; role={role}; action={action or 'none'}]\n{_body(message)}"


def _truncate(text: str, tokenizer, limit: int) -> tuple[str, int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= limit:
        return text, len(ids)
    # Preserve the issue at the head and the newest evidence at the tail.
    head = ids[: max(1, limit // 3)]
    tail = ids[-(limit - len(head)) :]
    merged = head + tail
    return tokenizer.decode(merged, skip_special_tokens=True), len(merged)


def serialize_example(row: dict, arm: str, landmark_tokens: int, max_monitor_tokens: int, agent_tokenizer, monitor_tokenizer) -> SerializedExample:
    if arm not in {"issue", "issue_window"}:
        raise ValueError(f"unknown arm: {arm}")
    issue = _issue(row)
    messages = list(row.get("messages") or row.get("trajectory") or [])
    kept, used = _complete_prefix(messages, landmark_tokens, agent_tokenizer)
    selected: list[tuple[int, dict]] = []
    if arm == "issue_window":
        completed = [(i, m) for i, m in kept if m.get("role") in {"assistant", "tool", "user"}]
        selected = completed[-8:]
        edit = next(
            ((i, m) for i, m in reversed(completed[:-8]) if (m.get("action") == "edit" or "str_replace_editor" in _body(m))),
            None,
        )
        if edit is not None:
            selected.insert(0, edit)
    sections = [f"[issue]\n{issue}"]
    sections.extend(_step_text(i, m) for i, m in selected)
    text, monitor_count = _truncate("\n\n".join(sections), monitor_tokenizer, max_monitor_tokens)
    indices = tuple(i for i, _ in selected)
    return SerializedExample(
        text=text,
        last_included_message=max((i for i, _ in kept), default=-1),
        landmark_token_count=used,
        monitor_token_count=monitor_count,
        included_message_indices=indices,
    )

