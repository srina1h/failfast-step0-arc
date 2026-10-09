import json

from failfast_arc.prefix import serialize_example


class SpaceTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(x) for x in ids)


def _row(messages):
    return {"instance_id": "task-1", "issue": "Fix the parser", "messages": messages}


def test_excludes_message_that_crosses_landmark():
    messages = [
        {"role": "assistant", "content": "one two three", "action": "search"},
        {"role": "tool", "content": "four five six"},
        {"role": "assistant", "content": "seven eight nine", "action": "edit"},
    ]
    out = serialize_example(_row(messages), "issue_window", 7, 4096, SpaceTokenizer(), SpaceTokenizer())
    assert "seven" not in out.text
    assert out.last_included_message == 1


def test_preserves_malformed_tool_arguments():
    messages = [{"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "run", "arguments": "{not-json"}}]}]
    out = serialize_example(_row(messages), "issue_window", 20_000, 4096, SpaceTokenizer(), SpaceTokenizer())
    assert "{not-json" in out.text


def test_issue_only_is_shared_by_attempts():
    a = serialize_example(_row([{"role": "assistant", "content": "alpha"}]), "issue", 20_000, 4096, SpaceTokenizer(), SpaceTokenizer())
    b = serialize_example(_row([{"role": "assistant", "content": "beta"}]), "issue", 20_000, 4096, SpaceTokenizer(), SpaceTokenizer())
    assert a.text == b.text


def test_window_has_eight_recent_steps_and_pinned_edit():
    messages = [{"role": "assistant", "content": f"step{i}", "action": "search"} for i in range(10)]
    messages.insert(1, {"role": "assistant", "content": "important-edit", "action": "edit"})
    out = serialize_example(_row(messages), "issue_window", 20_000, 4096, SpaceTokenizer(), SpaceTokenizer())
    assert "important-edit" in out.text
    assert "step9" in out.text
    assert "step0" not in out.text

