"""Native transcript comparison and in-memory stream assembly; never stores events."""

import json
from copy import deepcopy

ENDPOINTS = {"openai-completions": "/v1/chat/completions", "openai-responses": "/v1/responses"}


def incoming(body: dict, protocol: str) -> list:
    value = body.get("messages", []) if protocol == "openai-completions" else body.get("input", [])
    if isinstance(value, str) and protocol == "openai-responses":
        return [{"role": "user", "content": value}]
    if not isinstance(value, list) or not all(isinstance(x, dict) for x in value):
        raise ValueError("messages/input must be a list of objects (Responses also accepts a string)")
    return deepcopy(value)


def comparison(item: dict, protocol: str) -> dict:
    """Only documented text-only and optional-field equivalences. Unknown keys matter."""
    item = deepcopy(item)
    role = item.get("role")
    if protocol == "openai-completions":
        optional = ()
        if role == "assistant":
            optional = ("name", "refusal", "function_call", "tool_calls", "audio", "annotations")
        elif role in ("user", "system", "developer"):
            optional = ("name",)
        for key in optional:
            if item.get(key, 1) is None:
                item.pop(key)
        if role == "assistant" and item.get("content", 1) is None:
            item.pop("content")
    elif protocol == "openai-responses":
        # IDs remain significant; status is transport-only on these output items.
        kind = item.get("type")
        if kind in ("message", "function_call", "reasoning", "custom_tool_call"):
            if item.get("status") in (None, "completed", "in_progress", "incomplete"):
                item.pop("status", None)
        if role in ("user", "assistant", "system", "developer") and kind == "message":
            item.pop("type")
    else:
        return item
    content = item.get("content")
    if isinstance(content, list) and content:
        allowed_type = (
            "text" if protocol == "openai-completions" else ("output_text" if role == "assistant" else "input_text")
        )
        blocks = []
        for block in content:
            if not isinstance(block, dict):
                break
            b = dict(block)
            if protocol == "openai-responses" and b.get("type") == "output_text" and b.get("annotations") == []:
                b.pop("annotations")
            if set(b) != {"type", "text"} or b["type"] != allowed_type or not isinstance(b["text"], str):
                break
            blocks.append(b["text"])
        else:
            item["content"] = "".join(blocks)
    return item


def canonical(item: dict, protocol: str) -> str:
    # Python equality conflates JSON true and 1. Canonical JSON does not.
    return json.dumps(comparison(item, protocol), sort_keys=True, ensure_ascii=False)


def shared_prefix(a: list, b: list, protocol: str) -> int:
    for i, (left, right) in enumerate(zip(a, b)):
        if canonical(left, protocol) != canonical(right, protocol):
            return i
    return min(len(a), len(b))


def meaningful(items: list) -> bool:
    return any(x.get("role") not in ("system", "developer") for x in items)


def inherited_turns(turns: list, boundary: int) -> list:
    result = []
    for original in turns:
        if original["status"] == "in_progress" or original["input_end"] > boundary:
            break
        turn = deepcopy(original)
        if turn.get("output_end", turn["input_end"]) > boundary:
            turn.update(
                output_end=boundary,
                status="incomplete",
                partial_output=boundary > turn["output_start"],
                truncated_at_fork=True,
            )
            turn["response_id"] = None
            turn["usage"] = None
        result.append(turn)
    return result


def nonstream_output(data: dict, protocol: str) -> list[tuple[int, list, dict]]:
    if protocol == "openai-responses":
        status = data.get("status", "completed")
        status = status if status in {"completed", "failed", "incomplete", "cancelled"} else "incomplete"
        return [
            (
                0,
                deepcopy(data.get("output", [])),
                {
                    "status": status,
                    "response_status": data.get("status"),
                    "error": data.get("error"),
                    "incomplete_details": data.get("incomplete_details"),
                },
            )
        ]
    if not data.get("choices"):
        raise ValueError("Upstream returned no choices")
    return [
        (c.get("index", i), [deepcopy(c["message"])], {"finish_reason": c.get("finish_reason"), "status": "completed"})
        for i, c in enumerate(data.get("choices", []))
    ]


class ChatAssembler:
    def __init__(self):
        self.choices = {}
        self.model = self.response_id = self.usage = None

    def feed(self, chunk: dict):
        self.model = chunk.get("model", self.model)
        self.response_id = chunk.get("id", self.response_id)
        if chunk.get("usage") is not None:
            self.usage = chunk["usage"]
        for choice in chunk.get("choices", []):
            index = choice.get("index", 0)
            state = self.choices.setdefault(index, {"message": {}, "tools": {}, "finish_reason": None})
            if choice.get("finish_reason") is not None:
                state["finish_reason"] = choice["finish_reason"]
            for key, value in choice.get("delta", {}).items():
                if value is None:
                    continue
                if key == "tool_calls":
                    for call in value:
                        target = state["tools"].setdefault(call["index"], {})
                        self._merge(target, {k: v for k, v in call.items() if k != "index"})
                else:
                    self._merge(state["message"], {key: value})

    @staticmethod
    def _merge(target: dict, delta: dict):
        for key, value in delta.items():
            if isinstance(value, dict):
                ChatAssembler._merge(target.setdefault(key, {}), value)
            elif isinstance(value, str) and key in {"content", "refusal", "arguments", "name", "id"}:
                target[key] = target.get(key, "") + value
            else:
                target[key] = deepcopy(value)

    @property
    def complete(self):
        return bool(self.choices) and all(c["finish_reason"] is not None for c in self.choices.values())

    def outputs(self):
        result = []
        for index, state in sorted(self.choices.items()):
            message = deepcopy(state["message"])
            if state["tools"]:
                message["tool_calls"] = [deepcopy(t) for _, t in sorted(state["tools"].items())]
            if message:
                message.setdefault("role", "assistant")
            result.append((index, [message] if message else [], {"finish_reason": state["finish_reason"]}))
        return result


class ResponsesAssembler:
    TERMINAL = {"response.completed", "response.failed", "response.incomplete"}

    def __init__(self):
        self.items = {}
        self.model = self.response_id = self.usage = None
        self.terminal = None
        self.error = None
        self.response_status = None

    def feed(self, event: dict):
        kind = event.get("type", "")
        response = event.get("response")
        if isinstance(response, dict):
            self.model = response.get("model", self.model)
            self.response_id = response.get("id", self.response_id)
            self.response_status = response.get("status", self.response_status)
            if response.get("usage") is not None:
                self.usage = response["usage"]
            if response.get("output"):
                self.items = dict(enumerate(deepcopy(response["output"])))
        if kind in self.TERMINAL:
            self.terminal = deepcopy(event)
            # Failed terminal objects sometimes omit all output despite earlier
            # deltas. Retain the locally observed partial in that case.
            if kind == "response.completed" or response.get("output"):
                self.items = dict(enumerate(deepcopy(response.get("output", []))))
            self.error = response.get("error")
            return
        if kind == "error":
            self.error = event
            return
        index = event.get("output_index")
        if index is None:
            return
        if kind in {"response.output_item.added", "response.output_item.done"}:
            self.items[index] = deepcopy(event["item"])
            return
        function_event = kind.startswith("response.function_call_arguments.") and kind.endswith((".delta", ".done"))
        custom_tool_event = kind.startswith("response.custom_tool_call_input.") and kind.endswith((".delta", ".done"))
        part_event = kind in {
            "response.content_part.added",
            "response.content_part.done",
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_part.done",
        }
        text_event = kind.startswith(
            (
                "response.output_text.",
                "response.refusal.",
                "response.reasoning_text.",
                "response.reasoning_summary_text.",
            )
        ) and kind.endswith((".delta", ".done"))
        annotation_event = kind == "response.output_text.annotation.added"
        if not (function_event or custom_tool_event or part_event or text_event or annotation_event):
            return
        item = self.items.setdefault(index, {})
        if "item_id" in event:
            item.setdefault("id", event["item_id"])
        if function_event or custom_tool_event:
            field = "arguments" if function_event else "input"
            item.setdefault("type", "function_call" if function_event else "custom_tool_call")
            if kind.endswith(".delta"):
                item[field] = item.get(field, "") + event.get("delta", "")
            else:
                item[field] = event.get(field, item.get(field, ""))
            return
        summary = "reasoning_summary" in kind
        field = "summary" if summary else "content"
        part_index = event.get("summary_index" if summary else "content_index")
        if part_index is None:
            return
        parts = item.setdefault(field, [])
        while len(parts) <= part_index:
            parts.append({})
        if part_event:
            parts[part_index] = deepcopy(event["part"])
        elif annotation_event:
            annotations = parts[part_index].setdefault("annotations", [])
            annotation_index = event.get("annotation_index", len(annotations))
            while len(annotations) <= annotation_index:
                annotations.append({})
            annotations[annotation_index] = deepcopy(event["annotation"])
        elif text_event:
            key = "refusal" if "refusal" in kind else "text"
            part = parts[part_index]
            part_type = (
                "summary_text"
                if summary
                else "reasoning_text"
                if kind.startswith("response.reasoning_text.")
                else "refusal"
                if key == "refusal"
                else "output_text"
            )
            part.setdefault("type", part_type)
            if kind.endswith(".delta"):
                part[key] = part.get(key, "") + event.get("delta", "")
            else:
                part[key] = event.get(key, part.get(key, ""))

    @property
    def complete(self):
        return self.terminal is not None and self.terminal["type"] == "response.completed"

    def outputs(self):
        details = {"response_status": self.response_status}
        if self.terminal:
            details["incomplete_details"] = self.terminal["response"].get("incomplete_details")
        return [(0, [deepcopy(item) for _, item in sorted(self.items.items())], details)]
