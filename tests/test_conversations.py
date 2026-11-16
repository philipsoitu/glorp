from glorp.conversations import ChatAssembler, ResponsesAssembler, comparison, incoming, shared_prefix


def test_narrow_equivalences_and_semantic_differences():
    a = {"role": "assistant", "content": "hi", "refusal": None}
    b = {"content": [{"type": "text", "text": "hi"}], "role": "assistant"}
    assert comparison(a, "openai-completions") == comparison(b, "openai-completions")
    assert shared_prefix([a], [dict(b, unknown=None)], "openai-completions") == 0
    assert shared_prefix([a], [dict(a, content="changed")], "openai-completions") == 0
    c = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
    }
    d = dict(c, tool_calls=[dict(c["tool_calls"][0], id="c2")])
    assert shared_prefix([c], [d], "openai-completions") == 0
    response = {
        "id": "m1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "hi", "annotations": []}],
    }
    assert comparison(response, "openai-responses") == {"id": "m1", "role": "assistant", "content": "hi"}
    assert shared_prefix([response], [dict(response, id="m2")], "openai-responses") == 0
    assert incoming({"input": "hi"}, "openai-responses") == [{"role": "user", "content": "hi"}]


def test_chat_parallel_tools_and_refusals():
    assembler = ChatAssembler()
    for delta in [
        {
            "role": "assistant",
            "content": "h",
            "refusal": "n",
            "tool_calls": [
                {"index": 1, "id": "call2", "type": "function", "function": {"name": "g", "arguments": "{"}},
                {"index": 0, "id": "call1", "type": "function", "function": {"name": "f", "arguments": "{"}},
            ],
        },
        {
            "content": "i",
            "refusal": "o",
            "tool_calls": [{"index": 0, "function": {"arguments": "}"}}, {"index": 1, "function": {"arguments": "}"}}],
        },
    ]:
        assembler.feed({"id": "c", "model": "versioned", "choices": [{"index": 0, "delta": delta}]})
    assembler.feed(
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}], "usage": {"total_tokens": 5}}
    )
    message = assembler.outputs()[0][1][0]
    assert message["content"] == "hi" and message["refusal"] == "no"
    assert [t["id"] for t in message["tool_calls"]] == ["call1", "call2"]
    assert all(t["function"]["arguments"] == "{}" for t in message["tool_calls"])
    assert assembler.complete and assembler.usage == {"total_tokens": 5}


def test_responses_partial_assembly():
    assembler = ResponsesAssembler()
    assembler.feed(
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "message", "role": "assistant", "id": "m1", "content": []},
        }
    )
    assembler.feed(
        {
            "type": "response.content_part.added",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        }
    )
    for text in ("hel", "lo"):
        assembler.feed({"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": text})
    assembler.feed(
        {
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {"type": "function_call", "id": "fc", "call_id": "c", "name": "f", "arguments": ""},
        }
    )
    assembler.feed({"type": "response.function_call_arguments.delta", "output_index": 1, "delta": "{}"})
    output = assembler.outputs()[0][1]
    assert output[0]["content"][0]["text"] == "hello"
    assert output[1]["arguments"] == "{}"
    assert not assembler.complete
