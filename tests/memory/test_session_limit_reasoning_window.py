"""A limited session window must not start inside a stored model response."""

import pytest

from agents import Agent, RunConfig, Runner, SessionSettings, SQLiteSession
from agents.testing import ScriptedModel, assistant_message

ANSWER = {
    "type": "message",
    "id": "msg_1",
    "role": "assistant",
    "status": "completed",
    "content": [{"type": "output_text", "text": "a1", "annotations": []}],
}
HISTORY = [
    {"role": "user", "content": "q1"},
    {"type": "reasoning", "id": "rs_1", "summary": []},
    ANSWER,
    {"role": "user", "content": "q2"},
    {"type": "reasoning", "id": "rs_2", "summary": []},
    {
        **ANSWER,
        "id": "msg_2",
        "content": [{"type": "output_text", "text": "a2", "annotations": []}],
    },
]
CALL = [
    {"role": "user", "content": "q1"},
    {"type": "reasoning", "id": "rs_1", "summary": []},
    {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "lookup",
        "arguments": "{}",
    },
    {"type": "function_call_output", "call_id": "call_1", "output": "42"},
    {"role": "user", "content": "q2"},
]


async def _sent(items: list[dict], limit: int | None) -> list[object]:
    session = SQLiteSession("window")
    await session.add_items(items)  # type: ignore[arg-type]
    model = ScriptedModel([[assistant_message("ok")]])
    settings = SessionSettings(limit=limit) if limit else None
    await Runner.run(
        Agent(name="a", model=model),
        "q3",
        session=session,
        run_config=RunConfig(session_settings=settings),
    )
    return [item.get("id") or item.get("content") for item in model.calls[0].input]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("items", "limit", "expected"),
    [
        # The answer stays but the reasoning item that opened it was cut off.
        (HISTORY, 4, ["q2", "rs_2", "msg_2", "q3"]),
        # The call goes, and the output that answered it goes with it.
        (CALL, 3, ["q2", "q3"]),
        # A reasoning item may open a window; its answer stays with it.
        (HISTORY, 3, ["q2", "rs_2", "msg_2", "q3"]),
        # Without a window the history is sent as stored.
        (HISTORY[2:], None, ["msg_1", "q2", "rs_2", "msg_2", "q3"]),
    ],
)
async def test_window_start_does_not_orphan_an_answer(
    items: list[dict], limit: int | None, expected: list[object]
) -> None:
    assert await _sent(items, limit) == expected
