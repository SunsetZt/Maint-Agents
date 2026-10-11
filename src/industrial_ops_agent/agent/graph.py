"""One bounded LangGraph diagnosis loop; Temporal remains the retry owner."""

from __future__ import annotations

import asyncio
from collections.abc import Hashable
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any, Protocol, TypedDict, cast

from langgraph.graph import END, START, StateGraph

from industrial_ops_agent.agent.contracts import DIAGNOSIS_GRAPH_VERSION, budget_report


class DiagnosisGraphRuntime(Protocol):
    async def load(self) -> dict[str, Any]: ...
    async def enter(self, state: dict[str, Any], node: str) -> dict[str, Any]: ...
    async def commit(
        self, before: dict[str, Any], state: dict[str, Any], node: str
    ) -> dict[str, Any]: ...
    async def prepare(self, state: dict[str, Any]) -> dict[str, Any]: ...
    async def retrieve(self, state: dict[str, Any]) -> dict[str, Any]: ...
    async def decide(self, state: dict[str, Any]) -> dict[str, Any]: ...
    async def tool(self, state: dict[str, Any]) -> dict[str, Any]: ...
    async def finish(self, state: dict[str, Any]) -> dict[str, Any]: ...


class _Envelope(TypedDict):
    state: dict[str, Any]


class DiagnosisGraph:
    VERSION = DIAGNOSIS_GRAPH_VERSION

    def __init__(self, runtime: DiagnosisGraphRuntime) -> None:
        self._runtime = runtime
        graph = StateGraph(_Envelope)
        for name in ("prepare", "retrieve", "decide", "tool", "finish"):
            graph.add_node(name, self._node(name))
        destinations: dict[Hashable, str] = {
            name: name for name in ("prepare", "retrieve", "decide", "tool", "finish")
        }
        destinations["done"] = END
        graph.add_conditional_edges(START, self._next, destinations)
        for destination in destinations:
            if destination != "done":
                graph.add_conditional_edges(str(destination), self._next, destinations)
        self._compiled = graph.compile()  # No second checkpointer or retry policy.

    @staticmethod
    def _next(envelope: _Envelope) -> str:
        return str(envelope["state"]["next_node"])

    def _node(self, node: str) -> Any:
        async def execute(envelope: _Envelope) -> _Envelope:
            before = await self._runtime.enter(envelope["state"], node)
            state = deepcopy(before)
            if node in {"prepare", "retrieve", "tool"}:
                remaining = (
                    datetime.fromisoformat(state["deadline"]) - datetime.now(UTC)
                ).total_seconds()
                try:
                    async with asyncio.timeout(max(0.0, remaining)):
                        state = await getattr(self._runtime, node)(state)
                except TimeoutError:
                    state["report"] = budget_report("diagnosis_time_budget_exhausted")
            else:
                state = await getattr(self._runtime, node)(state)
            if state.get("report") is not None:
                state["next_node"] = "done"
            elif node == "prepare":
                state["next_node"] = "retrieve"
            elif node in {"retrieve", "tool"}:
                state["next_node"] = "decide"
            elif node == "decide":
                action = state["decision"]["action"]
                state["next_node"] = {
                    "RETRIEVE": "retrieve",
                    "CALL_TOOL": "tool",
                    "FINAL": "finish",
                    "NEEDS_INFORMATION": "finish",
                }[action]
                if action == "RETRIEVE":
                    state["retrieval_query"] = state["decision"]["query"]
                if (
                    action == "CALL_TOOL"
                    and state["tool_calls"] >= state["budget"]["max_tool_calls"]
                ):
                    state["report"] = budget_report("tool_budget_exhausted")
                    state["next_node"] = "done"
            return {"state": await self._runtime.commit(before, state, node)}

        return execute

    async def run(self) -> dict[str, Any]:
        state = await self._runtime.load()
        result = await self._compiled.ainvoke(
            {"state": state},
            config={"recursion_limit": max(8, 4 * int(state["budget"]["max_steps"]) + 8)},
        )
        return cast(dict[str, Any], result["state"])
