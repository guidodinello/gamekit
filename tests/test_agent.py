from __future__ import annotations

from gamekit import Agent


class Move:
    pass


class State:
    pass


type MyAgent = Agent[State, Move]


class GoodAgent:
    name = "good"

    def choose_action(
        self, state: State, legal_actions: list[Move], player_idx: int
    ) -> Move:
        return legal_actions[0]

    def reset(self) -> None:
        pass


class MissingName:
    def choose_action(
        self, state: State, legal_actions: list[Move], player_idx: int
    ) -> Move:
        return legal_actions[0]

    def reset(self) -> None:
        pass


def test_structural_agent_satisfies_the_protocol() -> None:
    assert isinstance(GoodAgent(), Agent)


def test_missing_name_attribute_fails_isinstance() -> None:
    assert not isinstance(MissingName(), Agent)


def test_agent_can_be_parameterised_for_annotations() -> None:
    assert MyAgent is not None
