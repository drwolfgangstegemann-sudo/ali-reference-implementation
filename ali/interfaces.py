from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Sequence

from .models import (
    BehaviourProposal,
    Event,
    ExecutionResult,
    NormEvaluation,
    Observation,
    ViabilityState,
)

if TYPE_CHECKING:
    from .configuration import ALIConfiguration


class _PluggableComponent:
    """Mixin that provides the from_config factory method for all ALI components.

    Every component class that participates in the plugin system must
    override from_config().  The default implementation raises
    NotImplementedError immediately, making a missing override visible
    at startup rather than at the first call.
    """

    @classmethod
    def from_config(
        cls,
        config: "ALIConfiguration",
        params: dict[str, Any],
    ) -> "_PluggableComponent":
        raise NotImplementedError(
            f"{cls.__name__} must implement from_config(config, params). "
            "See examples/llm_ego.py for a complete plugin example."
        )


class EnvironmentInterface(_PluggableComponent, ABC):
    @classmethod
    def from_config(
        cls,
        config: "ALIConfiguration",
        params: dict[str, Any],
    ) -> "EnvironmentInterface":
        raise NotImplementedError(f"{cls.__name__}.from_config() not implemented.")

    @abstractmethod
    def observe(self) -> list[Observation]: ...

    @abstractmethod
    def execute(self, proposal: BehaviourProposal) -> ExecutionResult: ...

    @abstractmethod
    def rollback(self, rollback_information: dict) -> None: ...

    @abstractmethod
    def is_inside_domain(self, path: str) -> bool: ...


class CausalCoreInterface(_PluggableComponent, ABC):
    @classmethod
    def from_config(
        cls,
        config: "ALIConfiguration",
        params: dict[str, Any],
    ) -> "CausalCoreInterface":
        raise NotImplementedError(f"{cls.__name__}.from_config() not implemented.")

    @abstractmethod
    def evaluate(
        self,
        observations: Sequence[Observation],
        *,
        unresolved_count: int = 0,
    ) -> ViabilityState: ...

    @abstractmethod
    def learn(self, event: Event) -> None: ...


class EgoInterface(_PluggableComponent, ABC):
    """Behavioural generation component.

    Plugin contract
    ---------------
    Implement from_config(config, params) to receive the full
    ALIConfiguration plus any extra parameters declared under
    config["components"]["ego"]["params"] in ali_config.json.
    """

    @classmethod
    def from_config(
        cls,
        config: "ALIConfiguration",
        params: dict[str, Any],
    ) -> "EgoInterface":
        raise NotImplementedError(f"{cls.__name__}.from_config() not implemented.")

    @abstractmethod
    def generate(
        self,
        observation: Observation,
        viability: ViabilityState,
        history: Sequence[Event],
    ) -> list[BehaviourProposal]: ...

    @abstractmethod
    def learn(self, event: Event) -> None: ...


class SuperEgoInterface(_PluggableComponent, ABC):
    @classmethod
    def from_config(
        cls,
        config: "ALIConfiguration",
        params: dict[str, Any],
    ) -> "SuperEgoInterface":
        raise NotImplementedError(f"{cls.__name__}.from_config() not implemented.")

    @abstractmethod
    def evaluate(
        self,
        proposal: BehaviourProposal,
        viability: ViabilityState,
    ) -> NormEvaluation: ...

    @abstractmethod
    def learn(self, event: Event) -> None: ...


class MemoryInterface(_PluggableComponent, ABC):
    @classmethod
    def from_config(
        cls,
        config: "ALIConfiguration",
        params: dict[str, Any],
    ) -> "MemoryInterface":
        raise NotImplementedError(f"{cls.__name__}.from_config() not implemented.")

    @abstractmethod
    def store(self, event: Event) -> None: ...

    @abstractmethod
    def retrieve_for_observation(
        self,
        observation: Observation,
        *,
        limit: int = 20,
    ) -> list[Event]: ...

    @abstractmethod
    def recent(self, *, limit: int = 20) -> list[Event]: ...
