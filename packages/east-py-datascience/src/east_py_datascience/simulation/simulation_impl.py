#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""Discrete Event Simulation (DES) platform functions for East.

Provides a generic priority-queue DES engine grounded in the REA
(Resources-Events-Agents) economic ontology. No external dependencies
beyond stdlib and numpy.
"""

import copy
import heapq
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import numpy as np
from east.runtime.errors import EastError
from east.runtime.platform import GenericPlatformFunction
from east.types.values import EastArray, EastStruct, EastVariant, is_east_variant

# ============================================================================
# Helper Functions
# ============================================================================


def _get_option(opt: EastVariant | None, default: Any) -> Any:
    """Extract value from Option variant, returning default if None."""
    if opt is None:
        return default
    if is_east_variant(opt) and opt.type == "some":
        return opt.value
    return default


def _datetime_key(dt: datetime) -> float:
    """Convert datetime to a sortable float (timestamp)."""
    return dt.timestamp()


# ============================================================================
# DES Core Engine
# ============================================================================


def _run_single(
    state: Any,
    initial_events: EastArray,
    process_fn: Callable[[Any, datetime, Any], EastStruct],
    max_events: int,
    end_date: datetime | None,
) -> EastStruct:
    """Run a single DES trajectory.

    Uses a heapq priority queue ordered by event date.
    Each event is processed by calling process_fn(state, date, event),
    which returns { state, events } — the new state and any follow-on events.
    """
    # Build initial priority queue: (timestamp, tiebreaker, scheduled_event)
    heap: list[tuple[float, int, Any]] = []
    counter = 0
    for scheduled in initial_events:
        dt = scheduled.get("date")
        heap.append((_datetime_key(dt), counter, scheduled))
        counter += 1
    heapq.heapify(heap)

    events_processed = 0
    last_date = datetime(1970, 1, 1, tzinfo=UTC)

    while heap and events_processed < max_events:
        _ts, _tie, scheduled = heapq.heappop(heap)
        event_date = scheduled.get("date")

        # Stop if past end_date
        if end_date is not None and event_date > end_date:
            break

        event = scheduled.get("event")
        result = process_fn(state, event_date, event)

        state = result.get("state")
        last_date = event_date
        events_processed += 1

        # Schedule follow-on events
        new_events = result.get("events")
        for new_scheduled in new_events:
            new_dt = new_scheduled.get("date")
            if new_dt < event_date:
                new_event = new_scheduled.get("event")
                event_tag = new_event.type if is_east_variant(new_event) else str(new_event)
                raise EastError(
                    f"simulation_run: process handler returned event '{event_tag}' "
                    f"scheduled at {new_dt.isoformat()} which is before the "
                    f"current event date {event_date.isoformat()}. "
                    f"Events must not be scheduled in the past."
                )
            heapq.heappush(heap, (_datetime_key(new_dt), counter, new_scheduled))
            counter += 1

    return EastStruct({
        "final_state": state,
        "events_processed": events_processed,
        "final_date": last_date,
    })


# ============================================================================
# Platform Function Implementations
# ============================================================================


def simulation_run_impl(
    initial_state: Any,
    initial_events: EastArray,
    process_fn: Callable[[Any, datetime, Any], EastStruct],
    config: EastStruct,
) -> EastStruct:
    """Run a single deterministic discrete event simulation."""
    max_events = int(_get_option(config.get("max_events"), 100_000))
    end_date = _get_option(config.get("end_date"), None)

    return _run_single(initial_state, initial_events, process_fn, max_events, end_date)


def simulation_run_trajectories_impl(
    initial_state: Any,
    initial_events: EastArray,
    process_fn: Callable[[Any, datetime, Any], EastStruct],
    config: EastStruct,
) -> EastStruct:
    """Run Monte Carlo simulation trajectories."""
    num_trajectories = int(config.get("trajectories"))
    base_seed = int(_get_option(config.get("seed"), 0))
    max_events = int(_get_option(config.get("max_events"), 100_000))
    end_date = _get_option(config.get("end_date"), None)

    trajectories: list[EastStruct] = []
    for i in range(num_trajectories):
        # Seed RNG for this trajectory
        np.random.seed(base_seed + i)

        # Deep copy state so each trajectory is independent
        state_copy = copy.deepcopy(initial_state)

        result = _run_single(state_copy, initial_events, process_fn, max_events, end_date)
        trajectories.append(result)

    return EastStruct({
        "trajectories": EastArray(None, trajectories),
    })


# ============================================================================
# Platform Function Registration
# ============================================================================

simulation_impl = [
    GenericPlatformFunction(
        name="simulation_run",
        type_parameters=["R", "E"],
        type="sync",
        fn=lambda R, E: simulation_run_impl,
    ),
    GenericPlatformFunction(
        name="simulation_run_trajectories",
        type_parameters=["R", "E"],
        type="sync",
        fn=lambda R, E: simulation_run_trajectories_impl,
    ),
]

__all__ = [
    "simulation_impl",
]
