from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from math import inf
from typing import Iterable, List, Sequence, Tuple


DAY_NAMES = (
    "Day 1",
    "Day 2",
    "Day 3",
    "Day 4",
    "Day 5",
    "Day 6",
    "Day 7",
)


@dataclass(frozen=True, order=True)
class Task:
    name: str
    workload_hours: int
    deadline_days: int

    def __post_init__(self) -> None:
        if self.workload_hours < 0:
            raise ValueError("workload_hours must be non-negative")
        if self.deadline_days <= 0:
            raise ValueError("deadline_days must be positive")


@dataclass(frozen=True)
class State:
    current_week_schedule: Tuple[Tuple[str, ...], ...]
    remaining_tasks: Tuple[Task, ...]

    def is_goal(self) -> bool:
        return not self.remaining_tasks

    def total_conflicts(self) -> int:
        return sum(max(0, len(set(day_blocks)) - 1) for day_blocks in self.current_week_schedule)


def _normalize_tasks(tasks: Iterable[Task], num_days: int) -> Tuple[Task, ...]:
    normalized = []
    for task in tasks:
        if task.deadline_days > num_days:
            normalized.append(Task(task.name, task.workload_hours, num_days))
        else:
            normalized.append(task)
    return tuple(sorted(normalized, key=lambda item: (item.deadline_days, -item.workload_hours, item.name)))


def initial_state(tasks: Iterable[Task], num_days: int = 7) -> State:
    schedule = tuple(() for _ in range(num_days))
    return State(schedule, _normalize_tasks(tasks, num_days))


def _replace_day(schedule: Tuple[Tuple[str, ...], ...], day_index: int, new_day: Tuple[str, ...]) -> Tuple[Tuple[str, ...], ...]:
    updated_schedule = list(schedule)
    updated_schedule[day_index] = new_day
    return tuple(updated_schedule)


def _replace_task(tasks: Tuple[Task, ...], task_index: int, updated_task: Task | None) -> Tuple[Task, ...]:
    updated_tasks = list(tasks)
    if updated_task is None:
        updated_tasks.pop(task_index)
    else:
        updated_tasks[task_index] = updated_task
    return tuple(sorted(updated_tasks, key=lambda item: (item.deadline_days, -item.workload_hours, item.name)))


def _day_incremental_conflict(day_blocks: Tuple[str, ...], task_name: str) -> int:
    if not day_blocks:
        return 0
    distinct_tasks = set(day_blocks)
    return 0 if task_name in distinct_tasks else 1


def _task_conflict_free_capacity(state: State, task: Task, daily_capacity: int) -> int:
    capacity = 0
    for day_index in range(task.deadline_days):
        day_blocks = state.current_week_schedule[day_index]
        if len(day_blocks) >= daily_capacity:
            continue
        distinct_tasks = set(day_blocks)
        if not distinct_tasks or distinct_tasks == {task.name}:
            capacity += daily_capacity - len(day_blocks)
    return capacity


def _remaining_capacity_before_deadline(state: State, task: Task, daily_capacity: int) -> int:
    capacity = 0
    for day_index in range(task.deadline_days):
        day_blocks = state.current_week_schedule[day_index]
        capacity += max(0, daily_capacity - len(day_blocks))
    return capacity


def is_feasible(state: State, daily_capacity: int) -> bool:
    for task in state.remaining_tasks:
        if _remaining_capacity_before_deadline(state, task, daily_capacity) < task.workload_hours:
            return False
    return True


def _candidate_days(state: State, task: Task, daily_capacity: int) -> List[int]:
    ranked_days = []
    for day_index in range(task.deadline_days):
        day_blocks = state.current_week_schedule[day_index]
        if len(day_blocks) >= daily_capacity:
            continue

        distinct_tasks = set(day_blocks)
        same_task_priority = 0 if task.name in distinct_tasks else 1
        empty_day_priority = 0 if not day_blocks else 1
        ranked_days.append((same_task_priority, empty_day_priority, day_index, day_index))

    ranked_days.sort()
    return [day_index for _, _, _, day_index in ranked_days]


def expand_state(state: State, daily_capacity: int) -> List[Tuple[State, int]]:
    successors: List[Tuple[State, int]] = []

    for task_index, task in enumerate(state.remaining_tasks):
        for day_index in _candidate_days(state, task, daily_capacity):
            day_blocks = state.current_week_schedule[day_index]
            incremental_conflict = _day_incremental_conflict(day_blocks, task.name)
            updated_day = tuple(sorted(day_blocks + (task.name,)))
            updated_schedule = _replace_day(state.current_week_schedule, day_index, updated_day)

            if task.workload_hours == 1:
                updated_tasks = _replace_task(state.remaining_tasks, task_index, None)
            else:
                updated_task = Task(task.name, task.workload_hours - 1, task.deadline_days)
                updated_tasks = _replace_task(state.remaining_tasks, task_index, updated_task)

            next_state = State(updated_schedule, updated_tasks)
            if is_feasible(next_state, daily_capacity):
                successors.append((next_state, incremental_conflict))

    return successors


def bfs_schedule(tasks: Sequence[Task], num_days: int = 7, daily_capacity: int = 4) -> State | None:
    start = initial_state(tasks, num_days)
    if not is_feasible(start, daily_capacity):
        return None

    queue = deque([start])
    visited = {start}
    best_goal: State | None = None
    best_conflicts = inf

    while queue:
        state = queue.popleft()
        current_conflicts = state.total_conflicts()

        if current_conflicts > best_conflicts:
            continue

        if state.is_goal():
            if current_conflicts < best_conflicts:
                best_goal = state
                best_conflicts = current_conflicts
            continue

        for next_state, _ in expand_state(state, daily_capacity):
            if next_state in visited:
                continue
            visited.add(next_state)
            queue.append(next_state)

    return best_goal


def _a_star_heuristic(state: State, daily_capacity: int) -> int:
    lower_bound = 0
    for task in state.remaining_tasks:
        conflict_free_capacity = _task_conflict_free_capacity(state, task, daily_capacity)
        if task.workload_hours > _remaining_capacity_before_deadline(state, task, daily_capacity):
            return inf
        if task.workload_hours > conflict_free_capacity:
            lower_bound += 1
    return lower_bound


def _slack_score(state: State, daily_capacity: int) -> int:
    score = 0
    for task in state.remaining_tasks:
        remaining_capacity = _remaining_capacity_before_deadline(state, task, daily_capacity)
        score += remaining_capacity - task.workload_hours
    return score


def astar_schedule(tasks: Sequence[Task], num_days: int = 7, daily_capacity: int = 4) -> State | None:
    start = initial_state(tasks, num_days)
    if not is_feasible(start, daily_capacity):
        return None

    frontier = []
    push_order = count()
    best_cost = {start: 0}
    start_h = _a_star_heuristic(start, daily_capacity)
    heappush(frontier, (start_h, _slack_score(start, daily_capacity), next(push_order), start))

    while frontier:
        _, _, _, state = heappop(frontier)
        current_cost = best_cost[state]

        if state.is_goal():
            return state

        for next_state, step_cost in expand_state(state, daily_capacity):
            new_cost = current_cost + step_cost
            if new_cost >= best_cost.get(next_state, inf):
                continue

            heuristic = _a_star_heuristic(next_state, daily_capacity)
            if heuristic == inf:
                continue

            best_cost[next_state] = new_cost
            heappush(
                frontier,
                (
                    new_cost + heuristic,
                    _slack_score(next_state, daily_capacity),
                    next(push_order),
                    next_state,
                ),
            )

    return None


def schedule_to_text(state: State | None) -> str:
    if state is None:
        return "No valid schedule was found."

    lines = []
    for day_index, day_blocks in enumerate(state.current_week_schedule):
        label = DAY_NAMES[day_index] if day_index < len(DAY_NAMES) else f"Day {day_index + 1}"
        if day_blocks:
            lines.append(f"{label}: {', '.join(day_blocks)}")
        else:
            lines.append(f"{label}: Free")
    lines.append(f"Conflicts: {state.total_conflicts()}")
    return "\n".join(lines)


if __name__ == "__main__":
    sample_tasks = [
        Task("Math", 4, 3),
        Task("Biology", 3, 5),
        Task("History", 2, 4),
    ]

    bfs_result = bfs_schedule(sample_tasks, num_days=7, daily_capacity=3)
    astar_result = astar_schedule(sample_tasks, num_days=7, daily_capacity=3)

    print("BFS schedule")
    print(schedule_to_text(bfs_result))
    print()
    print("A* schedule")
    print(schedule_to_text(astar_result))
