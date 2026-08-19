"""
Web UI for the simulation.

    uv run python server.py        # then open http://127.0.0.1:8000

The browser configures a run, the server streams it back frame by frame over a
websocket. The simulation itself is unchanged and unaware of any of this: the
server drives `World.iter_generation`, which pauses after every timestep, so
frames can be sent without blocking the event loop.

Sending a fresh `start` message at any point abandons the run in progress and
begins a new one, which is what makes the controls feel live.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import barriers
from capability_utils import Action, Sensor
from objectives import OBJECTIVES
from organism import World
from settings import Settings

HERE = Path(__file__).parent
STATIC = HERE / "static"

app = FastAPI(title="digital creatures")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/options")
async def options() -> dict[str, Any]:
    """
    Everything the UI needs to build its controls.

    The lists are derived from the enums and registries themselves, so adding a
    sense, an action, an objective or a barrier layout makes it appear in the
    browser with no front-end change at all.
    """
    defaults = Settings()
    return {
        "objectives": list(OBJECTIVES),
        "barriers": sorted(barriers.LAYOUTS),
        "sensors": [
            {"value": int(sensor), "label": str(sensor), "group": _sensor_group(sensor)}
            for sensor in Sensor
        ],
        "actions": [{"value": int(action), "label": str(action)} for action in Action],
        "defaults": {
            "objective": "left",
            "barriers": defaults.barrier_layout,
            "population": defaults.n_organisms,
            "steps": defaults.steps_per_generation,
            "generations": defaults.n_generations,
            "mutation_rate": defaults.point_mutation_rate,
            "n_genes": defaults.n_genes,
            "n_inner_neurons": defaults.n_inner_neurons,
        },
    }


def _sensor_group(sensor: Sensor) -> str:
    """Which heading a sense sits under in the capabilities menu."""
    name = sensor.name
    if name.startswith("PHEROMONE"):
        return "what can I smell"
    if name.startswith(("NEIGHBOUR", "BLOCKED", "POPULATION")):
        return "what is around me"
    if name.startswith(("X_", "Y_", "BORDER")):
        return "where am I"
    return "what was I doing"


# ----------------------------------------------------------------------------
# Turning a browser payload into Settings
# ----------------------------------------------------------------------------


def _clamp_int(value: Any, low: int, high: int, fallback: int) -> int:
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return fallback


def build_settings(payload: dict[str, Any]) -> Settings:
    """
    Build a `Settings` from whatever the browser sent.

    Every field is clamped rather than trusted: this server is meant to be run
    on a laptop, and a population of ten million would hang it just as
    effectively by accident as on purpose.
    """
    defaults = Settings()

    sensors = _selected(payload.get("sensors"), Sensor)
    actions = _selected(payload.get("actions"), Action)
    if not sensors:
        raise ValueError("at least one sense must be enabled")
    if not actions:
        raise ValueError("at least one action must be enabled")

    layout = payload.get("barriers", defaults.barrier_layout)
    if layout not in barriers.LAYOUTS:
        raise ValueError(f"unknown barrier layout {layout!r}")

    config = replace(
        defaults,
        n_organisms=_clamp_int(payload.get("population"), 2, 2000, defaults.n_organisms),
        steps_per_generation=_clamp_int(payload.get("steps"), 1, 2000, 200),
        n_generations=_clamp_int(payload.get("generations"), 1, 10_000, 100),
        n_genes=_clamp_int(payload.get("n_genes"), 1, 200, defaults.n_genes),
        n_inner_neurons=_clamp_int(payload.get("n_inner_neurons"), 1, 32, 4),
        barrier_layout=layout,
        point_mutation_rate=_clamp_float(
            payload.get("mutation_rate"), 0.0, 1.0, defaults.point_mutation_rate
        ),
    )
    return config.with_capabilities(sensors, actions)


def _clamp_float(value: Any, low: float, high: float, fallback: float) -> float:
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return fallback


def _selected(values: Any, enum: type[Sensor] | type[Action]) -> list[Any]:
    """Turn a list of raw numbers from the browser into valid enum members."""
    if not isinstance(values, list):
        return list(enum)

    chosen = []
    for value in values:
        try:
            chosen.append(enum(int(value)))
        except (TypeError, ValueError):
            continue
    return chosen


# ----------------------------------------------------------------------------
# Streaming a run
# ----------------------------------------------------------------------------


def _mask_to_base64(mask: np.ndarray) -> str:
    """Pack a grid into base64 so a frame stays small enough to send every step."""
    return base64.b64encode(np.ascontiguousarray(mask, dtype=np.uint8).tobytes()).decode("ascii")


def opening_frame(world: World, config: Settings) -> dict[str, Any]:
    """The layers that never change during a run, sent once."""
    return {
        "type": "start",
        "width": world.width,
        "height": world.height,
        "generations": config.n_generations,
        "steps": config.steps_per_generation,
        "population": config.n_organisms,
        "barriers": _mask_to_base64(world.barriers),
        "dynamic_zones": world.objective.dynamic,
        "zones": [
            {
                "label": shading.label,
                "colour": shading.colour,
                "mask": _mask_to_base64(shading.mask),
            }
            for shading in world.objective.zones(world)
        ],
        "sensors": [str(s) for s in config.enabled_sensors],
        "actions": [str(a) for a in config.enabled_actions],
    }


def step_frame(world: World, step: int, send_zones: bool) -> dict[str, Any]:
    """One timestep: where everything is, and how strong the scent is."""
    xs, ys = world.positions()
    frame: dict[str, Any] = {
        "type": "frame",
        "generation": world.generation,
        "step": step,
        "alive": len(xs),
        # Interleaved x, y keeps the payload half the size of a list of pairs.
        "positions": np.column_stack((xs, ys)).ravel().tolist(),
        "pheromone": _mask_to_base64(np.clip(world.pheromone, 0.0, 1.0) * 255),
    }
    if send_zones:
        frame["zones"] = [_mask_to_base64(s.mask) for s in world.objective.zones(world)]
    return frame


async def stream_run(websocket: WebSocket, payload: dict[str, Any]) -> None:
    """Run the simulation, sending a frame per timestep until it ends or is cancelled."""
    config = build_settings(payload)
    seed = payload.get("seed")
    if seed not in (None, ""):
        random.seed(int(seed))
        np.random.seed(int(seed) % (2**32))

    world = World(config=config, objective=payload.get("objective", "left"))
    await websocket.send_json(opening_frame(world, config))

    # How many timesteps to simulate between frames. Higher means a faster,
    # choppier run; the UI exposes it as a speed control.
    stride = _clamp_int(payload.get("stride"), 1, 200, 1)
    history: list[float] = []

    for _ in range(config.n_generations):
        generation = world.iter_generation()
        pending = 0
        while True:
            try:
                step = next(generation)
            except StopIteration as finished:
                survivors = finished.value
                break

            pending += 1
            if pending >= stride:
                pending = 0
                await websocket.send_json(step_frame(world, step, world.objective.dynamic))
                # Yield to the event loop so an incoming 'stop' is acted on
                # promptly rather than at the end of the generation.
                await asyncio.sleep(0)

        history.append(survivors / config.n_organisms)
        await websocket.send_json(
            {
                "type": "generation",
                "generation": world.generation,
                "survivors": survivors,
                "population": config.n_organisms,
                "history": history,
                "brain": world.organisms[0].brain.describe() if world.organisms else "",
            }
        )
        await asyncio.sleep(0)

    await websocket.send_json({"type": "done"})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    task: asyncio.Task[None] | None = None

    async def cancel_running() -> None:
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    try:
        while True:
            message = await websocket.receive_json()
            action = message.get("action")

            if action == "start":
                await cancel_running()
                task = asyncio.create_task(_guarded(websocket, message))
            elif action == "stop":
                await cancel_running()
                await websocket.send_json({"type": "stopped"})
    except WebSocketDisconnect:
        pass
    finally:
        await cancel_running()


async def _guarded(websocket: WebSocket, message: dict[str, Any]) -> None:
    """Run a simulation, reporting a bad configuration instead of dropping the socket."""
    try:
        await stream_run(websocket, message)
    except asyncio.CancelledError:
        raise
    except ValueError as invalid:
        await websocket.send_json({"type": "error", "message": str(invalid)})
    except WebSocketDisconnect:
        pass


def main() -> None:
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
