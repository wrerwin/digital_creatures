"""
Checks for the web layer. Run with the rest:

    uv run pytest

The simulation is tested in `test_simulation.py`; what matters here is that the
browser cannot configure a run the server should refuse, that the options the UI
builds its controls from stay in step with the Python enums, and that a run
actually streams frames.
"""

from __future__ import annotations

import base64
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

import barriers
import reproduction
from capability_utils import Action, Sensor
from objectives import OBJECTIVES
from server import app, build_settings
from settings import Settings


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def full_payload(**overrides: Any) -> dict[str, Any]:
    """A valid configuration, as the browser would send it."""
    payload: dict[str, Any] = {
        "action": "start",
        "objective": "left",
        "barriers": "none",
        "population": 30,
        "steps": 5,
        "generations": 1,
        "n_genes": 8,
        "n_inner_neurons": 2,
        "mutation_rate": 0.02,
        "reproduction": "asexual",
        "energy_enabled": True,
        "initial_energy": 140.0,
        "sense_cost": 0.04,
        "survival_zone_fraction": 0.12,
        "zone_shrink": 0.0,
        "offspring_per_survivor": 2.0,
        "carrying_capacity": 400,
        "mating_radius": 18,
        "stride": 1,
        "seed": 3,
        "sensors": [int(s) for s in Sensor],
        "actions": [int(a) for a in Action],
    }
    payload.update(overrides)
    return payload


# ----------------------------------------------------------------------------
# The options the UI is built from
# ----------------------------------------------------------------------------


def test_options_expose_every_capability(client: TestClient) -> None:
    """
    The UI builds its menus from this, so anything missing is invisible forever.

    A new sense added to the enum should appear in the browser with no
    front-end change at all, which only holds if this is derived and not typed
    out by hand.
    """
    options = client.get("/api/options").json()

    assert [item["label"] for item in options["sensors"]] == [str(s) for s in Sensor]
    assert [item["label"] for item in options["actions"]] == [str(a) for a in Action]
    assert options["objectives"] == list(OBJECTIVES)
    assert options["barriers"] == sorted(barriers.LAYOUTS)
    assert options["reproduction"] == list(reproduction.STRATEGIES)


def test_every_sensor_gets_a_menu_heading(client: TestClient) -> None:
    """An ungrouped sense would silently fall into the wrong section of the menu."""
    options = client.get("/api/options").json()
    for item in options["sensors"]:
        assert item["group"], f"{item['label']} has no group"


def test_the_page_is_served(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "digital creatures" in response.text


# ----------------------------------------------------------------------------
# Turning a browser payload into Settings
# ----------------------------------------------------------------------------


def test_a_full_payload_round_trips(client: TestClient) -> None:
    config = build_settings(full_payload())
    assert config.n_organisms == 30
    assert config.steps_per_generation == 5
    assert set(config.enabled_sensors) == set(Sensor)
    assert set(config.enabled_actions) == set(Action)


def test_capability_choices_reach_the_settings() -> None:
    """The checkboxes are the point of the UI, so pin that they take effect."""
    config = build_settings(full_payload(sensors=[int(Sensor.BIAS)], actions=[int(Action.STAY)]))
    assert config.enabled_sensors == (Sensor.BIAS,)
    assert config.enabled_actions == (Action.STAY,)


@pytest.mark.parametrize(
    ("field", "sent", "expected"),
    [
        ("population", 10**9, 2000),
        ("population", -5, 2),
        ("steps", 0, 1),
        ("generations", 10**9, 10_000),
        ("n_inner_neurons", 999, 32),
        ("mutation_rate", 5.0, 1.0),
        ("mutation_rate", -1.0, 0.0),
    ],
)
def test_absurd_numbers_are_clamped(field: str, sent: Any, expected: Any) -> None:
    """
    Anything from the browser is clamped rather than trusted.

    A population of a billion would hang the server just as effectively by
    accident as on purpose.
    """
    config = build_settings(full_payload(**{field: sent}))
    attribute = {
        "population": "n_organisms",
        "steps": "steps_per_generation",
        "generations": "n_generations",
        "n_inner_neurons": "n_inner_neurons",
        "mutation_rate": "point_mutation_rate",
    }[field]
    assert getattr(config, attribute) == expected


def test_nonsense_values_fall_back_to_defaults() -> None:
    """Junk in a numeric field should not crash the run."""
    defaults = Settings()
    config = build_settings(full_payload(population="not a number", mutation_rate=None))
    assert config.n_organisms == defaults.n_organisms
    assert config.point_mutation_rate == defaults.point_mutation_rate


def test_empty_capability_selections_are_refused() -> None:
    with pytest.raises(ValueError, match="at least one sense"):
        build_settings(full_payload(sensors=[]))
    with pytest.raises(ValueError, match="at least one action"):
        build_settings(full_payload(actions=[]))


def test_unknown_capability_numbers_are_ignored() -> None:
    """Out-of-range values are dropped, not crashed on."""
    config = build_settings(full_payload(sensors=[int(Sensor.BIAS), 999, "x"]))
    assert config.enabled_sensors == (Sensor.BIAS,)


def test_unknown_barrier_layout_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown barrier layout"):
        build_settings(full_payload(barriers="../../etc/passwd"))


# ----------------------------------------------------------------------------
# Streaming a run
# ----------------------------------------------------------------------------


def test_a_run_streams_frames_and_finishes(client: TestClient) -> None:
    """The end-to-end path: configure, stream every step, report the generation."""
    with client.websocket_connect("/ws") as socket:
        socket.send_json(full_payload(steps=4, generations=1, population=20))

        opening = socket.receive_json()
        assert opening["type"] == "start"
        assert (opening["width"], opening["height"]) == (100, 100)

        frames = []
        while True:
            message = socket.receive_json()
            if message["type"] == "frame":
                frames.append(message)
                continue
            if message["type"] == "generation":
                assert 0 <= message["survivors"] <= 20
                assert message["brain"], "no example brain reported"
                continue
            assert message["type"] == "done"
            break

        assert len(frames) == 4, "expected one frame per timestep"
        assert [frame["step"] for frame in frames] == [0, 1, 2, 3]


def test_frames_carry_positions_and_scent(client: TestClient) -> None:
    """Whatever the client needs to paint must actually be in the frame."""
    with client.websocket_connect("/ws") as socket:
        socket.send_json(full_payload(steps=1, generations=1, population=20))
        opening = socket.receive_json()
        frame = socket.receive_json()

        assert frame["type"] == "frame"
        # Interleaved x, y, so two numbers per living creature.
        assert len(frame["positions"]) == 2 * frame["alive"]
        assert frame["alive"] <= 20

        cells = opening["width"] * opening["height"]
        assert len(base64.b64decode(frame["pheromone"])) == cells
        assert len(base64.b64decode(opening["barriers"])) == cells


def test_masks_decode_to_the_grid_the_client_expects(client: TestClient) -> None:
    """
    Barrier bytes must line up with the layout, in the order the client unpacks.

    The client reads index [x * height + y]; getting this transposed would draw
    a plausible-looking but wrong world.
    """
    with client.websocket_connect("/ws") as socket:
        socket.send_json(full_payload(barriers="wall", steps=1, generations=1))
        opening = socket.receive_json()

    raw = np.frombuffer(base64.b64decode(opening["barriers"]), dtype=np.uint8)
    grid = raw.reshape(opening["width"], opening["height"]).astype(bool)
    assert np.array_equal(grid, barriers.build("wall", 100, 100))


def test_a_bad_configuration_reports_an_error_without_dropping_the_socket(
    client: TestClient,
) -> None:
    """A mistake in the form should say so, and leave the UI usable."""
    with client.websocket_connect("/ws") as socket:
        socket.send_json(full_payload(sensors=[]))
        message = socket.receive_json()
        assert message["type"] == "error"
        assert "sense" in message["message"]

        # The socket must still work afterwards.
        socket.send_json(full_payload(steps=1, generations=1, population=10))
        assert socket.receive_json()["type"] == "start"


def test_the_hazard_objective_streams_its_moving_zone(client: TestClient) -> None:
    """A dynamic objective has to resend its zones, or the danger never moves."""
    with client.websocket_connect("/ws") as socket:
        socket.send_json(full_payload(objective="hazard", steps=2, generations=1))
        opening = socket.receive_json()
        assert opening["dynamic_zones"] is True

        frame = socket.receive_json()
        assert frame["type"] == "frame"
        assert "zones" in frame, "a moving hazard must resend its zone each frame"


def test_a_static_objective_does_not_resend_its_zones(client: TestClient) -> None:
    """Zones that never change are sent once, to keep frames small."""
    with client.websocket_connect("/ws") as socket:
        socket.send_json(full_payload(objective="left", steps=2, generations=1))
        opening = socket.receive_json()
        assert opening["dynamic_zones"] is False
        assert opening["zones"], "a zone objective should describe its zone up front"

        frame = socket.receive_json()
        assert "zones" not in frame


def test_the_new_mechanics_reach_the_settings() -> None:
    """Every control the UI grew has to actually change the run."""
    config = build_settings(
        full_payload(
            reproduction="sexual",
            energy_enabled=False,
            initial_energy=55.0,
            sense_cost=0.5,
            zone_shrink=0.02,
            offspring_per_survivor=3.5,
            carrying_capacity=99,
            mating_radius=42,
        )
    )
    assert config.reproduction == "sexual"
    assert config.energy_enabled is False
    assert config.initial_energy == 55.0
    assert config.sense_cost == 0.5
    assert config.zone_shrink_per_generation == 0.02
    assert config.offspring_per_survivor == 3.5
    assert config.carrying_capacity == 99
    assert config.mating_radius == 42


def test_unknown_reproduction_strategy_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown reproduction strategy"):
        build_settings(full_payload(reproduction="mitosis"))


def test_a_generation_message_carries_the_population_picture(client: TestClient) -> None:
    """The readouts and the gene-expression chart are all fed from this one message."""
    with client.websocket_connect("/ws") as socket:
        socket.send_json(full_payload(steps=3, generations=1, population=30))
        socket.receive_json()

        while (message := socket.receive_json())["type"] != "generation":
            pass

        assert message["previous_population"] == 30
        assert message["capacity"] == 400
        assert message["zone_fraction"] == pytest.approx(0.12)
        assert len(message["populations"]) == 1

        expression = message["expression"]
        assert [item["label"] for item in expression["sensors"]] == [str(s) for s in Sensor]
        assert [item["label"] for item in expression["actions"]] == [str(a) for a in Action]
        assert expression["lineages"]["alive"] >= 1
        assert expression["mean_senses_used"] > 0


def test_an_extinct_run_says_so_and_stops(client: TestClient) -> None:
    """A population that dies out must report it rather than looking finished."""
    with client.websocket_connect("/ws") as socket:
        socket.send_json(
            full_payload(
                steps=8,
                generations=50,
                population=20,
                survival_zone_fraction=0.01,
                offspring_per_survivor=0.5,
                stride=50,
            )
        )
        socket.receive_json()

        while (message := socket.receive_json())["type"] != "done":
            pass
        assert message["extinct"] is True


def test_stride_thins_the_frames(client: TestClient) -> None:
    """The speed control must actually reduce how much is sent."""
    with client.websocket_connect("/ws") as socket:
        socket.send_json(full_payload(steps=20, generations=1, stride=5))
        socket.receive_json()

        frames = 0
        while True:
            message = socket.receive_json()
            if message["type"] == "frame":
                frames += 1
            elif message["type"] == "done":
                break

        assert frames == 4, f"expected 20/5 frames, got {frames}"
