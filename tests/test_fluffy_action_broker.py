from fluffy_action_broker import FluffyActionBroker, normalise_action_name


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return dict(self.payload)


class FakeSession:
    def __init__(self, payload=None):
        self.payload = payload or {"ok": True, "accepted": True}
        self.calls = []

    def post(self, url, json, timeout):
        self.calls.append((url, json, timeout))
        return FakeResponse(self.payload)


def test_aliases_are_normalised():
    assert normalise_action_name("High five!") == "high_five"
    assert normalise_action_name("push ups") == "push_up"
    assert normalise_action_name("Paw") == "hand_shake"


def test_unknown_action_never_reaches_robot():
    session = FakeSession()
    broker = FluffyActionBroker(
        "http://robot:8888",
        session=session,
        minimum_interval_seconds=0,
    )

    result = broker.execute("turn left", {})

    assert result["executed"] is False
    assert "not in" in result["reason"]
    assert session.calls == []


def test_tracking_state_blocks_stationary_action():
    session = FakeSession()
    broker = FluffyActionBroker(
        "http://robot:8888",
        session=session,
        minimum_interval_seconds=0,
    )

    result = broker.execute("nod", {"head_tracking_armed": True})

    assert result["executed"] is False
    assert "head tracking" in result["reason"]
    assert session.calls == []


def test_allowlisted_action_uses_fixed_daemon_payload():
    session = FakeSession()
    broker = FluffyActionBroker(
        "http://robot:8888/",
        session=session,
        minimum_interval_seconds=0,
    )

    result = broker.execute("push ups", {})

    assert result["executed"] is True
    assert session.calls == [
        (
            "http://robot:8888/command",
            {
                "cmd": "move_if_idle",
                "action": "push_up",
                "steps": 1,
                "speed": 70,
            },
            (1.0, 8.0),
        )
    ]


def test_busy_robot_is_reported_as_rejection():
    session = FakeSession({"ok": True, "accepted": False, "busy": True})
    broker = FluffyActionBroker(
        "http://robot:8888",
        session=session,
        minimum_interval_seconds=0,
    )

    result = broker.execute("sit", {})

    assert result["executed"] is False
    assert result["reason"] == "robot is busy"
