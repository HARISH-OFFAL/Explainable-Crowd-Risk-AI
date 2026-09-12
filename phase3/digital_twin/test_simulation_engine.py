import unittest

from .simulation_engine import simulate


def snapshot():
    return {"zones": {zone: {"current_people": 20, "capacity": 100, "inflow_rate": 12, "outflow_rate": 6, "relative_speed": 30} for zone in ("ZONE_A", "ZONE_B", "ZONE_C")}}


class SimulationTests(unittest.TestCase):
    def test_closed_system_no_action_conserves_initial_state(self):
        state = snapshot()
        for zone in state["zones"].values(): zone["inflow_rate"] = zone["outflow_rate"] = 0
        result = simulate(state, {"type": "NO_ACTION"}, 60)
        self.assertEqual(result["total_people"], 60)
        self.assertEqual([result["zones"][zone]["people"] for zone in ("ZONE_A", "ZONE_B", "ZONE_C")], [20, 20, 20])

    def test_closed_system_redirect_moves_people_without_creating_them(self):
        state = snapshot()
        for zone in state["zones"].values(): zone["inflow_rate"] = zone["outflow_rate"] = 0
        result = simulate(state, {"type": "REDIRECT", "redirect_people": 5, "from_zone": "ZONE_B", "to_zone": "ZONE_C"}, 15)
        self.assertEqual(result["total_people"], 60)
        self.assertEqual(result["zones"]["ZONE_B"]["people"], 15)
        self.assertEqual(result["zones"]["ZONE_C"]["people"], 25)

    def test_percentage_redirect_is_one_for_one(self):
        state = snapshot()
        for zone in state["zones"].values(): zone["outflow_rate"] = 0
        state["zones"]["ZONE_B"]["inflow_rate"] = 15
        before = simulate(state, {"type": "NO_ACTION"}, 60)
        redirected = simulate(state, {"type": "REDIRECT", "redirect_percent": 20, "from_zone": "ZONE_B", "to_zone": "ZONE_C"}, 60)
        self.assertEqual(redirected["total_people"], before["total_people"])
        self.assertAlmostEqual(redirected["zones"]["ZONE_C"]["people"] - before["zones"]["ZONE_C"]["people"], 3, places=4)

    def test_external_entry_uses_persons_per_minute(self):
        state = snapshot()
        for zone in state["zones"].values(): zone["inflow_rate"] = zone["outflow_rate"] = 0
        state["external_entry_rate"] = 10
        result = simulate(state, {"type": "NO_ACTION"}, 60)
        self.assertAlmostEqual(result["total_people"], 70, places=4)

    def test_external_exit_uses_persons_per_minute_and_never_negative(self):
        state = snapshot()
        for zone in state["zones"].values(): zone["inflow_rate"] = zone["outflow_rate"] = 0
        state["external_exit_rate"] = 10
        result = simulate(state, {"type": "NO_ACTION"}, 60)
        self.assertAlmostEqual(result["total_people"], 50, places=4)
        self.assertTrue(all(item["people"] >= 0 for item in result["zones"].values()))

    def test_redirect_respects_destination_capacity(self):
        state = snapshot(); state["zones"]["ZONE_B"]["current_people"] = 10; state["zones"]["ZONE_C"]["current_people"] = 99
        for zone in state["zones"].values(): zone["inflow_rate"] = zone["outflow_rate"] = 0
        result = simulate(state, {"type": "REDIRECT", "redirect_people": 5, "from_zone": "ZONE_B", "to_zone": "ZONE_C"}, 15)
        self.assertEqual(result["zones"]["ZONE_C"]["people"], 100)
        self.assertEqual(result["transfer_summary"]["effective_redirect_people"], 1)
        self.assertEqual(result["transfer_summary"]["blocked_redirect_people"], 4)
    def test_zero_flow_is_stable(self):
        state = snapshot()
        for zone in state["zones"].values(): zone["inflow_rate"] = zone["outflow_rate"] = 0
        result = simulate(state, {"type": "NO_ACTION"}, 15)
        self.assertEqual(result["zones"]["ZONE_A"]["people"], 20)

    def test_inflow_increases_population(self):
        self.assertGreater(simulate(snapshot(), {"type": "NO_ACTION"}, 15)["zones"]["ZONE_A"]["people"], 20)

    def test_population_never_negative(self):
        state = snapshot()
        for zone in state["zones"].values(): zone["inflow_rate"] = 0; zone["outflow_rate"] = 1000
        self.assertGreaterEqual(simulate(state, {"type": "NO_ACTION"}, 60)["zones"]["ZONE_A"]["people"], 0)

    def test_entry_restriction_reduces_inflow(self):
        baseline = simulate(snapshot(), {"type": "NO_ACTION"}, 15)
        restricted = simulate(snapshot(), {"type": "RESTRICT_ENTRY", "restrict_percent": 25}, 15)
        self.assertLess(restricted["zones"]["ZONE_A"]["people"], baseline["zones"]["ZONE_A"]["people"])

    def test_exit_capacity_reduces_exit_zone(self):
        baseline = simulate(snapshot(), {"type": "NO_ACTION"}, 30)
        opened = simulate(snapshot(), {"type": "INCREASE_EXIT", "additional_capacity": 30}, 30)
        self.assertLess(opened["zones"]["ZONE_C"]["people"], baseline["zones"]["ZONE_C"]["people"])

    def test_invalid_redirect_is_rejected(self):
        with self.assertRaises(ValueError): simulate(snapshot(), {"type": "REDIRECT", "redirect_percent": 20, "from_zone": "ZONE_A", "to_zone": "ZONE_C"}, 15)

    def test_deterministic(self):
        action = {"type": "INCREASE_EXIT", "additional_capacity": 12}
        self.assertEqual(simulate(snapshot(), action, 30), simulate(snapshot(), action, 30))


if __name__ == "__main__":
    unittest.main()
