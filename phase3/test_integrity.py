import copy
import unittest

from phase3.digital_twin.state_builder import build_current_state
from phase3.digital_twin.simulation_engine import simulate


def flow_result(session_id=117):
    return {
        "source": "demo.mp4",
        "session_id": session_id,
        "flow_state": "BALANCED FLOW",
        "snapshot_provenance": {"source_timestamp_sec": 9.6, "source_frame_index": 230},
        "zones": {
            "ZONE_A": {"people": 4, "inflow": 12, "outflow": 8, "net": 4, "average_movement": 10, "state": "Accumulating"},
            "ZONE_B": {"people": 3, "inflow": 8, "outflow": 8, "net": 0, "average_movement": 7, "state": "Balanced"},
            "ZONE_C": {"people": 2, "inflow": 4, "outflow": 6, "net": -2, "average_movement": 12, "state": "Dispersing"},
        },
    }


class DataIntegrityTests(unittest.TestCase):
    def test_snapshot_is_session_scoped_and_reconciles(self):
        state = build_current_state(flow_result(), event_id=11, monitoring_session_id=117, flow_analysis_id=130)
        self.assertEqual(state["event_id"], 11)
        self.assertEqual(state["monitoring_session_id"], 117)
        self.assertEqual(state["flow_analysis_id"], 130)
        self.assertEqual(state["total_people"], 9)

    def test_missing_zone_people_is_not_replaced_with_fake_zero(self):
        result = flow_result()
        del result["zones"]["ZONE_B"]["people"]
        with self.assertRaises(ValueError):
            build_current_state(result)

    def test_simulation_does_not_mutate_current_state(self):
        state = build_current_state(flow_result())
        before = copy.deepcopy(state)
        simulate(state, {"type": "RESTRICT_ENTRY", "restrict_percent": 25}, 15)
        self.assertEqual(state, before)


if __name__ == "__main__":
    unittest.main()
