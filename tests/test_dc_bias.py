import unittest
from unittest.mock import MagicMock, call, patch

from hardware.arduino_dc_bias import DCBiasController
from profiles import ProfileError, load_profile
from profiles.loader import _validate_dc_bias, _validate_qubit_dc_biases
from quam_config.components import ArduinoDCBias
from quam_config.create_machine_from_profile import create_machine_from_profile


class FakeSerialConnection:
    def __init__(self) -> None:
        self.in_waiting = 0
        self.writes: list[bytes] = []
        self.closed = False

    def write(self, data: bytes) -> int:
        self.writes.append(data)
        return len(data)

    def readline(self) -> bytes:
        return b""

    def close(self) -> None:
        self.closed = True


class DCBiasControllerTests(unittest.TestCase):
    def test_voltage_at_limit_is_sent(self):
        connection = FakeSerialConnection()
        controller = DCBiasController(connection, response_delay_s=0)

        controller.set_voltage(0, 0.01, verbose=False)

        self.assertEqual(connection.writes, [b"SET,0,0.01\r"])

    def test_voltage_above_limit_is_rejected_before_write(self):
        connection = FakeSerialConnection()
        controller = DCBiasController(connection, response_delay_s=0)

        with self.assertRaisesRegex(ValueError, "must not exceed 0.01 V"):
            controller.set_voltage(0, 0.010001, verbose=False)

        self.assertEqual(connection.writes, [])

    def test_limit_cannot_be_configured_above_hard_maximum(self):
        with self.assertRaisesRegex(ValueError, "no greater than 0.01 V"):
            DCBiasController(
                FakeSerialConnection(),
                max_abs_voltage_v=0.02,
                response_delay_s=0,
            )


class ArduinoDCBiasComponentTests(unittest.TestCase):
    def test_runtime_connection_is_not_serialized(self):
        bias = ArduinoDCBias()
        bias._controller = MagicMock(spec=DCBiasController)

        state = bias.to_dict()

        self.assertNotIn("_controller", state)
        self.assertNotIn("output_channel", state)
        self.assertEqual(state["max_abs_voltage_v"], 0.01)

    @patch("quam_config.components.dc_bias.open_controller")
    def test_applied_zeros_channel_and_disconnects(self, open_controller):
        controller = MagicMock(spec=DCBiasController)
        open_controller.return_value = controller
        bias = ArduinoDCBias(port="COM7")

        with bias.applied(channel=0, voltage_v=0.0025):
            self.assertTrue(bias.is_connected)

        self.assertEqual(
            controller.set_voltage.call_args_list,
            [call(0, 0.0025), call(0, 0.0)],
        )
        controller.close.assert_called_once_with()
        self.assertFalse(bias.is_connected)

    @patch("quam_config.components.dc_bias.open_controller")
    def test_out_of_range_voltage_is_rejected_before_connecting(
        self, open_controller
    ):
        bias = ArduinoDCBias(max_abs_voltage_v=0.01)

        with self.assertRaisesRegex(ValueError, "must not exceed 0.01 V"):
            bias.set_voltage(channel=0, voltage_v=0.02)

        open_controller.assert_not_called()

    @patch("quam_config.components.dc_bias.open_controller")
    def test_profile_bias_uses_hardcoded_channel_zero(self, open_controller):
        controller = MagicMock(spec=DCBiasController)
        open_controller.return_value = controller
        bias = ArduinoDCBias(qubit_biases_v={"q3": 0.0025})

        with bias.applied_for_qubit("q3"):
            pass

        self.assertEqual(
            controller.set_voltage.call_args_list,
            [call(0, 0.0025), call(0, 0.0)],
        )

    def test_profile_rejects_dc_bias_limit_above_hard_maximum(self):
        connectivity = {
            "dc_bias": {
                "port": "COM7",
                "baud_rate": 115200,
                "channel_count": 8,
                "max_abs_voltage_v": 0.02,
            }
        }

        with self.assertRaisesRegex(ProfileError, "no greater than 0.01 V"):
            _validate_dc_bias(connectivity)

    def test_profile_rejects_qubit_bias_above_configured_maximum(self):
        qubits = {"qubits": {"q3": {"dc_bias_v": 0.010001}}}
        connectivity = {"dc_bias": {"max_abs_voltage_v": 0.01}}

        with self.assertRaisesRegex(ProfileError, "exceeds the configured maximum"):
            _validate_qubit_dc_biases(qubits, connectivity)

    def test_all_single_qubit_entries_define_voltage_without_channel(self):
        profile = load_profile("single_qubit")

        for qubit in profile["qubits"]["qubits"].values():
            self.assertIn("dc_bias_v", qubit)
            self.assertNotIn("dc_bias", qubit)

    def test_profile_machine_contains_dc_bias_component_without_connecting(self):
        with patch("quam_config.components.dc_bias.open_controller") as open_controller:
            machine = create_machine_from_profile(
                "single_qubit", save=False, qubit="q3"
            )
            config = machine.generate_config()

        self.assertIsInstance(machine.dc_bias, ArduinoDCBias)
        self.assertEqual(machine.dc_bias.port, "COM7")
        self.assertEqual(machine.dc_bias.max_abs_voltage_v, 0.01)
        self.assertEqual(machine.dc_bias.output_channel, 0)
        self.assertEqual(machine.dc_bias.qubit_biases_v, {"q3": 0.0})
        self.assertIn("controllers", config)
        open_controller.assert_not_called()


if __name__ == "__main__":
    unittest.main()
