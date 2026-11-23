import unittest

import torch

from specforge.algorithms.common.hidden_states_data import build_collator


class DFlash2MropeCollationTest(unittest.TestCase):
    def test_position_triplets_survive_padding(self):
        def sample(length, offset):
            return {
                "input_ids": torch.arange(length).unsqueeze(0),
                "loss_mask": torch.ones(1, length),
                "hidden_states": torch.zeros(1, length, 8),
                "position_ids": torch.stack(
                    [torch.arange(length), torch.arange(length) + offset,
                     torch.arange(length) + 2 * offset], dim=-1
                ).unsqueeze(0),
            }

        batch = build_collator()([sample(2, 10), sample(3, 20)])
        self.assertEqual(tuple(batch["position_ids"].shape), (2, 3, 3))
        torch.testing.assert_close(batch["position_ids"][0, :2], sample(2, 10)["position_ids"][0])
        torch.testing.assert_close(batch["position_ids"][1], sample(3, 20)["position_ids"][0])
        torch.testing.assert_close(batch["position_ids"][0, 2], torch.zeros(3, dtype=torch.long))

    def test_missing_positions_do_not_silently_mix(self):
        with_positions = {
            "input_ids": torch.zeros(1, 2, dtype=torch.long),
            "loss_mask": torch.ones(1, 2),
            "hidden_states": torch.zeros(1, 2, 8),
            "position_ids": torch.zeros(1, 2, 3, dtype=torch.long),
        }
        without_positions = {key: value for key, value in with_positions.items() if key != "position_ids"}
        with self.assertRaises(KeyError):
            build_collator()([with_positions, without_positions])


if __name__ == "__main__":
    unittest.main()
