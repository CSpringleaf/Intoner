import importlib
import unittest

from intoner.paths import REPO_ROOT
from intoner.utils.hparams import set_hparams


class PisConfigTest(unittest.TestCase):
    def test_stage_configs_resolve_to_importable_tasks(self):
        expected = {
            "warmup": "intoner.pis.tasks.acoustic.FastSpeech2Task",
            "diffusion": "intoner.pis.tasks.diffusion.DiffSpeechTask",
        }
        for stage, task_cls in expected.items():
            with self.subTest(stage=stage):
                hparams = set_hparams(str(REPO_ROOT / f"configs/pis/{stage}.yaml"),
                                      print_hparams=False, global_hparams=False)
                self.assertEqual(hparams["task_cls"], task_cls)
                self.assertEqual(hparams["binary_data_dir"], "data/intoner/pis/binary")
                self.assertEqual(hparams["vocoder"], "intoner.vocoders.hifigan.HifiGAN")
                module, name = task_cls.rsplit(".", 1)
                self.assertTrue(hasattr(importlib.import_module(module), name))

    def test_diffusion_starts_from_warmup_checkpoint(self):
        hparams = set_hparams(str(REPO_ROOT / "configs/pis/diffusion.yaml"),
                              print_hparams=False, global_hparams=False)
        self.assertEqual(hparams["fs2_ckpt"], "checkpoints/pis_warmup/model_ckpt_steps_120000.ckpt")
        self.assertEqual((hparams["max_updates"], hparams["K_step"], hparams["max_beta"]), (160000, 71, 0.06))
        self.assertEqual(len(hparams["spec_min"]), 80)
        self.assertEqual(len(hparams["spec_max"]), 80)


if __name__ == "__main__":
    unittest.main()
