import importlib.util
import tempfile
import unittest
from pathlib import Path


@unittest.skipIf(
    importlib.util.find_spec("sentence_transformers") is None,
    "sentence_transformers not installed",
)
class TestModelResolution(unittest.TestCase):
    def test_resolves_model_from_models_dir(self):
        from infrastructure.config import ContainerConfig, _resolve_model_reference

        with tempfile.TemporaryDirectory() as tmp:
            model_ref = "google/flan-t5-small"
            model_path = Path(tmp) / model_ref
            model_path.mkdir(parents=True)
            cfg = ContainerConfig(models_dir=tmp)

            resolved = _resolve_model_reference(model_ref, cfg)

            self.assertEqual(resolved, str(model_path))

    def test_keeps_original_model_when_local_dir_missing(self):
        from infrastructure.config import ContainerConfig, _resolve_model_reference

        cfg = ContainerConfig(models_dir="/tmp/contextsearch-not-existing")

        resolved = _resolve_model_reference("sentence-transformers/all-MiniLM-L6-v2", cfg)

        self.assertEqual(resolved, "sentence-transformers/all-MiniLM-L6-v2")


if __name__ == "__main__":
    unittest.main()
