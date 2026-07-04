import json
import tempfile
import unittest
from pathlib import Path

from calibration_ai import CalibrationAIReviewer


class FakeClient:
    model = "fake-model"

    def __init__(self):
        self.prompt = None
        self.image_paths = None

    def chat_with_images(self, *, prompt, image_paths, max_tokens=2048, temperature=0.2):
        self.prompt = prompt
        self.image_paths = image_paths
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "summary": "Looks plausible.",
                                "pass_fail": "pass",
                                "confidence": 0.8,
                                "observations": ["Clear feature is visible."],
                                "suspected_issues": [],
                                "recommended_next_action": "Continue.",
                                "profile_update_recommendation": "approve",
                            }
                        )
                    }
                }
            ]
        }


class CalibrationAIReviewerTests(unittest.TestCase):
    def test_review_run_writes_artifacts_and_updates_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            run_directory = Path(directory) / "run"
            figures_directory = run_directory / "figures"
            figures_directory.mkdir(parents=True)
            (run_directory / "metadata.json").write_text(
                '{"experiment_name": "resonator_spectroscopy"}\n',
                encoding="utf-8",
            )
            (run_directory / "parameters.json").write_text('{"num_shots": 100}\n', encoding="utf-8")
            (figures_directory / "amplitude.png").write_bytes(b"not really a png")

            client = FakeClient()
            review = CalibrationAIReviewer(client).review_run(run_directory)

            self.assertEqual(review.json_path, run_directory / "ai_review.json")
            self.assertEqual(review.markdown_path, run_directory / "ai_review.md")
            saved = json.loads(review.json_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["pass_fail"], "pass")
            self.assertEqual(saved["figure_files"], ["figures/amplitude.png"])
            metadata = json.loads((run_directory / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["ai_review"], "ai_review.json")
            self.assertIn("resonator_spectroscopy", client.prompt)
            self.assertEqual(client.image_paths, [figures_directory / "amplitude.png"])

    def test_parse_review_content_extracts_json_from_prose(self):
        content = (
            "Here is the calibration review:\n"
            '{"summary": "Blob separation is marginal.", "pass_fail": "warning", '
            '"confidence": 0.62, "observations": ["Two clusters overlap."], '
            '"suspected_issues": [], "recommended_next_action": "Rerun IQ blobs.", '
            '"profile_update_recommendation": "rerun"}'
        )

        parsed = CalibrationAIReviewer._parse_review_content(content)

        self.assertEqual(parsed["pass_fail"], "warning")
        self.assertEqual(parsed["summary"], "Blob separation is marginal.")

    def test_parse_review_content_accepts_markdown_sections(self):
        content = (
            "**Summary:**\n"
            "The IQ blobs are separated.\n\n"
            "**Pass/Fail:**\n"
            "Pass\n\n"
            "**Confidence:**\n"
            "0.95\n\n"
            "**Observations:**\n"
            "- Ground and excited clusters are distinct.\n\n"
            "**Suspected Issues:**\n"
            "- Excited-state fidelity is lower.\n\n"
            "**Recommended Next Action:**\n"
            "- Rerun with more shots.\n\n"
            "**Profile Update Recommendation:**\n"
            "Hold - verify before applying."
        )

        parsed = CalibrationAIReviewer._parse_review_content(content)

        self.assertEqual(parsed["pass_fail"], "pass")
        self.assertEqual(parsed["confidence"], 0.95)
        self.assertEqual(parsed["profile_update_recommendation"], "hold")
        self.assertEqual(parsed["observations"], ["Ground and excited clusters are distinct."])

    def test_parse_review_content_handles_empty_markdown_sections(self):
        content = (
            "**Summary:**\n"
            "The run needs a closer look.\n\n"
            "**Pass/Fail:**\n\n"
            "**Confidence:**\n\n"
            "**Recommended Next Action:**\n"
            "- Inspect the raw response."
        )

        parsed = CalibrationAIReviewer._parse_review_content(content)

        self.assertEqual(parsed["pass_fail"], "warning")
        self.assertEqual(parsed["confidence"], 0.0)
        self.assertEqual(parsed["summary"], "The run needs a closer look.")


if __name__ == "__main__":
    unittest.main()
