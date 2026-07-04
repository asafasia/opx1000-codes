"""Review saved calibration figures with the NVIDIA Ising calibration model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .nvidia_ising_client import NvidiaIsingClient


REVIEW_JSON = "ai_review.json"
REVIEW_MARKDOWN = "ai_review.md"


@dataclass(frozen=True)
class CalibrationAIReview:
    """Paths created by an AI review."""

    json_path: Path
    markdown_path: Path
    figure_paths: tuple[Path, ...]


class CalibrationAIReviewer:
    """Create passive AI reviews for completed calibration run directories."""

    def __init__(self, client: NvidiaIsingClient | None = None) -> None:
        self.client = client or NvidiaIsingClient()

    def review_run(
        self,
        run_directory: str | Path,
        *,
        extra_context: Mapping[str, Any] | None = None,
        max_tokens: int = 2048,
    ) -> CalibrationAIReview:
        """Review figures in a saved run and write ``ai_review`` artifacts."""
        run_directory = Path(run_directory)
        if not run_directory.is_dir():
            raise FileNotFoundError(f"Calibration run directory does not exist: {run_directory}")

        figure_paths = tuple(self._figure_paths(run_directory))
        if not figure_paths:
            raise FileNotFoundError(f"No PNG/JPEG figures found under {run_directory / 'figures'}")

        prompt = self._build_prompt(run_directory, figure_paths, extra_context=extra_context)
        response = self.client.chat_with_images(
            prompt=prompt,
            image_paths=list(figure_paths),
            max_tokens=max_tokens,
        )
        content = self._response_content(response)
        review_payload = self._parse_review_content(content)
        review_payload.setdefault("raw_model_text", content)
        review_payload.setdefault("model", self.client.model)
        review_payload.setdefault(
            "figure_files",
            [path.relative_to(run_directory).as_posix() for path in figure_paths],
        )

        json_path = run_directory / REVIEW_JSON
        json_path.write_text(json.dumps(review_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        markdown_path = run_directory / REVIEW_MARKDOWN
        markdown_path.write_text(self._to_markdown(review_payload), encoding="utf-8")

        self._update_metadata(run_directory, REVIEW_JSON)
        return CalibrationAIReview(json_path=json_path, markdown_path=markdown_path, figure_paths=figure_paths)

    @staticmethod
    def _figure_paths(run_directory: Path) -> list[Path]:
        figures_directory = run_directory / "figures"
        if not figures_directory.is_dir():
            return []
        return sorted(
            [
                path
                for path in figures_directory.iterdir()
                if path.suffix.lower() in {".png", ".jpg", ".jpeg"} and path.is_file()
            ]
        )

    @staticmethod
    def _load_json(path: Path) -> Any | None:
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _build_prompt(
        self,
        run_directory: Path,
        figure_paths: tuple[Path, ...],
        *,
        extra_context: Mapping[str, Any] | None,
    ) -> str:
        metadata = self._load_json(run_directory / "metadata.json")
        parameters = self._load_json(run_directory / "parameters.json")
        analysis_result = self._load_json(run_directory / "analysis_result.json")
        context = {
            "run_directory_name": run_directory.name,
            "metadata": metadata,
            "parameters": parameters,
            "analysis_result": analysis_result,
            "figure_files": [path.relative_to(run_directory).as_posix() for path in figure_paths],
            "extra_context": dict(extra_context or {}),
        }
        return (
            "You are reviewing saved quantum-computer calibration figures from a lab QPU.\n"
            "Use the figures as the primary evidence. Use metadata, parameters, and deterministic "
            "analysis results as supporting context.\n\n"
            "Important safety rules:\n"
            "- Do not claim that the hardware profile was changed.\n"
            "- Do not recommend automatic profile changes unless the visual evidence is strong.\n"
            "- If a deterministic analysis result disagrees with the figure, flag the disagreement.\n"
            "- Prefer a rerun or narrower sweep when the figure is ambiguous.\n\n"
            "Return strict JSON with these keys:\n"
            "{\n"
            '  "summary": "one short paragraph",\n'
            '  "pass_fail": "pass|warning|fail",\n'
            '  "confidence": 0.0,\n'
            '  "observations": ["..."],\n'
            '  "suspected_issues": ["..."],\n'
            '  "recommended_next_action": "...",\n'
            '  "profile_update_recommendation": "approve|hold|rerun|none"\n'
            "}\n\n"
            f"Run context JSON:\n{json.dumps(context, indent=2, default=str)}"
        )

    @staticmethod
    def _response_content(response: Mapping[str, Any]) -> str:
        choices = response.get("choices")
        if not choices:
            return json.dumps(response)
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        return json.dumps(content)

    @staticmethod
    def _parse_review_content(content: str) -> dict[str, Any]:
        text = content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {"review": parsed}
        except json.JSONDecodeError:
            extracted = CalibrationAIReviewer._extract_json_object(text)
            if extracted is not None:
                return extracted
            markdown_review = CalibrationAIReviewer._parse_markdown_review(text)
            if markdown_review is not None:
                return markdown_review
            return {
                "summary": "The model did not return valid JSON.",
                "pass_fail": "warning",
                "confidence": 0.0,
                "observations": [],
                "suspected_issues": ["Non-JSON model response"],
                "recommended_next_action": "Read raw_model_text before acting on this review.",
                "profile_update_recommendation": "hold",
            }

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any] | None:
        """Extract the first balanced JSON object from a prose response."""
        start = text.find("{")
        while start != -1:
            depth = 0
            in_string = False
            escaped = False
            for index in range(start, len(text)):
                char = text[index]
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                    continue
                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:index + 1]
                        try:
                            parsed = json.loads(candidate)
                        except json.JSONDecodeError:
                            break
                        return parsed if isinstance(parsed, dict) else {"review": parsed}
            start = text.find("{", start + 1)
        return None

    @staticmethod
    def _parse_markdown_review(text: str) -> dict[str, Any] | None:
        """Parse common Markdown section output when the model ignores JSON."""
        sections: dict[str, list[str]] = {}
        current: str | None = None
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if line.startswith("**") and line.endswith("**"):
                current = line.strip("*: ").lower().replace("/", "_").replace(" ", "_")
                sections[current] = []
                continue
            if current is not None:
                sections[current].append(raw_line.rstrip())

        if not sections:
            return None

        def section_text(*names: str) -> str:
            for name in names:
                value = "\n".join(sections.get(name, [])).strip()
                if value:
                    return value
            return ""

        def bullet_items(*names: str) -> list[str]:
            text_value = section_text(*names)
            items = []
            for line in text_value.splitlines():
                item = line.strip()
                if item.startswith("-"):
                    item = item[1:].strip()
                if item:
                    items.append(item)
            return items

        def first_line(*names: str) -> str:
            lines = [line.strip() for line in section_text(*names).splitlines() if line.strip()]
            return lines[0] if lines else ""

        pass_fail = first_line("pass_fail").lower()
        if pass_fail not in {"pass", "warning", "fail"}:
            pass_fail = "warning"

        confidence = 0.0
        confidence_text = first_line("confidence")
        if confidence_text:
            try:
                confidence = float(confidence_text)
            except ValueError:
                confidence = 0.0

        recommendation = section_text("profile_update_recommendation").lower()
        if "approve" in recommendation:
            profile_update_recommendation = "approve"
        elif "rerun" in recommendation:
            profile_update_recommendation = "rerun"
        elif "hold" in recommendation:
            profile_update_recommendation = "hold"
        else:
            profile_update_recommendation = "none"

        summary = section_text("summary")
        if not summary and not any(sections.values()):
            return None

        return {
            "summary": summary or "The model returned a Markdown review.",
            "pass_fail": pass_fail,
            "confidence": confidence,
            "observations": bullet_items("observations"),
            "suspected_issues": bullet_items("suspected_issues"),
            "recommended_next_action": section_text("recommended_next_action"),
            "profile_update_recommendation": profile_update_recommendation,
        }

    @staticmethod
    def _to_markdown(review: Mapping[str, Any]) -> str:
        observations = "\n".join(f"- {item}" for item in review.get("observations", [])) or "- None"
        issues = "\n".join(f"- {item}" for item in review.get("suspected_issues", [])) or "- None"
        return (
            "# AI Calibration Review\n\n"
            f"**Status:** {review.get('pass_fail', 'unknown')}\n\n"
            f"**Confidence:** {review.get('confidence', 'unknown')}\n\n"
            f"{review.get('summary', '')}\n\n"
            "## Observations\n\n"
            f"{observations}\n\n"
            "## Suspected Issues\n\n"
            f"{issues}\n\n"
            "## Recommended Next Action\n\n"
            f"{review.get('recommended_next_action', '')}\n\n"
            "## Profile Update Recommendation\n\n"
            f"{review.get('profile_update_recommendation', 'hold')}\n"
        )

    @staticmethod
    def _update_metadata(run_directory: Path, review_filename: str) -> None:
        metadata_path = run_directory / "metadata.json"
        if not metadata_path.is_file():
            return
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["ai_review"] = review_filename
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
