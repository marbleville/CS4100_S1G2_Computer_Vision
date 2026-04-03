"""Shared helpers for evaluating and optimizing skin-fusion profiles."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from preprocessor.config.types import (
    SKIN_PRIOR_KEYS,
    LightingSwitchConfig,
    PreprocessorConfig,
    SkinFusionProfile,
)
from preprocessor.io.types import FramePacket
from preprocessor.pipeline.color import (
    build_default_low_light_skin_profile,
    build_default_normal_skin_profile,
    rgb_to_grayscale,
)
from preprocessor.pipeline.processor import PreprocessingPipeline

Box = tuple[int, int, int, int]
LightMode = Literal["normal", "low_light"]

DATASET_FILENAME = "estimated_bbox.json"
SCORE_EPSILON = 1e-12
DEFAULT_MAX_SWEEPS = 25
DEFAULT_SEED = 0
DEFAULT_BRIGHTNESS_BAND = 0.03

STEP_SIZES = {
    "mean": 0.02,
    "sigma": 0.02,
    "weight": 0.05,
    "foreground": 0.05,
    "cutoff": 0.02,
}
MIN_STEP_SIZES = {
    "mean": 0.005,
    "sigma": 0.005,
    "weight": 0.01,
    "foreground": 0.01,
    "cutoff": 0.005,
}


@dataclass(frozen=True, slots=True)
class DatasetSample:
    """One image fixture with expected boxes and a precomputed brightness score."""

    filename: str
    frame_rgb: np.ndarray
    expected_boxes: tuple[Box, ...]
    median_luma: float


@dataclass(frozen=True, slots=True)
class ProfileParameters:
    """The tunable values consumed by the scoring and search scripts."""

    normal_skin_profile: SkinFusionProfile
    low_light_skin_profile: SkinFusionProfile
    brightness_cutoff: float


@dataclass(frozen=True, slots=True)
class ScalarParameter:
    """One scalar coordinate in the optimization search space."""

    name: str
    profile_name: Literal["normal", "low_light"] | None
    kind: Literal["mean", "sigma", "weight", "foreground", "cutoff"]
    channel: str | None
    min_value: float
    max_value: float


@dataclass(frozen=True, slots=True)
class OptimizationOutcome:
    """Result of the local-search sweep."""

    parameters: ProfileParameters
    starting_score: float
    best_score: float
    sweeps_completed: int
    step_sizes: dict[str, float]

    @property
    def improved(self) -> bool:
        return self.best_score > self.starting_score + SCORE_EPSILON


def get_repo_root() -> Path:
    """Return the repository root that contains the preprocessor package."""
    return Path(__file__).resolve().parents[1]


def get_default_dataset_dir() -> Path:
    """Return the sample-frame dataset directory used for optimization."""
    return get_repo_root() / "data" / "test" / "sample_frames"


def get_default_optimization_report_path() -> Path:
    """Return the default report location for optimizer runs."""
    return (
        get_repo_root()
        / "artifacts"
        / "preprocessor_profile_optimization"
        / "latest_optimization_report.json"
    )


def load_dataset(dataset_dir: str | Path | None = None) -> list[DatasetSample]:
    """Load sample images and expected bounding boxes from disk."""
    root = Path(
        dataset_dir) if dataset_dir is not None else get_default_dataset_dir()
    metadata_path = root / DATASET_FILENAME
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    samples: list[DatasetSample] = []

    for record in payload["sample_files"]:
        image_path = root / record["filename"]
        frame_rgb = np.asarray(Image.open(
            image_path).convert("RGB"), dtype=np.uint8)
        median_luma = float(np.median(rgb_to_grayscale(frame_rgb)))
        expected_boxes = tuple(
            (
                int(bbox["bbox_x1"]),
                int(bbox["bbox_y1"]),
                int(bbox["bbox_x2"]),
                int(bbox["bbox_y2"]),
            )
            for bbox in record["bboxes"]
        )
        samples.append(
            DatasetSample(
                filename=record["filename"],
                frame_rgb=frame_rgb,
                expected_boxes=expected_boxes,
                median_luma=median_luma,
            )
        )

    return samples


def get_default_parameters() -> ProfileParameters:
    """Return the current built-in profiles and single-image brightness cutoff."""
    brightness_cutoff = LightingSwitchConfig().enter_low_light_threshold
    return ProfileParameters(
        normal_skin_profile=build_default_normal_skin_profile(),
        low_light_skin_profile=build_default_low_light_skin_profile(),
        brightness_cutoff=float(brightness_cutoff),
    )


def choose_light_mode(median_luma: float, brightness_cutoff: float) -> LightMode:
    """Classify one image into the low-light or normal profile bucket."""
    if median_luma < brightness_cutoff:
        return "low_light"
    return "normal"


def evaluate_parameters(
    parameters: ProfileParameters,
    samples: list[DatasetSample],
    dataset_dir: str | Path | None = None,
    include_per_image: bool = True,
) -> dict[str, Any]:
    """Score one parameter set against the dataset."""
    scores: list[float] = []
    per_image: list[dict[str, Any]] = []

    for frame_index, sample in enumerate(samples):
        light_mode = choose_light_mode(
            sample.median_luma, parameters.brightness_cutoff)
        predicted_boxes = predict_bounding_boxes(
            sample=sample,
            parameters=parameters,
            light_mode=light_mode,
            frame_index=frame_index,
        )
        image_score, matched_iou_sum = score_box_sets(
            sample.expected_boxes,
            predicted_boxes,
        )
        scores.append(image_score)

        if include_per_image:
            per_image.append(
                {
                    "filename": sample.filename,
                    "image_score": image_score,
                    "matched_iou_sum": matched_iou_sum,
                    "median_luma": sample.median_luma,
                    "light_mode": light_mode,
                    "expected_boxes": [list(box) for box in sample.expected_boxes],
                    "predicted_boxes": [list(box) for box in predicted_boxes],
                }
            )

    dataset_score = float(np.mean(scores)) if scores else 0.0
    report: dict[str, Any] = {
        **parameters_to_json_dict(parameters),
        "dataset_score": dataset_score,
        "dataset_dir": str(
            (Path(dataset_dir) if dataset_dir is not None else get_default_dataset_dir())
            .resolve()
        ),
        "sample_count": len(samples),
    }
    if include_per_image:
        report["per_image"] = per_image
    return report


def predict_bounding_boxes(
    sample: DatasetSample,
    parameters: ProfileParameters,
    light_mode: LightMode,
    frame_index: int,
) -> list[Box]:
    """Run the preprocessor on one image using a pinned lighting profile."""
    config = PreprocessorConfig(
        input_mode="webcam",
        normal_skin_profile=parameters.normal_skin_profile,
        low_light_skin_profile=parameters.low_light_skin_profile,
        lighting_switch=LightingSwitchConfig(mode=light_mode),
    )
    pipeline = PreprocessingPipeline(config)
    packet = FramePacket(
        frame_index=frame_index,
        timestamp_ms=frame_index,
        frame_rgb=sample.frame_rgb,
        source_id=sample.filename,
    )
    result = pipeline.process(packet)
    return [tuple(map(int, component.bbox_xyxy)) for component in result.candidates]


def score_box_sets(expected_boxes: tuple[Box, ...], predicted_boxes: list[Box]) -> tuple[float, float]:
    """Return the normalized similarity score and matched IoU sum."""
    expected_count = len(expected_boxes)
    predicted_count = len(predicted_boxes)
    if expected_count == 0 and predicted_count == 0:
        return 1.0, 0.0
    if expected_count == 0 or predicted_count == 0:
        return 0.0, 0.0

    matched_iou_sum = best_matched_iou_sum(
        expected_boxes, tuple(predicted_boxes))
    denominator = max(expected_count, predicted_count, 1)
    return matched_iou_sum / float(denominator), matched_iou_sum


def best_matched_iou_sum(expected_boxes: tuple[Box, ...], predicted_boxes: tuple[Box, ...]) -> float:
    """Compute the optimal one-to-one IoU sum with exhaustive search."""
    iou_matrix = tuple(
        tuple(box_iou(expected_box, predicted_box)
              for predicted_box in predicted_boxes)
        for expected_box in expected_boxes
    )

    @lru_cache(maxsize=None)
    def _search(expected_index: int, used_mask: int) -> float:
        if expected_index >= len(expected_boxes):
            return 0.0

        best = _search(expected_index + 1, used_mask)
        for predicted_index in range(len(predicted_boxes)):
            bit = 1 << predicted_index
            if used_mask & bit:
                continue
            candidate = iou_matrix[expected_index][predicted_index] + _search(
                expected_index + 1,
                used_mask | bit,
            )
            if candidate > best:
                best = candidate
        return best

    return _search(0, 0)


def box_iou(first_box: Box, second_box: Box) -> float:
    """Compute IoU for inclusive pixel-coordinate bounding boxes."""
    ax0, ay0, ax1, ay1 = first_box
    bx0, by0, bx1, by1 = second_box

    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)

    inter_width = max(0, inter_x1 - inter_x0 + 1)
    inter_height = max(0, inter_y1 - inter_y0 + 1)
    intersection = inter_width * inter_height
    if intersection <= 0:
        return 0.0

    area_a = max(0, ax1 - ax0 + 1) * max(0, ay1 - ay0 + 1)
    area_b = max(0, bx1 - bx0 + 1) * max(0, by1 - by0 + 1)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return float(intersection) / float(union)


def optimize_parameters(
    samples: list[DatasetSample],
    starting_parameters: ProfileParameters,
    max_sweeps: int = DEFAULT_MAX_SWEEPS,
    seed: int = DEFAULT_SEED,
) -> OptimizationOutcome:
    """Run best-improvement coordinate hill climbing over the search space."""
    rng = random.Random(seed)
    step_sizes = dict(STEP_SIZES)
    scalar_parameters = build_scalar_parameters()

    current_parameters = starting_parameters
    current_report = evaluate_parameters(
        current_parameters,
        samples,
        include_per_image=False,
    )
    current_score = float(current_report["dataset_score"])
    starting_score = current_score
    sweeps_completed = 0

    while sweeps_completed < max_sweeps and not step_sizes_below_threshold(step_sizes):
        sweeps_completed += 1
        best_neighbor_parameters: ProfileParameters | None = None
        best_neighbor_score = current_score

        shuffled_parameters = scalar_parameters[:]
        rng.shuffle(shuffled_parameters)
        for scalar in shuffled_parameters:
            directions = [-1.0, 1.0]
            rng.shuffle(directions)
            current_value = get_scalar_value(current_parameters, scalar)
            step_size = step_sizes[scalar.kind]

            for direction in directions:
                neighbor_value = clamp(
                    current_value + (direction * step_size),
                    scalar.min_value,
                    scalar.max_value,
                )
                if abs(neighbor_value - current_value) <= SCORE_EPSILON:
                    continue

                neighbor_parameters = set_scalar_value(
                    current_parameters,
                    scalar,
                    neighbor_value,
                )
                neighbor_report = evaluate_parameters(
                    neighbor_parameters,
                    samples,
                    include_per_image=False,
                )
                neighbor_score = float(neighbor_report["dataset_score"])
                if neighbor_score > best_neighbor_score + SCORE_EPSILON:
                    best_neighbor_parameters = neighbor_parameters
                    best_neighbor_score = neighbor_score

        if best_neighbor_parameters is None:
            for kind in step_sizes:
                step_sizes[kind] *= 0.5
            continue

        current_parameters = best_neighbor_parameters
        current_score = best_neighbor_score

    return OptimizationOutcome(
        parameters=current_parameters,
        starting_score=starting_score,
        best_score=current_score,
        sweeps_completed=sweeps_completed,
        step_sizes=step_sizes,
    )


def build_scalar_parameters() -> list[ScalarParameter]:
    """Enumerate the optimization coordinates in a stable order."""
    parameters: list[ScalarParameter] = []
    for profile_name in ("normal", "low_light"):
        for channel in SKIN_PRIOR_KEYS:
            parameters.append(
                ScalarParameter(
                    name=f"{profile_name}.gaussians.{channel}.mean",
                    profile_name=profile_name,
                    kind="mean",
                    channel=channel,
                    min_value=0.0,
                    max_value=1.0,
                )
            )
            parameters.append(
                ScalarParameter(
                    name=f"{profile_name}.gaussians.{channel}.sigma",
                    profile_name=profile_name,
                    kind="sigma",
                    channel=channel,
                    min_value=0.02,
                    max_value=0.50,
                )
            )
        for channel in SKIN_PRIOR_KEYS:
            parameters.append(
                ScalarParameter(
                    name=f"{profile_name}.weights.{channel}",
                    profile_name=profile_name,
                    kind="weight",
                    channel=channel,
                    min_value=0.0,
                    max_value=1.0,
                )
            )
        parameters.append(
            ScalarParameter(
                name=f"{profile_name}.foreground_weight",
                profile_name=profile_name,
                kind="foreground",
                channel=None,
                min_value=0.0,
                max_value=1.0,
            )
        )

    parameters.append(
        ScalarParameter(
            name="brightness_cutoff",
            profile_name=None,
            kind="cutoff",
            channel=None,
            min_value=0.0,
            max_value=1.0,
        )
    )
    return parameters


def step_sizes_below_threshold(step_sizes: dict[str, float]) -> bool:
    """Return True once all step sizes have shrunk below their stop threshold."""
    return all(
        step_sizes[kind] < MIN_STEP_SIZES[kind]
        for kind in MIN_STEP_SIZES
    )


def get_scalar_value(parameters: ProfileParameters, scalar: ScalarParameter) -> float:
    """Read one scalar from the optimization parameter set."""
    if scalar.kind == "cutoff":
        return parameters.brightness_cutoff

    profile = get_profile(parameters, scalar.profile_name)
    if scalar.kind == "foreground":
        return profile.foreground_weight
    if scalar.kind == "weight":
        assert scalar.channel is not None
        return profile.weights[scalar.channel]

    assert scalar.channel is not None
    mean, sigma = profile.gaussians[scalar.channel]
    if scalar.kind == "mean":
        return mean
    return sigma


def set_scalar_value(
    parameters: ProfileParameters,
    scalar: ScalarParameter,
    new_value: float,
) -> ProfileParameters:
    """Return a copy with one scalar updated and clamped."""
    clamped_value = clamp(new_value, scalar.min_value, scalar.max_value)
    if scalar.kind == "cutoff":
        return ProfileParameters(
            normal_skin_profile=parameters.normal_skin_profile,
            low_light_skin_profile=parameters.low_light_skin_profile,
            brightness_cutoff=clamped_value,
        )

    updated_normal = parameters.normal_skin_profile
    updated_low_light = parameters.low_light_skin_profile
    if scalar.profile_name == "normal":
        updated_normal = set_profile_scalar(
            updated_normal, scalar, clamped_value)
    else:
        updated_low_light = set_profile_scalar(
            updated_low_light, scalar, clamped_value)

    return ProfileParameters(
        normal_skin_profile=updated_normal,
        low_light_skin_profile=updated_low_light,
        brightness_cutoff=parameters.brightness_cutoff,
    )


def get_profile(
    parameters: ProfileParameters,
    profile_name: Literal["normal", "low_light"] | None,
) -> SkinFusionProfile:
    """Return one profile from the parameter bundle."""
    if profile_name == "normal":
        return parameters.normal_skin_profile
    if profile_name == "low_light":
        return parameters.low_light_skin_profile
    raise ValueError("Expected `profile_name` to be `normal` or `low_light`.")


def set_profile_scalar(
    profile: SkinFusionProfile,
    scalar: ScalarParameter,
    new_value: float,
) -> SkinFusionProfile:
    """Return a copy of one profile with a single scalar updated."""
    gaussians = dict(profile.gaussians)
    weights = dict(profile.weights)
    foreground_weight = profile.foreground_weight

    if scalar.kind == "foreground":
        foreground_weight = new_value
    elif scalar.kind == "weight":
        assert scalar.channel is not None
        weights[scalar.channel] = new_value
    else:
        assert scalar.channel is not None
        mean, sigma = gaussians[scalar.channel]
        if scalar.kind == "mean":
            gaussians[scalar.channel] = (new_value, sigma)
        else:
            gaussians[scalar.channel] = (mean, new_value)

    return SkinFusionProfile(
        gaussians=gaussians,
        weights=weights,
        foreground_weight=foreground_weight,
    )


def parameters_to_json_dict(parameters: ProfileParameters) -> dict[str, Any]:
    """Serialize parameters into a JSON-friendly dictionary."""
    return {
        "normal_skin_profile": skin_profile_to_json_dict(parameters.normal_skin_profile),
        "low_light_skin_profile": skin_profile_to_json_dict(parameters.low_light_skin_profile),
        "brightness_cutoff": parameters.brightness_cutoff,
    }


def skin_profile_to_json_dict(profile: SkinFusionProfile) -> dict[str, Any]:
    """Serialize one profile into a JSON-friendly dictionary."""
    return {
        "gaussians": {
            channel: {
                "mean": profile.gaussians[channel][0],
                "sigma": profile.gaussians[channel][1],
            }
            for channel in SKIN_PRIOR_KEYS
        },
        "weights": {
            channel: profile.weights[channel]
            for channel in SKIN_PRIOR_KEYS
        },
        "foreground_weight": profile.foreground_weight,
    }


def parameters_from_json_dict(payload: dict[str, Any]) -> ProfileParameters:
    """Deserialize parameters from either a bare candidate or a report."""
    return ProfileParameters(
        normal_skin_profile=skin_profile_from_json_dict(
            payload["normal_skin_profile"]),
        low_light_skin_profile=skin_profile_from_json_dict(
            payload["low_light_skin_profile"]),
        brightness_cutoff=float(payload["brightness_cutoff"]),
    )


def skin_profile_from_json_dict(payload: dict[str, Any]) -> SkinFusionProfile:
    """Deserialize one JSON profile into a SkinFusionProfile."""
    raw_gaussians = payload["gaussians"]
    gaussians: dict[str, tuple[float, float]] = {}
    for channel in SKIN_PRIOR_KEYS:
        raw_value = raw_gaussians[channel]
        if isinstance(raw_value, dict):
            mean = float(raw_value["mean"])
            sigma = float(raw_value["sigma"])
        else:
            mean, sigma = raw_value
            mean = float(mean)
            sigma = float(sigma)
        gaussians[channel] = (mean, sigma)

    weights = {
        channel: float(payload["weights"][channel])
        for channel in SKIN_PRIOR_KEYS
    }
    return SkinFusionProfile(
        gaussians=gaussians,
        weights=weights,
        foreground_weight=float(payload["foreground_weight"]),
    )


def load_parameters_from_json(path: str | Path) -> ProfileParameters:
    """Load parameters from a JSON file produced by the scripts."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return parameters_from_json_dict(payload)


def save_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write JSON to disk with stable formatting."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(
        payload, indent=2) + "\n", encoding="utf-8")
    return output_path


def runtime_thresholds_from_cutoff(brightness_cutoff: float) -> tuple[float, float]:
    """Convert the single-image cutoff into runtime enter/exit thresholds."""
    enter_low_light = max(0.0, brightness_cutoff - DEFAULT_BRIGHTNESS_BAND)
    exit_low_light = min(
        1.0,
        max(brightness_cutoff + DEFAULT_BRIGHTNESS_BAND, enter_low_light + 0.01),
    )
    return enter_low_light, exit_low_light


def rewrite_profile_defaults(parameters: ProfileParameters) -> dict[str, Any]:
    """Rewrite built-in source defaults to match the supplied parameters."""
    color_path = get_repo_root() / "preprocessor" / "pipeline" / "color.py"
    config_path = get_repo_root() / "preprocessor" / "config" / "types.py"

    color_text = color_path.read_text(encoding="utf-8")
    color_text = replace_dict_assignment(
        color_text,
        "SKIN_PRIOR_GAUSSIANS",
        "dict[str, tuple[float, float]]",
        format_gaussian_dict("SKIN_PRIOR_GAUSSIANS",
                             parameters.normal_skin_profile.gaussians),
    )
    color_text = replace_dict_assignment(
        color_text,
        "SKIN_PRIOR_WEIGHTS",
        "dict[str, float]",
        format_weight_dict("SKIN_PRIOR_WEIGHTS",
                           parameters.normal_skin_profile.weights),
    )
    color_text = replace_float_assignment(
        color_text,
        "DEFAULT_FOREGROUND_WEIGHT",
        parameters.normal_skin_profile.foreground_weight,
    )
    color_text = replace_dict_assignment(
        color_text,
        "LOW_LIGHT_SKIN_PRIOR_GAUSSIANS",
        "dict[str, tuple[float, float]]",
        format_gaussian_dict(
            "LOW_LIGHT_SKIN_PRIOR_GAUSSIANS",
            parameters.low_light_skin_profile.gaussians,
        ),
    )
    color_text = replace_dict_assignment(
        color_text,
        "LOW_LIGHT_SKIN_PRIOR_WEIGHTS",
        "dict[str, float]",
        format_weight_dict(
            "LOW_LIGHT_SKIN_PRIOR_WEIGHTS",
            parameters.low_light_skin_profile.weights,
        ),
    )
    color_text = replace_float_assignment(
        color_text,
        "LOW_LIGHT_FOREGROUND_WEIGHT",
        parameters.low_light_skin_profile.foreground_weight,
    )
    color_path.write_text(color_text, encoding="utf-8")

    enter_low_light, exit_low_light = runtime_thresholds_from_cutoff(
        parameters.brightness_cutoff
    )
    config_text = config_path.read_text(encoding="utf-8")
    config_text = replace_dataclass_default(
        config_text,
        "enter_low_light_threshold",
        enter_low_light,
    )
    config_text = replace_dataclass_default(
        config_text,
        "exit_low_light_threshold",
        exit_low_light,
    )
    config_path.write_text(config_text, encoding="utf-8")

    return {
        "color_path": str(color_path.resolve()),
        "config_path": str(config_path.resolve()),
        "runtime_thresholds": {
            "enter_low_light_threshold": enter_low_light,
            "exit_low_light_threshold": exit_low_light,
        },
    }


def format_gaussian_dict(
    name: str,
    gaussians: dict[str, tuple[float, float]],
) -> str:
    """Format a gaussian assignment block in the source-file style."""
    lines = [f"{name}: dict[str, tuple[float, float]] = {{"]
    for channel in SKIN_PRIOR_KEYS:
        mean, sigma = gaussians[channel]
        lines.append(
            f'    "{channel}": ({format_float_literal(mean)}, {format_float_literal(sigma)}),'
        )
    lines.append("}")
    return "\n".join(lines)


def format_weight_dict(name: str, weights: dict[str, float]) -> str:
    """Format a weight assignment block in the source-file style."""
    lines = [f"{name}: dict[str, float] = {{"]
    for channel in SKIN_PRIOR_KEYS:
        lines.append(
            f'    "{channel}": {format_float_literal(weights[channel])},')
    lines.append("}")
    return "\n".join(lines)


def replace_dict_assignment(
    source_text: str,
    name: str,
    annotation: str,
    replacement_block: str,
) -> str:
    """Replace a named dictionary assignment block by scanning its braces."""
    prefix = f"{name}: {annotation} = "
    assignment_start = source_text.index(prefix)
    brace_start = source_text.index("{", assignment_start)
    brace_depth = 0
    brace_end = brace_start
    for brace_end in range(brace_start, len(source_text)):
        char = source_text[brace_end]
        if char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
            if brace_depth == 0:
                break

    return (
        source_text[:assignment_start]
        + replacement_block
        + source_text[brace_end + 1:]
    )


def replace_float_assignment(source_text: str, name: str, value: float) -> str:
    """Replace a simple module-level float assignment."""
    pattern = re.compile(
        rf"^(?P<prefix>{re.escape(name)} = )(?P<value>[-+]?[0-9]*\.?[0-9]+)$",
        re.MULTILINE,
    )
    return pattern.sub(
        rf"\g<prefix>{format_float_literal(value)}",
        source_text,
        count=1,
    )


def replace_dataclass_default(source_text: str, field_name: str, value: float) -> str:
    """Replace one dataclass field default in config/types.py."""
    pattern = re.compile(
        rf"^(?P<prefix>\s*{re.escape(field_name)}: float = )(?P<value>[-+]?[0-9]*\.?[0-9]+)$",
        re.MULTILINE,
    )
    return pattern.sub(
        rf"\g<prefix>{format_float_literal(value)}",
        source_text,
        count=1,
    )


def format_float_literal(value: float) -> str:
    """Format floats for stable source rewrites."""
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return text


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp a scalar into a closed interval."""
    return max(min_value, min(max_value, float(value)))
