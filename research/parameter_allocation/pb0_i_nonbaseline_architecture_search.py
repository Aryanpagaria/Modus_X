from __future__ import annotations

import csv
import hashlib
import importlib.util
import inspect
import json
import math
import sys
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


# ==============================================================================
# PATHS
# ==============================================================================

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

MODELS_PATH = PROJECT_ROOT / "language" / "models.py"

OUTPUT_DIR = PROJECT_ROOT / "research" / "parameter_allocation" / "outputs"

PB0_F_PATH = OUTPUT_DIR / "PB0_F_PARAMETER_FORMULA_CALIBRATION.json"
PB0_G_PATH = OUTPUT_DIR / "PB0_G_CALIBRATED_CANDIDATE_SEARCH.json"
PB0_H_PATH = OUTPUT_DIR / "PB0_H_EXACT_PARAMETER_TREE_RECOUNT.json"
B0_CENSUS_PATH = OUTPUT_DIR / "b0_parameter_census.json"

OUTPUT_JSON = OUTPUT_DIR / "PB0_I_NONBASELINE_ARCHITECTURE_SEARCH.json"
OUTPUT_MD = OUTPUT_DIR / "PB0_I_NONBASELINE_ARCHITECTURE_SEARCH.md"
OUTPUT_CSV = OUTPUT_DIR / "PB0_I_NONBASELINE_CANDIDATES.csv"
OUTPUT_VERIFICATION_CSV = (
    OUTPUT_DIR / "PB0_I_NONBASELINE_VERIFICATION.csv"
)


# ==============================================================================
# SEARCH CONFIGURATION
# ==============================================================================

BASELINE_R = 512
BASELINE_N = 512
BASELINE_H = 32

PARAMETER_TOLERANCE = 50_000

MAX_STORED_CANDIDATES = 500
MAX_REAL_INITIALIZER_VERIFICATIONS = 25

# Architecture search range.
#
# We deliberately keep dimensions aligned to multiples of 8 because:
#
# - this is friendlier to accelerator kernels
# - it avoids arbitrary irregular dimensions
# - the previous PB0 calibration used this granularity
#
R_VALUES = list(range(64, 769, 8))
N_VALUES = list(range(64, 769, 8))
H_VALUES = list(range(8, 129, 4))


# ==============================================================================
# DATA MODELS
# ==============================================================================

@dataclass(frozen=True)
class Formula:
    intercept: int
    r_coefficient: int
    n_coefficient: int
    h_coefficient: int
    feedback_r_coefficient: int
    feedback_constant_coefficient: int
    feedback_cap: int

    def feedback_rank(self, r: int, n: int) -> int:
        return min(self.feedback_cap, r, n)

    def predict(
        self,
        r: int,
        n: int,
        h: int,
    ) -> int:
        f = self.feedback_rank(r, n)

        return (
            self.intercept
            + self.r_coefficient * r
            + self.n_coefficient * n
            + self.h_coefficient * h
            + self.feedback_r_coefficient * f * r
            + self.feedback_constant_coefficient * f
        )


@dataclass(frozen=True)
class Candidate:
    r: int
    n: int
    h: int
    f: int
    predicted_parameters: int
    parameter_error: int
    absolute_parameter_error: int

    matrix_ratio: float
    vector_ratio: float
    router_ratio: float

    matrix_change_ratio: float
    vector_change_ratio: float
    router_change_ratio: float

    nonbaseline_distance: float
    structural_change_score: float
    balance_score: float
    ranking_score: float

    exact: bool


@dataclass
class VerificationResult:
    candidate_rank: int
    r: int
    n: int
    h: int
    f_formula: int

    predicted_parameters: int
    actual_parameters: int | None

    prediction_error: int | None
    exact_parameter_match: bool | None

    initialization_succeeded: bool
    error: str | None


# ==============================================================================
# GENERAL HELPERS
# ==============================================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def print_header(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input file does not exist:\n{path}"
        )

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise TypeError(
            f"Expected JSON object at top level in:\n{path}"
        )

    return data


def require_int(
    mapping: dict[str, Any],
    key: str,
    source_name: str,
) -> int:
    if key not in mapping:
        raise KeyError(
            f"Required field '{key}' was not found in {source_name}."
        )

    value = mapping[key]

    if isinstance(value, bool):
        raise TypeError(
            f"Field '{key}' in {source_name} must be an integer."
        )

    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Field '{key}' in {source_name} could not be converted "
            f"to an integer. Value: {value!r}"
        ) from exc


def safe_relative_change(
    value: int,
    baseline: int,
) -> float:
    if baseline == 0:
        raise ZeroDivisionError(
            "Baseline dimension cannot be zero."
        )

    return (value - baseline) / baseline


# ==============================================================================
# PB0-F FORMULA LOADING
# ==============================================================================

def load_formula() -> Formula:
    data = load_json(PB0_F_PATH)

    status = data.get("status")

    if status != "passed":
        raise RuntimeError(
            "PB0-F calibration did not pass.\n"
            f"PB0-F status: {status!r}\n"
            "Do not perform PB0-I until the calibrated formula "
            "has passed validation."
        )

    if data.get("canonical_exact_match") is not True:
        raise RuntimeError(
            "PB0-F canonical architecture did not exactly match "
            "the B0 parameter count."
        )

    if data.get("held_out_validation_exact") is not True:
        raise RuntimeError(
            "PB0-F held-out validation was not exact."
        )

    coefficients = data.get("formula_coefficients")

    if not isinstance(coefficients, dict):
        raise KeyError(
            "Could not find 'formula_coefficients' in:\n"
            f"{PB0_F_PATH}"
        )

    return Formula(
        intercept=require_int(
            coefficients,
            "intercept",
            str(PB0_F_PATH),
        ),
        r_coefficient=require_int(
            coefficients,
            "matrix_rank",
            str(PB0_F_PATH),
        ),
        n_coefficient=require_int(
            coefficients,
            "vector_dim",
            str(PB0_F_PATH),
        ),
        h_coefficient=require_int(
            coefficients,
            "router_hidden",
            str(PB0_F_PATH),
        ),
        feedback_r_coefficient=require_int(
            coefficients,
            "feedback_rank_times_matrix_rank",
            str(PB0_F_PATH),
        ),
        feedback_constant_coefficient=require_int(
            coefficients,
            "feedback_rank",
            str(PB0_F_PATH),
        ),
        feedback_cap=32,
    )


# ==============================================================================
# B0 TARGET LOADING
# ==============================================================================

def load_target_parameter_count() -> int:
    data = load_json(B0_CENSUS_PATH)

    for key in (
        "actual_parameter_count",
        "expected_parameter_count",
        "parameter_count",
        "total_parameters",
    ):
        value = data.get(key)

        if isinstance(value, int) and value > 0:
            return value

    raise KeyError(
        "Could not find the canonical B0 parameter count in:\n"
        f"{B0_CENSUS_PATH}"
    )


# ==============================================================================
# BASELINE VALIDATION
# ==============================================================================
def validate_previous_gates(
    pb0_f_data: dict,
    pb0_g_data: dict,
    pb0_h_data: dict,
    b0_data: dict,
) -> None:
    """
    Validate that all prerequisite PB0 gates completed successfully before
    performing the non-baseline architecture search.

    PB0-F:
        The calibrated parameter formula must be exact.

    PB0-G:
        The calibrated candidate search must have completed successfully.

    PB0-H:
        The real JAX parameter tree recount must have passed and all exact
        verification checks must be true.

    B0:
        The canonical frozen parameter count must be available.
    """

    # ------------------------------------------------------------------
    # PB0-F validation
    # ------------------------------------------------------------------

    pb0_f_status = pb0_f_data.get("status")

    if pb0_f_status != "passed":
        raise RuntimeError(
            "PB0-F did not pass parameter formula calibration.\n\n"
            f"PB0-F status: {pb0_f_status!r}"
        )

    canonical_exact_match = pb0_f_data.get(
        "canonical_exact_match"
    )

    if canonical_exact_match is not True:
        raise RuntimeError(
            "PB0-F canonical architecture verification was not exact.\n\n"
            f"canonical_exact_match: "
            f"{canonical_exact_match!r}"
        )

    held_out_validation_exact = pb0_f_data.get(
        "held_out_validation_exact"
    )

    if held_out_validation_exact is not True:
        raise RuntimeError(
            "PB0-F held-out formula validation was not exact.\n\n"
            f"held_out_validation_exact: "
            f"{held_out_validation_exact!r}"
        )

    # -------------------------------------------------------------------------
    # PB0-G compatibility validation
    # -------------------------------------------------------------------------
    # PB0-G output schemas from earlier research versions may not contain a
    # top-level "status" field. Validate the actual search evidence instead.
    # -------------------------------------------------------------------------

    pb0_g_status = pb0_g_data.get("status")

    pb0_g_exact_matches = pb0_g_data.get("exact_matches")
    pb0_g_candidates = pb0_g_data.get("candidates")

    if pb0_g_exact_matches is None:
        pb0_g_exact_matches = pb0_g_data.get(
            "exact_match_count"
        )

    if pb0_g_candidates is None:
        pb0_g_candidates = pb0_g_data.get(
            "stored_candidates"
        )

    pb0_g_completed = False

    # Newer schema explicitly records status.
    if pb0_g_status == "passed":
        pb0_g_completed = True

    # Compatibility with schemas that store an exact-match count.
    elif isinstance(pb0_g_exact_matches, int) and pb0_g_exact_matches >= 1:
        pb0_g_completed = True

    # Compatibility with schemas that store candidate lists.
    elif isinstance(pb0_g_candidates, list) and len(pb0_g_candidates) > 0:
        pb0_g_completed = True

    # Compatibility with summary/result structures.
    else:
        summary = pb0_g_data.get("summary")

        if isinstance(summary, dict):   
            summary_exact_matches = summary.get(
                "exact_matches",
                summary.get("exact_match_count"),
            )

            if (
                isinstance(summary_exact_matches, int)
                and summary_exact_matches >= 1
            ):
                pb0_g_completed = True


    if not pb0_g_completed:
        raise RuntimeError(
            "PB0-G did not provide valid calibrated candidate-search evidence.\n\n"
            f"PB0-G status: {pb0_g_status!r}\n"
            f"PB0-G top-level keys: {sorted(pb0_g_data.keys())}"
        )

    # ------------------------------------------------------------------
    # PB0-H validation
    # ------------------------------------------------------------------

    pb0_h_status = pb0_h_data.get("status")

    if pb0_h_status != "passed":
        raise RuntimeError(
            "PB0-H did not pass exact parameter-tree recount.\n\n"
            f"PB0-H status: {pb0_h_status!r}"
        )

    verification = pb0_h_data.get("verification")

    if not isinstance(verification, dict):
        raise RuntimeError(
            "PB0-H output does not contain a valid 'verification' object."
        )

    all_exact = verification.get("all_exact")

    if all_exact is not True:
        raise RuntimeError(
            "PB0-H did not pass exact parameter-tree recount.\n\n"
            f"PB0-H verification.all_exact: {all_exact!r}\n"
            f"PB0-H status: {pb0_h_status!r}"
        )

    required_pb0_h_checks = {
        "tree_matches_pb0_g_prediction": True,
        "tree_matches_b0_budget": True,
        "models_count_params_matches_tree": True,
        "feedback_rank_matches_pb0_g": True,
    }

    failed_pb0_h_checks = []

    for field_name, expected_value in (
        required_pb0_h_checks.items()
    ):
        actual_value = verification.get(field_name)

        if actual_value is not expected_value:
            failed_pb0_h_checks.append(
                f"{field_name}={actual_value!r}"
            )

    if failed_pb0_h_checks:
        raise RuntimeError(
            "PB0-H verification contains failed exact checks.\n\n"
            "Failed checks:\n"
            + "\n".join(
                f"  - {item}"
                for item in failed_pb0_h_checks
            )
        )

    # ------------------------------------------------------------------
    # B0 validation
    # ------------------------------------------------------------------

    b0_parameter_count = b0_data.get(
        "actual_parameter_count"
    )

    if not isinstance(b0_parameter_count, int):
        b0_parameter_count = b0_data.get(
            "expected_parameter_count"
        )

    if (
        not isinstance(b0_parameter_count, int)
        or b0_parameter_count <= 0
    ):
        raise RuntimeError(
            "B0 does not contain a valid canonical parameter count.\n\n"
            "Expected 'actual_parameter_count' or "
            "'expected_parameter_count' to contain a positive integer."
        )
# ==============================================================================
# CANDIDATE SCORING
# ==============================================================================

def calculate_structural_scores(
    r: int,
    n: int,
    h: int,
    parameter_error: int,
    target_parameters: int,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]:
    matrix_ratio = r / BASELINE_R
    vector_ratio = n / BASELINE_N
    router_ratio = h / BASELINE_H

    matrix_change_ratio = abs(
        safe_relative_change(r, BASELINE_R)
    )

    vector_change_ratio = abs(
        safe_relative_change(n, BASELINE_N)
    )

    router_change_ratio = abs(
        safe_relative_change(h, BASELINE_H)
    )

    # Euclidean normalized distance from baseline.
    #
    # Router width receives a smaller contribution because router
    # changes alone are relatively weak architectural changes compared
    # with matrix-rank and vector-state changes.
    nonbaseline_distance = math.sqrt(
        matrix_change_ratio ** 2
        + vector_change_ratio ** 2
        + 0.25 * router_change_ratio ** 2
    )

    # Structural score rewards meaningful changes to the two major
    # recurrent capacity axes.
    structural_change_score = (
        0.50 * matrix_change_ratio
        + 0.50 * vector_change_ratio
        + 0.10 * min(router_change_ratio, 1.0)
    )

    # Balance score rewards architectures where matrix and vector
    # capacity remain reasonably balanced instead of collapsing one
    # pathway while expanding the other.
    pathway_ratio = r / n

    log_balance = abs(math.log(pathway_ratio))

    balance_score = 1.0 / (1.0 + log_balance)

    # Parameter closeness is included only as a weak ranking signal
    # because all retained candidates are already inside tolerance.
    parameter_closeness = 1.0 - min(
        abs(parameter_error) / max(PARAMETER_TOLERANCE, 1),
        1.0,
    )

    ranking_score = (
        0.50 * structural_change_score
        + 0.30 * balance_score
        + 0.20 * parameter_closeness
    )

    return (
        matrix_ratio,
        vector_ratio,
        router_ratio,
        nonbaseline_distance,
        structural_change_score,
        balance_score,
        ranking_score,
    )


# ==============================================================================
# SEARCH
# ==============================================================================

def generate_candidates(
    formula: Formula,
    target_parameters: int,
) -> list[Candidate]:
    candidates: list[Candidate] = []

    for r in R_VALUES:
        for n in N_VALUES:
            f = formula.feedback_rank(r, n)

            # Solve the linear router-width equation approximately
            # before checking neighboring discrete h values.
            #
            # P = base + h_coefficient * h
            #
            # This avoids blindly evaluating every h for every
            # (r, n) pair.
            base_without_h = (
                formula.intercept
                + formula.r_coefficient * r
                + formula.n_coefficient * n
                + formula.feedback_r_coefficient * f * r
                + formula.feedback_constant_coefficient * f
            )

            ideal_h = (
                target_parameters - base_without_h
            ) / formula.h_coefficient

            nearest_h = (
                round(ideal_h / 4) * 4
            )

            h_candidates = {
                nearest_h - 8,
                nearest_h - 4,
                nearest_h,
                nearest_h + 4,
                nearest_h + 8,
            }

            for h in sorted(h_candidates):
                if h not in H_VALUES:
                    continue

                if (
                    r == BASELINE_R
                    and n == BASELINE_N
                    and h == BASELINE_H
                ):
                    continue

                predicted = formula.predict(r, n, h)

                error = predicted - target_parameters
                absolute_error = abs(error)

                if absolute_error > PARAMETER_TOLERANCE:
                    continue

                (
                    matrix_ratio,
                    vector_ratio,
                    router_ratio,
                    nonbaseline_distance,
                    structural_change_score,
                    balance_score,
                    ranking_score,
                ) = calculate_structural_scores(
                    r=r,
                    n=n,
                    h=h,
                    parameter_error=error,
                    target_parameters=target_parameters,
                )

                candidates.append(
                    Candidate(
                        r=r,
                        n=n,
                        h=h,
                        f=f,
                        predicted_parameters=predicted,
                        parameter_error=error,
                        absolute_parameter_error=absolute_error,
                        matrix_ratio=matrix_ratio,
                        vector_ratio=vector_ratio,
                        router_ratio=router_ratio,
                        matrix_change_ratio=abs(
                            safe_relative_change(
                                r,
                                BASELINE_R,
                            )
                        ),
                        vector_change_ratio=abs(
                            safe_relative_change(
                                n,
                                BASELINE_N,
                            )
                        ),
                        router_change_ratio=abs(
                            safe_relative_change(
                                h,
                                BASELINE_H,
                            )
                        ),
                        nonbaseline_distance=nonbaseline_distance,
                        structural_change_score=(
                            structural_change_score
                        ),
                        balance_score=balance_score,
                        ranking_score=ranking_score,
                        exact=error == 0,
                    )
                )

    candidates.sort(
        key=lambda candidate: (
            -candidate.ranking_score,
            candidate.absolute_parameter_error,
            -candidate.nonbaseline_distance,
            candidate.r,
            candidate.n,
            candidate.h,
        )
    )

    return candidates


# ==============================================================================
# REAL MODEL IMPORT
# ==============================================================================

def import_models_module() -> Any:
    if not MODELS_PATH.exists():
        raise FileNotFoundError(
            f"Canonical models source does not exist:\n{MODELS_PATH}"
        )

    module_name = "modus_x_pb0_i_models"

    spec = importlib.util.spec_from_file_location(
        module_name,
        MODELS_PATH,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not create import specification for:\n{MODELS_PATH}"
        )

    module = importlib.util.module_from_spec(spec)

    sys.modules[module_name] = module

    spec.loader.exec_module(module)

    return module


# ==============================================================================
# JAX TREE COUNTING
# ==============================================================================

def recursive_count_parameters(tree: Any) -> int:
    try:
        import jax
    except ImportError as exc:
        raise RuntimeError(
            "JAX is required for PB0-I real architecture verification."
        ) from exc

    leaves = jax.tree_util.tree_leaves(tree)

    total = 0

    for leaf in leaves:
        shape = getattr(leaf, "shape", None)

        if shape is None:
            continue

        size = 1

        for dimension in shape:
            size *= int(dimension)

        total += size

    return total


# ==============================================================================
# MODEL CONSTRUCTION
# ==============================================================================

def instantiate_candidate(
    models: Any,
    r: int,
    n: int,
    h: int,
) -> Any:
    model_config_cls = getattr(
        models,
        "ModelConfig",
        None,
    )

    make_model = getattr(
        models,
        "make_model",
        None,
    )

    if model_config_cls is None:
        raise AttributeError(
            "language/models.py does not expose ModelConfig."
        )

    if make_model is None:
        raise AttributeError(
            "language/models.py does not expose make_model."
        )

    canonical_config = {
        "vocab_size": 256,
        "embed_dim": 512,
        "hidden_dim": 1536,
        "ax_res": r,
        "n_layers": 12,
        "n_heads_attn": 8,
        "seq_len": 512,
        "mamba_state_dim": n,
        "vector_router": True,
        "router_hidden": h,
    }

    try:
        config = model_config_cls(**canonical_config)
    except TypeError as exc:
        signature = inspect.signature(model_config_cls)

        accepted = {
            name
            for name in signature.parameters
        }

        filtered = {
            key: value
            for key, value in canonical_config.items()
            if key in accepted
        }

        try:
            config = model_config_cls(**filtered)
        except Exception as nested_exc:
            raise RuntimeError(
                "Could not instantiate ModelConfig for PB0-I.\n"
                f"Original error: {exc}\n"
                f"Filtered config error: {nested_exc}\n"
                f"Config attempted: {canonical_config}"
            ) from nested_exc

    try:
        return make_model(config)
    except TypeError:
        return make_model(
            config=config
        )


# ==============================================================================
# REAL VERIFICATION
# ==============================================================================

def verify_candidates(
    models: Any,
    candidates: list[Candidate],
) -> list[VerificationResult]:
    verification_results: list[VerificationResult] = []

    for rank, candidate in enumerate(
        candidates[:MAX_REAL_INITIALIZER_VERIFICATIONS],
        start=1,
    ):
        try:
            parameter_tree = instantiate_candidate(
                models=models,
                r=candidate.r,
                n=candidate.n,
                h=candidate.h,
            )

            actual_parameters = recursive_count_parameters(
                parameter_tree
            )

            prediction_error = (
                actual_parameters
                - candidate.predicted_parameters
            )

            verification_results.append(
                VerificationResult(
                    candidate_rank=rank,
                    r=candidate.r,
                    n=candidate.n,
                    h=candidate.h,
                    f_formula=candidate.f,
                    predicted_parameters=(
                        candidate.predicted_parameters
                    ),
                    actual_parameters=actual_parameters,
                    prediction_error=prediction_error,
                    exact_parameter_match=(
                        prediction_error == 0
                    ),
                    initialization_succeeded=True,
                    error=None,
                )
            )

        except Exception as exc:
            verification_results.append(
                VerificationResult(
                    candidate_rank=rank,
                    r=candidate.r,
                    n=candidate.n,
                    h=candidate.h,
                    f_formula=candidate.f,
                    predicted_parameters=(
                        candidate.predicted_parameters
                    ),
                    actual_parameters=None,
                    prediction_error=None,
                    exact_parameter_match=None,
                    initialization_succeeded=False,
                    error=(
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
            )

    return verification_results


# ==============================================================================
# OUTPUT WRITING
# ==============================================================================

def candidate_to_row(
    candidate: Candidate,
) -> dict[str, Any]:
    return {
        "matrix_rank_r": candidate.r,
        "vector_dimension_n": candidate.n,
        "router_hidden_h": candidate.h,
        "feedback_rank_f": candidate.f,
        "predicted_parameters": (
            candidate.predicted_parameters
        ),
        "parameter_error": candidate.parameter_error,
        "absolute_parameter_error": (
            candidate.absolute_parameter_error
        ),
        "exact": candidate.exact,
        "matrix_ratio": candidate.matrix_ratio,
        "vector_ratio": candidate.vector_ratio,
        "router_ratio": candidate.router_ratio,
        "matrix_change_ratio": (
            candidate.matrix_change_ratio
        ),
        "vector_change_ratio": (
            candidate.vector_change_ratio
        ),
        "router_change_ratio": (
            candidate.router_change_ratio
        ),
        "nonbaseline_distance": (
            candidate.nonbaseline_distance
        ),
        "structural_change_score": (
            candidate.structural_change_score
        ),
        "balance_score": candidate.balance_score,
        "ranking_score": candidate.ranking_score,
    }


def write_csv(
    path: Path,
    rows: Iterable[dict[str, Any]],
) -> None:
    rows_list = list(rows)

    if not rows_list:
        raise RuntimeError(
            f"Cannot write empty CSV:\n{path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(rows_list[0].keys())

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows_list)


def build_fingerprint(
    candidates: list[Candidate],
    verifications: list[VerificationResult],
) -> str:
    payload = {
        "candidates": [
            candidate_to_row(candidate)
            for candidate in candidates
        ],
        "verifications": [
            asdict(result)
            for result in verifications
        ],
    }

    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


def write_markdown(
    path: Path,
    target_parameters: int,
    formula: Formula,
    candidates: list[Candidate],
    verifications: list[VerificationResult],
    fingerprint: str,
) -> None:
    exact_candidates = [
        candidate
        for candidate in candidates
        if candidate.exact
    ]

    verified_exact = [
        result
        for result in verifications
        if (
            result.initialization_succeeded
            and result.exact_parameter_match
        )
    ]

    best_candidate = (
        candidates[0]
        if candidates
        else None
    )

    lines: list[str] = []

    lines.append(
        "# PB0-I Non-Baseline Architecture Search"
    )
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "PB0-I searches for genuinely non-baseline Modus_X "
        "architectures near the frozen canonical parameter budget."
    )
    lines.append("")
    lines.append(
        "The search uses the exact PB0-F calibrated parameter formula "
        "and verifies the highest-ranked candidates through the real "
        "Modus_X initializer."
    )
    lines.append("")
    lines.append(
        "No model source, checkpoint, dataset, or training run is modified."
    )
    lines.append("")
    lines.append("## Frozen Budget")
    lines.append("")
    lines.append(
        f"- Target parameters: **{target_parameters:,}**"
    )
    lines.append(
        f"- Search tolerance: **±{PARAMETER_TOLERANCE:,}**"
    )
    lines.append("")
    lines.append("## Baseline Excluded")
    lines.append("")
    lines.append(
        f"- r = {BASELINE_R}"
    )
    lines.append(
        f"- n = {BASELINE_N}"
    )
    lines.append(
        f"- h = {BASELINE_H}"
    )
    lines.append("")
    lines.append("## Calibrated Formula")
    lines.append("")
    lines.append(
        "P = "
        f"{formula.intercept} "
        f"+ ({formula.r_coefficient} * r) "
        f"+ ({formula.n_coefficient} * n) "
        f"+ ({formula.h_coefficient} * h) "
        f"+ ({formula.feedback_r_coefficient} * f * r) "
        f"+ ({formula.feedback_constant_coefficient} * f)"
    )
    lines.append("")
    lines.append(
        f"where f = min({formula.feedback_cap}, r, n)"
    )
    lines.append("")
    lines.append("## Search Result")
    lines.append("")
    lines.append(
        f"- Non-baseline candidates within tolerance: **{len(candidates):,}**"
    )
    lines.append(
        f"- Exact non-baseline matches: **{len(exact_candidates):,}**"
    )
    lines.append(
        f"- Real initializer verifications: **{len(verifications):,}**"
    )
    lines.append(
        f"- Exact formula/tree matches: **{len(verified_exact):,}**"
    )
    lines.append("")

    if best_candidate is not None:
        lines.append(
            "## Highest-Ranked Non-Baseline Candidate"
        )
        lines.append("")
        lines.append(
            f"- Matrix rank r: **{best_candidate.r}**"
        )
        lines.append(
            f"- Vector dimension n: **{best_candidate.n}**"
        )
        lines.append(
            f"- Router hidden h: **{best_candidate.h}**"
        )
        lines.append(
            f"- Feedback rank f: **{best_candidate.f}**"
        )
        lines.append(
            f"- Predicted parameters: "
            f"**{best_candidate.predicted_parameters:,}**"
        )
        lines.append(
            f"- Parameter error: "
            f"**{best_candidate.parameter_error:+,}**"
        )
        lines.append(
            f"- Structural change score: "
            f"**{best_candidate.structural_change_score:.6f}**"
        )
        lines.append(
            f"- Balance score: "
            f"**{best_candidate.balance_score:.6f}**"
        )
        lines.append(
            f"- Overall ranking score: "
            f"**{best_candidate.ranking_score:.6f}**"
        )
        lines.append("")

    lines.append(
        "## Interpretation"
    )
    lines.append("")

    if exact_candidates:
        lines.append(
            "At least one genuinely non-baseline architecture exactly "
            "matches the frozen parameter budget under the calibrated formula."
        )
    else:
        lines.append(
            "No genuinely non-baseline architecture exactly matches the "
            "frozen parameter budget inside the searched discrete space."
        )

    lines.append("")

    lines.append(
        "An architecture should not be declared better than the baseline "
        "from parameter arithmetic alone. Parameter-count equivalence only "
        "establishes a fair budget comparison. Performance requires "
        "subsequent controlled evaluation."
    )

    lines.append("")
    lines.append("## Verification Results")
    lines.append("")

    if verifications:
        lines.append(
            "| Rank | r | n | h | Predicted | Actual | Error | Exact | Status |"
        )
        lines.append(
            "|---:|---:|---:|---:|---:|---:|---:|---|---|"
        )

        for result in verifications:
            actual = (
                f"{result.actual_parameters:,}"
                if result.actual_parameters is not None
                else "N/A"
            )

            error = (
                f"{result.prediction_error:+,}"
                if result.prediction_error is not None
                else "N/A"
            )

            exact = (
                str(result.exact_parameter_match)
                if result.exact_parameter_match is not None
                else "N/A"
            )

            status = (
                "initialized"
                if result.initialization_succeeded
                else "failed"
            )

            lines.append(
                f"| {result.candidate_rank} "
                f"| {result.r} "
                f"| {result.n} "
                f"| {result.h} "
                f"| {result.predicted_parameters:,} "
                f"| {actual} "
                f"| {error} "
                f"| {exact} "
                f"| {status} |"
            )
    else:
        lines.append(
            "No candidate verification was performed because no "
            "non-baseline candidate fell within the configured tolerance."
        )

    lines.append("")
    lines.append("## Reproducibility")
    lines.append("")
    lines.append(
        f"- Search fingerprint: `{fingerprint}`"
    )
    lines.append(
        f"- Generated UTC: `{utc_now()}`"
    )
    lines.append("")

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write("\n".join(lines))


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    print_header(
        "MODUS_X PB0-I NON-BASELINE ARCHITECTURE SEARCH"
    )

    print(
        f"Project root : {PROJECT_ROOT}"
    )
    print(
        f"Models source: {MODELS_PATH}"
    )
    print(
        f"PB0-F input  : {PB0_F_PATH}"
    )
    print(
        f"PB0-G input  : {PB0_G_PATH}"
    )
    print(
        f"PB0-H input  : {PB0_H_PATH}"
    )
    print(
        f"B0 input     : {B0_CENSUS_PATH}"
    )
    print(
        f"Output dir   : {OUTPUT_DIR}"

    )
    print_header("LOADING PREVIOUS RESEARCH OUTPUTS")

    pb0_f_data = load_json(PB0_F_PATH)
    pb0_g_data = load_json(PB0_G_PATH)
    pb0_h_data = load_json(PB0_H_PATH)
    b0_data = load_json(B0_CENSUS_PATH)

    target_parameters = int(
        b0_data["actual_parameter_count"]
    )

    print_header(
        "VALIDATING PREVIOUS GATES"
    )

    target_parameters = load_target_parameter_count()
    

    validate_previous_gates(
        pb0_f_data=pb0_f_data,
        pb0_g_data=pb0_g_data,
        pb0_h_data=pb0_h_data,
    b0_data=b0_data,
    )

    print(
        "PB0-F formula status : required"
    )
    print(
        "PB0-G search         : validated"
    )
    print(
        "PB0-H tree recount   : validated"
    )
    print(
        f"Frozen B0 budget     : {target_parameters:,}"
    )

    print_header(
        "LOADING CALIBRATED PARAMETER FORMULA"
    )

    formula = load_formula()

    print(
        f"Intercept                    : "
        f"{formula.intercept:,}"
    )
    print(
        f"r coefficient                : "
        f"{formula.r_coefficient:,}"
    )
    print(
        f"n coefficient                : "
        f"{formula.n_coefficient:,}"
    )
    print(
        f"h coefficient                : "
        f"{formula.h_coefficient:,}"
    )
    print(
        f"feedback r coefficient       : "
        f"{formula.feedback_r_coefficient:,}"
    )
    print(
        f"feedback constant coefficient: "
        f"{formula.feedback_constant_coefficient:,}"
    )
    print(
        f"feedback cap                 : "
        f"{formula.feedback_cap}"
    )

    print_header(
        "SEARCHING NON-BASELINE ARCHITECTURE SPACE"
    )

    total_rn_pairs = (
        len(R_VALUES)
        * len(N_VALUES)
    )

    print(
        f"Matrix rank values    : {len(R_VALUES):,}"
    )
    print(
        f"Vector dimension values: {len(N_VALUES):,}"
    )
    print(
        f"Router hidden values   : {len(H_VALUES):,}"
    )
    print(
        f"Approximate r/n pairs  : {total_rn_pairs:,}"
    )
    print(
        f"Tolerance              : ±{PARAMETER_TOLERANCE:,}"
    )
    print(
        "Baseline candidate excluded:"
    )
    print(
        f"  r={BASELINE_R} "
        f"n={BASELINE_N} "
        f"h={BASELINE_H}"
    )

    candidates = generate_candidates(
        formula=formula,
        target_parameters=target_parameters,
    )

    print()
    print(
        f"Non-baseline candidates within tolerance: "
        f"{len(candidates):,}"
    )

    exact_count = sum(
        1
        for candidate in candidates
        if candidate.exact
    )

    print(
        f"Exact non-baseline matches: "
        f"{exact_count:,}"
    )

    if not candidates:
        raise RuntimeError(
            "PB0-I found no non-baseline candidate within the "
            f"configured ±{PARAMETER_TOLERANCE:,} tolerance."
        )

    print_header(
        "TOP NON-BASELINE CANDIDATES"
    )

    for index, candidate in enumerate(
        candidates[:20],
        start=1,
    ):
        print(
            f"{index:03d} "
            f"r={candidate.r:3d} "
            f"n={candidate.n:3d} "
            f"h={candidate.h:3d} "
            f"f={candidate.f:2d} "
            f"params={candidate.predicted_parameters:,} "
            f"error={candidate.parameter_error:+,} "
            f"change={candidate.structural_change_score:.4f} "
            f"balance={candidate.balance_score:.4f} "
            f"score={candidate.ranking_score:.4f} "
            f"exact={candidate.exact}"
        )

    print_header(
        "IMPORTING ACTUAL MODUS_X ARCHITECTURE"
    )

    models = import_models_module()

    print(
        f"Imported module: {MODELS_PATH}"
    )

    print_header(
        "VERIFYING TOP NON-BASELINE CANDIDATES"
    )

    verification_results = verify_candidates(
        models=models,
        candidates=candidates,
    )

    for result in verification_results:
        if result.initialization_succeeded:
            print(
                f"{result.candidate_rank:03d} "
                f"r={result.r:3d} "
                f"n={result.n:3d} "
                f"h={result.h:3d} "
                f"predicted={result.predicted_parameters:,} "
                f"actual={result.actual_parameters:,} "
                f"error={result.prediction_error:+,} "
                f"exact={result.exact_parameter_match}"
            )
        else:
            print(
                f"{result.candidate_rank:03d} "
                f"r={result.r:3d} "
                f"n={result.n:3d} "
                f"h={result.h:3d} "
                f"FAILED: {result.error}"
            )

    verified_exact = [
        result
        for result in verification_results
        if (
            result.initialization_succeeded
            and result.exact_parameter_match
        )
    ]

    stored_candidates = candidates[
        :MAX_STORED_CANDIDATES
    ]

    fingerprint = build_fingerprint(
        candidates=stored_candidates,
        verifications=verification_results,
    )

    print_header(
        "WRITING PB0-I OUTPUTS"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "generated_at_utc": utc_now(),
        "project_root": str(PROJECT_ROOT),
        "models_path": str(MODELS_PATH),
        "scope": (
            "Search for genuinely non-baseline Modus_X architecture "
            "configurations under the frozen canonical parameter budget. "
            "No model source, checkpoint, dataset, or training run modified."
        ),
        "baseline": {
            "matrix_rank_r": BASELINE_R,
            "vector_dimension_n": BASELINE_N,
            "router_hidden_h": BASELINE_H,
            "parameters": target_parameters,
        },
        "target_parameter_count": target_parameters,
        "parameter_tolerance": PARAMETER_TOLERANCE,
        "formula": {
            "intercept": formula.intercept,
            "matrix_rank_coefficient": (
                formula.r_coefficient
            ),
            "vector_dimension_coefficient": (
                formula.n_coefficient
            ),
            "router_hidden_coefficient": (
                formula.h_coefficient
            ),
            "feedback_rank_times_matrix_rank_coefficient": (
                formula.feedback_r_coefficient
            ),
            "feedback_rank_coefficient": (
                formula.feedback_constant_coefficient
            ),
            "feedback_rank_rule": (
                f"min({formula.feedback_cap}, r, n)"
            ),
        },
        "search_space": {
            "matrix_rank_values": {
                "minimum": min(R_VALUES),
                "maximum": max(R_VALUES),
                "step": 8,
                "count": len(R_VALUES),
            },
            "vector_dimension_values": {
                "minimum": min(N_VALUES),
                "maximum": max(N_VALUES),
                "step": 8,
                "count": len(N_VALUES),
            },
            "router_hidden_values": {
                "minimum": min(H_VALUES),
                "maximum": max(H_VALUES),
                "step": 4,
                "count": len(H_VALUES),
            },
            "baseline_excluded": True,
        },
        "search_results": {
            "candidates_within_tolerance": (
                len(candidates)
            ),
            "exact_nonbaseline_matches": exact_count,
            "stored_candidates": (
                len(stored_candidates)
            ),
            "real_initializer_verifications": (
                len(verification_results)
            ),
            "exact_formula_tree_matches": (
                len(verified_exact)
            ),
        },
        "top_candidate": (
            candidate_to_row(candidates[0])
            if candidates
            else None
        ),
        "candidates": [
            candidate_to_row(candidate)
            for candidate in stored_candidates
        ],
        "real_initializer_verification": [
            asdict(result)
            for result in verification_results
        ],
        "fingerprint": fingerprint,
        "status": "passed",
        "next_gate": (
            "PB0-J should compare the strongest verified non-baseline "
            "candidate against the canonical baseline using controlled "
            "evaluation. Do not claim architectural superiority from "
            "parameter-count analysis alone."
        ),
    }

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
        )

    write_csv(
        OUTPUT_CSV,
        [
            candidate_to_row(candidate)
            for candidate in stored_candidates
        ],
    )

    write_csv(
        OUTPUT_VERIFICATION_CSV,
        [
            asdict(result)
            for result in verification_results
        ],
    )

    write_markdown(
        path=OUTPUT_MD,
        target_parameters=target_parameters,
        formula=formula,
        candidates=stored_candidates,
        verifications=verification_results,
        fingerprint=fingerprint,
    )

    print_header(
        "PB0-I NON-BASELINE ARCHITECTURE SEARCH COMPLETE"
    )

    print(
        f"Target parameter count          : "
        f"{target_parameters:,}"
    )
    print(
        f"Candidates within tolerance     : "
        f"{len(candidates):,}"
    )
    print(
        f"Exact non-baseline matches      : "
        f"{exact_count:,}"
    )
    print(
        f"Real initializer verifications  : "
        f"{len(verification_results):,}"
    )
    print(
        f"Exact formula/tree matches      : "
        f"{len(verified_exact):,}"
    )

    best = candidates[0]

    print()
    print(
        "Best non-baseline candidate:"
    )
    print(
        f"  r = {best.r}"
    )
    print(
        f"  n = {best.n}"
    )
    print(
        f"  h = {best.h}"
    )
    print(
        f"  f = {best.f}"
    )
    print(
        f"  parameters = "
        f"{best.predicted_parameters:,}"
    )
    print(
        f"  error      = "
        f"{best.parameter_error:+,}"
    )
    print(
        f"  exact      = "
        f"{best.exact}"
    )
    print(
        f"  ranking score = "
        f"{best.ranking_score:.6f}"
    )

    print()
    print(
        "Output files:"
    )
    print(
        f"  {OUTPUT_JSON}"
    )
    print(
        f"  {OUTPUT_MD}"
    )
    print(
        f"  {OUTPUT_CSV}"
    )
    print(
        f"  {OUTPUT_VERIFICATION_CSV}"
    )

    print()
    print(
        "NEXT GATE:"
    )

    if verified_exact:
        print(
            "At least one non-baseline candidate was verified "
            "against the real parameter tree."
        )
        print(
            "The next question is empirical performance, not "
            "parameter arithmetic."
        )
    else:
        print(
            "No verified exact non-baseline candidate was found "
            "among the highest-ranked candidates."
        )
        print(
            "Inspect the PB0-I report before deciding whether "
            "the search space or fixed budget should be reconsidered."
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print_header(
            "PB0-I NON-BASELINE ARCHITECTURE SEARCH FAILED"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        print()
        traceback.print_exc()

        raise