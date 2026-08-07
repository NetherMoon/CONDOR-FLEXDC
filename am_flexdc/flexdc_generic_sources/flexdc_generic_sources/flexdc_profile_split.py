"""Profile-level dataset splitting for the focused J=2 FlexDC experiment.

The master dataset contains every two-profile workload built from four model
families (ResNet, GPT-2, Llama, Bloom) and two modes (Inference, Training).
For a selected model specification, workloads containing only visible profiles
are split by complete base configuration into 80% train, 10% validation, and
10% seen-profile baseline test. Workloads containing a held-out profile are
strictly test-only and are labelled as one- or two-held-out-profile tests.

All simulator repeats sharing a base configuration are assigned together.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

PROFILE_FAMILIES = ("ResNet", "GPT-2", "Llama", "Bloom")
PROFILE_MODES = ("Inference", "Training")
ALL_PROFILES = tuple(f"{family} {mode}" for mode in PROFILE_MODES for family in PROFILE_FAMILIES)


@dataclass(frozen=True)
class ProfileHoldoutModelSpec:
    model_id: str
    display_name: str
    split_type: str
    visible_families: tuple[str, ...]
    heldout_families: tuple[str, ...]

    @property
    def visible_profiles(self) -> tuple[str, ...]:
        return tuple(
            f"{family} {mode}"
            for mode in PROFILE_MODES
            for family in self.visible_families
        )

    @property
    def heldout_profiles(self) -> tuple[str, ...]:
        return tuple(
            f"{family} {mode}"
            for mode in PROFILE_MODES
            for family in self.heldout_families
        )

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["visible_profiles"] = list(self.visible_profiles)
        payload["heldout_profiles"] = list(self.heldout_profiles)
        return payload


MODEL_SPECS: dict[str, ProfileHoldoutModelSpec] = {
    "3x1_llama_holdout": ProfileHoldoutModelSpec(
        model_id="3x1_llama_holdout",
        display_name="3x1 - Llama Held Out",
        split_type="3x1",
        visible_families=("ResNet", "GPT-2", "Bloom"),
        heldout_families=("Llama",),
    ),
    "3x1_resnet_holdout": ProfileHoldoutModelSpec(
        model_id="3x1_resnet_holdout",
        display_name="3x1 - ResNet Held Out",
        split_type="3x1",
        visible_families=("GPT-2", "Llama", "Bloom"),
        heldout_families=("ResNet",),
    ),
    "2x2_train_resnet_gpt2": ProfileHoldoutModelSpec(
        model_id="2x2_train_resnet_gpt2",
        display_name="2x2 - Train ResNet/GPT-2; Hold Out Llama/Bloom",
        split_type="2x2",
        visible_families=("ResNet", "GPT-2"),
        heldout_families=("Llama", "Bloom"),
    ),
    "2x2_train_llama_bloom": ProfileHoldoutModelSpec(
        model_id="2x2_train_llama_bloom",
        display_name="2x2 - Train Llama/Bloom; Hold Out ResNet/GPT-2",
        split_type="2x2",
        visible_families=("Llama", "Bloom"),
        heldout_families=("ResNet", "GPT-2"),
    ),
}

_MODEL_ALIASES = {
    "3x1-llama": "3x1_llama_holdout",
    "3x1_llama": "3x1_llama_holdout",
    "llama": "3x1_llama_holdout",
    "3x1-resnet": "3x1_resnet_holdout",
    "3x1_resnet": "3x1_resnet_holdout",
    "resnet": "3x1_resnet_holdout",
    "2x2-resnet-gpt2": "2x2_train_resnet_gpt2",
    "2x2_resnet_gpt2": "2x2_train_resnet_gpt2",
    "resnet_gpt2": "2x2_train_resnet_gpt2",
    "2x2-llama-bloom": "2x2_train_llama_bloom",
    "2x2_llama_bloom": "2x2_train_llama_bloom",
    "llama_bloom": "2x2_train_llama_bloom",
}


def resolve_model_spec(model: str | ProfileHoldoutModelSpec) -> ProfileHoldoutModelSpec:
    if isinstance(model, ProfileHoldoutModelSpec):
        return model
    key = str(model).strip()
    key = _MODEL_ALIASES.get(key.lower(), key)
    if key not in MODEL_SPECS:
        raise KeyError(
            f"Unknown model specification {model!r}. Choose one of: "
            + ", ".join(MODEL_SPECS)
        )
    return MODEL_SPECS[key]


def normalize_family(text: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "", str(text).lower())
    if token in {"resnet", "resnet50"}:
        return "ResNet"
    if token in {"gpt2"}:
        return "GPT-2"
    if token in {"llama", "llama7b"}:
        return "Llama"
    if token in {"bloom", "bloom560m"}:
        return "Bloom"
    raise ValueError(f"Unknown profile family: {text!r}")


def normalize_mode(text: str) -> str:
    token = re.sub(r"[^a-z]+", "", str(text).lower())
    if token in {"inf", "infer", "inference"}:
        return "Inference"
    if token in {"train", "training"}:
        return "Training"
    raise ValueError(f"Unknown profile mode: {text!r}")


def canonical_profile(family: str, mode: str) -> str:
    return f"{normalize_family(family)} {normalize_mode(mode)}"


_PROFILE_PATTERN = re.compile(
    r"(?P<family>resnet(?:50)?|gpt[-_ ]?2|llama(?:7b)?|bloom(?:560m)?)"
    r"[-_ ]*(?P<mode>inf(?:erence)?|train(?:ing)?)",
    flags=re.IGNORECASE,
)


def parse_profiles_from_text(value: object) -> list[str]:
    """Extract canonical profiles from a workload name/path or JSON list."""
    if isinstance(value, (list, tuple, np.ndarray)):
        raw_items = list(value)
    else:
        text = str(value).strip()
        raw_items = []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    raw_items = parsed
            except Exception:
                pass
        if not raw_items:
            matches = list(_PROFILE_PATTERN.finditer(Path(text).stem))
            return [canonical_profile(m.group("family"), m.group("mode")) for m in matches]

    profiles: list[str] = []
    for item in raw_items:
        item_text = str(item)
        matches = list(_PROFILE_PATTERN.finditer(item_text))
        if matches:
            profiles.extend(
                canonical_profile(m.group("family"), m.group("mode"))
                for m in matches
            )
            continue
        parts = item_text.rsplit(" ", 1)
        if len(parts) == 2:
            profiles.append(canonical_profile(parts[0], parts[1]))
        else:
            raise ValueError(f"Could not parse profile from {item!r}")
    return profiles


def parse_workload_profiles(row: Mapping[str, object], expected_jobs: int | None = 2) -> list[str]:
    """Resolve profile identities from explicit metadata, name, or config path."""
    explicit_columns = (
        "Job_Profiles",
        "job_profiles",
        "Profiles",
        "profiles",
    )
    candidates: list[object] = []
    for column in explicit_columns:
        if column in row and not pd.isna(row[column]):
            candidates.append(row[column])
    for column in (
        "Workload_Name",
        "workload_name",
        "workload_config",
        "Workload_Config",
        "workload_config_path",
    ):
        if column in row and not pd.isna(row[column]):
            candidates.append(row[column])

    failures: list[str] = []
    for candidate in candidates:
        try:
            profiles = parse_profiles_from_text(candidate)
        except Exception as exc:
            failures.append(f"{candidate!r}: {exc}")
            continue
        if expected_jobs is None or len(profiles) == expected_jobs:
            return profiles
    raise ValueError(
        "Could not resolve the workload profiles"
        + (f" (expected {expected_jobs})" if expected_jobs is not None else "")
        + ". Candidates/errors: "
        + "; ".join(failures or [repr(x) for x in candidates])
    )


def workload_type(profiles: Sequence[str]) -> str:
    modes = [profile.rsplit(" ", 1)[1] for profile in profiles]
    if all(mode == "Inference" for mode in modes):
        return "Inference-Only"
    if all(mode == "Training" for mode in modes):
        return "Training-Only"
    return "Mixed"


def stable_hash(value: str, seed: int = 0) -> int:
    digest = hashlib.sha256(f"{int(seed)}|{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _first_present(frame: pd.DataFrame, names: Iterable[str]) -> str | None:
    lower = {str(column).lower(): str(column) for column in frame.columns}
    for name in names:
        if name in frame.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def ensure_base_group_id(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    frame = frame.copy()
    existing = _first_present(frame, ("Base_Plan_Row_ID", "base_plan_row_id"))
    if existing is not None:
        frame["Base_Plan_Row_ID"] = frame[existing].astype(str)
        return frame, "Base_Plan_Row_ID"

    plan_id = _first_present(frame, ("Plan_Row_ID", "plan_row_id"))
    seed_column = _first_present(frame, ("Simulation_Seed", "simulation_seed", "random_seed"))
    identity_candidates = [
        _first_present(frame, ("Workload_Name", "workload_name", "workload_config")),
        _first_present(frame, ("server_count", "Server_Count")),
        _first_present(frame, ("utilization", "Utilization")),
        _first_present(frame, ("Pbar_kw_per_server", "Pbar")),
        _first_present(frame, ("R_kw_per_server", "R")),
        _first_present(frame, ("weights",)),
    ]
    identity_columns = [column for column in identity_candidates if column is not None]
    if len(identity_columns) >= 5:
        def build_identity(row: pd.Series) -> str:
            parts = []
            for column in identity_columns:
                value = row[column]
                if isinstance(value, float):
                    parts.append(f"{value:.12g}")
                else:
                    parts.append(str(value))
            return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:24]
        frame["Base_Plan_Row_ID"] = frame.apply(build_identity, axis=1)
        return frame, "Base_Plan_Row_ID"

    if plan_id is not None:
        frame["Base_Plan_Row_ID"] = frame[plan_id].astype(str)
        if seed_column is not None:
            # Only a last-resort fallback. Exact-plan datasets should provide a
            # true base ID when seed replicates exist.
            frame["Base_Plan_Row_ID"] = frame["Base_Plan_Row_ID"].str.replace(
                r"(?:[_:-]seed)?[_:-]?\d+$", "", regex=True
            )
        return frame, "Base_Plan_Row_ID"
    raise KeyError("Cannot construct a base configuration ID.")


def _context_series(frame: pd.DataFrame) -> pd.Series:
    workload_col = _first_present(frame, ("Workload_Name", "workload_name"))
    n_col = _first_present(frame, ("server_count", "Server_Count"))
    u_col = _first_present(frame, ("utilization", "Utilization"))
    if workload_col is None:
        raise KeyError("Workload_Name is required for profile holdout splitting.")
    n_text = frame[n_col].astype(str) if n_col else pd.Series("NA", index=frame.index)
    u_text = (
        pd.to_numeric(frame[u_col], errors="coerce").map(lambda x: f"{x:.8g}")
        if u_col else pd.Series("NA", index=frame.index)
    )
    return frame[workload_col].astype(str) + "|N=" + n_text + "|U=" + u_text


def _allocate_seen_group_splits(
    group_ids: Sequence[str],
    *,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> dict[str, str]:
    total_fraction = train_fraction + validation_fraction + test_fraction
    if not np.isclose(total_fraction, 1.0):
        raise ValueError(f"Split fractions must sum to 1.0, got {total_fraction}")
    unique = sorted(set(map(str, group_ids)), key=lambda x: (stable_hash(x, seed), x))
    n = len(unique)
    if n < 3:
        raise ValueError("Each seen-profile context needs at least three base groups.")
    n_validation = max(1, int(round(n * validation_fraction)))
    n_test = max(1, int(round(n * test_fraction)))
    if n_validation + n_test >= n:
        n_validation = 1
        n_test = 1
    n_train = n - n_validation - n_test
    if n_train < 1:
        raise ValueError("Split allocation left no training groups.")

    assignments: dict[str, str] = {}
    for group_id in unique[:n_train]:
        assignments[group_id] = "train"
    for group_id in unique[n_train:n_train + n_validation]:
        assignments[group_id] = "validation"
    for group_id in unique[n_train + n_validation:]:
        assignments[group_id] = "test"
    return assignments


def assign_profile_holdout_split(
    dataframe: pd.DataFrame,
    model: str | ProfileHoldoutModelSpec,
    *,
    split_seed: int = 20260804,
    train_fraction: float = 0.80,
    validation_fraction: float = 0.10,
    baseline_test_fraction: float = 0.10,
    expected_jobs: int | None = 2,
    strict_master_coverage: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Create one strict profile-holdout dataset and audit report."""
    spec = resolve_model_spec(model)
    frame, _ = ensure_base_group_id(dataframe)
    frame = frame.copy().reset_index(drop=True)

    workload_col = _first_present(frame, ("Workload_Name", "workload_name"))
    if workload_col is None:
        raise KeyError("Workload_Name is required.")
    frame["Workload_Name"] = frame[workload_col].astype(str)
    if "Plan_Row_ID" not in frame.columns:
        plan_col = _first_present(frame, ("plan_row_id",))
        if plan_col is not None:
            frame["Plan_Row_ID"] = frame[plan_col].astype(str)
        else:
            frame["Plan_Row_ID"] = [f"row_{index:09d}" for index in range(len(frame))]

    workload_metadata: dict[str, dict] = {}
    for workload_name, row in frame.groupby("Workload_Name", sort=True).first().iterrows():
        row_map = row.to_dict()
        row_map["Workload_Name"] = workload_name
        profiles = parse_workload_profiles(row_map, expected_jobs=expected_jobs)
        heldout = [profile for profile in profiles if profile in spec.heldout_profiles]
        visible = [profile for profile in profiles if profile in spec.visible_profiles]
        if len(heldout) + len(visible) != len(profiles):
            unknown = [p for p in profiles if p not in set(ALL_PROFILES)]
            raise ValueError(f"Unknown profiles in {workload_name}: {unknown}")
        workload_metadata[str(workload_name)] = {
            "profiles": profiles,
            "heldout": heldout,
            "visible": visible,
            "workload_type": workload_type(profiles),
        }

    meta = frame["Workload_Name"].map(workload_metadata)
    frame["Job_Profiles"] = meta.map(lambda x: json.dumps(x["profiles"]))
    for index in range(expected_jobs or max(len(x["profiles"]) for x in workload_metadata.values())):
        frame[f"Job_Profile_{index + 1}"] = meta.map(
            lambda x, idx=index: x["profiles"][idx] if idx < len(x["profiles"]) else None
        )
    frame["Workload_Type"] = meta.map(lambda x: x["workload_type"])
    frame["Heldout_Profile_Count"] = meta.map(lambda x: len(x["heldout"])).astype(int)
    frame["Heldout_Profiles"] = meta.map(lambda x: json.dumps(x["heldout"]))
    frame["Visible_Profiles_In_Workload"] = meta.map(lambda x: json.dumps(x["visible"]))
    frame["Context_ID"] = _context_series(frame)

    assignments: dict[str, str] = {}
    seen = frame[frame["Heldout_Profile_Count"] == 0]
    for context_id, group in seen.groupby("Context_ID", sort=True):
        context_seed = stable_hash(str(context_id), split_seed) % (2**31 - 1)
        assignments.update(
            _allocate_seen_group_splits(
                group["Base_Plan_Row_ID"].astype(str).unique(),
                seed=context_seed,
                train_fraction=train_fraction,
                validation_fraction=validation_fraction,
                test_fraction=baseline_test_fraction,
            )
        )

    original_split_col = _first_present(frame, ("Data_Split", "data_split"))
    if original_split_col is not None:
        frame["Original_Data_Split"] = frame[original_split_col].astype(str)
    else:
        frame["Original_Data_Split"] = ""

    def data_split_for_row(row: pd.Series) -> str:
        if int(row["Heldout_Profile_Count"]) > 0:
            return "test"
        return assignments[str(row["Base_Plan_Row_ID"])]

    frame["Data_Split"] = frame.apply(data_split_for_row, axis=1)
    frame["Benchmark_Type"] = np.select(
        [
            frame["Data_Split"].eq("train"),
            frame["Data_Split"].eq("validation"),
            frame["Heldout_Profile_Count"].eq(0),
            frame["Heldout_Profile_Count"].eq(1),
            frame["Heldout_Profile_Count"].ge(2),
        ],
        [
            "Training",
            "Validation",
            "Seen-Profile Baseline",
            "One Held-Out Profile",
            "Two Held-Out Profiles",
        ],
        default="Unknown",
    )
    frame["Model_ID"] = spec.model_id
    frame["Model_Display_Name"] = spec.display_name
    frame["Split_Type"] = spec.split_type
    frame["Profiles_Visible_During_Training"] = json.dumps(spec.visible_profiles)
    frame["Profiles_Completely_Held_Out"] = json.dumps(spec.heldout_profiles)
    frame["Profile_Split_Seed"] = int(split_seed)

    group_split_counts = frame.groupby("Base_Plan_Row_ID")["Data_Split"].nunique()
    crossing_groups = group_split_counts[group_split_counts != 1]
    train_validation = frame[frame["Data_Split"].isin(["train", "validation"])]
    leaked_rows = train_validation[train_validation["Heldout_Profile_Count"] > 0]
    heldout_non_test = frame[
        (frame["Heldout_Profile_Count"] > 0) & (frame["Data_Split"] != "test")
    ]

    workload_summary = (
        frame.groupby(
            ["Workload_Name", "Workload_Type", "Heldout_Profile_Count", "Benchmark_Type", "Data_Split"],
            dropna=False,
            sort=True,
        )
        .agg(Rows=("Plan_Row_ID", "size"), Base_Groups=("Base_Plan_Row_ID", "nunique"))
        .reset_index()
    )
    split_summary = (
        frame.groupby(["Data_Split", "Benchmark_Type"], sort=True)
        .agg(
            Rows=("Plan_Row_ID", "size"),
            Base_Groups=("Base_Plan_Row_ID", "nunique"),
            Workloads=("Workload_Name", "nunique"),
        )
        .reset_index()
    )

    unique_workloads = frame[["Workload_Name", "Heldout_Profile_Count"]].drop_duplicates()
    actual_workload_counts = {
        "seen_profile_workloads": int((unique_workloads["Heldout_Profile_Count"] == 0).sum()),
        "one_heldout_workloads": int((unique_workloads["Heldout_Profile_Count"] == 1).sum()),
        "two_heldout_workloads": int((unique_workloads["Heldout_Profile_Count"] >= 2).sum()),
        "total_workloads": int(len(unique_workloads)),
    }
    expected_seen = 15 if spec.split_type == "3x1" else 6
    expected_total = 28
    errors: list[str] = []
    warnings: list[str] = []
    if len(crossing_groups):
        errors.append(f"{len(crossing_groups)} base configuration groups cross split boundaries.")
    if len(leaked_rows):
        errors.append(f"{len(leaked_rows)} held-out-profile rows leaked into train/validation.")
    if len(heldout_non_test):
        errors.append(f"{len(heldout_non_test)} held-out-profile rows are not test-only.")
    if set(frame["Data_Split"]) != {"train", "validation", "test"}:
        errors.append(f"Expected train/validation/test; found {sorted(set(frame['Data_Split']))}.")
    if actual_workload_counts["seen_profile_workloads"] != expected_seen:
        errors.append(
            f"Seen-profile workload definitions={actual_workload_counts['seen_profile_workloads']}, "
            f"expected={expected_seen} for {spec.model_id}."
        )
    if strict_master_coverage and actual_workload_counts["total_workloads"] != expected_total:
        errors.append(
            f"Master workload definitions={actual_workload_counts['total_workloads']}, expected={expected_total}."
        )

    audit = {
        "status": "PASS" if not errors else "FAIL",
        "model_spec": spec.to_dict(),
        "split_seed": int(split_seed),
        "fractions_for_seen_profiles": {
            "train": float(train_fraction),
            "validation": float(validation_fraction),
            "seen_profile_baseline_test": float(baseline_test_fraction),
        },
        "rows": int(len(frame)),
        "base_groups": int(frame["Base_Plan_Row_ID"].nunique()),
        "contexts": int(frame["Context_ID"].nunique()),
        "workload_counts": actual_workload_counts,
        "split_summary": split_summary.to_dict(orient="records"),
        "workload_summary": workload_summary.to_dict(orient="records"),
        "base_groups_crossing_splits": int(len(crossing_groups)),
        "heldout_rows_in_train_or_validation": int(len(leaked_rows)),
        "heldout_rows_not_in_test": int(len(heldout_non_test)),
        "errors": errors,
        "warnings": warnings,
    }
    if errors:
        raise ValueError("Profile holdout split audit failed:\n- " + "\n- ".join(errors))
    return frame, audit


def write_model_specs_json(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({key: spec.to_dict() for key, spec in MODEL_SPECS.items()}, indent=2),
        encoding="utf-8",
    )
    return path
