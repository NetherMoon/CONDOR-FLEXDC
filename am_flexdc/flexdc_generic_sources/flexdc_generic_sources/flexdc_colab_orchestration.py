"""Colab/local orchestration helpers for the generic FlexDC/CONDOR notebooks.

This module deliberately preserves the operational behavior of the original
project notebooks: clone/update controls, optional direct upload, verified ZIP
extraction, runtime-config installation, checkpoint discovery, repository
fingerprinting, artifact packaging, and optional Colab download.

No training or inference semantics live here.  It only coordinates files and
processes so model logic remains in the original-style dependency modules.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class RepositoryState:
    name: str
    root: str
    url: str
    branch: str
    commit: str
    dirty: bool
    action: str

    def to_dict(self) -> dict:
        return asdict(self)


def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_checked(
    command: Sequence[str],
    *,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    capture_output: bool = False,
    timeout: int | None = None,
) -> subprocess.CompletedProcess:
    printable = " ".join(str(part) for part in command)
    print("$", printable)
    return subprocess.run(
        [str(part) for part in command],
        cwd=None if cwd is None else str(cwd),
        env=None if env is None else dict(env),
        check=True,
        text=True,
        capture_output=capture_output,
        timeout=timeout,
    )


def _git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args],
        text=True,
    ).strip()


def clone_or_update_repository(
    *,
    name: str,
    url: str,
    destination: str | Path,
    branch: str = "main",
    force_reclone: bool = False,
    update_existing: bool = False,
    preserve_local_changes: bool = True,
    depth: int | None = None,
) -> RepositoryState:
    """Clone a repository or reuse/update an existing checkout.

    The original notebooks intentionally left existing local files unchanged.
    That remains the default.  Set ``update_existing=True`` to fetch and
    fast-forward a clean checkout.  A dirty checkout is never overwritten when
    ``preserve_local_changes=True``.
    """
    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    if force_reclone and destination.exists():
        print(f"Removing existing {name} checkout: {destination}")
        shutil.rmtree(destination)

    action = "reused"
    if not destination.exists():
        command = ["git", "clone", "--branch", branch]
        if depth is not None:
            command += ["--depth", str(depth)]
        command += [url, str(destination)]
        run_checked(command)
        action = "cloned"
    elif not (destination / ".git").exists():
        raise RuntimeError(
            f"{destination} exists but is not a git checkout.  Set force_reclone=True "
            "or point the notebook at a valid repository root."
        )
    elif update_existing:
        dirty = bool(_git_output(destination, "status", "--porcelain"))
        if dirty and preserve_local_changes:
            print(f"{name} has local changes; leaving checkout unchanged: {destination}")
            action = "reused_dirty"
        else:
            run_checked(["git", "-C", str(destination), "fetch", "origin", branch])
            run_checked(["git", "-C", str(destination), "checkout", branch])
            run_checked(["git", "-C", str(destination), "merge", "--ff-only", f"origin/{branch}"])
            action = "updated"
    else:
        print(f"Repository already exists; leaving local files unchanged: {destination}")

    commit = _git_output(destination, "rev-parse", "HEAD")
    dirty = bool(_git_output(destination, "status", "--porcelain"))
    return RepositoryState(
        name=name,
        root=str(destination),
        url=url,
        branch=branch,
        commit=commit,
        dirty=dirty,
        action=action,
    )


def verify_zip(path: str | Path, *, reject_exact_25_mib: bool = True) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if reject_exact_25_mib and size == 25 * 1024 * 1024:
        raise RuntimeError(
            f"{path.name} is exactly 25 MiB and appears truncated.  Re-upload the "
            "original file rather than training from an incomplete archive."
        )
    if not zipfile.is_zipfile(path):
        with path.open("rb") as handle:
            header = handle.read(32)
        raise zipfile.BadZipFile(
            f"Not a valid ZIP: {path}\nFirst bytes: {header!r}"
        )
    with zipfile.ZipFile(path, "r") as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise zipfile.BadZipFile(f"CRC failure in {path}: {bad_member}")
        names = archive.namelist()
    return {
        "path": str(path),
        "size_bytes": int(size),
        "sha256": sha256_file(path),
        "entries": int(len(names)),
        "names": names,
    }


def extract_zip_verified(
    path: str | Path,
    destination: str | Path,
    *,
    clean_destination: bool = True,
    required_fragments: Sequence[str] = (),
    reject_exact_25_mib: bool = True,
) -> tuple[Path, dict]:
    info = verify_zip(path, reject_exact_25_mib=reject_exact_25_mib)
    names = info["names"]
    missing = [fragment for fragment in required_fragments if not any(fragment in name for name in names)]
    if missing:
        raise RuntimeError(
            "ZIP is valid but is missing required content:\n"
            + "\n".join(f" - {fragment}" for fragment in missing)
        )
    destination = Path(destination).expanduser().resolve()
    if clean_destination and destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "r") as archive:
        archive.extractall(destination)
    return destination, info


def upload_files_direct(*, prompt: str, multiple: bool = True) -> list[Path]:
    if not is_colab():
        raise RuntimeError("Direct upload is available only inside Google Colab.")
    from google.colab import files

    print(prompt)
    uploaded = files.upload()
    if not multiple and len(uploaded) != 1:
        raise RuntimeError(f"Expected exactly one uploaded file, received: {list(uploaded)}")
    paths: list[Path] = []
    for name, contents in uploaded.items():
        path = Path("/content") / Path(name).name
        if not path.exists() or path.read_bytes() != contents:
            path.write_bytes(contents)
        paths.append(path.resolve())
    return paths


def resolve_file_or_upload(
    configured_path: str | Path | None,
    *,
    description: str,
    allowed_suffixes: Sequence[str] = (),
    auto_upload: bool = True,
    multiple: bool = False,
) -> Path | list[Path]:
    if configured_path is not None:
        path = Path(configured_path).expanduser().resolve()
        if path.exists():
            if allowed_suffixes and path.suffix.lower() not in {suffix.lower() for suffix in allowed_suffixes}:
                raise ValueError(f"Unexpected suffix for {description}: {path}")
            return path
    if not auto_upload:
        raise FileNotFoundError(f"Could not find {description}: {configured_path}")
    paths = upload_files_direct(prompt=f"Upload {description}.", multiple=multiple)
    if allowed_suffixes:
        allowed = {suffix.lower() for suffix in allowed_suffixes}
        bad = [path for path in paths if path.suffix.lower() not in allowed]
        if bad:
            raise ValueError(f"Unexpected uploaded files for {description}: {bad}")
    return paths if multiple else paths[0]


def find_bundle_root(search_roots: Iterable[str | Path], marker: str = "flexdc_generic_sources") -> Path:
    checked: list[Path] = []
    for root_value in search_roots:
        if root_value is None:
            continue
        root = Path(root_value).expanduser().resolve()
        checked.append(root)
        direct = root / marker
        if direct.exists():
            return root
        if root.exists() and root.is_dir():
            for candidate in root.rglob(marker):
                if candidate.is_dir():
                    return candidate.parent.resolve()
    raise FileNotFoundError(
        f"Could not locate {marker}. Checked: " + ", ".join(str(path) for path in checked)
    )


def install_runtime_bundle(source_root: str | Path, flexdc_root: str | Path) -> dict:
    """Install packaged FlexDC runtime files into a cloned FlexDC checkout.

    Dataset bundles produced by the repaired preparation script store custom
    files under ``runtime_bundle/FlexDC``.  This includes generated J2 workload
    INIs, the experiment/gradient/cluster configs, and reproducibility-patched
    scripts that are not guaranteed to exist in a fresh public clone.
    """
    source_root = Path(source_root).expanduser().resolve()
    flexdc_root = Path(flexdc_root).expanduser().resolve()
    candidates = []
    if (source_root / "runtime_bundle" / "FlexDC").exists():
        candidates.append(source_root / "runtime_bundle" / "FlexDC")
    if source_root.exists() and source_root.is_dir():
        candidates.extend(path for path in source_root.rglob("runtime_bundle/FlexDC") if path.is_dir())
    unique = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    if not unique:
        return {"installed": False, "source": None, "files": []}
    if len(unique) > 1:
        print("Multiple runtime bundles found; using:", unique[0])
    runtime_root = unique[0]
    installed: list[dict] = []
    for source in sorted(path for path in runtime_root.rglob("*") if path.is_file()):
        relative = source.relative_to(runtime_root)
        destination = flexdc_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        installed.append(
            {
                "relative_path": relative.as_posix(),
                "destination": str(destination),
                "sha256": sha256_file(destination),
            }
        )
    print(f"Installed {len(installed)} packaged FlexDC runtime files from {runtime_root}")
    return {"installed": True, "source": str(runtime_root), "files": installed}


def discover_checkpoints(root: str | Path) -> dict[str, list[Path]]:
    root = Path(root).expanduser().resolve()
    roles = ["latest", "best_loss", "best_objective", "best_feasibility", "final"]
    found = {role: [] for role in roles}
    if not root.exists():
        return found
    for path in root.rglob("*.pt"):
        lower = path.name.lower()
        for role in roles:
            if role in lower:
                found[role].append(path.resolve())
    for role in roles:
        found[role] = sorted(found[role], key=lambda path: (path.stat().st_mtime, path.name))
    return found


def select_checkpoint(
    root: str | Path,
    *,
    preferred_role: str = "best_feasibility",
    explicit_path: str | Path | None = None,
) -> Path:
    if explicit_path is not None:
        path = Path(explicit_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    found = discover_checkpoints(root)
    order = [preferred_role, "best_loss", "best_objective", "final", "latest"]
    for role in order:
        if found.get(role):
            return found[role][-1]
    all_pt = sorted(Path(root).expanduser().resolve().rglob("*.pt"))
    if len(all_pt) == 1:
        return all_pt[0].resolve()
    raise FileNotFoundError(f"Could not select a checkpoint under {root}. Found: {all_pt}")


def write_json(path: str | Path, payload) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def package_paths(
    output_zip: str | Path,
    paths: Iterable[str | Path],
    *,
    root: str | Path | None = None,
    include_missing: bool = False,
) -> Path:
    output_zip = Path(output_zip).expanduser().resolve()
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    root_path = None if root is None else Path(root).expanduser().resolve()
    seen: set[str] = set()
    seen_sources: set[str] = set()
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for value in paths:
            path = Path(value).expanduser().resolve()
            if not path.exists():
                if include_missing:
                    print("Skipping missing artifact:", path)
                continue
            iterable = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
            for item in iterable:
                source_key = str(item.resolve())
                if source_key in seen_sources:
                    continue
                seen_sources.add(source_key)
                if root_path is not None:
                    try:
                        arcname = item.relative_to(root_path).as_posix()
                    except ValueError:
                        arcname = item.name
                elif path.is_dir():
                    arcname = f"{path.name}/{item.relative_to(path).as_posix()}"
                else:
                    arcname = item.name
                if arcname in seen:
                    stem = Path(arcname).stem
                    suffix = Path(arcname).suffix
                    parent = Path(arcname).parent
                    counter = 2
                    while str(parent / f"{stem}_{counter}{suffix}") in seen:
                        counter += 1
                    arcname = str(parent / f"{stem}_{counter}{suffix}")
                seen.add(arcname)
                archive.write(item, arcname=arcname)
    return output_zip


def download_if_colab(path: str | Path, *, enabled: bool) -> None:
    path = Path(path).expanduser().resolve()
    if not enabled:
        print("Automatic download disabled. Artifact:", path)
        return
    if not is_colab():
        print("Not running in Colab. Artifact:", path)
        return
    from google.colab import files
    files.download(str(path))


def repository_manifest(states: Sequence[RepositoryState], output: str | Path) -> Path:
    return write_json(output, {"repositories": [state.to_dict() for state in states]})


def reassemble_multipart_file(
    part_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_size_bytes: int | None = None,
) -> Path:
    """Reassemble deterministic ``.partNNN`` uploads and verify the result."""
    if not part_paths:
        raise ValueError("No multipart files were supplied.")
    paths = [Path(path).expanduser().resolve() for path in part_paths]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)

    def part_number(path: Path) -> int:
        match = __import__("re").search(r"\.part(\d+)$", path.name, flags=__import__("re").IGNORECASE)
        if not match:
            raise ValueError(f"Multipart file does not end in .partNNN: {path}")
        return int(match.group(1))

    paths = sorted(paths, key=part_number)
    numbers = [part_number(path) for path in paths]
    expected_numbers = list(range(1, len(paths) + 1))
    if numbers != expected_numbers:
        raise RuntimeError(f"Multipart sequence is incomplete: found {numbers}, expected {expected_numbers}")
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as destination:
        for path in paths:
            with path.open("rb") as source:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
    actual_size = output_path.stat().st_size
    actual_sha = sha256_file(output_path)
    if expected_size_bytes is not None and actual_size != int(expected_size_bytes):
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"Reassembled size {actual_size} != expected {expected_size_bytes}")
    if expected_sha256 is not None and actual_sha.lower() != str(expected_sha256).lower():
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"Reassembled SHA-256 {actual_sha} != expected {expected_sha256}")
    print(f"Reassembled {len(paths)} parts -> {output_path} ({actual_size:,} bytes)")
    return output_path


def reassemble_from_manifest(
    manifest_path: str | Path,
    *,
    search_dir: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Reassemble a dataset ZIP from the manifest emitted by the prep script."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = Path(search_dir).expanduser().resolve() if search_dir is not None else manifest_path.parent
    parts = payload.get("parts", [])
    part_paths = [base / (item["name"] if isinstance(item, dict) else str(item)) for item in parts]
    if output_path is None:
        output_path = base / payload.get(
            "archive_name",
            payload.get("original_name", payload.get("output_name", "reassembled_dataset.zip")),
        )
    return reassemble_multipart_file(
        part_paths,
        output_path,
        expected_sha256=payload.get("archive_sha256", payload.get("sha256")),
        expected_size_bytes=payload.get("archive_size_bytes", payload.get("size_bytes")),
    )


def install_generic_sources_to_condor(
    source_dir: str | Path,
    condor_train_dir: str | Path,
    *,
    overwrite_generic_names: bool = True,
) -> dict:
    """Install generic modules beside—not over—the repository's versioned files.

    The original inference notebook generated V3-named copies from V2 sources.
    The repaired package removes that fragile string-replacement step, but still
    integrates the complete generic source set into ``am_flexdc/train`` so the
    cloned repository remains a runnable, inspectable workspace.
    """
    source_dir = Path(source_dir).expanduser().resolve()
    condor_train_dir = Path(condor_train_dir).expanduser().resolve()
    condor_train_dir.mkdir(parents=True, exist_ok=True)
    installed = []
    for source in sorted(source_dir.glob("*.py")):
        destination = condor_train_dir / source.name
        if destination.exists() and not overwrite_generic_names:
            continue
        shutil.copy2(source, destination)
        installed.append(
            {
                "source": str(source),
                "destination": str(destination),
                "sha256": sha256_file(destination),
            }
        )
    print(f"Installed {len(installed)} generic CONDOR-FlexDC source files into {condor_train_dir}")
    return {"source_dir": str(source_dir), "train_dir": str(condor_train_dir), "files": installed}
