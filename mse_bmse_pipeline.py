#!/usr/bin/env python3
"""Calculate ROI-wise MSE and pairwise bMSE from one or more 4D fMRI files."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import re
import shutil
import sys
import tempfile
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np
import pandas as pd


from entropy_metrics import BMSE, CalcMSE, default_n_jobs


class InputError(RuntimeError):
    """An input file or its geometry is invalid."""


class OutputConflict(RuntimeError):
    """Existing output does not match the requested analysis."""


@dataclass(frozen=True)
class Run:
    fmri: Path
    subject: str
    session: str
    run_id: str

    @property
    def parts(self) -> tuple[str, str, str]:
        return (safe_component(self.subject), safe_component(self.session),
                safe_component(self.run_id))


@dataclass
class Mask:
    name: str
    path: Path
    selected: np.ndarray
    voxel_count: int
    shape: tuple[int, ...]
    affine: np.ndarray
    zooms: tuple[float, ...]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def nifti_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    cleaned = cleaned.strip(".-")
    if not cleaned or cleaned in {"..", "."}:
        raise InputError(f"Invalid output identifier: {value!r}")
    return cleaned


def entity(text: str, key: str) -> str | None:
    match = re.search(rf"(?:^|[_/]){re.escape(key)}-([^_/]+)", text)
    return match.group(1) if match else None


def derive_run(path: Path, subject: str | None = None,
               session: str | None = None, run_id: str | None = None) -> Run:
    text = path.as_posix()
    stem = nifti_stem(path)
    sub = subject or entity(text, "sub") or "unknown"
    ses = session or entity(text, "ses") or "none"
    if run_id:
        rid = run_id
    else:
        task = entity(stem, "task")
        run_number = entity(stem, "run")
        parts = []
        if task:
            parts.append(f"task-{task}")
        if run_number:
            parts.append(f"run-{run_number}")
        rid = "_".join(parts) or stem
    return Run(path.resolve(), f"sub-{sub.removeprefix('sub-')}",
               f"ses-{ses.removeprefix('ses-')}", rid)


def discover_runs(args: argparse.Namespace) -> list[Run]:
    if args.fmri:
        path = args.fmri.expanduser().resolve()
        if not path.is_file():
            raise InputError(f"fMRI file not found: {path}")
        return [derive_run(path, args.subject_id, args.session_id, args.run_id)]

    root = args.input_root.expanduser().resolve()
    if not root.is_dir():
        raise InputError(f"Input root not found: {root}")
    paths = sorted(p.resolve() for p in root.glob(args.pattern) if p.is_file())
    if not paths:
        raise InputError(f"No fMRI files match {args.pattern!r} below {root}")
    runs = [derive_run(path) for path in paths]
    destinations = [run.parts for run in runs]
    if len(destinations) != len(set(destinations)):
        raise InputError("Discovered fMRI files map to duplicate output IDs")
    return runs


def find_mask_paths(mask_dir: Path) -> list[Path]:
    paths = sorted(
        (p.resolve() for p in mask_dir.iterdir()
         if p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz"))),
        key=lambda p: nifti_stem(p),
    )
    if len(paths) < 2:
        raise InputError(f"At least two .nii/.nii.gz masks are required in {mask_dir}")
    names = [nifti_stem(path) for path in paths]
    if len(names) != len(set(names)):
        raise InputError("Mask names are not unique after removing NIfTI suffixes")
    return paths


def load_masks(mask_dir: Path, strict: bool) -> tuple[list[Mask], list[str]]:
    mask_dir = mask_dir.expanduser().resolve()
    if not mask_dir.is_dir():
        raise InputError(f"Mask directory not found: {mask_dir}")
    masks: list[Mask] = []
    warnings: list[str] = []
    for path in find_mask_paths(mask_dir):
        image = nib.load(str(path))
        if len(image.shape) != 3:
            raise InputError(f"Mask must be 3D: {path} has shape {image.shape}")
        data = np.asarray(image.get_fdata(dtype=np.float32))
        if not np.isfinite(data).all():
            raise InputError(f"Mask contains non-finite values: {path}")
        is_binary = np.logical_or(np.isclose(data, 0.0), np.isclose(data, 1.0)).all()
        if not is_binary:
            message = f"Mask is not binary and will be thresholded at > 0.5: {path}"
            if strict:
                raise InputError(message)
            warnings.append(message)
        selected = data > 0.5
        count = int(selected.sum())
        if count == 0:
            raise InputError(f"Mask contains no voxels above 0.5: {path}")
        masks.append(Mask(
            name=nifti_stem(path), path=path, selected=selected,
            voxel_count=count, shape=tuple(image.shape),
            affine=np.asarray(image.affine),
            zooms=tuple(float(v) for v in image.header.get_zooms()[:3]),
        ))
    return masks, warnings


def validate_geometry(image: nib.spatialimages.SpatialImage, masks: Sequence[Mask],
                      fmri: Path, m: int, scales: int) -> None:
    if len(image.shape) != 4:
        raise InputError(f"fMRI must be 4D: {fmri} has shape {image.shape}")
    spatial_shape = tuple(image.shape[:3])
    n_timepoints = int(image.shape[3])
    minimum = scales * (m + 2)
    if n_timepoints < minimum:
        raise InputError(
            f"Too few timepoints for m={m}, a={scales}: {fmri} has "
            f"{n_timepoints}, requires at least {minimum}"
        )
    fmri_zooms = tuple(float(v) for v in image.header.get_zooms()[:3])
    for mask in masks:
        if mask.shape != spatial_shape:
            raise InputError(
                f"Shape mismatch: {mask.path} {mask.shape} != {fmri} {spatial_shape}"
            )
        if not np.allclose(mask.affine, image.affine, rtol=1e-5, atol=1e-4):
            raise InputError(f"Affine mismatch between {mask.path} and {fmri}")
        if not np.allclose(mask.zooms, fmri_zooms, rtol=1e-5, atol=1e-5):
            raise InputError(f"Voxel-size mismatch between {mask.path} and {fmri}")


def extract_timeseries(image: nib.spatialimages.SpatialImage,
                       masks: Sequence[Mask], fmri: Path) -> pd.DataFrame:
    data = np.asarray(image.get_fdata(dtype=np.float32))
    columns: dict[str, np.ndarray] = {}
    for mask in masks:
        voxels = data[mask.selected]
        if not np.isfinite(voxels).all():
            raise InputError(f"Non-finite BOLD values inside ROI {mask.name}: {fmri}")
        series = np.mean(voxels, axis=0, dtype=np.float64)
        if not np.isfinite(series).all():
            raise InputError(f"Non-finite ROI time series for {mask.name}: {fmri}")
        if float(np.std(series)) <= np.finfo(float).eps:
            raise InputError(f"Constant ROI time series for {mask.name}: {fmri}")
        columns[mask.name] = series
    return pd.DataFrame(columns)


def fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns}


def package_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for package in ("numpy", "pandas", "nibabel", "scipy", "joblib", "EntropyHub"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return versions


def expected_files(scales: int, save_timeseries: bool) -> set[str]:
    result = {"mse.csv", "bmse_CI.csv", "run_manifest.json"}
    result.update(f"bmse_a{i}.csv" for i in range(1, scales + 1))
    if save_timeseries:
        result.add("roi_timeseries.csv")
    return result


def analysis_signature(run: Run, masks: Sequence[Mask], args: argparse.Namespace,
                       jobs: int) -> dict[str, Any]:
    return {
        "fmri": fingerprint(run.fmri),
        "masks": [fingerprint(mask.path) for mask in masks],
        "parameters": {"m": args.m, "mse_r": args.mse_r,
                       "bmse_r": args.bmse_r, "a": args.a, "jobs": jobs},
        "save_timeseries": bool(args.save_timeseries),
    }


def completion_state(run_dir: Path, signature: dict[str, Any], scales: int,
                     save_timeseries: bool) -> str:
    if not run_dir.exists() or not any(run_dir.iterdir()):
        return "absent"
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return "conflict"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "conflict"
    complete = all((run_dir / name).is_file()
                   for name in expected_files(scales, save_timeseries))
    if (manifest.get("status") == "success" and
            manifest.get("signature") == signature and complete):
        return "complete"
    return "conflict"


def publish_results(temp_dir: Path, run_dir: Path, scales: int,
                    save_timeseries: bool) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    generated = {"mse.csv", "bmse_CI.csv", "run_manifest.json",
                 "roi_timeseries.csv"}
    generated.update(p.name for p in run_dir.glob("bmse_a*.csv"))
    generated.update(expected_files(scales, save_timeseries))
    # Publish data first and the success manifest last.
    manifest = temp_dir / "run_manifest.json"
    for source in sorted(temp_dir.iterdir()):
        if source.name != "run_manifest.json":
            os.replace(source, run_dir / source.name)
    for name in generated:
        target = run_dir / name
        if name not in {p.name for p in temp_dir.iterdir()} and target.exists() \
                and name not in expected_files(scales, save_timeseries):
            target.unlink()
    os.replace(manifest, run_dir / "run_manifest.json")


def process_run(run: Run, masks: Sequence[Mask], common_warnings: Sequence[str],
                args: argparse.Namespace, jobs: int) -> str:
    run_dir = args.output.resolve().joinpath(*run.parts)
    signature = analysis_signature(run, masks, args, jobs)
    state = completion_state(run_dir, signature, args.a, args.save_timeseries)
    if state == "complete" and not args.force:
        print(f"[skip] {run.fmri} -> {run_dir}")
        return "skipped"
    if state == "conflict" and not args.force:
        raise OutputConflict(
            f"Existing output is incomplete or has different inputs/parameters: "
            f"{run_dir} (use --force to replace generated files)"
        )

    started = utc_now()
    image = nib.load(str(run.fmri))
    validate_geometry(image, masks, run.fmri, args.m, args.a)
    tr = float(image.header.get_zooms()[3])
    if args.dry_run:
        print(f"[dry-run] {run.fmri} -> {run_dir} "
              f"({len(masks)} ROIs, shape={image.shape}, TR={tr:g})")
        image.uncache()
        return "successful"

    print(f"[run] {run.fmri} -> {run_dir}")
    timeseries = extract_timeseries(image, masks, run.fmri)
    temp_parent = run_dir.parent
    temp_parent.mkdir(parents=True, exist_ok=True)
    temp_path = Path(tempfile.mkdtemp(prefix=f".{run.parts[-1]}.tmp-",
                                     dir=temp_parent))
    try:
        CalcMSE(run.subject, timeseries, str(temp_path) + os.sep,
                m=args.m, r=args.mse_r, scales=args.a, n_jobs=jobs).process()
        BMSE(m=args.m, r=args.bmse_r, a=args.a, n_jobs=jobs).process_data(
            timeseries, str(temp_path / "bmse")
        )
        if args.save_timeseries:
            timeseries.to_csv(temp_path / "roi_timeseries.csv", index=False)
        manifest = {
            "status": "success", "started_at": started, "finished_at": utc_now(),
            "signature": signature,
            "identifiers": {"subject": run.subject, "session": run.session,
                            "run": run.run_id},
            "image": {"shape": list(image.shape), "timepoints": image.shape[3],
                      "tr": tr, "affine": np.asarray(image.affine).tolist()},
            "rois": [{"name": mask.name, "path": str(mask.path),
                      "voxel_count": mask.voxel_count} for mask in masks],
            "versions": package_versions(), "warnings": list(common_warnings),
        }
        (temp_path / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        publish_results(temp_path, run_dir, args.a, args.save_timeseries)
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)
        del timeseries
        image.uncache()
        gc.collect()
    print(f"[done] {run_dir}")
    return "successful"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def nonzero_int(value: str) -> int:
    parsed = int(value)
    if parsed == 0:
        raise argparse.ArgumentTypeError("must not be 0")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and > 0")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fmri", type=Path, help="one 4D NIfTI file")
    source.add_argument("--input-root", type=Path,
                        help="root containing multiple subject/session files")
    parser.add_argument("--pattern", default="sub-*/ses-*/func/*_bold.nii.gz",
                        help="relative glob used with --input-root")
    parser.add_argument("--masks", type=Path, required=True,
                        help="directory containing binary ROI masks")
    parser.add_argument("--output", type=Path, required=True, help="output root")
    parser.add_argument("--subject-id", help="single-file subject override")
    parser.add_argument("--session-id", help="single-file session override")
    parser.add_argument("--run-id", help="single-file run override")
    parser.add_argument("--m", type=positive_int, default=2)
    parser.add_argument("--r", type=positive_float,
                        help="set both MSE and bMSE tolerance")
    parser.add_argument("--mse-r", type=positive_float, default=None)
    parser.add_argument("--bmse-r", type=positive_float, default=None)
    parser.add_argument("--a", type=positive_int, default=10,
                        help="number of scales (default: 10)")
    parser.add_argument("--jobs", type=nonzero_int, default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--save-timeseries", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="discover and validate without calculating or writing")
    args = parser.parse_args(argv)
    if args.r is not None and (args.mse_r is not None or args.bmse_r is not None):
        parser.error("--r cannot be combined with --mse-r or --bmse-r")
    args.mse_r = args.r if args.r is not None else (args.mse_r or 0.3)
    args.bmse_r = args.r if args.r is not None else (args.bmse_r or 0.6)
    if args.input_root is not None and any(
            value is not None for value in
            (args.subject_id, args.session_id, args.run_id)):
        parser.error("ID overrides are supported only with --fmri")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    jobs = args.jobs if args.jobs is not None else default_n_jobs()
    try:
        runs = discover_runs(args)
        masks, mask_warnings = load_masks(args.masks, args.strict)
        for warning in mask_warnings:
            print(f"[warning] {warning}", file=sys.stderr)
        print(f"Inputs: {len(runs)} run(s), {len(masks)} ROI(s); "
              f"m={args.m}, mse-r={args.mse_r:g}, bmse-r={args.bmse_r:g}, "
              f"a={args.a}, jobs={jobs}")
    except InputError as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 3

    counts = {"successful": 0, "skipped": 0, "failed": 0}
    failure_kinds: list[type[Exception]] = []
    for run in runs:
        try:
            outcome = process_run(run, masks, mask_warnings, args, jobs)
            counts[outcome] += 1
        except (InputError, OutputConflict, Exception) as exc:
            counts["failed"] += 1
            failure_kinds.append(type(exc))
            print(f"[failed] {run.fmri}: {exc}", file=sys.stderr)
            if args.fail_fast:
                if os.environ.get("MSE_BMSE_TRACEBACK"):
                    traceback.print_exc()
                break
    print("Summary: " + ", ".join(f"{key}={value}" for key, value in counts.items()))
    if counts["failed"]:
        return 3 if all(issubclass(kind, (InputError, OutputConflict))
                        for kind in failure_kinds) else 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
