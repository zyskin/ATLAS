#!/usr/bin/env python3
"""Apply every saved FeO model to real M3 spectra with domain diagnostics.

Input can be either a CSV containing m3_b01 ... m3_b74, or a Stage-1 directory
containing PREFIX_band_1.tsv ... PREFIX_band_74.tsv.  Raw predictions are never
clipped.  The program separately records physical-range, training-range,
principal-component distance, model-disagreement, and predicted-FeO flags.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import joblib
import numpy as np
import pandas as pd


BAND_COLUMNS = [f"m3_b{band:02d}" for band in range(1, 75)]
FLAG_INVALID_INPUT = 1
FLAG_REFLECTANCE_PHYSICAL_RANGE = 2
FLAG_TOO_MANY_BANDS_OUTSIDE_TRAINING_RANGE = 4
FLAG_PCA_DISTANCE = 8
FLAG_NEGATIVE_FEO = 16
FLAG_EXCESSIVE_FEO = 32
FLAG_MODEL_DISAGREEMENT = 64


def load_bundle(bundle_dir: Path):
    manifest_path = bundle_dir / "deployment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("band_columns") != BAND_COLUMNS:
        raise ValueError("Deployment bundle band definition is not m3_b01 ... m3_b74")
    models = {
        name: joblib.load(bundle_dir / relative)
        for name, relative in manifest["model_files"].items()
    }
    domain = joblib.load(bundle_dir / manifest["domain_reference_file"])
    return manifest, models, domain


def domain_diagnostics(x: np.ndarray, domain: dict[str, Any]):
    finite = np.isfinite(x).all(axis=1)
    safe = np.where(np.isfinite(x), x, 0.0)
    outside_minmax = np.sum(
        (safe < domain["band_minimum"]) | (safe > domain["band_maximum"]), axis=1
    ).astype(np.uint8)
    outside_q01_q99 = np.sum(
        (safe < domain["band_q01"]) | (safe > domain["band_q99"]), axis=1
    ).astype(np.uint8)
    standardized = domain["scaler"].transform(safe)
    scores = domain["pca"].transform(standardized)
    variance = np.maximum(
        domain["pca_score_variance"], domain["pca_variance_floor"]
    )
    distance_squared = np.sum(
        (scores - domain["pca_score_mean"]) ** 2 / variance, axis=1
    )
    distance_squared[~finite] = np.nan
    return finite, outside_minmax, outside_q01_q99, distance_squared


def predict_block(
    x: np.ndarray,
    models: dict[str, Any],
    domain: dict[str, Any],
    physical_reflectance_minimum: float,
    physical_reflectance_maximum: float,
    maximum_outside_bands: int,
    pca_threshold_name: str,
    physical_feo_maximum: float,
    disagreement_threshold: float,
) -> dict[str, np.ndarray]:
    count = len(x)
    finite, outside_minmax, outside_quantile, distance_squared = domain_diagnostics(x, domain)
    flags = np.zeros(count, dtype=np.uint16)
    flags[~finite] |= FLAG_INVALID_INPUT
    physical_reflectance = np.any(
        (x < physical_reflectance_minimum) | (x > physical_reflectance_maximum), axis=1
    )
    flags[finite & physical_reflectance] |= FLAG_REFLECTANCE_PHYSICAL_RANGE
    flags[finite & (outside_minmax > maximum_outside_bands)] |= (
        FLAG_TOO_MANY_BANDS_OUTSIDE_TRAINING_RANGE
    )
    pca_threshold = domain["pca_distance_squared_thresholds"][pca_threshold_name]
    flags[finite & (distance_squared > pca_threshold)] |= FLAG_PCA_DISTANCE

    safe = np.where(np.isfinite(x), x, 0.0)
    predictions = {}
    for name, model in models.items():
        value = np.asarray(model.predict(safe), dtype=float).reshape(-1)
        value[~finite] = np.nan
        predictions[name] = value
    matrix = np.column_stack([predictions[name] for name in models])
    with np.errstate(all="ignore"):
        consensus_mean = np.nanmean(matrix, axis=1)
        consensus_median = np.nanmedian(matrix, axis=1)
        disagreement = np.nanstd(matrix, axis=1, ddof=1)
        prediction_minimum = np.nanmin(matrix, axis=1)
        prediction_maximum = np.nanmax(matrix, axis=1)
    flags[finite & (prediction_minimum < 0.0)] |= FLAG_NEGATIVE_FEO
    flags[finite & (prediction_maximum > physical_feo_maximum)] |= FLAG_EXCESSIVE_FEO
    flags[finite & (disagreement > disagreement_threshold)] |= FLAG_MODEL_DISAGREEMENT
    strict_reject_bits = (
        FLAG_INVALID_INPUT
        | FLAG_REFLECTANCE_PHYSICAL_RANGE
        | FLAG_TOO_MANY_BANDS_OUTSIDE_TRAINING_RANGE
        | FLAG_PCA_DISTANCE
        | FLAG_NEGATIVE_FEO
        | FLAG_EXCESSIVE_FEO
        | FLAG_MODEL_DISAGREEMENT
    )
    screened = consensus_median.copy()
    screened[(flags & strict_reject_bits) != 0] = np.nan
    return {
        **{f"prediction_{name}": value for name, value in predictions.items()},
        "prediction_consensus_mean": consensus_mean,
        "prediction_consensus_median": consensus_median,
        "prediction_consensus_screened": screened,
        "model_prediction_standard_deviation": disagreement,
        "model_prediction_minimum": prediction_minimum,
        "model_prediction_maximum": prediction_maximum,
        "bands_outside_training_minimum_maximum": outside_minmax,
        "bands_outside_training_q01_q99": outside_quantile,
        "training_pca_distance_squared": distance_squared,
        "diagnostic_flag_mask": flags,
    }


def summarize_flags(flag_values: np.ndarray) -> dict[str, int]:
    definitions = {
        "invalid_input": FLAG_INVALID_INPUT,
        "reflectance_physical_range": FLAG_REFLECTANCE_PHYSICAL_RANGE,
        "too_many_bands_outside_training_range": FLAG_TOO_MANY_BANDS_OUTSIDE_TRAINING_RANGE,
        "pca_distance": FLAG_PCA_DISTANCE,
        "negative_feo": FLAG_NEGATIVE_FEO,
        "excessive_feo": FLAG_EXCESSIVE_FEO,
        "model_disagreement": FLAG_MODEL_DISAGREEMENT,
    }
    return {
        name: int(np.count_nonzero((flag_values & bit) != 0))
        for name, bit in definitions.items()
    }


def write_csv_predictions(
    input_csv: Path,
    output_dir: Path,
    models: dict[str, Any],
    domain: dict[str, Any],
    chunk_rows: int,
    diagnostic_arguments: dict[str, Any],
) -> dict[str, Any]:
    output_path = output_dir / "m3_feo_predictions.csv"
    rows_total = 0
    flags_all = []
    prediction_columns = None
    write_header = True
    for chunk in pd.read_csv(input_csv, chunksize=chunk_rows):
        missing = [column for column in BAND_COLUMNS if column not in chunk.columns]
        if missing:
            raise ValueError(f"CSV is missing bands: {', '.join(missing)}")
        x = chunk[BAND_COLUMNS].to_numpy(float)
        result = predict_block(x, models, domain, **diagnostic_arguments)
        if prediction_columns is None:
            prediction_columns = list(result)
        identity = chunk.drop(columns=BAND_COLUMNS)
        output = pd.concat(
            [identity.reset_index(drop=True), pd.DataFrame(result)], axis=1
        )
        output.to_csv(output_path, mode="w" if write_header else "a", header=write_header, index=False)
        write_header = False
        rows_total += len(chunk)
        flags_all.append(result["diagnostic_flag_mask"])
    combined_flags = np.concatenate(flags_all) if flags_all else np.array([], dtype=np.uint16)
    return {
        "input_mode": "csv",
        "input_rows": rows_total,
        "output": str(output_path.resolve()),
        "flag_counts": summarize_flags(combined_flags),
        "unflagged_rows": int(np.count_nonzero(combined_flags == 0)),
    }


def inspect_stage1(stage1_dir: Path, prefix: str):
    paths = [stage1_dir / f"{prefix}_band_{band}.tsv" for band in range(1, 75)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"Missing {len(missing)} Stage-1 band files; first is {missing[0]}")
    lines = 0
    columns = None
    with paths[0].open("r", encoding="utf-8") as handle:
        for line in handle:
            values = np.fromstring(line.rstrip("\n"), sep="\t")
            if columns is None:
                columns = len(values)
            elif len(values) != columns:
                raise ValueError(f"Inconsistent column count in {paths[0]} at line {lines + 1}")
            lines += 1
    if lines == 0 or not columns:
        raise ValueError(f"Empty Stage-1 band table: {paths[0]}")
    return paths, lines, int(columns)


def stage1_blocks(
    handles: list[Any], columns: int, chunk_lines: int
) -> Iterator[tuple[int, np.ndarray]]:
    line_offset = 0
    while True:
        band_blocks = []
        current_lines = None
        for band_number, handle in enumerate(handles, 1):
            rows = []
            for _ in range(chunk_lines):
                line = handle.readline()
                if not line:
                    break
                values = np.fromstring(line.rstrip("\n"), sep="\t")
                if len(values) != columns:
                    raise ValueError(
                        f"Band {band_number}, line {line_offset + len(rows) + 1}: "
                        f"expected {columns} columns, found {len(values)}"
                    )
                rows.append(values)
            if current_lines is None:
                current_lines = len(rows)
            elif len(rows) != current_lines:
                raise ValueError("Stage-1 band files have different row counts")
            band_blocks.append(np.asarray(rows, dtype=float))
        if not current_lines:
            break
        cube = np.stack(band_blocks, axis=-1)
        yield line_offset, cube.reshape(current_lines * columns, len(BAND_COLUMNS))
        line_offset += current_lines


def write_stage1_predictions(
    stage1_dir: Path,
    prefix: str,
    output_dir: Path,
    models: dict[str, Any],
    domain: dict[str, Any],
    chunk_lines: int,
    diagnostic_arguments: dict[str, Any],
) -> dict[str, Any]:
    paths, lines, columns = inspect_stage1(stage1_dir, prefix)
    output_names = [
        *[f"prediction_{name}" for name in models],
        "prediction_consensus_mean",
        "prediction_consensus_median",
        "prediction_consensus_screened",
        "model_prediction_standard_deviation",
        "model_prediction_minimum",
        "model_prediction_maximum",
        "bands_outside_training_minimum_maximum",
        "bands_outside_training_q01_q99",
        "training_pca_distance_squared",
        "diagnostic_flag_mask",
    ]
    arrays = {}
    integer_names = {
        "bands_outside_training_minimum_maximum": np.uint8,
        "bands_outside_training_q01_q99": np.uint8,
        "diagnostic_flag_mask": np.uint16,
    }
    for name in output_names:
        dtype = integer_names.get(name, np.float32)
        arrays[name] = np.lib.format.open_memmap(
            output_dir / f"{prefix}_{name}.npy",
            mode="w+", dtype=dtype, shape=(lines, columns),
        )

    with ExitStack() as stack:
        handles = [stack.enter_context(path.open("r", encoding="utf-8")) for path in paths]
        for line_offset, x in stage1_blocks(handles, columns, chunk_lines):
            result = predict_block(x, models, domain, **diagnostic_arguments)
            block_lines = len(x) // columns
            for name, values in result.items():
                arrays[name][line_offset:line_offset + block_lines, :] = values.reshape(
                    block_lines, columns
                )
            print(
                f"Processed lines {line_offset + 1}-{line_offset + block_lines} of {lines}",
                flush=True,
            )
    for array in arrays.values():
        array.flush()
    flag_array = np.asarray(arrays["diagnostic_flag_mask"])
    return {
        "input_mode": "stage1_directory",
        "prefix": prefix,
        "shape": [lines, columns],
        "pixels": lines * columns,
        "output_directory": str(output_dir.resolve()),
        "flag_counts": summarize_flags(flag_array.reshape(-1)),
        "unflagged_pixels": int(np.count_nonzero(flag_array == 0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-csv", type=Path)
    source.add_argument("--stage1-dir", type=Path)
    parser.add_argument("--prefix", help="Required with --stage1-dir")
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chunk-rows", type=int, default=50000)
    parser.add_argument("--chunk-lines", type=int, default=64)
    parser.add_argument("--reflectance-minimum", type=float, default=-0.05)
    parser.add_argument("--reflectance-maximum", type=float, default=1.2)
    parser.add_argument("--maximum-outside-bands", type=int, default=2)
    parser.add_argument("--pca-threshold", choices=("p95", "p99", "maximum"), default="p99")
    parser.add_argument("--physical-feo-maximum", type=float, default=30.0)
    parser.add_argument("--model-disagreement-threshold", type=float, default=5.0)
    args = parser.parse_args()
    if args.stage1_dir and not args.prefix:
        raise ValueError("--prefix is required with --stage1-dir")
    if args.chunk_rows < 1 or args.chunk_lines < 1:
        raise ValueError("Chunk sizes must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    deployment_manifest, models, domain = load_bundle(args.bundle_dir)
    diagnostic_arguments = {
        "physical_reflectance_minimum": args.reflectance_minimum,
        "physical_reflectance_maximum": args.reflectance_maximum,
        "maximum_outside_bands": args.maximum_outside_bands,
        "pca_threshold_name": args.pca_threshold,
        "physical_feo_maximum": args.physical_feo_maximum,
        "disagreement_threshold": args.model_disagreement_threshold,
    }
    if args.input_csv:
        summary = write_csv_predictions(
            args.input_csv, args.output_dir, models, domain,
            args.chunk_rows, diagnostic_arguments,
        )
    else:
        summary = write_stage1_predictions(
            args.stage1_dir, args.prefix, args.output_dir, models, domain,
            args.chunk_lines, diagnostic_arguments,
        )
    manifest = {
        "schema": "m3-feo-orbital-application-1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_directory": str(args.bundle_dir.resolve()),
        "training_dataset_sha256": deployment_manifest["dataset_sha256"],
        "models": list(models),
        "diagnostic_arguments": diagnostic_arguments,
        "flag_bit_definitions": {
            "1": "invalid or non-finite input band",
            "2": "reflectance outside configured broad physical interval",
            "4": "too many bands outside laboratory training minimum/maximum",
            "8": "principal-component distance beyond configured training threshold",
            "16": "at least one model predicted negative FeO",
            "32": "at least one model exceeded configured physical FeO maximum",
            "64": "between-model standard deviation exceeded threshold",
        },
        "important_interpretation": [
            "Raw model outputs are retained and never clipped.",
            "A screened consensus is omitted when any configured diagnostic flag is set.",
            "Between-model disagreement is not a calibrated uncertainty interval.",
            "Passing these checks does not prove scientific validity; failing them identifies unsupported extrapolation.",
        ],
        "summary": summary,
    }
    manifest_path = args.output_dir / "m3_feo_application_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
