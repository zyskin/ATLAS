#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(
    description="Convert the M3 global bandpass IMG file to a wide CSV table."
)
parser.add_argument("input_img", type=Path)
parser.add_argument("output_csv", type=Path)
args = parser.parse_args()

number_of_wavelengths = 2701
number_of_bands = 85
first_wavelength_nm = 350

# <f4 means little-endian 32-bit floating point.
values = np.fromfile(args.input_img, dtype="<f4")

expected = number_of_wavelengths * number_of_bands
if values.size != expected:
    raise SystemExit(
        f"Unexpected file size: found {values.size} values; "
        f"expected {expected}."
    )

bandpasses = values.reshape(number_of_wavelengths, number_of_bands)
wavelengths = np.arange(
    first_wavelength_nm,
    first_wavelength_nm + number_of_wavelengths,
)

with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(
        ["wavelength_nm"]
        + [f"M3_global_band_{band}" for band in range(1, 86)]
    )

    for wavelength, responses in zip(wavelengths, bandpasses):
        writer.writerow(
            [int(wavelength)]
            + [format(float(value), ".9g") for value in responses]
        )

print("Output:", args.output_csv)
print("Matrix shape:", bandpasses.shape)
print("Minimum value:", float(np.min(bandpasses)))
print("Maximum value:", float(np.max(bandpasses)))
print("First five band-column sums:")
print(bandpasses.sum(axis=0)[:5])
