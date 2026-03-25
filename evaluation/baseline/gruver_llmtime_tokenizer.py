# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import numpy as np
from functools import partial
from dataclasses import dataclass

# Please check the original code at https://github.com/ngruver/llmtime/blob/main/data/serialize.py
@dataclass
class SerializerSettings:
    """
    Settings for time series serialization and deserialization.
    """
    base: int = 10                  # Numeric base for representation
    prec: int = 3                  # Number of digits after the 'decimal' point
    signed: bool = True            # Whether to include a sign for positive values
    fixed_length: bool = False     # If True, pad to a fixed number of digits
    max_val: float = 1e7           # Maximum absolute value allowed
    time_sep: str = ' ,'           # Separator between time steps
    bit_sep: str = ' '             # Separator between individual digits
    plus_sign: str = ''            # String to prepend for positive values
    minus_sign: str = ' -'         # String to prepend for negative values
    half_bin_correction: bool = True  # Apply half-bin correction on deserialization
    decimal_point: str = ''        # Literal to mark decimal point in serialized string
    missing_str: str = ' Nan'      # Representation for missing (NaN) values


def vec_num2repr(val: np.ndarray, base: int, prec: int, max_val: float):
    """
    Convert an array of floats into sign and digit representation in the given base.
    """
    base = float(base)
    sign = np.where(val >= 0, 1, -1)
    mag = np.abs(val)
    max_bit_pos = int(np.ceil(np.log(max_val) / np.log(base)).item())

    # Compute digits before the "decimal"
    before = []
    rem = mag.copy()
    for i in range(max_bit_pos):
        power = base ** (max_bit_pos - i - 1)
        digit = (rem / power).astype(int)
        before.append(digit)
        rem = rem - digit * power
    before = np.stack(before, axis=-1)

    # Compute digits after the "decimal"
    if prec > 0:
        after = []
        for i in range(prec):
            power = base ** (-(i + 1))
            digit = (rem / power).astype(int)
            after.append(digit)
            rem = rem - digit * power
        after = np.stack(after, axis=-1)
        digits = np.concatenate([before, after], axis=-1)
    else:
        digits = before

    return sign, digits


def vec_repr2num(sign: np.ndarray, digits: np.ndarray, base: int, prec: int, half_bin_correction: bool = True) -> np.ndarray:
    """
    Convert sign and digit arrays back into floats.
    """
    base = float(base)
    # Reverse the digit order for positional weights
    flipped = np.flip(digits, axis=-1)
    total_digits = digits.shape[1]
    # Compute positional powers
    powers = -np.arange(-prec, total_digits-prec)
    mags = np.sum(flipped / (base ** powers), axis=-1)
    if half_bin_correction:
        mags = mags + (0.5 / (base ** prec))
    return sign * mags


def serialize_arr(arr: np.ndarray, settings: SerializerSettings) -> str:
    """
    Serialize a 1D numpy array into a digit-level string using the given settings.
    """
    # Validate range
    clean = np.where(np.isnan(arr), 0.0, arr)
    assert np.all(np.abs(clean) <= settings.max_val), \
        f"Values must be within ±{settings.max_val}"

    # Convert numbers to sign and digits
    to_repr = partial(vec_num2repr, base=settings.base, prec=settings.prec, max_val=settings.max_val)
    sign_arr, digits_arr = to_repr(clean)
    is_nan = np.isnan(arr)

    def format_digits(digits):
        # Optionally strip leading zeros
        if not settings.fixed_length:
            nz = np.where(digits != 0)[0]
            if nz.size > 0:
                digits = digits[nz[0]:]
            else:
                digits = np.array([0], dtype=int)
        # Insert decimal point if specified
        if settings.decimal_point and settings.prec > 0:
            point_idx = len(digits) - settings.prec
            digits = np.concatenate([digits[:point_idx], [-1], digits[point_idx:]])
        # Join with bit separator
        return settings.bit_sep.join(str(d) for d in digits if d != -1)

    parts = []
    for s, digs, missing in zip(sign_arr, digits_arr, is_nan):
        if missing:
            parts.append(settings.missing_str)
        else:
            sign_str = settings.plus_sign if s > 0 else settings.minus_sign
            digit_str = format_digits(digs)
            parts.append(sign_str + digit_str)

    result = settings.time_sep.join(parts) + settings.time_sep
    return result


def deserialize_str(bit_str: str, settings: SerializerSettings, ignore_last: bool = False, steps: int = None) -> np.ndarray:
    """
    Deserialize a serialized string back into a numpy array of floats.
    """
    tokens = [t for t in bit_str.split(settings.time_sep) if t]
    if ignore_last:
        tokens = tokens[:-1]
    if steps is not None:
        tokens = tokens[:steps]

    sign_list = []
    digit_list = []
    for tok in tokens:
        if tok == settings.missing_str.strip():
            sign_list.append(1)
            digit_list.append([0])
            continue
        # Determine sign
        if settings.signed and tok.startswith(settings.minus_sign):
            sign_list.append(-1)
            tok = tok[len(settings.minus_sign):]
        else:
            sign_list.append(1)
            if settings.signed and tok.startswith(settings.plus_sign):
                tok = tok[len(settings.plus_sign):]
        # Split digits
        digs = [int(ch) for ch in tok.split(settings.bit_sep) if ch.isdigit()]
        digit_list.append(digs)

    # Pad to equal length
    maxlen = max(len(d) for d in digit_list)
    padded = np.array([([0]*(maxlen - len(d)) + d) for d in digit_list])

    nums = vec_repr2num(
        np.array(sign_list), padded,
        base=settings.base, prec=settings.prec,
        half_bin_correction=settings.half_bin_correction
    )
    return nums

gpt_settings = SerializerSettings(base=10, prec=3, signed=True, time_sep=', ', bit_sep=' ', minus_sign='-')
llama = SerializerSettings(base=10, prec=3, signed=True, half_bin_correction=True)

# Then wrap serialize_arr so it only takes the array
def gpt_formatter(arr: np.ndarray) -> str:
    return serialize_arr(arr, gpt_settings)

def llama_formatter(arr: np.ndarray) -> str:
    return serialize_arr(arr, llama)

# Backward compatibility: default to llama_formatter
# (or you can set to gpt_formatter if you prefer)
gruver_et_al_formatter = llama_formatter