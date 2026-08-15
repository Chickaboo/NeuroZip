"""A small forward binary arithmetic/range coder.

The implementation uses integer arithmetic over a 64-bit coding interval and
accepts a fresh positive-count CDF for every symbol. It is intentionally
simple and auditable for V0; a faster rANS implementation can be added behind
the same codec interface later.
"""

from __future__ import annotations

from bisect import bisect_right
from typing import Sequence

from .cdf import validate_cdf


STATE_BITS = 64
FULL = 1 << STATE_BITS
HALF = FULL >> 1
FIRST_QUARTER = FULL >> 2
THIRD_QUARTER = FIRST_QUARTER * 3


class _BitWriter:
    def __init__(self) -> None:
        self._bytes = bytearray()
        self._current = 0
        self._bits = 0

    def write(self, bit: int) -> None:
        self._current = (self._current << 1) | (bit & 1)
        self._bits += 1
        if self._bits == 8:
            self._bytes.append(self._current)
            self._current = 0
            self._bits = 0

    def finish(self) -> bytes:
        if self._bits:
            self._bytes.append(self._current << (8 - self._bits))
            self._current = 0
            self._bits = 0
        return bytes(self._bytes)


class _BitReader:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._byte_index = 0
        self._bit_index = 0
        self.exhausted = False

    def read(self) -> int:
        if self._byte_index >= len(self._payload):
            self.exhausted = True
            return 0
        value = (self._payload[self._byte_index] >> (7 - self._bit_index)) & 1
        self._bit_index += 1
        if self._bit_index == 8:
            self._bit_index = 0
            self._byte_index += 1
        return value


def _check_symbol(cdf: Sequence[int], symbol: int) -> None:
    if symbol < 0 or symbol + 1 >= len(cdf):
        raise ValueError(f"symbol {symbol} is outside CDF alphabet")


class RangeEncoder:
    """Forward arithmetic coder using an integer CDF per symbol."""

    def __init__(self, *, validate_cdfs: bool = True) -> None:
        self.low = 0
        self.high = FULL - 1
        self.pending = 0
        self.validate_cdfs = validate_cdfs
        self._writer = _BitWriter()

    def _emit(self, bit: int) -> None:
        self._writer.write(bit)
        while self.pending:
            self._writer.write(1 - bit)
            self.pending -= 1

    def encode_symbol(self, cdf: Sequence[int], symbol: int) -> None:
        if self.validate_cdfs:
            validate_cdf(cdf)
        _check_symbol(cdf, symbol)
        total = cdf[-1]
        interval = self.high - self.low + 1
        self.high = self.low + (interval * cdf[symbol + 1]) // total - 1
        self.low = self.low + (interval * cdf[symbol]) // total
        if self.high < self.low:
            raise ArithmeticError("range interval collapsed")

        while True:
            if self.high < HALF:
                self._emit(0)
            elif self.low >= HALF:
                self._emit(1)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= FIRST_QUARTER and self.high < THIRD_QUARTER:
                self.pending += 1
                self.low -= FIRST_QUARTER
                self.high -= FIRST_QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1

    def finish(self) -> bytes:
        self.pending += 1
        if self.low < FIRST_QUARTER:
            self._emit(0)
        else:
            self._emit(1)
        return self._writer.finish()


class RangeDecoder:
    """Mirror of :class:`RangeEncoder` for a known symbol count."""

    def __init__(self, payload: bytes, *, validate_cdfs: bool = True) -> None:
        self.low = 0
        self.high = FULL - 1
        self.validate_cdfs = validate_cdfs
        self._reader = _BitReader(payload)
        self.code = 0
        for _ in range(STATE_BITS):
            self.code = (self.code << 1) | self._reader.read()

    @property
    def exhausted(self) -> bool:
        return self._reader.exhausted

    def decode_symbol(self, cdf: Sequence[int]) -> int:
        if self.validate_cdfs:
            validate_cdf(cdf)
        total = cdf[-1]
        interval = self.high - self.low + 1
        scaled = ((self.code - self.low + 1) * total - 1) // interval
        symbol = bisect_right(cdf, scaled) - 1
        if symbol < 0 or symbol + 1 >= len(cdf):
            raise ValueError("decoded symbol is outside CDF alphabet")

        self.high = self.low + (interval * cdf[symbol + 1]) // total - 1
        self.low = self.low + (interval * cdf[symbol]) // total
        if self.high < self.low:
            raise ArithmeticError("range interval collapsed during decode")

        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.low -= HALF
                self.high -= HALF
                self.code -= HALF
            elif self.low >= FIRST_QUARTER and self.high < THIRD_QUARTER:
                self.low -= FIRST_QUARTER
                self.high -= FIRST_QUARTER
                self.code -= FIRST_QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1
            self.code = (self.code << 1) | self._reader.read()
        return symbol
