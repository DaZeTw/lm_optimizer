"""Logical operator enum."""

from enum import Enum


class Op(str, Enum):
    DECOMPOSE = "DECOMPOSE"
    I = "I"  # Isolate / retrieve
    TRANSFORM = "TRANSFORM"
    COMPOSE = "COMPOSE"
    UNION = "UNION"
    DIFF = "DIFF"
    RANK = "RANK"
    AGGREGATE = "AGGREGATE"
    VERIFY = "VERIFY"
