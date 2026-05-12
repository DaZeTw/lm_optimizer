"""
Recursive descent parser for algebraic plan expressions.

Grammar:
    expr     := OP_NAME '(' arglist ')'
              | STRING
    arglist  := (arg (',' arg)*)?
    arg      := KWARG | expr
    KWARG    := IDENT '=' STRING
    OP_NAME  := 'I' | 'TRANSFORM' | 'COMPOSE' | 'UNION' | 'DIFF'
              | 'RANK' | 'AGGREGATE' | 'VERIFY' | 'DECOMPOSE'
    STRING   := '"..."' | "'...'"

Examples of valid input:
    I("what method is proposed?")
    AGGREGATE(I("main contribution"), goal="summarize")
    VERIFY(AGGREGATE(UNION(I("q1"), I("q2")), goal="answer"), constraints="grounded")
"""

from __future__ import annotations

import re
from typing import Any

from ir.nodes import LogicalNode
from ir.ops import Op

# ── Token types ────────────────────────────────────────────────────

_OP_NAMES = {op.value for op in Op}

_TOKEN_RE = re.compile(
    r"\s*(?:"
    r'(?P<STRING>"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\')'  # "..." or '...'
    r"|(?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)"  # identifiers / op names
    r"|(?P<LPAREN>\()"
    r"|(?P<RPAREN>\))"
    r"|(?P<COMMA>,)"
    r"|(?P<EQ>=)"
    r")\s*"
)


class ParseError(Exception):
    pass


# ── Tokenizer ──────────────────────────────────────────────────────


def _tokenize(text: str) -> list[tuple[str, str]]:
    """Return list of (token_type, token_value) pairs."""
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(text):
        if text[pos] in (" ", "\t", "\n", "\r"):
            pos += 1
            continue
        m = _TOKEN_RE.match(text, pos)
        if not m:
            raise ParseError(f"Unexpected character {text[pos]!r} at position {pos}")
        kind = m.lastgroup
        value = m.group(kind)
        tokens.append((kind, value))
        pos = m.end()
    return tokens


def _unquote(s: str) -> str:
    """Strip surrounding quotes and unescape common escape sequences."""
    inner = s[1:-1]
    return inner.replace('\\"', '"').replace("\\'", "'").replace("\\\\", "\\")


# ── Recursive descent parser ───────────────────────────────────────


class _Parser:
    def __init__(self, tokens: list[tuple[str, str]]):
        self.tokens = tokens
        self.pos = 0

    def _peek(self) -> tuple[str, str] | None:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def _consume(self, expected_kind: str | None = None) -> tuple[str, str]:
        tok = self._peek()
        if tok is None:
            raise ParseError("Unexpected end of input")
        if expected_kind and tok[0] != expected_kind:
            raise ParseError(
                f"Expected {expected_kind}, got {tok[0]}={tok[1]!r} "
                f"at token {self.pos}"
            )
        self.pos += 1
        return tok

    # ── top-level ──────────────────────────────────────────────────

    def parse_expr(self) -> LogicalNode:
        tok = self._peek()
        if tok is None:
            raise ParseError("Empty input — expected an operator call")

        kind, value = tok

        if kind != "IDENT":
            raise ParseError(f"Expected operator name, got {kind}={value!r}")
        if value not in _OP_NAMES:
            raise ParseError(
                f"Unknown operator {value!r}. Valid operators: {sorted(_OP_NAMES)}"
            )

        self._consume("IDENT")  # consume op name
        self._consume("LPAREN")  # consume '('
        positional, kwargs = self._parse_arglist()
        self._consume("RPAREN")  # consume ')'

        return self._build_node(value, positional, kwargs)

    # ── arglist ────────────────────────────────────────────────────

    def _parse_arglist(self) -> tuple[list[LogicalNode], dict[str, str]]:
        """Parse comma-separated args. Returns (positional_nodes, kwargs)."""
        positional: list[LogicalNode] = []
        kwargs: dict[str, str] = {}

        tok = self._peek()
        if tok is None or tok[0] == "RPAREN":
            return positional, kwargs

        while True:
            self._parse_one_arg(positional, kwargs)
            tok = self._peek()
            if tok is None or tok[0] == "RPAREN":
                break
            if tok[0] == "COMMA":
                self._consume("COMMA")
            else:
                raise ParseError(f"Expected ',' or ')' but got {tok!r}")

        return positional, kwargs

    def _parse_one_arg(
        self,
        positional: list[LogicalNode],
        kwargs: dict[str, str],
    ) -> None:
        """Parse a single argument — either kwarg (key=value) or a sub-expression."""
        tok = self._peek()
        if tok is None:
            raise ParseError("Unexpected end of input in argument list")

        # Look ahead: IDENT followed by '=' → keyword argument
        if (
            tok[0] == "IDENT"
            and tok[1] not in _OP_NAMES
            and self.pos + 1 < len(self.tokens)
            and self.tokens[self.pos + 1][0] == "EQ"
        ):
            key_tok = self._consume("IDENT")
            self._consume("EQ")
            val_tok = self._consume("STRING")
            kwargs[key_tok[1]] = _unquote(val_tok[1])
            return

        # Otherwise it must be a sub-expression (another operator call or string)
        if tok[0] == "STRING":
            # bare string used as a shorthand for I("...")
            # (only valid for I — enforced in _build_node)
            self._consume("STRING")
            positional.append(LogicalNode.isolate(_unquote(tok[1])))
            return

        # Recurse
        node = self.parse_expr()
        positional.append(node)

    # ── node builder ───────────────────────────────────────────────

    def _build_node(
        self,
        op_name: str,
        positional: list[LogicalNode],
        kwargs: dict[str, str],
    ) -> LogicalNode:
        op = Op(op_name)

        if op == Op.I:
            # I("query text")  — exactly one positional string arg
            if len(positional) != 1:
                raise ParseError(
                    f"I() expects exactly 1 positional argument, "
                    f"got {len(positional)}"
                )
            # The positional arg was already constructed as an isolate node
            # by the bare-string branch above; unwrap its query param.
            raw_node = positional[0]
            if raw_node.op != Op.I:
                raise ParseError("I() argument must be a quoted string")
            return raw_node  # already correctly built

        if op == Op.DECOMPOSE:
            if len(positional) != 1 or positional[0].op != Op.I:
                raise ParseError("DECOMPOSE() expects exactly 1 string argument")
            query = positional[0].params["query"]
            return LogicalNode.decompose(query)

        if op == Op.TRANSFORM:
            if len(positional) != 1:
                raise ParseError("TRANSFORM() expects exactly 1 child node")
            return LogicalNode.transform(positional[0], schema=kwargs.get("schema", ""))

        if op == Op.RANK:
            if len(positional) != 1:
                raise ParseError("RANK() expects exactly 1 child node")
            return LogicalNode.rank(
                positional[0], criterion=kwargs.get("criterion", "")
            )

        if op == Op.AGGREGATE:
            if len(positional) != 1:
                raise ParseError("AGGREGATE() expects exactly 1 child node")
            return LogicalNode.aggregate(positional[0], goal=kwargs.get("goal", ""))

        if op == Op.VERIFY:
            if len(positional) != 1:
                raise ParseError("VERIFY() expects exactly 1 child node")
            return LogicalNode.verify(
                positional[0], constraints=kwargs.get("constraints", "")
            )

        if op == Op.COMPOSE:
            if len(positional) != 2:
                raise ParseError("COMPOSE() expects exactly 2 child nodes")
            return LogicalNode.compose(
                positional[0], positional[1], condition=kwargs.get("condition", "")
            )

        if op == Op.DIFF:
            if len(positional) != 2:
                raise ParseError("DIFF() expects exactly 2 child nodes")
            return LogicalNode.diff(positional[0], positional[1])

        if op == Op.UNION:
            if len(positional) < 2:
                raise ParseError("UNION() expects at least 2 child nodes")
            return LogicalNode.union(*positional)

        raise ParseError(f"Unhandled operator: {op_name}")  # should never reach here


# ── Public API ─────────────────────────────────────────────────────


def parse_task_strategy(text: str) -> dict:
    """
    Parse the LLM's three-section Task Strategy Template into a plain dict.

    Expected input format (produced by task_prompts.TASK_SYSTEM_PROMPT)::

        LOGICAL SKELETON
        VERIFY(
          AGGREGATE(
            RANK(I({QUERY}), criterion="{RANK_CRITERION}"),
            goal="{AGGREGATION_GOAL}"
          ),
          constraints="{VERIFY_CONSTRAINTS}"
        )

        PHYSICAL POLICY
        I_1        | I        | HybridRetrieve   | {}
        RANK_1     | RANK     | CrossEncoderRank  | top_k=5

        ADAPTATION POLICY
        mutable_slots: QUERY, RANK_CRITERION
        immutable_slots: VERIFY_CONSTRAINTS
        mutable_ops: RANK_1
        immutable_ops: VERIFY_1
        allowed_rewrites: may insert TRANSFORM if chunks are verbose
        forbidden_rewrites: must not drop VERIFY

    Returns a plain dict::

        {
            "logical_skeleton": {
                "template": "<raw algebraic expression text with {SLOT} holes>",
                "slots": ["QUERY", "RANK_CRITERION", ...],
            },
            "physical_policy": {
                # keyed by op_id
                "I_1":    {"op_name": "I",    "variant": "HybridRetrieve",  "params": {}},
                "RANK_1": {"op_name": "RANK", "variant": "CrossEncoderRank","params": {"top_k": "5"}},
                ...
            },
            "adaptation_policy": {
                "mutable_slots":     ["QUERY", "RANK_CRITERION"],
                "immutable_slots":   ["VERIFY_CONSTRAINTS"],
                "mutable_ops":       ["RANK_1"],
                "immutable_ops":     ["VERIFY_1"],
                "allowed_rewrites":  ["may insert TRANSFORM if chunks are verbose"],
                "forbidden_rewrites":["must not drop VERIFY"],
            },
        }

    Raises:
        ParseError: If any required section is missing or a line is malformed.
    """
    # ── 1. Split into the three named sections ──────────────────────
    _SECTION_HEADERS = ("LOGICAL SKELETON", "PHYSICAL POLICY", "ADAPTATION POLICY")

    # Strip markdown fences the LLM might have added
    text = (
        re.sub(r"```[a-z]*", "", text, flags=re.IGNORECASE).strip().strip("`").strip()
    )

    # Find header positions
    positions: dict[str, int] = {}
    for header in _SECTION_HEADERS:
        idx = text.find(header)
        if idx == -1:
            raise ParseError(f"Missing required section header: {header!r}")
        positions[header] = idx

    # Extract raw section bodies (text between each header and the next)
    ordered = sorted(positions.items(), key=lambda kv: kv[1])
    section_bodies: dict[str, str] = {}
    for i, (header, start) in enumerate(ordered):
        body_start = start + len(header)
        body_end = ordered[i + 1][1] if i + 1 < len(ordered) else len(text)
        section_bodies[header] = text[body_start:body_end].strip()

    # ── 2. Parse LOGICAL SKELETON ───────────────────────────────────
    skeleton_text = section_bodies["LOGICAL SKELETON"]

    # Extract {SLOT_NAME} placeholders preserving order, deduplicating
    slot_pattern = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
    seen: set[str] = set()
    slots: list[str] = []
    for m in slot_pattern.finditer(skeleton_text):
        name = m.group(1)
        if name not in seen:
            slots.append(name)
            seen.add(name)

    logical_skeleton: dict = {
        "template": skeleton_text,
        "slots": slots,
    }

    # ── 3. Parse PHYSICAL POLICY ────────────────────────────────────
    # Each non-blank line: op_id | op_name | variant | params
    physical_policy: dict[str, dict] = {}
    for line in section_bodies["PHYSICAL POLICY"].splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 4:
            raise ParseError(
                f"PHYSICAL POLICY line must have exactly 4 pipe-separated fields, "
                f"got {len(parts)} in: {line!r}"
            )
        op_id, op_name, variant, raw_params = parts

        # Parse params: "{}" → empty dict; "key=val, key2=val2" → dict
        params: dict[str, str] = {}
        raw_params = raw_params.strip()
        if raw_params and raw_params != "{}":
            for pair in raw_params.split(","):
                pair = pair.strip()
                if "=" not in pair:
                    raise ParseError(
                        f"PHYSICAL POLICY param {pair!r} is not in key=value format "
                        f"in line: {line!r}"
                    )
                k, v = pair.split("=", 1)
                params[k.strip()] = v.strip()

        physical_policy[op_id] = {
            "op_name": op_name,
            "variant": variant,
            "params": params,
        }

    if not physical_policy:
        raise ParseError("PHYSICAL POLICY section is empty")

    # ── 4. Parse ADAPTATION POLICY ──────────────────────────────────
    # Keys with a single list value:   mutable_slots, immutable_slots,
    #                                  mutable_ops,   immutable_ops
    # Keys that accumulate per line:   allowed_rewrites, forbidden_rewrites
    _MULTI_VALUE_KEYS = {
        "mutable_slots",
        "immutable_slots",
        "mutable_ops",
        "immutable_ops",
    }
    _ACCUMULATE_KEYS = {"allowed_rewrites", "forbidden_rewrites"}

    adaptation_policy: dict[str, list[str]] = {
        "mutable_slots": [],
        "immutable_slots": [],
        "mutable_ops": [],
        "immutable_ops": [],
        "allowed_rewrites": [],
        "forbidden_rewrites": [],
    }

    for line in section_bodies["ADAPTATION POLICY"].splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, raw_value = line.partition(":")
        key = key.strip().lower().replace(" ", "_")
        value = raw_value.strip()

        if key in _MULTI_VALUE_KEYS:
            # Comma-separated list; filter out empty strings
            adaptation_policy[key] = [v.strip() for v in value.split(",") if v.strip()]
        elif key in _ACCUMULATE_KEYS:
            if value:
                adaptation_policy[key].append(value)
        # Unknown keys are silently ignored so forward-compatible TSTs don't break

    return {
        "logical_skeleton": logical_skeleton,
        "physical_policy": physical_policy,
        "adaptation_policy": adaptation_policy,
    }


def parse_expression(text: str) -> LogicalNode:
    """
    Parse an algebraic plan expression into a LogicalNode DAG.

    Args:
        text: An algebraic expression such as:
              'AGGREGATE(I("what method?"), goal="summarize")'

    Returns:
        The root LogicalNode of the plan DAG.

    Raises:
        ParseError: If the expression is syntactically or structurally invalid.
    """
    # Strip markdown fences if the LLM wrapped its output
    text = re.sub(r"```[a-z]*", "", text).strip().strip("`").strip()

    if not text:
        raise ParseError("Empty plan expression")

    tokens = _tokenize(text)
    parser = _Parser(tokens)
    node = parser.parse_expr()

    # Ensure the entire input was consumed
    remaining = parser._peek()
    if remaining is not None:
        raise ParseError(f"Unexpected trailing tokens starting with {remaining!r}")

    return node
