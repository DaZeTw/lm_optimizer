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
