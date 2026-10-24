"""Numeric claim extraction and cross-verification.

Pipeline:
    report (Markdown / plain text / PDF)  -> extract_claims()  -> [Claim]
    script output (stdout / file)         -> parse_truths()    -> [Truth]
    cross_verify(claims, truths)          -> AuditReport (MATCH / MISMATCH / MISSING)

Matching is label-driven: each claim carries the words around the number
("Total rows after cleaning: 127" -> {total, row, after, cleaning}); each truth
carries its key ("rows_after_cleaning=118" -> {row, after, cleaning}). The best
overlapping claim is compared to the truth with rounding-aware tolerance, so a
report stating "0.85" accepts a computed 0.8512, while "127" vs 118 is flagged.

CLI:
    python -m truth_auditor.numeric_diff report.pdf --truth out.txt
    python -m truth_auditor.numeric_diff report.md --run "bash pipeline.sh"
Exit code: 0 clean, 1 mismatch/missing, 2 usage error.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

# --------------------------------------------------------------------------- #
# Lexing
# --------------------------------------------------------------------------- #

# Tokens that look numeric but are not claims: dates, clock times, versions,
# unit codes / identifiers glued to letters (FIT1043, A2, sha1, Q3).
_NOISE = re.compile(
    r"""
      \b\d{4}-\d{1,2}-\d{1,2}(?:[T ]\d{1,2}:\d{2}(?::\d{2})?)?\b   # ISO dates
    | \b\d{1,2}/\d{1,2}/\d{2,4}\b                                 # d/m/y
    | \b\d{1,2}:\d{2}(?::\d{2})?\b                                # clock
    | \bv?\d+\.\d+\.\d+(?:\.\d+)*\b                               # versions
    | \b[A-Za-z_]+\d[\w]*\b                                       # FIT1043, A2
    | \b\d+(?![eE][-+]?\d)[A-Za-z_]\w*\b                           # 3rd, 2x
    """,
    re.VERBOSE,
)

_NUMBER = re.compile(
    r"""
    (?<![\w.])
    (?P<sign>[-+\u2212])?
    (?P<body>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)
    (?P<exp>[eE][-+]?\d+)?
    (?P<pct>\s?%)?
    (?![\w])
    """,
    re.VERBOSE,
)

_WORD = re.compile(r"[A-Za-z][A-Za-z']*")
_STOP = frozenset(
    "a an the of in on at to for from by with and or is are was were be been "
    "this that these those it its we our there here as has have had which "
    "value number s".split()
)
_KV_LINE = re.compile(r"^\s*[-*>|#\s]*(?P<key>[A-Za-z][\w .()/%-]{0,80}?)\s*(?:=|:|\|)\s*(?P<val>.+?)\s*$")


def _stem(w: str) -> str:
    w = w.lower().strip("'")
    for suf in ("ies", "es", "s"):
        if len(w) > 3 and w.endswith(suf) and not w.endswith("ss"):
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def tokenize_label(text: str) -> frozenset:
    """Normalise a label: split snake/camel case, drop stopwords, crude stem."""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text).replace("_", " ")
    return frozenset(t for t in (_stem(w) for w in _WORD.findall(text)) if t and t not in _STOP)


def parse_number(raw: str) -> Optional[float]:
    """Parse a single numeric literal ('1,234.5', '-3e-2', '85.1%', '−7')."""
    m = _NUMBER.search(raw.strip())
    if not m:
        return None
    return _to_float(m)


def _to_float(m: re.Match) -> float:
    s = m.group("body").replace(",", "") + (m.group("exp") or "")
    v = float(s)
    if m.group("sign") in ("-", "\u2212"):
        v = -v
    return v


def _decimals(m: re.Match) -> int:
    body = m.group("body")
    d = len(body.split(".", 1)[1]) if "." in body else 0
    exp = m.group("exp")
    return d - int(exp[1:]) if exp else d


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Claim:
    value: float
    raw: str
    label: frozenset
    line_no: int
    context: str
    decimals: int = 0
    percent: bool = False

    def tolerance(self) -> float:
        """Half a unit in the last stated digit: '0.85' covers [0.845, 0.855]."""
        return 0.5 * 10 ** (-self.decimals)


@dataclass(frozen=True)
class Truth:
    key: str
    value: float
    label: frozenset = field(default_factory=frozenset)

    @staticmethod
    def of(key: str, value: float) -> "Truth":
        return Truth(key, float(value), tokenize_label(key))


@dataclass
class Finding:
    key: str
    expected: float
    status: str  # MATCH | MISMATCH | MISSING
    claim: Optional[Claim] = None
    score: float = 0.0
    note: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.claim is not None:
            d["claim"] = {
                "value": self.claim.value,
                "raw": self.claim.raw,
                "line": self.claim.line_no,
                "context": self.claim.context,
            }
        return d


@dataclass
class AuditReport:
    findings: List[Finding]

    @property
    def ok(self) -> bool:
        return all(f.status == "MATCH" for f in self.findings)

    def by_status(self, status: str) -> List[Finding]:
        return [f for f in self.findings if f.status == status]

    def render(self) -> str:
        lines = []
        for f in self.findings:
            where = f"L{f.claim.line_no} '{f.claim.raw}'" if f.claim else "-"
            lines.append(f"[{f.status:<8}] {f.key:<32} expected={_fmt(f.expected):<14} report={where} {f.note}".rstrip())
        c = {s: len(self.by_status(s)) for s in ("MATCH", "MISMATCH", "MISSING")}
        lines.append(f"summary: {c['MATCH']} match, {c['MISMATCH']} mismatch, {c['MISSING']} missing")
        return "\n".join(lines)


def _fmt(v: float) -> str:
    return str(int(v)) if float(v).is_integer() and abs(v) < 1e15 else f"{v:.6g}"


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #


def _mask(line: str) -> str:
    return _NOISE.sub(lambda m: " " * len(m.group(0)), line)


def extract_claims(text: str, window: int = 6) -> List[Claim]:
    """Extract every numeric claim with a label built from surrounding words.

    Label = key of a 'key: value' / table row if present, else up to `window`
    words before the number plus up to 2 words after it (captures '118 rows').
    """
    claims: List[Claim] = []
    for line_no, line in enumerate(text.splitlines(), 1):
        masked = _mask(line)
        kv = _KV_LINE.match(line)
        kv_label = tokenize_label(kv.group("key")) if kv and not _NUMBER.fullmatch(kv.group("key").strip()) else frozenset()
        for m in _NUMBER.finditer(masked):
            before = _WORD.findall(masked[: m.start()])[-window:]
            after = _WORD.findall(masked[m.end():])[:2]
            label = tokenize_label(" ".join(before + after)) | kv_label
            claims.append(
                Claim(
                    value=_to_float(m),
                    raw=line[m.start(): m.end()].strip(),
                    label=frozenset(label),
                    line_no=line_no,
                    context=line.strip()[:160],
                    decimals=_decimals(m),
                    percent=bool(m.group("pct")),
                )
            )
    return claims


def read_report(path: str | Path) -> str:
    """Read Markdown/text directly; PDFs via pypdf (pages joined with form feed)."""
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as e:  # pragma: no cover - environment dependent
            raise RuntimeError("pypdf is required for PDF reports: pip install pypdf") from e
        return "\n\f\n".join((page.extract_text() or "") for page in PdfReader(str(p)).pages)
    return p.read_text(encoding="utf-8", errors="replace")


def parse_truths(output: str) -> List[Truth]:
    """Parse deterministic script output into ground-truth values.

    Accepts JSON objects (flattened with '.'), or lines like `key=value`,
    `key: value`, `key<TAB>value`, and `wc -l`-style `127 file.csv`.
    Later duplicates overwrite earlier ones (last write wins, like a log).
    """
    stripped = output.strip()
    if stripped.startswith("{"):
        try:
            return [Truth.of(k, v) for k, v in _flatten(json.loads(stripped)).items()]
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    found: dict = {}
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^(?P<key>[A-Za-z_][\w .()/-]*?)\s*(?:=|:|\t)\s*(?P<val>\S.*)$", line)
        if m:
            v = parse_number(m.group("val"))
            if v is not None:
                found[m.group("key").strip()] = v
            continue
        m = re.match(r"^(?P<val>[-+]?[\d,]*\.?\d+)\s+(?P<key>\S.*)$", line)  # wc -l / uniq -c
        if m:
            v = parse_number(m.group("val"))
            if v is not None:
                found[m.group("key").strip()] = v
    return [Truth.of(k, v) for k, v in found.items()]


def _flatten(obj, prefix: str = "") -> dict:
    out: dict = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}{k}."))
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)) and math.isfinite(obj):
        out[prefix.rstrip(".")] = float(obj)
    return out


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #


def values_agree(claim: Claim, expected: float, rel_tol: float = 1e-9) -> bool:
    """Rounding-aware equality; percentages also match fractional truths."""
    candidates = [expected]
    if claim.percent:
        candidates.append(expected * 100.0)
    tol = claim.tolerance() + 1e-12
    return any(
        abs(claim.value - e) <= tol or math.isclose(claim.value, e, rel_tol=rel_tol)
        for e in candidates
    )


def _score(truth: Truth, claim: Claim) -> float:
    if not truth.label or not claim.label:
        return 0.0
    inter = len(truth.label & claim.label)
    return inter / len(truth.label)  # recall of the truth key's tokens


def cross_verify(
    claims: Sequence[Claim],
    truths: Iterable[Truth],
    min_score: float = 0.5,
) -> AuditReport:
    """For each truth pick the most label-similar claims and judge agreement.

    Among claims tied at the best score, any agreeing value yields MATCH
    (a report may restate a number); otherwise the top candidate is reported
    as MISMATCH. No candidate above `min_score` -> MISSING.
    """
    findings: List[Finding] = []
    for t in truths:
        scored = sorted(((_score(t, c), c) for c in claims), key=lambda x: -x[0])
        scored = [(s, c) for s, c in scored if s >= min_score]
        if not scored:
            findings.append(Finding(t.key, t.value, "MISSING", note="no labelled claim in report"))
            continue
        best = scored[0][0]
        top = [c for s, c in scored if s == best]
        hit = next((c for c in top if values_agree(c, t.value)), None)
        if hit is not None:
            findings.append(Finding(t.key, t.value, "MATCH", hit, best))
        else:
            c = top[0]
            delta = c.value - (t.value * 100 if c.percent and abs(t.value) <= 1 else t.value)
            findings.append(Finding(t.key, t.value, "MISMATCH", c, best, note=f"delta={_fmt(delta)}"))
    return AuditReport(findings)


def audit(report_text: str, script_output: str, min_score: float = 0.5) -> AuditReport:
    return cross_verify(extract_claims(report_text), parse_truths(script_output), min_score)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="numeric_diff", description=__doc__.split("\n\n")[0])
    ap.add_argument("report", help="report file (.md/.txt/.pdf)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--truth", help="file holding deterministic script output")
    src.add_argument("--run", help="shell command whose stdout is the ground truth")
    ap.add_argument("--min-score", type=float, default=0.5)
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)

    try:
        text = read_report(args.report)
        if args.run:
            proc = subprocess.run(args.run, shell=True, capture_output=True, text=True, timeout=600)
            if proc.returncode != 0:
                print(proc.stderr, file=sys.stderr)
                return 2
            output = proc.stdout
        else:
            output = Path(args.truth).read_text(encoding="utf-8")
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    report = audit(text, output, args.min_score)
    if args.json:
        print(json.dumps({"ok": report.ok, "findings": [f.to_dict() for f in report.findings]}, indent=2, default=list))
    else:
        print(report.render())
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
