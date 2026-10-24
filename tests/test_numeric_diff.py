import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from truth_auditor.numeric_diff import (  # noqa: E402
    Truth,
    audit,
    cross_verify,
    extract_claims,
    main,
    parse_number,
    parse_truths,
    read_report,
    tokenize_label,
)

REPORT = """\
# FIT1043 A2 Report
Generated 2026-10-24 23:55 (v1.2.3)

After cleaning, the dataset contains 127 rows across 12 columns.
Model accuracy: 0.85
Missing rate is 3.4% of all records.

| metric      | value |
|-------------|-------|
| mean_salary | 72,450.5 |
"""

TRUTH = """\
rows_after_cleaning=118
columns: 12
accuracy=0.8512
missing_rate=0.0341
mean_salary\t72450.5
"""


# ----------------------------------------------------------------- parsing


@pytest.mark.parametrize(
    "raw,expected",
    [("1,234.5", 1234.5), ("-3e-2", -0.03), ("85.1%", 85.1), ("\u22127", -7.0), (".5", 0.5), ("abc", None)],
)
def test_parse_number(raw, expected):
    assert parse_number(raw) == expected


def test_tokenize_label_normalises_case_plurals_and_stopwords():
    assert tokenize_label("rows_after_cleaning") == {"row", "after", "cleaning"}
    assert tokenize_label("meanSalary") == {"mean", "salary"}
    assert tokenize_label("the number of Classes") == {"class"}


def test_noise_tokens_are_not_claims():
    vals = [c.value for c in extract_claims(REPORT)]
    for noise in (2026, 10, 24, 23, 55, 1043, 2, 1.2):
        assert noise not in vals
    assert {127, 12, 0.85, 3.4, 72450.5} <= set(vals)


def test_claim_metadata():
    claims = {c.raw: c for c in extract_claims(REPORT)}
    acc = claims["0.85"]
    assert acc.decimals == 2 and acc.tolerance() == pytest.approx(0.005)
    assert "accuracy" in acc.label and acc.line_no == 5
    assert claims["3.4%"].percent
    assert "row" in claims["127"].label  # word after the number is captured
    assert "salary" in claims["72,450.5"].label  # table key propagates


def test_parse_truths_kv_tab_and_wc_style():
    truths = {t.key: t.value for t in parse_truths(TRUTH + "  127 data/raw.csv\n# comment\nnoise line\n")}
    assert truths["rows_after_cleaning"] == 118
    assert truths["columns"] == 12
    assert truths["mean_salary"] == 72450.5
    assert truths["data/raw.csv"] == 127
    assert "noise line" not in truths


def test_parse_truths_json_flattens_and_skips_non_numeric():
    out = json.dumps({"model": {"accuracy": 0.9, "name": "rf", "ok": True}, "rows": 10, "nan": None})
    truths = {t.key: t.value for t in parse_truths(out)}
    assert truths == {"model.accuracy": 0.9, "rows": 10.0}


def test_parse_truths_last_write_wins():
    assert parse_truths("rows=1\nrows=2\n")[0].value == 2


# ------------------------------------------------------------ verification


def test_audit_flags_127_vs_118_and_accepts_rounding():
    rep = audit(REPORT, TRUTH)
    status = {f.key: f.status for f in rep.findings}
    assert status == {
        "rows_after_cleaning": "MISMATCH",
        "columns": "MATCH",
        "accuracy": "MATCH",
        "missing_rate": "MATCH",  # 3.4% vs 0.0341 fraction
        "mean_salary": "MATCH",
    }
    mm = rep.by_status("MISMATCH")[0]
    assert mm.claim.value == 127 and "delta=9" in mm.note
    assert not rep.ok


def test_rounding_tolerance_is_bounded_by_stated_precision():
    claims = extract_claims("accuracy: 0.85")
    assert cross_verify(claims, [Truth.of("accuracy", 0.8549)]).ok
    assert not cross_verify(claims, [Truth.of("accuracy", 0.8551)]).ok
    assert not cross_verify(extract_claims("accuracy: 0.850"), [Truth.of("accuracy", 0.8512)]).ok


def test_missing_when_no_labelled_claim():
    rep = cross_verify(extract_claims("We report 42 things."), [Truth.of("timezone_offset", 10)])
    assert rep.findings[0].status == "MISSING"


def test_restated_number_any_agreeing_tied_claim_matches():
    text = "rows: 120 (draft)\nrows: 118"
    assert cross_verify(extract_claims(text), [Truth.of("rows", 118)]).ok


def test_render_summary_line():
    out = audit(REPORT, TRUTH).render()
    assert "MISMATCH" in out and out.splitlines()[-1] == "summary: 4 match, 1 mismatch, 0 missing"


# --------------------------------------------------------------------- I/O


def test_read_report_pdf(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    fitz = pytest.importorskip("pymupdf")
    pdf = tmp_path / "r.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Total rows after cleaning: 127")
    doc.save(str(pdf))
    assert pypdf  # used by read_report
    text = read_report(pdf)
    rep = audit(text, "rows_after_cleaning=118")
    assert rep.findings[0].status == "MISMATCH"


def test_cli_exit_codes_and_json(tmp_path, capsys):
    r = tmp_path / "r.md"
    r.write_text(REPORT)
    t = tmp_path / "t.txt"
    t.write_text(TRUTH)
    assert main([str(r), "--truth", str(t)]) == 1
    assert "summary:" in capsys.readouterr().out

    assert main([str(r), "--run", "printf 'columns=12\\n'", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] and payload["findings"][0]["status"] == "MATCH"

    assert main([str(r), "--run", "exit 3"]) == 2
    assert main([str(tmp_path / "nope.md"), "--truth", str(t)]) == 2
