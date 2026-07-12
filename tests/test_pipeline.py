# =============================================================================
# tests/test_pipeline.py — Anomaly Detection and Alert Agent
# =============================================================================
# Phase 5: VALIDATE
#
# Run from the project root with (venv) active:
#   pytest tests/test_pipeline.py -v
#
# Nine test classes covering the full detection surface:
#   1. Data loading            — close and history load and validate
#   2. safe_variance            — the near-zero denominator guard
#   3. is_material              — the three-part materiality rule
#   4. thresholds_for           — per-account thresholds
#   5. volatility_context       — the modified z-score, robust to outliers
#   6. detect_anomalies         — integration across all accounts
#   7. JSON parsing             — structured triage extraction
#   8. name normalisation       — tolerant account matching
#   9. output writing           — dual hashes, audit, disposition CSV
#
# No real API calls. The Claude call is mocked so the suite runs fast and
# costs nothing.
# =============================================================================

import json
import pytest
import pandas as pd

from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.step1_data_loader import load_close, load_history
from src.step2_detection_engine import (
    thresholds_for, safe_variance, is_material,
    volatility_context, detect_anomalies,
)
from src.step3_triage_agent import parse_triage_json, strip_json_block
from src.step4_output_writer import (
    write_output, export_anomalies_csv,
    _normalise_account, _build_triage_lookup,
)
from config import CLOSE_FILE, HISTORY_FILE


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def close_df():
    return load_close(CLOSE_FILE)


@pytest.fixture
def history():
    return load_history(HISTORY_FILE)


@pytest.fixture
def detection(close_df, history):
    return detect_anomalies(close_df, history)


@pytest.fixture
def flagged(detection):
    return detection[0]


@pytest.fixture
def detection_flags(detection):
    return detection[1]


@pytest.fixture
def tmp_dirs(tmp_path):
    out_dir = tmp_path / "output"
    audit   = out_dir / "audit_log.jsonl"
    out_dir.mkdir()
    with patch("src.step4_output_writer.OUTPUT_DIR", out_dir), \
         patch("src.step4_output_writer.AUDIT_LOG",  audit):
        yield out_dir, audit


# =============================================================================
# CLASS 1: Data loading
# =============================================================================

class TestDataLoading:

    def test_close_account_count(self, close_df):
        assert len(close_df) == 7

    def test_close_no_duplicates(self, close_df):
        assert not close_df["account"].duplicated().any()

    def test_close_required_columns(self, close_df):
        for col in ("account", "actual", "budget", "prior", "forecast"):
            assert col in close_df.columns

    def test_history_is_dict(self, history):
        assert isinstance(history, dict)

    def test_history_ordered_oldest_first(self, history):
        # Revenue history starts 2025-06 (1,150,000) and ends 2026-05 (1,320,000)
        assert history["Revenue"][0]  == 1150000
        assert history["Revenue"][-1] == 1320000

    def test_history_twelve_months(self, history):
        assert len(history["Revenue"]) == 12

    def test_prior_reconciles_to_history(self, close_df, history):
        # The close 'prior' must equal the last history month for every account
        for _, row in close_df.iterrows():
            assert row["prior"] == history[row["account"]][-1]


# =============================================================================
# CLASS 2: safe_variance — the denominator guard
# =============================================================================

class TestSafeVariance:

    def test_normal_dollar(self):
        dollar, pct, base = safe_variance(1290000, 1300000)
        assert dollar == -10000

    def test_normal_pct(self):
        dollar, pct, base = safe_variance(1290000, 1300000)
        assert abs(pct - (-10000 / 1300000)) < 1e-9

    def test_zero_benchmark_no_crash(self):
        dollar, pct, base = safe_variance(8000, 0)
        assert pct is None
        assert base == "near_zero_base"

    def test_near_zero_benchmark(self):
        dollar, pct, base = safe_variance(8000, 500)
        assert pct is None

    def test_above_floor_computes_pct(self):
        dollar, pct, base = safe_variance(8000, 5000)
        assert pct is not None
        assert base == "ok"

    def test_dollar_always_computed(self):
        dollar, pct, base = safe_variance(8000, 0)
        assert dollar == 8000


# =============================================================================
# CLASS 3: is_material — the three-part rule
# =============================================================================

class TestIsMaterial:

    def test_big_dollar_low_pct(self):
        # 80k move on Revenue at 2% — flags via big_dollar (75k threshold)
        material, reason = is_material("Revenue", 80000, 0.02)
        assert material and reason == "big_dollar"

    def test_pct_and_floor(self):
        # Travel: 19k at 158% — clears default floor and band
        material, reason = is_material("Travel & Ent", 19000, 1.58)
        assert material and reason == "pct+floor"

    def test_tiny_dollar_high_pct_not_material(self):
        # 2k at 300% — the floor kills it (tiny base noise)
        material, reason = is_material("Legal", 2000, 3.0)
        assert not material

    def test_both_rules(self):
        material, reason = is_material("Marketing Spend", 60000, 0.30)
        assert material and "big_dollar" in reason

    def test_near_zero_big_dollar_passes(self):
        # pct is None (near-zero base), big dollar clears
        material, reason = is_material("Legal", 60000, None)
        assert material and reason == "big_dollar"

    def test_near_zero_big_dollar_fails(self):
        material, reason = is_material("Legal", 2000, None)
        assert not material

    def test_always_returns_two_tuple(self):
        result = is_material("Anything", 100, 0.01)
        assert isinstance(result, tuple) and len(result) == 2


# =============================================================================
# CLASS 4: thresholds_for — per-account
# =============================================================================

class TestThresholds:

    def test_revenue_tight_band(self):
        assert thresholds_for("Revenue")["pct_band"] == 0.05

    def test_personnel_tight_band(self):
        assert thresholds_for("Personnel Cost")["pct_band"] == 0.05

    def test_unknown_account_uses_default(self):
        assert thresholds_for("Office Supplies")["pct_band"] == 0.10


# =============================================================================
# CLASS 5: volatility_context — the modified z-score
# =============================================================================

class TestVolatilityContext:

    def test_stable_account_moved(self):
        label, mz = volatility_context(51000, [45000] * 12)
        assert label == "stable_account_moved"

    def test_stable_account_no_move(self):
        label, mz = volatility_context(45000, [45000] * 12)
        assert label == "stable_no_move"

    def test_genuine_outlier(self):
        history = [100, 110, 105, 95, 100, 102, 98, 101, 99, 103, 100, 97]
        label, mz = volatility_context(5000000, history)
        assert label == "unusual_for_account"

    def test_median_robust_to_history_outlier(self):
        # History has genuine spread plus one extreme outlier (9999).
        # The median stays ~100 while a mean would be dragged to ~924.
        # So a value of 100 is correctly seen as normal.
        history = [95, 98, 102, 100, 101, 99, 103, 97, 100, 102, 98, 9999]
        label, mz = volatility_context(100, history)
        assert label == "within_normal_range"

    def test_normal_variation(self):
        history = [100, 110, 105, 95, 100, 102, 98, 101, 99, 103, 100, 97]
        label, mz = volatility_context(101, history)
        assert label == "within_normal_range"


# =============================================================================
# CLASS 6: detect_anomalies — integration
# =============================================================================

class TestDetectAnomalies:

    def test_three_accounts_flagged(self, flagged):
        assert len(flagged) == 3

    def test_marketing_flagged(self, flagged):
        assert "Marketing Spend" in [f["account"] for f in flagged]

    def test_it_flagged(self, flagged):
        assert "IT Infrastructure" in [f["account"] for f in flagged]

    def test_travel_flagged(self, flagged):
        assert "Travel & Ent" in [f["account"] for f in flagged]

    def test_revenue_not_flagged(self, flagged):
        assert "Revenue" not in [f["account"] for f in flagged]

    def test_cogs_not_flagged(self, flagged):
        assert "COGS" not in [f["account"] for f in flagged]

    def test_it_flagged_via_stable_moved(self, flagged):
        it = next(f for f in flagged if f["account"] == "IT Infrastructure")
        assert it["stable_moved"] is True
        assert it["material_benchmarks"] == []

    def test_each_flagged_has_three_comparisons(self, flagged):
        for f in flagged:
            assert len(f["comparisons"]) == 3

    def test_no_missing_history_flags(self, detection_flags):
        assert len(detection_flags) == 0


# =============================================================================
# CLASS 7: JSON parsing
# =============================================================================

class TestJSONParsing:

    def test_valid_json_parses(self):
        text = 'narrative ```json\n{"triage":[{"account":"A","priority":"HIGH"}]}\n``` end'
        result = parse_triage_json(text)
        assert result is not None
        assert result[0]["account"] == "A"

    def test_missing_json_returns_none(self):
        assert parse_triage_json("just prose, no json block") is None

    def test_malformed_json_returns_none(self):
        assert parse_triage_json("```json\n{not valid}\n```") is None

    def test_strip_removes_block(self):
        text = 'keep this ```json\n{"triage":[]}\n``` and this'
        stripped = strip_json_block(text)
        assert "```json" not in stripped
        assert "keep this" in stripped
        assert "and this" in stripped


# =============================================================================
# CLASS 8: name normalisation
# =============================================================================

class TestNameNormalisation:

    def test_ampersand_matches_and(self):
        assert _normalise_account("Travel & Ent") == _normalise_account("Travel and Ent")

    def test_case_insensitive(self):
        assert _normalise_account("REVENUE") == _normalise_account("revenue")

    def test_whitespace_collapsed(self):
        assert _normalise_account("A  B") == _normalise_account("A B")

    def test_lookup_finds_reworded_name(self):
        triage_json = [{"account": "Travel and Ent", "priority": "HIGH"}]
        lookup = _build_triage_lookup(triage_json)
        found = lookup.get(_normalise_account("Travel & Ent"))
        assert found is not None
        assert found["priority"] == "HIGH"


# =============================================================================
# CLASS 9: output writing
# =============================================================================

class TestOutputWriting:

    MOCK_TRIAGE = (
        "Summary. Three accounts flagged.\n\n"
        "Marketing Spend [HIGH]\nLikely: permanent.\n"
    )
    MOCK_JSON = [
        {"account": "Marketing Spend", "priority": "HIGH", "assessment": "permanent",
         "headline": "46% over budget", "story": "off all three",
         "hypothesis": "new campaign", "confidence": "medium"},
        {"account": "IT Infrastructure", "priority": "MEDIUM", "assessment": "permanent",
         "headline": "EUR 6k on a fixed line", "story": "never moves",
         "hypothesis": "contract change", "confidence": "high"},
        {"account": "Travel & Ent", "priority": "HIGH", "assessment": "timing",
         "headline": "94% over prior", "story": "small base",
         "hypothesis": "conference, likely reverses", "confidence": "medium"},
    ]

    def test_text_file_created(self, flagged, detection_flags, tmp_dirs):
        out_dir, audit_log = tmp_dirs
        path, _ = write_output(self.MOCK_TRIAGE, flagged, detection_flags,
                               1500, 800, "end_turn")
        assert path.exists()
        assert path.stat().st_size > 100

    def test_audit_has_dual_hashes(self, flagged, detection_flags, tmp_dirs):
        out_dir, audit_log = tmp_dirs
        write_output(self.MOCK_TRIAGE, flagged, detection_flags,
                     1500, 800, "end_turn")
        record = json.loads(audit_log.read_text(encoding="utf-8").strip())
        assert record["close_hash"].startswith("sha256:")
        assert record["history_hash"].startswith("sha256:")
        assert record["close_hash"] != record["history_hash"]

    def test_audit_flagged_accounts(self, flagged, detection_flags, tmp_dirs):
        out_dir, audit_log = tmp_dirs
        write_output(self.MOCK_TRIAGE, flagged, detection_flags,
                     1500, 800, "end_turn")
        record = json.loads(audit_log.read_text(encoding="utf-8").strip())
        assert record["accounts_flagged"] == 3
        assert "Marketing Spend" in record["flagged_accounts"]

    def test_flagged_requires_review(self, flagged, detection_flags, tmp_dirs):
        out_dir, audit_log = tmp_dirs
        write_output(self.MOCK_TRIAGE, flagged, detection_flags,
                     1500, 800, "end_turn")
        record = json.loads(audit_log.read_text(encoding="utf-8").strip())
        assert record["requires_review"] is True

    def test_csv_has_disposition_column(self, flagged, tmp_dirs):
        out_dir, audit_log = tmp_dirs
        csv_path = export_anomalies_csv(flagged, self.MOCK_JSON)
        df = pd.read_csv(csv_path)
        assert "disposition" in df.columns
        assert "reviewed_by" in df.columns
        # disposition is blank for the controller to fill
        assert df["disposition"].isna().all() or (df["disposition"] == "").all()

    def test_csv_row_per_flagged(self, flagged, tmp_dirs):
        out_dir, audit_log = tmp_dirs
        csv_path = export_anomalies_csv(flagged, self.MOCK_JSON)
        df = pd.read_csv(csv_path)
        assert len(df) == len(flagged)

    def test_csv_travel_gets_agent_data(self, flagged, tmp_dirs):
        # The name-normalisation fix: Travel & Ent picks up its agent triage
        out_dir, audit_log = tmp_dirs
        csv_path = export_anomalies_csv(flagged, self.MOCK_JSON)
        df = pd.read_csv(csv_path)
        travel = df[df["account"] == "Travel & Ent"].iloc[0]
        assert travel["agent_priority"] == "HIGH"
        assert str(travel["agent_assessment"]) == "timing"
