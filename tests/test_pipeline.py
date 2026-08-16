# =============================================================================
# tests/test_pipeline.py — Anomaly Detection and Alert Agent
# =============================================================================
# Phase 6: VALIDATE
#
# Run from the project root with (venv) active:
#   pytest tests/test_pipeline.py -v
#
# Seventeen test classes covering the full detection surface:
#   1. Data loading             — close and history load and validate
#   2. safe_variance            — the near-zero denominator guard
#   3. is_material              — the three-part materiality rule
#   4. thresholds_for           — per-account thresholds
#   5. volatility_context       — the modified z-score, robust to outliers
#   6. detect_anomalies         — integration across all accounts
#   6b. threshold overrides     — prove parameters flow through
#   7. JSON parsing             — structured triage extraction
#   8. name normalisation       — tolerant account matching
#   9. output writing           — dual hashes, audit, disposition CSV
#  10. web data load            — data contract for the web layer
#  11. sensitivity mapping      — slider-to-threshold mapping
#  12. sensitivity + detection  — integration at each sensitivity level
#  13. triage helpers           — fingerprint, matching, sorting, escaping
#  14. triage cost              — cost display formatting
#  15. in-memory builders       — PDF bytes, CSV bytes, effective thresholds
#  16. governance and example   — fabrication guards, session audit, example
#  17. currency wiring          — CURRENCY_SYMBOL/CODE drive PDF and prompt
#  18. two-mode redesign        — mode mapping, isolation, example, stale-clear
#  19. volatility-only account  — Utilities flags on z-score, not materiality
#
# No real API calls. The Claude call is mocked so the suite runs fast and
# costs nothing.
# =============================================================================

import io
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
from src.step3_triage_agent import parse_triage_json, strip_json_block, build_prompt
from src.step4_output_writer import (
    write_output, export_anomalies_csv,
    build_pdf_bytes, build_csv_bytes, file_sha256,
    _normalise_account, _build_triage_lookup,
)
from streamlit_app.lib.sensitivity import (
    LEVELS, BASELINE_INDEX, MATERIALITY_FACTORS, Z_CUTOFFS,
    scale_threshold, effective_thresholds, patch_thresholds,
)
from streamlit_app.lib.triage_helpers import (
    triage_fingerprint, match_triage_to_flagged,
    sort_by_priority, escape_text,
)
from streamlit_app.lib.cost import format_cost
from streamlit_app.lib.theme import priority_badge_html
from config import (
    CLOSE_FILE, HISTORY_FILE,
    GLOBAL_THRESHOLD, ACCOUNT_THRESHOLDS,
    MODIFIED_Z_CUTOFF, NEAR_ZERO,
)


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
        assert len(close_df) == 8

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
# CLASS 6b: threshold overrides — prove the parameters flow through
# =============================================================================

class TestThresholdOverrides:
    """The default-path tests (class 6) would still pass if a helper
    accidentally read a module global instead of its parameter, because
    the default value equals the global. These tests pass deliberately
    non-default thresholds and assert the flagged set changes, proving
    the parameters actually take effect."""

    def test_high_z_cutoff_changes_volatility_label(self, close_df, history):
        flagged, _ = detect_anomalies(close_df, history, z_cutoff=100)
        travel = next((f for f in flagged if f["account"] == "Travel & Ent"), None)
        if travel:
            assert travel["volatility"] != "unusual_for_account"

    def test_high_z_cutoff_does_not_deflag_material_accounts(self, close_df, history):
        flagged, _ = detect_anomalies(close_df, history, z_cutoff=100)
        names = [f["account"] for f in flagged]
        assert "Marketing Spend" in names
        assert "Travel & Ent" in names

    def test_fully_loose_thresholds_leave_only_stable_moved(
            self, close_df, history):
        loose = {"pct_band": 0.90, "min_dollar_floor": 999999, "big_dollar": 999999}
        flagged, _ = detect_anomalies(
            close_df, history,
            account_thresholds={}, global_threshold=loose, z_cutoff=100,
        )
        names = [f["account"] for f in flagged]
        assert names == ["IT Infrastructure"]
        assert flagged[0]["stable_moved"] is True
        assert flagged[0]["material_benchmarks"] == []

    def test_loose_materiality_shrinks_material_benchmarks(self, close_df, history):
        loose = {"pct_band": 0.90, "min_dollar_floor": 999999, "big_dollar": 999999}
        flagged, _ = detect_anomalies(
            close_df, history,
            account_thresholds={}, global_threshold=loose,
        )
        for f in flagged:
            assert f["material_benchmarks"] == []

    def test_loose_materiality_fewer_flags_than_default(self, close_df, history):
        default_flagged, _ = detect_anomalies(close_df, history)
        loose = {"pct_band": 0.90, "min_dollar_floor": 999999, "big_dollar": 999999}
        loose_flagged, _ = detect_anomalies(
            close_df, history,
            account_thresholds={}, global_threshold=loose, z_cutoff=100,
        )
        assert len(loose_flagged) < len(default_flagged)

    def test_tight_thresholds_flag_all_accounts(self, close_df, history):
        tight = {"pct_band": 0.001, "min_dollar_floor": 1, "big_dollar": 1}
        tight_flagged, _ = detect_anomalies(
            close_df, history,
            account_thresholds={}, global_threshold=tight, z_cutoff=0.1,
        )
        assert len(tight_flagged) == len(close_df)

    def test_near_zero_override_changes_pct_computation(self, close_df, history):
        high_nz_flagged, _ = detect_anomalies(close_df, history, near_zero=2000000)
        for f in high_nz_flagged:
            for bench in f["comparisons"]:
                assert f["comparisons"][bench]["pct"] is None
                assert f["comparisons"][bench]["base"] == "near_zero_base"


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


# =============================================================================
# CLASS 10: web data load contract
# =============================================================================

class TestWebDataLoad:
    """Guard the data contract the web layer builds on: the sample files
    load into the shapes detect_anomalies expects."""

    def test_close_has_eight_accounts(self, close_df):
        assert len(close_df) == 8

    def test_close_has_required_columns(self, close_df):
        for col in ("account", "actual", "budget", "prior", "forecast"):
            assert col in close_df.columns

    def test_history_has_all_close_accounts(self, close_df, history):
        for account in close_df["account"]:
            assert account in history

    def test_history_has_twelve_months_each(self, history):
        for account, values in history.items():
            assert len(values) == 12

    def test_history_total_rows(self, history):
        total = sum(len(v) for v in history.values())
        assert total == 96


# =============================================================================
# CLASS 11: sensitivity — slider-to-threshold mapping
# =============================================================================

class TestSensitivityMapping:
    """Prove that the sensitivity module maps slider levels to the correct
    threshold values and that the scaling preserves the per-account structure."""

    def test_medium_equals_config_defaults(self):
        at, gt, zc, nz = effective_thresholds("Medium", "Medium")
        assert gt == GLOBAL_THRESHOLD
        assert zc == MODIFIED_Z_CUTOFF
        assert nz == NEAR_ZERO
        for account in ACCOUNT_THRESHOLDS:
            assert at[account] == ACCOUNT_THRESHOLDS[account]

    def test_high_materiality_halves_thresholds(self):
        at, gt, zc, nz = effective_thresholds("High", "Medium")
        assert gt["pct_band"] == pytest.approx(GLOBAL_THRESHOLD["pct_band"] * 0.5)
        assert gt["min_dollar_floor"] == pytest.approx(
            GLOBAL_THRESHOLD["min_dollar_floor"] * 0.5)
        assert gt["big_dollar"] == pytest.approx(
            GLOBAL_THRESHOLD["big_dollar"] * 0.5)

    def test_low_materiality_doubles_thresholds(self):
        at, gt, zc, nz = effective_thresholds("Low", "Medium")
        assert gt["pct_band"] == pytest.approx(GLOBAL_THRESHOLD["pct_band"] * 2.0)

    def test_volatility_slider_sets_z_cutoff(self):
        for level, expected_zc in Z_CUTOFFS.items():
            _, _, zc, _ = effective_thresholds("Medium", level)
            assert zc == pytest.approx(expected_zc)

    def test_per_account_structure_preserved(self):
        at, gt, _, _ = effective_thresholds("Very high", "Medium")
        for account in ACCOUNT_THRESHOLDS:
            assert account in at
            base = ACCOUNT_THRESHOLDS[account]
            factor = MATERIALITY_FACTORS["Very high"]
            assert at[account]["pct_band"] == pytest.approx(
                base["pct_band"] * factor)

    def test_scale_threshold_pure(self):
        base = {"pct_band": 0.10, "min_dollar_floor": 10000, "big_dollar": 50000}
        scaled = scale_threshold(base, 3.0)
        assert scaled["pct_band"] == pytest.approx(0.30)
        assert scaled["min_dollar_floor"] == pytest.approx(30000)
        assert scaled["big_dollar"] == pytest.approx(150000)
        assert base["pct_band"] == 0.10  # original unchanged

    def test_very_low_factor_is_3x(self):
        assert MATERIALITY_FACTORS["Very low"] == 3.0

    def test_very_high_factor_is_quarter(self):
        assert MATERIALITY_FACTORS["Very high"] == 0.25


# =============================================================================
# CLASS 12: sensitivity + detection integration
# =============================================================================

class TestSensitivityDetection:
    """Run detection at each sensitivity level on the sample data and verify
    the flagged set changes as expected."""

    def test_medium_flags_three(self, close_df, history):
        at, gt, zc, nz = effective_thresholds("Medium", "Medium")
        flagged, _ = detect_anomalies(close_df, history, at, gt, zc, nz)
        assert len(flagged) == 3

    def test_very_high_sensitivity_flags_more(self, close_df, history):
        at, gt, zc, nz = effective_thresholds("Very high", "Very high")
        flagged, _ = detect_anomalies(close_df, history, at, gt, zc, nz)
        assert len(flagged) >= 4

    def test_very_low_sensitivity_flags_fewer(self, close_df, history):
        at, gt, zc, nz = effective_thresholds("Very low", "Very low")
        flagged, _ = detect_anomalies(close_df, history, at, gt, zc, nz)
        assert len(flagged) < 3

    def test_low_materiality_removes_travel_material_benchmarks(self, close_df, history):
        at, gt, zc, nz = effective_thresholds("Low", "Medium")
        flagged, _ = detect_anomalies(close_df, history, at, gt, zc, nz)
        travel = next(f for f in flagged if f["account"] == "Travel & Ent")
        assert travel["material_benchmarks"] == []
        assert travel["volatility"] == "unusual_for_account"

    def test_high_materiality_adds_cogs(self, close_df, history):
        at, gt, zc, nz = effective_thresholds("High", "Medium")
        flagged, _ = detect_anomalies(close_df, history, at, gt, zc, nz)
        names = [f["account"] for f in flagged]
        assert "COGS" in names

    def test_it_infrastructure_always_flagged(self, close_df, history):
        for mat in LEVELS:
            for vol in LEVELS:
                at, gt, zc, nz = effective_thresholds(mat, vol)
                flagged, _ = detect_anomalies(close_df, history, at, gt, zc, nz)
                names = [f["account"] for f in flagged]
                assert "IT Infrastructure" in names, (
                    "IT Infrastructure must be flagged at {}/{}".format(mat, vol))

    def test_it_infrastructure_stable_moved_at_all_levels(self, close_df, history):
        for mat in LEVELS:
            for vol in LEVELS:
                at, gt, zc, nz = effective_thresholds(mat, vol)
                flagged, _ = detect_anomalies(close_df, history, at, gt, zc, nz)
                it = next(f for f in flagged if f["account"] == "IT Infrastructure")
                assert it["stable_moved"] is True

    def test_patch_overrides_global(self, close_df, history):
        at, gt, zc, nz = effective_thresholds("Medium", "Medium")
        patches = {
            "global": {"pct_band": 0.001, "min_dollar_floor": 1, "big_dollar": 1},
            "accounts": {},
        }
        p_at, p_gt, p_zc, p_nz = patch_thresholds(at, gt, zc, nz, patches)
        assert p_gt["pct_band"] == 0.001
        flagged, _ = detect_anomalies(close_df, history, p_at, p_gt, p_zc, p_nz)
        assert len(flagged) >= 3

    def test_patch_overrides_account(self, close_df, history):
        at, gt, zc, nz = effective_thresholds("Very low", "Very low")
        flagged_before, _ = detect_anomalies(close_df, history, at, gt, zc, nz)
        names_before = {f["account"] for f in flagged_before}
        patches = {
            "global": {},
            "accounts": {
                "Marketing Spend": {
                    "pct_band": 0.001, "min_dollar_floor": 1, "big_dollar": 1,
                },
            },
        }
        p_at, p_gt, p_zc, p_nz = patch_thresholds(at, gt, zc, nz, patches)
        flagged_after, _ = detect_anomalies(close_df, history, p_at, p_gt, p_zc, p_nz)
        names_after = {f["account"] for f in flagged_after}
        assert "Marketing Spend" in names_after

    def test_patch_z_cutoff(self, close_df, history):
        at, gt, zc, nz = effective_thresholds("Medium", "Medium")
        patches = {"z_cutoff": 0.1}
        p_at, p_gt, p_zc, p_nz = patch_thresholds(at, gt, zc, nz, patches)
        assert p_zc == 0.1

    def test_keyless_detection_works(self, close_df, history):
        """Detection and sensitivity run without any API key or client."""
        at, gt, zc, nz = effective_thresholds("High", "High")
        flagged, flags = detect_anomalies(close_df, history, at, gt, zc, nz)
        assert len(flagged) >= 3
        assert isinstance(flags, list)


# =============================================================================
# CLASS 13: triage helpers — fingerprint, matching, sorting, escaping
# =============================================================================

class TestTriageHelpers:
    """Pure-function tests for the web triage helpers. No API calls."""

    MOCK_FLAGGED = [
        {"account": "Marketing Spend"},
        {"account": "IT Infrastructure"},
        {"account": "Travel & Ent"},
    ]

    MOCK_TRIAGE_JSON = [
        {"account": "Marketing Spend", "priority": "HIGH",
         "assessment": "permanent", "headline": "46% over budget",
         "story": "off all three", "hypothesis": "new campaign",
         "confidence": "medium"},
        {"account": "IT Infrastructure", "priority": "MEDIUM",
         "assessment": "permanent", "headline": "EUR 6k on a fixed line",
         "story": "never moves", "hypothesis": "contract change",
         "confidence": "high"},
        {"account": "Travel & Ent", "priority": "HIGH",
         "assessment": "timing", "headline": "94% over prior",
         "story": "small base", "hypothesis": "conference",
         "confidence": "medium"},
    ]

    def test_fingerprint_same_flagged(self, close_df, history):
        at, gt, zc, nz = effective_thresholds("Medium", "Medium")
        flagged1, _ = detect_anomalies(close_df, history, at, gt, zc, nz)
        flagged2, _ = detect_anomalies(close_df, history, at, gt, zc, nz)
        assert triage_fingerprint(flagged1) == triage_fingerprint(flagged2)

    def test_fingerprint_account_added(self, close_df, history):
        """Adding an account to the flagged set changes the fingerprint."""
        at1, gt1, zc1, nz1 = effective_thresholds("Medium", "Medium")
        at2, gt2, zc2, nz2 = effective_thresholds("High", "Medium")
        flagged1, _ = detect_anomalies(close_df, history, at1, gt1, zc1, nz1)
        flagged2, _ = detect_anomalies(close_df, history, at2, gt2, zc2, nz2)
        names1 = {f["account"] for f in flagged1}
        names2 = {f["account"] for f in flagged2}
        assert names1 != names2
        assert triage_fingerprint(flagged1) != triage_fingerprint(flagged2)

    def test_fingerprint_account_removed(self, close_df, history):
        """Removing an account from the flagged set changes the fingerprint."""
        at1, gt1, zc1, nz1 = effective_thresholds("Medium", "Medium")
        at2, gt2, zc2, nz2 = effective_thresholds("Low", "Low")
        flagged1, _ = detect_anomalies(close_df, history, at1, gt1, zc1, nz1)
        flagged2, _ = detect_anomalies(close_df, history, at2, gt2, zc2, nz2)
        assert len(flagged1) > len(flagged2)
        assert triage_fingerprint(flagged1) != triage_fingerprint(flagged2)

    def test_threshold_nudge_keeps_fingerprint(self, close_df, history):
        """A threshold nudge that does not change any flagging decision
        leaves the fingerprint unchanged (the cards stay)."""
        at, gt, zc, nz = effective_thresholds("Medium", "Medium")
        flagged1, _ = detect_anomalies(close_df, history, at, gt, zc, nz)

        gt2 = dict(gt)
        gt2["big_dollar"] = 49000  # no account crosses this boundary
        flagged2, _ = detect_anomalies(close_df, history, at, gt2, zc, nz)

        assert len(flagged1) == len(flagged2)
        assert triage_fingerprint(flagged1) == triage_fingerprint(flagged2)

    def test_same_accounts_different_material_reasons(self, close_df, history):
        """Same accounts flagged but one loses a material benchmark:
        fingerprint changes (the subtle correctness case).

        Travel & Ent dollar variances: vs prior 15k, vs budget 13k,
        vs forecast 16k. A per-account floor of 14k drops vs budget
        (13k < 14k) while prior and forecast stay material.
        """
        at, gt, zc, nz = effective_thresholds("Medium", "Medium")
        flagged1, _ = detect_anomalies(close_df, history, at, gt, zc, nz)

        at2 = {k: dict(v) for k, v in at.items()}
        at2["Travel & Ent"] = {
            "pct_band": gt["pct_band"],
            "min_dollar_floor": 14000,
            "big_dollar": gt["big_dollar"],
        }
        flagged2, _ = detect_anomalies(close_df, history, at2, gt, zc, nz)

        travel1 = next(f for f in flagged1 if f["account"] == "Travel & Ent")
        travel2 = next(f for f in flagged2 if f["account"] == "Travel & Ent")
        assert "budget" in travel1["material_benchmarks"]
        assert "budget" not in travel2["material_benchmarks"]
        assert len(travel2["material_benchmarks"]) >= 1

        assert triage_fingerprint(flagged1) != triage_fingerprint(flagged2)

    def test_volatility_label_change_fingerprint(self, close_df, history):
        """A z-cutoff crossing flips Travel's volatility label from
        unusual_for_account (z=4.99 > 3.5) to within_normal_range
        (4.99 < 6.0). Travel stays flagged via material benchmarks,
        but the fingerprint changes because the prompt would differ."""
        at, gt, _, nz = effective_thresholds("Medium", "Medium")
        flagged1, _ = detect_anomalies(close_df, history, at, gt,
                                       z_cutoff=3.5, near_zero=nz)
        flagged2, _ = detect_anomalies(close_df, history, at, gt,
                                       z_cutoff=6.0, near_zero=nz)

        travel1 = next(f for f in flagged1 if f["account"] == "Travel & Ent")
        travel2 = next(f for f in flagged2 if f["account"] == "Travel & Ent")
        assert travel1["volatility"] == "unusual_for_account"
        assert travel2["volatility"] == "within_normal_range"

        assert triage_fingerprint(flagged1) != triage_fingerprint(flagged2)

    def test_match_all_present(self):
        matched, missing = match_triage_to_flagged(
            self.MOCK_TRIAGE_JSON, self.MOCK_FLAGGED)
        assert len(matched) == 3
        assert len(missing) == 0
        assert "Marketing Spend" in matched

    def test_match_missing_account(self):
        incomplete = [t for t in self.MOCK_TRIAGE_JSON
                      if t["account"] != "IT Infrastructure"]
        matched, missing = match_triage_to_flagged(
            incomplete, self.MOCK_FLAGGED)
        assert len(matched) == 2
        assert "IT Infrastructure" in missing

    def test_match_empty_triage(self):
        matched, missing = match_triage_to_flagged(None, self.MOCK_FLAGGED)
        assert len(matched) == 0
        assert len(missing) == 3

    def test_match_tolerates_name_variation(self):
        varied = [{"account": "Travel and Ent", "priority": "HIGH"}]
        flagged = [{"account": "Travel & Ent"}]
        matched, missing = match_triage_to_flagged(varied, flagged)
        assert len(matched) == 1
        assert len(missing) == 0

    def test_sort_by_priority(self):
        entries = [
            {"priority": "LOW"},
            {"priority": "HIGH"},
            {"priority": "MEDIUM"},
        ]
        sorted_e = sort_by_priority(entries)
        assert sorted_e[0]["priority"] == "HIGH"
        assert sorted_e[1]["priority"] == "MEDIUM"
        assert sorted_e[2]["priority"] == "LOW"

    def test_sort_handles_missing_priority(self):
        entries = [{"priority": "HIGH"}, {"name": "no priority"}]
        sorted_e = sort_by_priority(entries)
        assert sorted_e[0]["priority"] == "HIGH"

    def test_escape_script_tag(self):
        hostile = "<script>alert('xss')</script>"
        safe = escape_text(hostile)
        assert "<script>" not in safe
        assert "&lt;script&gt;" in safe

    def test_escape_ampersand(self):
        assert "&amp;" in escape_text("Travel & Ent")

    def test_badge_known_priorities(self):
        for p in ("HIGH", "MEDIUM", "LOW"):
            badge = priority_badge_html(p)
            assert "sc-badge" in badge
            assert p in badge

    def test_badge_unexpected_priority_no_inject(self):
        hostile = "<script>alert(1)</script>"
        badge = priority_badge_html(hostile)
        assert "<script>" not in badge.lower()
        assert "&lt;" in badge
        assert "sc-badge" not in badge

    def test_badge_case_insensitive(self):
        badge = priority_badge_html("high")
        assert "sc-badge" in badge
        assert "HIGH" in badge

    def test_parse_and_strip_reuse(self):
        """The parse and strip helpers produce the narrative and the list."""
        response = (
            "Summary. Three accounts flagged.\n\n"
            "Marketing Spend [HIGH]\nLikely: permanent.\n\n"
            '```json\n{"triage": [{"account": "Marketing Spend", '
            '"priority": "HIGH"}]}\n```\n'
            "End.")
        narrative = strip_json_block(response)
        triage = parse_triage_json(response)
        assert "Three accounts flagged" in narrative
        assert "```json" not in narrative
        assert triage is not None
        assert triage[0]["account"] == "Marketing Spend"


# =============================================================================
# CLASS 14: triage cost display
# =============================================================================

class TestTriageCost:

    def test_sub_cent_cost(self):
        result = format_cost(100, 50)
        assert "less than EUR 0.01" in result
        assert "billed by Anthropic" in result

    def test_above_cent_cost(self):
        result = format_cost(10000, 5000)
        assert "about EUR" in result
        assert "billed by Anthropic" in result

    def test_cost_includes_token_counts(self):
        result = format_cost(1500, 800)
        assert "1,500 in" in result
        assert "800 out" in result


# =============================================================================
# CLASS 15: in-memory PDF and CSV builders (Phase 5)
# =============================================================================

class TestInMemoryBuilders:
    """Verify the in-memory PDF and CSV builders produce valid output
    matching the CLI artefacts' structure."""

    MOCK_TRIAGE_JSON = [
        {"account": "Marketing Spend", "priority": "HIGH",
         "assessment": "permanent", "headline": "46% over budget",
         "story": "off all three", "hypothesis": "new campaign",
         "confidence": "medium"},
        {"account": "IT Infrastructure", "priority": "MEDIUM",
         "assessment": "permanent", "headline": "EUR 6k on a fixed line",
         "story": "never moves", "hypothesis": "contract change",
         "confidence": "high"},
        {"account": "Travel & Ent", "priority": "HIGH",
         "assessment": "timing", "headline": "94% over prior",
         "story": "small base", "hypothesis": "conference",
         "confidence": "medium"},
    ]

    def test_pdf_bytes_non_empty(self, flagged):
        pdf = build_pdf_bytes(
            "Summary.", flagged, self.MOCK_TRIAGE_JSON, 1500, 800)
        assert isinstance(pdf, bytes)
        assert len(pdf) > 500

    def test_pdf_starts_with_header(self, flagged):
        pdf = build_pdf_bytes(
            "Summary.", flagged, self.MOCK_TRIAGE_JSON, 1500, 800)
        assert pdf[:5] == b"%PDF-"

    def test_csv_bytes_non_empty(self, flagged):
        csv_bytes = build_csv_bytes(flagged, self.MOCK_TRIAGE_JSON)
        assert isinstance(csv_bytes, bytes)
        assert len(csv_bytes) > 100

    def test_csv_has_18_columns(self, flagged):
        csv_bytes = build_csv_bytes(flagged, self.MOCK_TRIAGE_JSON)
        df = pd.read_csv(io.BytesIO(csv_bytes))
        assert len(df.columns) == 18

    def test_csv_expected_columns(self, flagged):
        csv_bytes = build_csv_bytes(flagged, self.MOCK_TRIAGE_JSON)
        df = pd.read_csv(io.BytesIO(csv_bytes))
        expected = [
            "account", "actual",
            "vs_prior_eur", "vs_prior_pct",
            "vs_budget_eur", "vs_budget_pct",
            "vs_forecast_eur", "vs_forecast_pct",
            "material_benchmarks", "volatility",
            "agent_priority", "agent_assessment",
            "agent_headline", "agent_story",
            "agent_hypothesis", "agent_confidence",
            "disposition", "reviewed_by",
        ]
        assert list(df.columns) == expected

    def test_csv_disposition_blank(self, flagged):
        csv_bytes = build_csv_bytes(flagged, self.MOCK_TRIAGE_JSON)
        df = pd.read_csv(io.BytesIO(csv_bytes))
        assert df["disposition"].isna().all() or (df["disposition"] == "").all()
        assert df["reviewed_by"].isna().all() or (df["reviewed_by"] == "").all()

    def test_csv_row_per_flagged(self, flagged):
        csv_bytes = build_csv_bytes(flagged, self.MOCK_TRIAGE_JSON)
        df = pd.read_csv(io.BytesIO(csv_bytes))
        assert len(df) == len(flagged)

    def test_csv_agent_columns_populated(self, flagged):
        csv_bytes = build_csv_bytes(flagged, self.MOCK_TRIAGE_JSON)
        df = pd.read_csv(io.BytesIO(csv_bytes))
        travel = df[df["account"] == "Travel & Ent"].iloc[0]
        assert travel["agent_priority"] == "HIGH"
        assert travel["agent_assessment"] == "timing"

    def test_pdf_records_effective_thresholds(self, flagged):
        pdf_default = build_pdf_bytes(
            "Summary.", flagged, self.MOCK_TRIAGE_JSON, 1500, 800)
        custom_at = {"Revenue": {"pct_band": 0.371, "min_dollar_floor": 99999,
                                 "big_dollar": 88888}}
        custom_gt = {"pct_band": 0.456, "min_dollar_floor": 77777,
                     "big_dollar": 66666}
        pdf_custom = build_pdf_bytes(
            "Summary.", flagged, self.MOCK_TRIAGE_JSON, 1500, 800,
            effective_thresholds=(custom_at, custom_gt, 7.7, 500))
        assert len(pdf_custom) > len(pdf_default)
        assert pdf_custom != pdf_default

    def test_pdf_without_thresholds_omits_table(self, flagged):
        pdf = build_pdf_bytes(
            "Summary.", flagged, self.MOCK_TRIAGE_JSON, 1500, 800)
        assert b"Effective thresholds" not in pdf


# =============================================================================
# CLASS 16: governance and example — fabrication guards, audit, keyless example
# =============================================================================

class TestGovernanceAndExample:
    """Phase 6: fabrication guards prove the worked example is honest,
    session audit records the right fields, and the example loads correctly."""

    def test_fabrication_detection_matches_sample(self, close_df, history):
        """Re-run detection at default thresholds; the flagged accounts
        must match the example's flagged set exactly."""
        from streamlit_app.lib.example import load_example
        ex = load_example()
        flagged, _ = detect_anomalies(close_df, history)
        ex_names = sorted(f["account"] for f in ex["flagged"])
        det_names = sorted(f["account"] for f in flagged)
        assert ex_names == det_names

    def test_fabrication_triage_verbatim(self):
        """The captured narrative contains text that also appears in the
        committed sample_output.txt, proving it was not fabricated."""
        from streamlit_app.lib.example import SAMPLE_TRIAGE
        with open(SAMPLE_TRIAGE, encoding="utf-8") as f:
            captured = json.load(f)
        sample_txt = Path("docs/sample_output.txt").read_text(encoding="utf-8")
        assert "Three accounts need attention" in captured["narrative"]
        assert "Three accounts need attention" in sample_txt

    def test_fabrication_accounts_match_triage(self):
        """The triage entries' account names match the flagged accounts
        from default detection."""
        from streamlit_app.lib.example import load_example
        ex = load_example()
        triage_accounts = sorted(t["account"] for t in ex["triage"])
        flagged_accounts = sorted(f["account"] for f in ex["flagged"])
        assert triage_accounts == flagged_accounts

    def test_fabrication_hashes_honest(self):
        """Data file hashes match the known values from the CLI audit log."""
        close_hash = file_sha256(str(CLOSE_FILE))
        history_hash = file_sha256(str(HISTORY_FILE))
        assert close_hash == (
            "sha256:ab606f7fdb5aafb1a6a080ac03455ee870fe7cb6c8de5e121de622c"
            "436ed6bd7")
        assert history_hash == (
            "sha256:245d70f881afe78c10737584a876c521a39d1d0bf46316b5228e380a"
            "9f319550")

    def test_audit_record_fields(self, flagged):
        """build_audit_record produces all governance fields."""
        from streamlit_app.lib.audit import build_audit_record
        eff = (ACCOUNT_THRESHOLDS, GLOBAL_THRESHOLD,
               MODIFIED_Z_CUTOFF, NEAR_ZERO)
        record = build_audit_record(
            flagged, eff,
            "sha256:abc", "sha256:def",
            1500, 800, "end_turn")
        assert record["n_flagged"] == 3
        assert "Marketing Spend" in record["flagged_accounts"]
        assert record["effective_thresholds"]["z_cutoff"] == MODIFIED_Z_CUTOFF
        assert record["effective_thresholds"]["global_threshold"] == GLOBAL_THRESHOLD
        assert record["effective_thresholds"]["account_thresholds"] == ACCOUNT_THRESHOLDS
        assert record["effective_thresholds"]["near_zero"] == NEAR_ZERO
        assert record["close_hash"] == "sha256:abc"
        assert record["history_hash"] == "sha256:def"
        assert record["tokens_in"] == 1500
        assert record["tokens_out"] == 800
        assert record["requires_review"] is True
        assert "timestamp" in record
        assert "cost" in record

    def test_audit_records_non_default_thresholds(self, flagged):
        """A run at non-default sensitivity records those thresholds,
        so the audit distinguishes sensitivities."""
        from streamlit_app.lib.audit import build_audit_record
        custom_at, custom_gt, custom_zc, custom_nz = effective_thresholds(
            "High", "High")
        eff = (custom_at, custom_gt, custom_zc, custom_nz)
        record = build_audit_record(
            flagged, eff,
            "sha256:abc", "sha256:def",
            1500, 800, "end_turn")
        assert record["effective_thresholds"]["global_threshold"] == custom_gt
        assert record["effective_thresholds"]["account_thresholds"] == custom_at
        assert record["effective_thresholds"]["z_cutoff"] == custom_zc
        assert record["effective_thresholds"]["near_zero"] == custom_nz
        assert custom_gt != GLOBAL_THRESHOLD
        assert custom_zc != MODIFIED_Z_CUTOFF

    def test_audit_requires_review_max_tokens(self, flagged):
        """requires_review is True when stop_reason is max_tokens."""
        from streamlit_app.lib.audit import build_audit_record
        eff = (ACCOUNT_THRESHOLDS, GLOBAL_THRESHOLD,
               MODIFIED_Z_CUTOFF, NEAR_ZERO)
        record = build_audit_record(
            flagged, eff,
            "sha256:abc", "sha256:def",
            1500, 800, "max_tokens")
        assert record["requires_review"] is True

    def test_audit_requires_review_short_output(self):
        """requires_review is True when output is suspiciously short."""
        from streamlit_app.lib.audit import build_audit_record
        eff = ({}, GLOBAL_THRESHOLD, MODIFIED_Z_CUTOFF, NEAR_ZERO)
        record = build_audit_record(
            [], eff,
            "sha256:abc", "sha256:def",
            500, 50, "end_turn")
        assert record["requires_review"] is True

    def test_example_loads_three_flagged(self):
        """Example loads with three flagged accounts at default thresholds."""
        from streamlit_app.lib.example import load_example
        ex = load_example()
        assert len(ex["flagged"]) == 3
        assert ex["fingerprint"]
        assert len(ex["matched"]) == 3
        assert len(ex["missing"]) == 0

    def test_example_fingerprint_matches_default(self, close_df, history):
        """Example fingerprint matches detection at default thresholds."""
        from streamlit_app.lib.example import load_example
        ex = load_example()
        flagged, _ = detect_anomalies(close_df, history)
        assert ex["fingerprint"] == triage_fingerprint(flagged)

    def test_example_non_default_clears(self, close_df, history):
        """Moving sliders from default produces a different fingerprint."""
        from streamlit_app.lib.example import load_example
        ex = load_example()
        at, gt, zc, nz = effective_thresholds("High", "High")
        flagged, _ = detect_anomalies(close_df, history, at, gt, zc, nz)
        assert triage_fingerprint(flagged) != ex["fingerprint"]

    def test_file_sha256_deterministic(self):
        """file_sha256 is pure: same file, same hash."""
        h1 = file_sha256(str(CLOSE_FILE))
        h2 = file_sha256(str(CLOSE_FILE))
        assert h1 == h2
        assert h1.startswith("sha256:")


# =============================================================================
# CLASS 17: currency wiring — CURRENCY_SYMBOL and CURRENCY_CODE drive the PDF
#           and the triage prompt; the cost display is unchanged
# =============================================================================

class TestCurrencyWiring:
    """Prove the display currency constants drive the PDF and triage prompt.
    At the default (EUR/euro sign), output is unchanged. With a USD swap,
    the prompt and PDF reflect dollars, and the cost display stays EUR."""

    MOCK_TRIAGE = [
        {"account": "Marketing Spend", "priority": "HIGH",
         "assessment": "permanent", "headline": "46% over budget",
         "story": "off all three", "hypothesis": "new campaign",
         "confidence": "medium"},
        {"account": "IT Infrastructure", "priority": "MEDIUM",
         "assessment": "permanent", "headline": "6k on a fixed line",
         "story": "never moves", "hypothesis": "contract change",
         "confidence": "high"},
        {"account": "Travel & Ent", "priority": "HIGH",
         "assessment": "timing", "headline": "94% over prior",
         "story": "small base", "hypothesis": "conference",
         "confidence": "medium"},
    ]

    def test_prompt_default_uses_eur(self, flagged):
        sys_p, usr_p = build_prompt(flagged, "Test Entity", "2026-06")
        assert "EUR terms" in sys_p
        assert "EUR 45k" in sys_p
        assert "actual EUR " in usr_p

    def test_prompt_usd_swap(self, flagged):
        with patch('src.step3_triage_agent.CURRENCY_CODE', 'USD'):
            sys_p, usr_p = build_prompt(flagged, "Test Entity", "2026-06")
        assert "USD terms" in sys_p
        assert "USD 45k" in sys_p
        assert "actual USD " in usr_p
        assert "EUR" not in sys_p
        assert "EUR" not in usr_p

    def test_pdf_usd_swap_changes_output(self, flagged):
        pdf_eur = build_pdf_bytes(
            "Summary.", flagged, self.MOCK_TRIAGE, 1500, 800,
            effective_thresholds=(ACCOUNT_THRESHOLDS, GLOBAL_THRESHOLD,
                                 MODIFIED_Z_CUTOFF, NEAR_ZERO))
        with patch('src.step4_output_writer.CURRENCY_SYMBOL', '$'), \
             patch('src.step4_output_writer.CURRENCY_CODE', 'USD'):
            pdf_usd = build_pdf_bytes(
                "Summary.", flagged, self.MOCK_TRIAGE, 1500, 800,
                effective_thresholds=(ACCOUNT_THRESHOLDS, GLOBAL_THRESHOLD,
                                     MODIFIED_Z_CUTOFF, NEAR_ZERO))
        assert pdf_eur != pdf_usd

    def test_cost_unchanged_by_currency_swap(self):
        result = format_cost(1500, 800)
        assert "EUR" in result


# =============================================================================
# CLASS 18: two-mode radio redesign — mode-to-threshold mapping, isolation,
#           example fingerprint, and stale-clear across mode switches
# =============================================================================

class TestTwoModeRedesign:
    """Guard the two-mode radio redesign. The modes (sensitivity sliders and
    per-account thresholds) are two independent ways to produce the same four
    threshold variables fed to detect_anomalies. These tests verify:

    1. Each mode produces the expected thresholds and flagged set.
    2. Mode isolation: detection depends only on the four threshold variables,
       not on which mode produced them. (Widget-level isolation, where Streamlit
       only returns values for rendered widgets, is covered by the render-confirm.)
    3. Both modes reproduce the worked example at defaults.
    4. Stale-clear: a mode switch that changes the flagged set changes the
       fingerprint (clearing triage), while one that does not keeps it.
    """

    # -- 1. Mode-to-thresholds mapping -----------------------------------------

    def test_slider_medium_produces_config_defaults(self):
        at, gt, zc, nz = effective_thresholds("Medium", "Medium")
        assert gt == GLOBAL_THRESHOLD
        assert zc == MODIFIED_Z_CUTOFF
        assert nz == NEAR_ZERO
        for account in ACCOUNT_THRESHOLDS:
            assert at[account] == ACCOUNT_THRESHOLDS[account]

    def test_slider_mode_flags_three_at_medium(self, close_df, history):
        at, gt, zc, nz = effective_thresholds("Medium", "Medium")
        flagged, _ = detect_anomalies(close_df, history, at, gt, zc, nz)
        names = sorted(f["account"] for f in flagged)
        assert names == ["IT Infrastructure", "Marketing Spend", "Travel & Ent"]

    def test_slider_mode_high_flags_more(self, close_df, history):
        at, gt, zc, nz = effective_thresholds("High", "High")
        flagged, _ = detect_anomalies(close_df, history, at, gt, zc, nz)
        assert len(flagged) > 3

    def test_per_account_config_defaults_flag_three(self, close_df, history):
        flagged, _ = detect_anomalies(
            close_df, history,
            account_thresholds=ACCOUNT_THRESHOLDS,
            global_threshold=GLOBAL_THRESHOLD,
            z_cutoff=MODIFIED_Z_CUTOFF,
            near_zero=NEAR_ZERO,
        )
        names = sorted(f["account"] for f in flagged)
        assert names == ["IT Infrastructure", "Marketing Spend", "Travel & Ent"]

    def test_per_account_custom_values_change_flags(self, close_df, history):
        tight = {"pct_band": 0.001, "min_dollar_floor": 1, "big_dollar": 1}
        flagged, _ = detect_anomalies(
            close_df, history,
            account_thresholds={},
            global_threshold=tight,
            z_cutoff=0.1,
            near_zero=NEAR_ZERO,
        )
        assert len(flagged) == len(close_df)

    # -- 2. Mode isolation -----------------------------------------------------
    # Detection is driven only by (account_thresholds, global_threshold,
    # z_cutoff, near_zero). Values that "would belong to the other mode"
    # cannot leak in because detect_anomalies is a pure function of its
    # parameters. This test proves the API boundary: identical parameters
    # produce identical results regardless of how they were derived.

    def test_identical_thresholds_produce_identical_flags(self, close_df, history):
        slider_at, slider_gt, slider_zc, slider_nz = effective_thresholds(
            "Medium", "Medium")
        direct_at = {k: dict(v) for k, v in ACCOUNT_THRESHOLDS.items()}
        direct_gt = dict(GLOBAL_THRESHOLD)
        direct_zc = MODIFIED_Z_CUTOFF
        direct_nz = NEAR_ZERO

        flagged_slider, _ = detect_anomalies(
            close_df, history, slider_at, slider_gt, slider_zc, slider_nz)
        flagged_direct, _ = detect_anomalies(
            close_df, history, direct_at, direct_gt, direct_zc, direct_nz)

        assert (sorted(f["account"] for f in flagged_slider) ==
                sorted(f["account"] for f in flagged_direct))
        assert (triage_fingerprint(flagged_slider) ==
                triage_fingerprint(flagged_direct))

    def test_detection_ignores_extra_account_overrides(self, close_df, history):
        at, gt, zc, nz = effective_thresholds("Medium", "Medium")
        flagged_before, _ = detect_anomalies(close_df, history, at, gt, zc, nz)

        at_extra = {k: dict(v) for k, v in at.items()}
        at_extra["Nonexistent Account"] = {
            "pct_band": 0.001, "min_dollar_floor": 1, "big_dollar": 1}
        flagged_after, _ = detect_anomalies(close_df, history, at_extra, gt, zc, nz)

        assert (sorted(f["account"] for f in flagged_before) ==
                sorted(f["account"] for f in flagged_after))

    # -- 3. Both modes reproduce the example -----------------------------------

    def test_slider_medium_matches_example_fingerprint(self, close_df, history):
        from streamlit_app.lib.example import load_example
        ex = load_example()
        at, gt, zc, nz = effective_thresholds("Medium", "Medium")
        flagged, _ = detect_anomalies(close_df, history, at, gt, zc, nz)
        assert triage_fingerprint(flagged) == ex["fingerprint"]

    def test_per_account_defaults_match_example_fingerprint(self, close_df, history):
        from streamlit_app.lib.example import load_example
        ex = load_example()
        flagged, _ = detect_anomalies(
            close_df, history,
            account_thresholds=ACCOUNT_THRESHOLDS,
            global_threshold=GLOBAL_THRESHOLD,
            z_cutoff=MODIFIED_Z_CUTOFF,
            near_zero=NEAR_ZERO,
        )
        assert triage_fingerprint(flagged) == ex["fingerprint"]

    def test_slider_non_default_clears_example(self, close_df, history):
        from streamlit_app.lib.example import load_example
        ex = load_example()
        at, gt, zc, nz = effective_thresholds("High", "High")
        flagged, _ = detect_anomalies(close_df, history, at, gt, zc, nz)
        assert triage_fingerprint(flagged) != ex["fingerprint"]

    def test_per_account_non_default_clears_example(self, close_df, history):
        from streamlit_app.lib.example import load_example
        ex = load_example()
        tight = {"pct_band": 0.001, "min_dollar_floor": 1, "big_dollar": 1}
        flagged, _ = detect_anomalies(
            close_df, history,
            account_thresholds={},
            global_threshold=tight,
            z_cutoff=0.1,
            near_zero=NEAR_ZERO,
        )
        assert triage_fingerprint(flagged) != ex["fingerprint"]

    # -- 4. Stale-clear across mode switches -----------------------------------

    def test_same_defaults_same_fingerprint_across_modes(self, close_df, history):
        at_slider, gt_slider, zc_slider, nz_slider = effective_thresholds(
            "Medium", "Medium")
        flagged_slider, _ = detect_anomalies(
            close_df, history, at_slider, gt_slider, zc_slider, nz_slider)

        flagged_direct, _ = detect_anomalies(
            close_df, history,
            account_thresholds=ACCOUNT_THRESHOLDS,
            global_threshold=GLOBAL_THRESHOLD,
            z_cutoff=MODIFIED_Z_CUTOFF,
            near_zero=NEAR_ZERO,
        )
        assert triage_fingerprint(flagged_slider) == triage_fingerprint(flagged_direct)

    def test_mode_switch_with_changed_flags_changes_fingerprint(
            self, close_df, history):
        at, gt, zc, nz = effective_thresholds("Medium", "Medium")
        flagged_default, _ = detect_anomalies(close_df, history, at, gt, zc, nz)

        tight = {"pct_band": 0.001, "min_dollar_floor": 1, "big_dollar": 1}
        flagged_tight, _ = detect_anomalies(
            close_df, history,
            account_thresholds={},
            global_threshold=tight,
            z_cutoff=0.1,
            near_zero=NEAR_ZERO,
        )
        assert len(flagged_default) != len(flagged_tight)
        assert triage_fingerprint(flagged_default) != triage_fingerprint(flagged_tight)

    def test_mode_switch_same_accounts_same_reasons_keeps_fingerprint(
            self, close_df, history):
        at, gt, zc, nz = effective_thresholds("Medium", "Medium")
        flagged1, _ = detect_anomalies(close_df, history, at, gt, zc, nz)

        flagged2, _ = detect_anomalies(
            close_df, history,
            account_thresholds=ACCOUNT_THRESHOLDS,
            global_threshold=GLOBAL_THRESHOLD,
            z_cutoff=MODIFIED_Z_CUTOFF,
            near_zero=NEAR_ZERO,
        )

        assert (sorted(f["account"] for f in flagged1) ==
                sorted(f["account"] for f in flagged2))
        assert triage_fingerprint(flagged1) == triage_fingerprint(flagged2)


# =============================================================================
# CLASS 19: volatility-only account (Utilities)
# =============================================================================

class TestVolatilityOnlyAccount:
    """Utilities is tuned to flag on the volatility (z-score) test alone.
    Its z-score lands between 2.5 and 3.5: it flags at High volatility
    sensitivity but not at Medium (the default), so the default flagged set
    and the captured example are unchanged."""

    def test_utilities_not_flagged_at_default(self, close_df, history):
        flagged, _ = detect_anomalies(close_df, history)
        names = [f["account"] for f in flagged]
        assert "Utilities" not in names

    def test_default_flagged_set_unchanged(self, close_df, history):
        flagged, _ = detect_anomalies(close_df, history)
        names = sorted(f["account"] for f in flagged)
        assert names == ["IT Infrastructure", "Marketing Spend", "Travel & Ent"]

    def test_utilities_z_score_in_target_band(self, history):
        label, mz = volatility_context(8450, history["Utilities"], 3.5)
        assert 2.5 < mz < 3.5

    def test_utilities_not_material_against_any_benchmark(self, close_df, history):
        flagged, _ = detect_anomalies(
            close_df, history, z_cutoff=0.1)
        util = next(f for f in flagged if f["account"] == "Utilities")
        assert util["material_benchmarks"] == []

    def test_utilities_flagged_at_high_volatility(self, close_df, history):
        at, gt, zc, nz = effective_thresholds("Medium", "High")
        flagged, _ = detect_anomalies(close_df, history, at, gt, zc, nz)
        names = [f["account"] for f in flagged]
        assert "Utilities" in names

    def test_utilities_flagged_via_volatility_label(self, close_df, history):
        at, gt, zc, nz = effective_thresholds("Medium", "High")
        flagged, _ = detect_anomalies(close_df, history, at, gt, zc, nz)
        util = next(f for f in flagged if f["account"] == "Utilities")
        assert util["volatility"] == "unusual_for_account"
        assert util["material_benchmarks"] == []

    def test_utilities_not_flagged_at_medium_volatility(self, close_df, history):
        at, gt, zc, nz = effective_thresholds("Medium", "Medium")
        flagged, _ = detect_anomalies(close_df, history, at, gt, zc, nz)
        names = [f["account"] for f in flagged]
        assert "Utilities" not in names

    def test_slider_high_volatility_adds_utilities(self, close_df, history):
        at_med, gt_med, zc_med, nz_med = effective_thresholds("Medium", "Medium")
        flagged_med, _ = detect_anomalies(
            close_df, history, at_med, gt_med, zc_med, nz_med)

        at_hi, gt_hi, zc_hi, nz_hi = effective_thresholds("Medium", "High")
        flagged_hi, _ = detect_anomalies(
            close_df, history, at_hi, gt_hi, zc_hi, nz_hi)

        names_med = {f["account"] for f in flagged_med}
        names_hi = {f["account"] for f in flagged_hi}
        assert "Utilities" not in names_med
        assert "Utilities" in names_hi
        assert names_hi - names_med == {"Utilities"}

    def test_example_fingerprint_still_valid(self, close_df, history):
        from streamlit_app.lib.example import load_example
        ex = load_example()
        flagged, _ = detect_anomalies(close_df, history)
        assert triage_fingerprint(flagged) == ex["fingerprint"]
