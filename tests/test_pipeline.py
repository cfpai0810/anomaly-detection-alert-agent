# =============================================================================
# tests/test_pipeline.py: Anomaly Detection and Alert Agent
# =============================================================================
# Phase 6: VALIDATE
#
# Run from the project root with (venv) active:
#   pytest tests/test_pipeline.py -v
#
# The first twenty test classes cover the detection surface; the classes
# after them (not listed) are regression tests for later fixes:
#   1. Data loading             — close and history load and validate
#   2. safe_variance            — the near-zero denominator guard
#   3. is_material              — the three-part materiality rule
#   4. thresholds_for           — per-account thresholds
#   5. volatility_context       — the modified z-score, stable against outliers
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
# No real API calls. Most tests never reach the model; the few that do use
# stand-in clients, so the suite costs nothing and needs no key.
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

    def test_example_shares_an_opening_sentence_with_sample_output(self):
        """The saved narrative and docs/sample_output.txt both contain the
        sentence "Three accounts need attention". This checks that one
        sentence only, not the rest of the text."""
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
            "sha256:97875ae3b9aa9935fdaf86b416f0939decb437020742fa80c5cc807"
            "3243a4ceb")
        assert history_hash == (
            "sha256:21e99d6e6fb2d9c71003f227929d38c73925971727c3f1161ffbf7b5"
            "d509c0a6")

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


class TestDeploymentPins:
    """The web server stack the live app installs."""

    def test_item1_starlette_pinned_for_streamlit_1_60(self):
        """Streamlit 1.60 cannot serve images with starlette 1.8 or later (item 1)."""
        from importlib.metadata import version
        requirements = Path(__file__).resolve().parent.parent / "requirements.txt"
        assert "starlette==1.7.0" in requirements.read_text(encoding="utf-8-sig").split()
        if version("streamlit").startswith("1.60."):
            major, minor = (int(part) for part in version("starlette").split(".")[:2])
            assert (major, minor) < (1, 8)


# =============================================================================
# Step 2 fixes: plain failure messages and the key gate (items 2, 4, 17, 55)
# =============================================================================

import anthropic
import httpx

from streamlit_app.lib.errors import (
    friendly_message, failure_code, model_failure_message,
    MODEL_FAILURE_MESSAGES, KEY_LOOKS_COMPLETE,
)
from src.step3_triage_agent import call_claude

APP_HOME = Path(__file__).resolve().parent.parent / "streamlit_app" / "Home.py"
TEST_KEY = "sk-ant-api03-" + "x" * 90


def _sdk_status_error(cls, status):
    """A stand-in SDK error for one HTTP status, built without the network."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status, request=request)
    return cls("stand-in", response=response, body=None)


def _sdk_connection_error():
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.APIConnectionError(request=request)


class _RaisingMessages:
    def __init__(self, exc):
        self._exc = exc

    def create(self, **kwargs):
        raise self._exc


class _RaisingClient:
    """A stand-in client whose model call fails with a given SDK error."""

    def __init__(self, exc):
        self.messages = _RaisingMessages(exc)


def _app_test():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(str(APP_HOME), default_timeout=60).run()


SDK_CASES = [
    (anthropic.AuthenticationError, 401, "key_rejected"),
    (anthropic.PermissionDeniedError, 403, "key_forbidden"),
    (anthropic.NotFoundError, 404, "model_not_found"),
    (anthropic.RateLimitError, 429, "rate_limited"),
    (anthropic.BadRequestError, 400, "bad_request"),
    (anthropic.InternalServerError, 500, "service_unavailable"),
    (anthropic.APIStatusError, 529, "service_unavailable"),
]


class TestPlainFailureMessages:
    """One plain message per kind of failed model call (items 17 and 55)."""

    @pytest.mark.parametrize("cls,status,code", SDK_CASES)
    def test_item17_sdk_error_maps_to_code(self, cls, status, code):
        exc = _sdk_status_error(cls, status)
        assert failure_code(exc) == code
        message = friendly_message(exc)
        assert message == model_failure_message(code)
        assert message.endswith("Code: `{}`".format(code))
        assert cls.__name__ not in message
        assert "Error" not in message

    def test_item17_connection_error_maps_to_service_unavailable(self):
        exc = _sdk_connection_error()
        assert failure_code(exc) == "service_unavailable"
        assert "APIConnectionError" not in friendly_message(exc)

    @pytest.mark.parametrize("cls,status,code", SDK_CASES)
    def test_item55_engine_wrap_still_maps(self, cls, status, code):
        """call_claude wraps the SDK error in RuntimeError; the cause chain
        still leads to the right plain line, never the generic one."""
        client = _RaisingClient(_sdk_status_error(cls, status))
        with pytest.raises(RuntimeError) as info:
            call_claude("system", "user", client=client)
        assert isinstance(info.value.__cause__, cls)
        assert friendly_message(info.value) == model_failure_message(code)

    def test_item55_wrapped_connection_error_maps(self):
        client = _RaisingClient(_sdk_connection_error())
        with pytest.raises(RuntimeError) as info:
            call_claude("system", "user", client=client)
        assert failure_code(info.value) == "service_unavailable"

    def test_item17_wording_rules(self):
        for code in ("bad_request", "model_not_found"):
            assert "usually temporary" not in model_failure_message(code)
        assert "claude-" in model_failure_message("model_not_found")
        generic = friendly_message(ValueError("boom"))
        assert generic == "Something went wrong running that request. Please try again."
        texts = list(MODEL_FAILURE_MESSAGES.values()) + [generic]
        for text in texts:
            assert "example prompts" not in text
            assert "above" not in text

    def test_item17_no_key_message_points_to_sidebar(self):
        from streamlit_app.lib.claude_client import NoKeyError
        message = friendly_message(NoKeyError("No API key provided."))
        assert "in the sidebar" in message
        assert "above" not in message

    def test_item55_close_scan_shows_key_rejected_line(self):
        """A rejected key on the Close Scan page shows the key_rejected line,
        not the generic one, and records no audit run."""
        rejected = _sdk_status_error(anthropic.AuthenticationError, 401)

        def stand_in_anthropic(api_key=None, **kwargs):
            return _RaisingClient(rejected)

        with patch("anthropic.Anthropic", stand_in_anthropic):
            at = _app_test()
            at.sidebar.text_input(key="byo_api_key").input(TEST_KEY).run()
            at.switch_page("views/close_scan.py").run()
            at.button(key="run_triage").click().run()
        errors = [e.value for e in at.error]
        assert errors == [model_failure_message("key_rejected")]
        assert "scan_audit" not in at.session_state


class TestKeyGate:
    """The sidebar key gate: privacy wording and persistence (items 2 and 4)."""

    NOTE = ("Your key is held in the server's memory for your session only, "
            "used to make the call, and never stored or logged. If you would "
            "rather your key never leave your own machine, download the "
            "project from GitHub and run it locally.")

    def test_item2_privacy_wording(self):
        at = _app_test()
        box = at.sidebar.text_input(key="byo_api_key")
        assert box.help == ("Held in the server's memory for this session "
                            "only. Never stored, never logged.")
        sidebar_text = " ".join(m.value for m in at.sidebar.markdown)
        assert self.NOTE in sidebar_text
        assert "browser" not in sidebar_text

    def test_item2_success_line_is_not_a_verdict(self):
        at = _app_test()
        at.sidebar.text_input(key="byo_api_key").input(TEST_KEY).run()
        assert [s.value for s in at.sidebar.success] == [KEY_LOOKS_COMPLETE]

    def test_item2_header_comment(self):
        source = (APP_HOME.parent / "lib" / "key_gate.py").read_text(encoding="utf-8")
        assert "never stored server-side" not in source
        assert "browser" not in source

    def test_item4_key_persists_across_pages(self):
        """A key entered on Home is still in effect on the Close Scan page,
        after a visit to a page that has no key gate. No network call: the
        client is only constructed."""
        at = _app_test()
        at.sidebar.text_input(key="byo_api_key").input(TEST_KEY).run()
        assert isinstance(at.session_state["live_client"], anthropic.Anthropic)
        at.switch_page("views/how_it_works.py").run()
        at.switch_page("views/close_scan.py").run()
        assert not at.exception
        assert isinstance(at.session_state["live_client"], anthropic.Anthropic)
        assert [b.label for b in at.button] == ["Run AI triage"]


# =============================================================================
# Step 2 fixes: signs, PDF text, the CLI cost line, the sidebar (items 35, 53,
# 57, 58)
# =============================================================================

import base64
import re
import zlib
from types import SimpleNamespace

from src.step4_output_writer import format_money, pdf_text

SAMPLE_TRIAGE_FILE = Path(__file__).resolve().parent.parent / "data" / "sample_triage.json"

_STREAM = re.compile(rb"stream\r?\n(.*?)endstream", re.DOTALL)
_TEXT_OP = re.compile(rb"\(((?:\\.|[^\\)])*)\)\s*Tj|T\*|\bTd\b|\bTD\b|\bTm\b|\bET\b", re.DOTALL)
_PDF_ESCAPE = re.compile(rb"\\([0-7]{1,3}|.)", re.DOTALL)


def _extract_pdf_text(pdf_bytes):
    """The text a reader sees in a reportlab PDF, one line per text line, with
    no extra dependency: each content stream is ASCII85 then Flate encoded,
    and its text is the strings drawn by Tj."""
    def unescape(raw):
        def one(match):
            code = match.group(1)
            if code[:1].isdigit():
                return bytes([int(code, 8) & 0xFF])
            return {b"n": b"\n", b"r": b"\r", b"t": b"\t"}.get(code, code)
        return _PDF_ESCAPE.sub(one, raw).decode("cp1252")

    lines, current = [], []
    for raw in _STREAM.findall(pdf_bytes):
        try:
            content = zlib.decompress(base64.a85decode(raw.strip(), adobe=True))
        except ValueError:
            continue
        for match in _TEXT_OP.finditer(content):
            if match.group(1) is not None:
                current.append(unescape(match.group(1)))
            elif current:
                lines.append("".join(current))
                current = []
    if current:
        lines.append("".join(current))
    return "\n".join(lines)


def _sample_capture():
    return json.loads(SAMPLE_TRIAGE_FILE.read_text(encoding="utf-8"))


def _canned_reply(text, tokens_in=1191, tokens_out=930):
    """A stand-in for the SDK's message object."""
    return SimpleNamespace(
        content=[SimpleNamespace(text=text)],
        usage=SimpleNamespace(input_tokens=tokens_in, output_tokens=tokens_out),
        stop_reason="end_turn",
    )


class _ReplyingClient:
    """A stand-in client whose model call returns a canned reply."""

    def __init__(self, reply):
        self.messages = SimpleNamespace(create=lambda **kwargs: reply)


def _reply_from_capture(triage):
    capture = _sample_capture()
    return _canned_reply(
        capture["narrative"] + "\n\n```json\n"
        + json.dumps({"triage": triage}) + "\n```")


def _close_scan_after_run(reply):
    """Run the Close Scan page with a stand-in client and click Run AI triage."""
    with patch("anthropic.Anthropic", lambda api_key=None, **kw: _ReplyingClient(reply)):
        at = _app_test()
        at.sidebar.text_input(key="byo_api_key").input(TEST_KEY).run()
        at.switch_page("views/close_scan.py").run()
        at.button(key="run_triage").click().run()
    return at


def _download_labels(at):
    return [el.proto.label for el in at.get("download_button")]


class TestSignBeforeSymbol:
    """Amounts read -€30,000, never €-30,000 (item 58). The sign is the
    arithmetic of actual less benchmark. P3 shows no arrow or colour for the
    direction of a change, so no polarity rule applies."""

    def test_item58_revenue_loss(self):
        # Revenue: actual 1,290,000 against prior 1,320,000
        assert format_money(1290000 - 1320000, signed=True) == "-€30,000"

    def test_item58_revenue_gain(self):
        assert format_money(25000, signed=True) == "+€25,000"

    def test_item58_cost_rise(self):
        # Marketing Spend: actual 143,000 against prior 96,000
        assert format_money(143000 - 96000, signed=True) == "+€47,000"

    def test_item58_cost_fall(self):
        assert format_money(-12000, signed=True) == "-€12,000"

    def test_item58_unsigned_amounts(self):
        assert format_money(143000) == "€143,000"
        assert format_money(-5000) == "-€5,000"
        assert format_money(-0.4, signed=True) == "+€0"

    def test_item58_close_scan_table(self):
        at = _app_test()
        at.switch_page("views/close_scan.py").run()
        table = next(d.value for d in at.dataframe if "vs Prior" in d.value.columns)
        rows = table.set_index("Account")
        assert rows.loc["Revenue", "vs Prior"] == "-€30,000 (-2.3%)"
        assert rows.loc["Revenue", "vs Budget"] == "-€10,000 (-0.8%)"
        assert rows.loc["COGS", "vs Prior"] == "+€12,000 (+2.2%)"
        assert rows.loc["Marketing Spend", "vs Prior"] == "+€47,000 (+49.0%)"
        cells = " ".join(str(v) for v in table.to_numpy().ravel())
        assert "€-" not in cells and "€+" not in cells


class TestPdfText:
    """Model, data and reviewer text is escaped for the PDF (item 35), and the
    PDF's summary and sentences read correctly (item 58)."""

    MARKUP = [
        {"account": "Travel & Ent", "priority": "HIGH", "assessment": "timing",
         "headline": "<1k & up", "story": "a<b and <b unclosed",
         "hypothesis": "R&D <5% over plan.", "confidence": "<low>"},
    ]

    def test_item35_helper_escapes_ampersand_first(self):
        assert pdf_text("R&D <b> & x > y") == "R&amp;D &lt;b&gt; &amp; x &gt; y"

    def test_item35_model_markup_builds_pdf(self, flagged):
        pdf = build_pdf_bytes(
            "# Title\n\nThree <5% accounts & a <b tag need review.\n",
            flagged, self.MARKUP, 10, 20)
        assert pdf[:5] == b"%PDF-"
        text = _extract_pdf_text(pdf)
        assert "<1k & up" in text
        assert "a<b and <b unclosed." in text
        assert "R&D <5% over plan." in text
        assert "Summary. Three <5% accounts & a <b tag need review." in text
        assert "&amp;" not in text and "&lt;" not in text

    def test_item35_markup_in_account_and_thresholds(self, flagged):
        triage = [dict(self.MARKUP[0], account="<unknown> & co")]
        custom_at = {"<b>Odd & name": {"pct_band": 0.1, "min_dollar_floor": 1,
                                        "big_dollar": 2}}
        pdf = build_pdf_bytes(
            "Summary.", flagged, triage, 10, 20,
            effective_thresholds=(custom_at, GLOBAL_THRESHOLD, 3.5, 1000))
        text = _extract_pdf_text(pdf)
        assert "<unknown> & co" in text
        assert "<b>Odd & name" in text

    def test_item58_pdf_summary_is_the_summary_sentence(self, flagged):
        capture = _sample_capture()
        text = _extract_pdf_text(build_pdf_bytes(
            capture["narrative"], flagged, capture["triage"], 1191, 930))
        summary = [l for l in text.splitlines() if l.startswith("Summary.")]
        assert summary == ["Summary. Three accounts need attention: two are "
                           "likely timing and one is likely permanent."]

    def test_item58_pdf_no_double_full_stop(self, flagged):
        capture = _sample_capture()
        text = _extract_pdf_text(build_pdf_bytes(
            capture["narrative"], flagged, capture["triage"], 1191, 930))
        assert "capture. Confidence medium." in text
        assert ".." not in text

    def test_item58_pdf_has_no_rule_line(self, flagged):
        capture = _sample_capture()
        assert "\n---" in capture["narrative"]
        text = _extract_pdf_text(build_pdf_bytes(
            capture["narrative"], flagged, capture["triage"], 1191, 930))
        assert "---" not in text

    def test_item58_pdf_actual_sign_before_symbol(self, flagged):
        negative = [dict(f, actual=-5000) for f in flagged]
        text = _extract_pdf_text(build_pdf_bytes("Summary.", negative, [], 10, 20))
        assert "actual -€5,000" in text
        assert "€-" not in text

    def test_item35_close_scan_pdf_with_markup_downloads(self):
        """A live run whose reply has <b, & and < offers both downloads."""
        capture = _sample_capture()
        triage = [dict(t) for t in capture["triage"]]
        triage[0]["hypothesis"] = "R&D <b unclosed <5% over plan"
        at = _close_scan_after_run(_reply_from_capture(triage))
        assert not at.exception
        assert [e.value for e in at.error] == []
        # Since Step 5b the review PDF is a DRAFT until a reviewer signs it;
        # since Step 6b it sits in the Sign-off section, after the CSV.
        assert _download_labels(at) == ["Download anomalies CSV", "Download the DRAFT PDF"]

    def test_item35_close_scan_pdf_failure_keeps_csv(self):
        """If the PDF cannot be built, a plain message shows and the CSV still
        downloads."""
        capture = _sample_capture()
        reply = _reply_from_capture(capture["triage"])
        with patch("src.step4_output_writer.build_pdf_bytes",
                   side_effect=ValueError("paraparser: syntax error")):
            at = _close_scan_after_run(reply)
        assert not at.exception
        errors = [e.value for e in at.error]
        assert errors == ["The review PDF could not be built. Something went "
                          "wrong running that request. Please try again."]
        assert _download_labels(at) == ["Download anomalies CSV"]


class TestCliCostLine:
    """The CLI cost line is in the currency it is computed in (item 57)."""

    def test_item57_cli_cost_line_is_usd(self, capsys):
        call_claude("system", "user", client=_ReplyingClient(_canned_reply("x " * 200)))
        out = capsys.readouterr().out
        line = next(l for l in out.splitlines() if "Approx cost" in l)
        # 1,191 x 3.00 / 1M + 930 x 15.00 / 1M = 0.003573 + 0.013950 = 0.017523
        assert line.strip() == "Approx cost:   USD 0.0175"
        assert "EUR" not in line

    def test_item57_cli_cost_reads_config_rates(self, capsys):
        with patch("src.step3_triage_agent.COST_INPUT_USD_PER_MTOK", 6.0):
            call_claude("system", "user", client=_ReplyingClient(_canned_reply("x " * 200)))
        line = next(l for l in capsys.readouterr().out.splitlines() if "Approx cost" in l)
        # 1,191 x 6.00 / 1M + 930 x 15.00 / 1M = 0.007146 + 0.013950 = 0.021096
        assert line.strip() == "Approx cost:   USD 0.0211"


class TestSidebarState:
    """The sidebar opens on a wide screen and stays closed on a phone (item 53)."""

    def test_item53_initial_sidebar_state_auto(self):
        source = APP_HOME.read_text(encoding="utf-8")
        assert 'initial_sidebar_state="auto"' in source
        assert 'initial_sidebar_state="expanded"' not in source


# =============================================================================
# Step 2 UX: the key gate in an expander, the worked-example button, tabular
# figures only on numbers (UX1, UX2, UX4)
# =============================================================================

from streamlit_app.lib.key_gate import GATE_LABEL, LIVE_ON, LIVE_OFF
from streamlit_app.lib.theme import inject_css


class TestKeyGateExpander:
    """The whole gate sits in a closed sidebar expander with a constant label,
    and one line below it says whether live mode is on (UX1)."""

    def test_ux1_gate_in_closed_expander(self):
        at = _app_test()
        expanders = at.sidebar.expander
        assert [e.label for e in expanders] == [GATE_LABEL]
        assert expanders[0].proto.expanded is False
        gate = expanders[0]
        assert [t.key for t in gate.text_input] == ["byo_api_key"]
        assert len(gate.caption) == 2
        assert any("sc-keynote" in m.value for m in gate.markdown)
        assert not any("Run it yourself" in m.value for m in at.sidebar.markdown)

    def test_ux1_live_mode_line(self):
        at = _app_test()
        assert at.sidebar.caption[-1].value == LIVE_OFF
        at.sidebar.text_input(key="byo_api_key").input(TEST_KEY).run()
        assert at.sidebar.caption[-1].value == LIVE_ON
        assert [e.label for e in at.sidebar.expander] == [GATE_LABEL]
        assert [s.value for s in at.sidebar.expander[0].success] == [KEY_LOOKS_COMPLETE]

    def test_ux1_warning_inside_expander(self):
        at = _app_test()
        at.sidebar.text_input(key="byo_api_key").input("sk-ant-short").run()
        assert len(at.sidebar.expander[0].warning) == 1
        assert at.sidebar.caption[-1].value == LIVE_OFF


class TestWorkedExampleButton:
    """Home has one button straight to the worked example, no key (UX2)."""

    def test_ux2_button_opens_close_scan_example(self):
        at = _app_test()
        assert [b.label for b in at.button] == ["See the worked example"]
        at.button(key="see_worked_example").click().run()
        assert not at.exception
        assert any("<h1>Close scan</h1>" in m.value for m in at.markdown)
        assert [t.label for t in at.tabs] == ["View a worked example", "Run it yourself"]
        example_tab = at.tabs[0]
        assert any(c.value.startswith("This is a captured model output")
                   for c in example_tab.caption)
        assert at.session_state["live_client"] is None


class TestTabularFigures:
    """Tabular figures only where numbers line up (UX4)."""

    def test_ux4_global_rule_has_no_tnum(self):
        css = inject_css()
        global_rule = css[css.index("html, body"):]
        global_rule = global_rule[:global_rule.index("}")]
        assert "tnum" not in global_rule
        assert '"cv05" 1' in global_rule
        # Lists, bold, italics and captions in Inter too, as Project 6's
        # theme sets them (5b fix round)
        assert "th, ul, ol, li, strong, em, caption {" in global_rule

    def test_ux4_numeric_rule_lists_table(self):
        css = inject_css()
        start = css.index("/* Tabular figures only where numbers line up")
        rule = css[start:css.index("}", start)]
        selector = rule[rule.index("*/") + 2:rule.index("{")]
        assert "table" in [s.strip() for s in selector.split(",")]
        assert '"tnum" 1' in rule


# =============================================================================
# Step 3: the Home button opens the worked example tab (UX2)
# =============================================================================

def _blocks_of_type(node, block_type):
    """Every block of one type under an AppTest node, in page order."""
    found = []
    for child in getattr(node, "children", {}).values():
        if getattr(child, "type", None) == block_type:
            found.append(child)
        found.extend(_blocks_of_type(child, block_type))
    return found


class TestWorkedExampleTab:
    """Home's button opens Close scan on its "View a worked example" tab, and
    so does the sidebar link: since Step 5a it is the first tab (UX2, V5)."""

    def test_ux2_button_opens_the_example_tab(self):
        at = _app_test()
        at.button(key="see_worked_example").click().run()
        assert not at.exception
        assert any("<h1>Close scan</h1>" in m.value for m in at.markdown)
        tab_sets = _blocks_of_type(at.main, "tab_container")
        assert len(tab_sets) == 1
        assert [t.label for t in at.tabs] == ["View a worked example", "Run it yourself"]
        assert tab_sets[0].proto.tab_container.default_tab_index == 0
        # A fixed key keeps the chosen tab when a rerun changes the page
        # around the tabs.
        assert tab_sets[0].proto.id.endswith("-cs_triage_tabs")
        # The button only switches page: it sets no one-off flag.
        assert "cs_open_example" not in at.session_state
        assert at.session_state["live_client"] is None

    def test_ux2_sidebar_link_still_opens_the_first_tab(self):
        at = _app_test()
        at.switch_page("views/close_scan.py").run()
        assert not at.exception
        tab_sets = _blocks_of_type(at.main, "tab_container")
        assert len(tab_sets) == 1
        assert tab_sets[0].proto.tab_container.default_tab_index == 0


# =============================================================================
# Step 3 fix: no horizontal-rule line in the Recommended actions callout
# =============================================================================

class TestActionsCallout:
    """The narrative ends with "\n---", which the Recommended actions callout
    used to show as a literal "---" line."""

    def test_fix_no_rule_line_in_actions(self):
        import re
        at = _app_test()
        at.button(key="see_worked_example").click().run()
        assert not at.exception
        callouts = [m.value for m in at.tabs[0].markdown
                    if m.value.startswith('<div class="sc-narrative">')]
        actions = [c for c in callouts if "RECOMMENDED ACTIONS" in c]
        assert len(callouts) == 2 and len(actions) == 1
        for callout in callouts:
            body = callout[len('<div class="sc-narrative">'):-len("</div>")]
            lines = [line.strip() for line in body.split("<br>")]
            assert not any(re.fullmatch(r"([-*_])\1{2,}", line) for line in lines)
        assert actions[0].endswith("misposted invoice.</div>")


class TestDependencies:
    """The pinned packages (items 21 and 22): test tools only in the dev file,
    no byte order mark, one anthropic version, patched versions."""

    @staticmethod
    def _pins(name):
        text = (Path(__file__).resolve().parent.parent / name).read_text(encoding="utf-8-sig")
        pins = {}
        for line in text.splitlines():
            line = line.strip()
            if line and not line.startswith(("#", "-r")) and "==" in line:
                package, version = line.split("==", 1)
                pins[package.strip().lower()] = version.strip()
        return text, pins

    @staticmethod
    def _version(text):
        return tuple(int(part) for part in text.split(".")[:3])

    def test_item21_test_tools_only_in_dev_requirements(self):
        _, runtime = self._pins("requirements.txt")
        dev_text, dev = self._pins("requirements-dev.txt")
        assert not {"pytest", "pluggy", "iniconfig", "pygments", "colorama"} & set(runtime)
        assert "-r requirements.txt" in dev_text and "pytest" in dev

    def test_item21_no_byte_order_mark(self):
        root = Path(__file__).resolve().parent.parent
        for name in ("requirements.txt", "requirements-dev.txt"):
            assert not (root / name).read_bytes().startswith(b"\xef\xbb\xbf"), name

    def test_item21_one_anthropic_version(self):
        assert self._pins("requirements.txt")[1]["anthropic"] == "0.116.0"

    def test_item22_patched_versions(self):
        pins = self._pins("requirements.txt")[1]
        assert self._version(pins["anyio"]) >= (4, 14, 2)
        assert self._version(pins["pillow"]) >= (12, 3, 0)

    def test_item22_certifi_has_the_fixed_root_bundle(self):
        # certifi before 2026.10.18 trusts emSign ECC Root CA G3, whose
        # intermediates had compromised keys (GHSA-6q48-6c7w-9gw2).
        text = (Path(__file__).resolve().parent.parent / "requirements.txt").read_text(
            encoding="utf-8-sig")
        assert "certifi==2026.10.19" in text.split()


# =============================================================================
# Step 5a: Project 6's structure in the series navy (V1 to V5, page icon)
# =============================================================================

import html
import tomllib

from streamlit_app.lib import theme
from streamlit_app.lib.theme import DARK_BLUE, status_chip_html

# Each page as AppTest opens it (None is Home), with the name its banner shows
BANNERS = {
    None: "Anomaly Detection and Alert Agent",
    "views/how_it_works.py": "How it works",
    "views/close_scan.py": "Close scan",
    "views/your_own_data.py": "Your own data",
    "views/how_its_built.py": "How it's built",
}


def _page(path=None):
    """One page of the app, opened from Home with no key."""
    at = _app_test()
    if path:
        at.switch_page(path).run()
    assert not at.exception
    return at


def _navigation():
    """The pages, and the page_icon, exactly as Home hands them to Streamlit."""
    import streamlit as st
    pages, config = [], {}
    real_nav, real_config = st.navigation, st.set_page_config

    def spy_nav(entries, **kwargs):
        pages[:] = list(entries)
        return real_nav(entries, **kwargs)

    def spy_config(*args, **kwargs):
        config.update(kwargs)
        return real_config(*args, **kwargs)

    with patch("streamlit.navigation", spy_nav), \
            patch("streamlit.set_page_config", spy_config):
        at = _app_test()
    assert not at.exception
    return pages, config


class TestStep5Look:
    """The look and layout of Project 6, with P3's navy kept: the theme, one
    banner pattern, the Home blocks, sentence-case names with unchanged
    addresses, and the worked example first (V1 to V5)."""

    def test_v1_primary_colour_is_the_navy(self):
        root = Path(__file__).resolve().parent.parent
        config = tomllib.loads(
            (root / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
        assert config["theme"]["primaryColor"] == DARK_BLUE == "#1A3A5C"
        assert config["client"]["toolbarMode"] == "minimal"

    @pytest.mark.parametrize("path,title", list(BANNERS.items()))
    def test_v2_every_page_has_a_banner(self, path, title):
        at = _page(path)
        banners = [m.value for m in at.main.markdown
                   if m.value.startswith('<div class="sc-header">')]
        assert len(banners) == 1
        assert "<h1>{}</h1>".format(html.escape(title)) in banners[0]
        # The banner is the page's only title: no second title line under it.
        assert not [m.value for m in at.main.markdown if m.value.startswith("#")
                    and m.value.lstrip("#").strip().startswith(title)]

    def test_v2_close_scan_title_line_is_a_caption(self):
        at = _page("views/close_scan.py")
        assert at.main.caption[0].value == "Valencia Operations, Close 2026-06"
        assert not [m.value for m in at.markdown if "Valencia Operations" in m.value
                    and m.value.startswith("#")]

    def test_v2_draft_and_final_chips(self):
        assert theme.DRAFT_CHIP == "#B45F06" and theme.FINAL_CHIP == "#2E6B3F"
        assert status_chip_html("draft") == '<span class="sc-status draft">DRAFT</span>'
        assert status_chip_html("final") == '<span class="sc-status final">FINAL</span>'
        assert status_chip_html("FINAL") == '<span class="sc-status draft">DRAFT</span>'
        assert status_chip_html(None) == '<span class="sc-status draft">DRAFT</span>'
        css = inject_css()
        assert ".sc-status.draft { background: #B45F06; }" in css
        assert ".sc-status.final { background: #2E6B3F; }" in css
        # The blue draft badge is gone; the priority badges stay.
        assert not hasattr(theme, "badge_html") and not hasattr(theme, "BADGE")

    def test_v3_home_blocks_and_page_links(self):
        at = _page()
        assert [b.label for b in at.button] == ["See the worked example"]
        headings = [m.value for m in at.main.markdown if m.value.startswith("#")]
        assert headings == ["#### What it does", "#### The governance promise",
                            "#### Pages"]
        promise = next(m.value for m in at.main.markdown
                       if m.value.startswith("- **Python detects.**"))
        assert [line.split("**")[1] for line in promise.splitlines()] == [
            "Python detects.", "A rule flags.", "The model triages.",
            "The reviewer signs."]
        links = at.get("page_link")
        assert [link.proto.page for link in links] == [
            "how_it_works", "close_scan", "your_own_data", "how_its_built"]
        assert [link.proto.label.split(": ")[0] for link in links] == [
            "How it works", "Close scan", "Your own data", "How it's built"]
        assert [link.proto.icon for link in links] == [
            ":material/menu_book:", ":material/search:",
            ":material/upload_file:", ":material/build:"]
        # The links wrap rather than clip, as in Project 6.
        css = inject_css()
        rule = css[css.index("/* Page links wrap rather than clip on a phone */"):]
        assert 'white-space: normal !important' in rule[:rule.index("}")]
        # The key explainer stays, closed, with its cost figures.
        explainer = at.main.expander
        assert [e.label for e in explainer] == ["Using your own API key"]
        assert explainer[0].proto.expanded is False
        assert any("1.8 US cents" in m.value and "1,191 input and 930 output" in m.value
                   for m in explainer[0].markdown)
        assert at.main.caption[-1].value == (
            "Sample data only: Valencia Operations is a fictional company. "
            "All figures are illustrative.")
        # No green "loaded" box; the warning only when a file is missing.
        assert not at.success and not at.warning

    def test_v3_home_warns_only_when_a_data_file_is_missing(self):
        real_exists = Path.exists

        def exists(self, *args, **kwargs):
            if self.name == "account_history.csv":
                return False
            return real_exists(self, *args, **kwargs)

        with patch.object(Path, "exists", exists):
            at = _app_test()
        assert [w.value for w in at.warning] == [
            "The sample data is incomplete. Missing: account_history.csv. The "
            "tool page needs these files in the project's data folder."]

    def test_v4_titles_are_sentence_case_and_urls_unchanged(self):
        pages, _ = _navigation()
        # Every page address before Step 5a (main at 06b2f31), as Streamlit
        # assigned them: the titles change, the addresses must not.
        assert [p.url_path for p in pages] == [
            "", "how_it_works", "close_scan", "your_own_data", "how_its_built"]
        assert {p.url_path: p.title for p in pages} == {
            "": "Home",
            "how_it_works": "How it works",
            "close_scan": "Close scan",
            "your_own_data": "Your own data",
            "how_its_built": "How it's built",
        }
        for page in pages:
            assert page.title[1:] == page.title[1:].lower()

    def test_v5_example_tab_first_and_no_open_example_flag(self):
        at = _page("views/close_scan.py")
        assert [t.label for t in at.tabs] == ["View a worked example", "Run it yourself"]
        tab_sets = _blocks_of_type(at.main, "tab_container")
        assert len(tab_sets) == 1
        assert tab_sets[0].proto.tab_container.default_tab_index == 0
        assert tab_sets[0].proto.id.endswith("-cs_triage_tabs")
        assert any(c.value.startswith("This is a captured model output")
                   for c in at.tabs[0].caption)
        app = APP_HOME.parent
        for source in app.rglob("*.py"):
            assert "_open_example" not in source.read_text(encoding="utf-8"), source
        page = (app / "views" / "close_scan.py").read_text(encoding="utf-8")
        call = page[page.index("st.tabs("):]
        assert "default=" not in call[:call.index(")")]

    def test_v5_page_icon_is_the_tool_page_icon(self):
        pages, config = _navigation()
        tool = next(p for p in pages if p.url_path == "close_scan")
        assert config["page_icon"] == tool.icon == ":material/search:"


def _tab_place(at):
    """Where the tab bar sits among the page's top-level blocks, and its id.
    While both stay the same, the browser keeps the tab the reader chose."""
    for index, child in at.main.children.items():
        if getattr(child, "type", None) == "tab_container":
            return index, child.proto.id
    return None


class TestStep5TabsAtTheTop:
    """Everything a live run uses sits inside Run it yourself, so the tab bar
    is directly under the banner and stays put through the controls and a run
    (V5 points 3 and 4)."""

    def test_v5_tab_bar_directly_under_the_banner(self):
        at = _page("views/close_scan.py")
        kinds = [getattr(c, "type", None) for c in at.main.children.values()]
        assert kinds[:4] == ["markdown", "markdown", "caption", "tab_container"]
        example, live = at.tabs
        assert [r.key for r in live.radio] == ["threshold_mode"]
        assert [s.key for s in live.select_slider] == ["mat_level", "vol_level"]
        columns = [list(d.value.columns) for d in live.dataframe]
        assert ["Account", "Actual", "Prior month", "Budget", "Forecast"] in columns
        assert any("vs Prior" in c for c in columns)
        assert "#### AI triage" in [m.value for m in live.markdown]
        # The example tab keeps its own content and no live control.
        assert not example.radio and not example.select_slider
        assert not example.dataframe

    def test_v5_tab_stays_put_after_the_controls_and_a_run(self):
        reply = _reply_from_capture(_sample_capture()["triage"])
        with patch("anthropic.Anthropic",
                   lambda api_key=None, **kw: _ReplyingClient(reply)):
            at = _app_test()
            at.sidebar.text_input(key="byo_api_key").input(TEST_KEY).run()
            at.switch_page("views/close_scan.py").run()
            place = _tab_place(at)
            assert place[0] == 3 and place[1].endswith("-cs_triage_tabs")
            at.select_slider(key="mat_level").set_value("High").run()
            assert _tab_place(at) == place
            at.select_slider(key="vol_level").set_value("Low").run()
            assert _tab_place(at) == place
            at.radio(key="threshold_mode").set_value("Custom thresholds").run()
            assert _tab_place(at) == place
            at.radio(key="threshold_mode").set_value("Sensitivity sliders").run()
            at.select_slider(key="mat_level").set_value("Medium").run()
            at.select_slider(key="vol_level").set_value("Medium").run()
            assert _tab_place(at) == place
            at.button(key="run_triage").click().run()
        assert not at.exception
        assert _tab_place(at) == place
        assert _download_labels(at) == ["Download anomalies CSV", "Download the DRAFT PDF"]

    def test_v5_example_tab_names_where_the_controls_are(self):
        at = _page("views/close_scan.py")
        at.select_slider(key="mat_level").set_value("High").run()
        assert [i.value for i in at.tabs[0].info] == [
            "The captured triage was produced at the default sensitivity. "
            "Move the sliders in Run it yourself back to Medium to see it, or "
            "enter an API key in the sidebar to run live triage."]


# =============================================================================
# Step 5b: the Sign-off section, after Project 6 (the rule in src/signoff.py)
# =============================================================================

from src.signoff import (
    ALREADY_SIGNED, NAME_LIMIT, NAME_REFUSAL, SCOPE_NOTE, STALE_CAPTION,
    WORD_HELP, WORD_LABEL, WORD_REFUSAL, clean_name, pdf_name, when_text,
)
from streamlit_app.lib.triage_helpers import flag_reasons, outstanding_lines

_SIGN_MEANS = ("Signing says you have read the result against the figures "
               "above. The figures come from Python. This version does not "
               "check the numbers in the model's wording.")
_DRAFT_CHIP = '<span class="sc-status draft">DRAFT</span>'
_FINAL_CHIP = '<span class="sc-status final">FINAL</span>'
# The flagged accounts at the default sensitivity, as the section lists them
_DEFAULT_FLAGS = [
    "Marketing Spend: material vs prior, budget, forecast; unusual for account.",
    "IT Infrastructure: stable account moved.",
    "Travel & Ent: material vs prior, budget, forecast; unusual for account.",
]
# Materiality Medium, volatility Low: the flagged set and its reasons are the
# same at volatility Very low, so a triage run at one stays on screen at the
# other (the model would be sent the same prompt).
_LOW_VOL_FLAGS = [
    "Marketing Spend: material vs prior, budget, forecast.",
    "IT Infrastructure: stable account moved.",
    "Travel & Ent: material vs prior, budget, forecast.",
]


@pytest.fixture
def stand_in(monkeypatch):
    """Every client the page builds, on every rerun, is a stand-in that
    replies with the worked example's capture: no key, no network."""
    reply = _reply_from_capture(_sample_capture()["triage"])
    monkeypatch.setattr("anthropic.Anthropic",
                        lambda api_key=None, **kw: _ReplyingClient(reply))


@pytest.fixture
def downloads(monkeypatch):
    """Reads the file behind a download button: while the test runs, every
    file the page hands to one is kept by its media id."""
    from streamlit.runtime.memory_media_file_storage import MemoryMediaFileStorage
    files = {}
    load = MemoryMediaFileStorage.load_and_get_id

    def _load(self, data, mimetype, kind, filename=None):
        file_id = load(self, data, mimetype, kind, filename)
        files[file_id] = self.get_file(file_id).content
        return file_id

    monkeypatch.setattr(MemoryMediaFileStorage, "load_and_get_id", _load)

    def read(at, label):
        (button,) = [d for d in at.get("download_button") if d.proto.label == label]
        return files[button.proto.url.rsplit("/", 1)[-1].split(".")[0]]
    return read


def _triaged(vol=None):
    """Close scan after Run AI triage with the stand-in client, optionally at
    another volatility sensitivity."""
    at = _app_test()
    at.sidebar.text_input(key="byo_api_key").input(TEST_KEY).run()
    at.switch_page("views/close_scan.py").run()
    if vol:
        at.select_slider(key="vol_level").set_value(vol).run()
    at.button(key="run_triage").click().run()
    assert not at.exception
    return at


def _signoff_section(at):
    """The Sign-off section's text, from its heading on (Run it yourself)."""
    values = [m.value for m in at.tabs[1].markdown]
    return values[values.index("#### Sign-off"):]


def _numbered(markdown):
    """The lines of a numbered markdown list, without their numbers."""
    lines = markdown.splitlines()
    assert [line.split(". ", 1)[0] for line in lines] == [
        str(i) for i in range(1, len(lines) + 1)]
    return [line.split(". ", 1)[1] for line in lines]


def _sign(at, name="José Núñez", word="SIGN"):
    at.text_input(key="cs_reviewer").input(name).run()
    at.text_input(key="cs_sign").input(word).run()
    at.button(key="cs_signoff").click().run()
    assert not at.exception


def _boxes(at):
    return [(t.key, t.value) for t in at.text_input
            if t.key in ("cs_reviewer", "cs_sign")]


def _signoff_buttons(at):
    return [b for b in at.button if b.key == "cs_signoff"]


def _audit_rows(at):
    return at.session_state["scan_audit"] if "scan_audit" in at.session_state else []


class TestStep5bSignOff:
    """The Sign-off section: the rule in src/signoff.py, drawn as in Project 6,
    after the live triage and its downloads, before the session audit."""

    def test_5b_no_section_without_a_triage(self, stand_in):
        at = _page("views/close_scan.py")
        assert "#### Sign-off" not in [m.value for m in at.markdown]
        at.sidebar.text_input(key="byo_api_key").input(TEST_KEY).run()
        assert [b.key for b in at.button] == ["run_triage"]
        assert "#### Sign-off" not in [m.value for m in at.markdown]
        assert not _boxes(at) and not _signoff_buttons(at)
        # The worked example has no section and no download, as before.
        assert not [m for m in at.tabs[0].markdown if "Sign-off" in m.value]
        assert _download_labels(at) == []

    def test_5b_from_run_to_final(self, stand_in, downloads):
        at = _triaged()
        live = at.tabs[1]
        # After the triage and its downloads, at the level of its siblings
        headings = [m.value for m in live.markdown if m.value.startswith("#")]
        assert headings == ["#### The close", "#### Detection sensitivity",
                            "#### Scan results", "#### AI triage", "#### Sign-off"]
        section = _signoff_section(at)
        assert section == ["#### Sign-off", _SIGN_MEANS, _DRAFT_CHIP,
                           "Outstanding before you sign:", section[4]]
        assert _numbered(section[4]) == _DEFAULT_FLAGS
        assert [(t.label, t.help) for t in live.text_input
                if t.key in ("cs_reviewer", "cs_sign")] == [
            ("Reviewer name", ""), (WORD_LABEL, WORD_HELP)]
        (button,) = _signoff_buttons(at)
        assert button.label == "Sign off" and button.proto.type == "primary"
        assert not button.disabled
        captions = [c.value for c in live.caption]
        assert captions.count(SCOPE_NOTE) == 1 and STALE_CAPTION not in captions
        assert _download_labels(at) == ["Download anomalies CSV", "Download the DRAFT PDF"]
        draft = _extract_pdf_text(downloads(at, "Download the DRAFT PDF")).splitlines()
        assert draft[0] == "MONTH-END CLOSE ANOMALY REVIEW"
        assert draft[2].startswith("Generated ")
        assert draft[3:5] == ["DRAFT: not signed off", "AI TRIAGE ASSESSMENT"]

        # Refused: no name, a wrong word, the word in lower case. Each stays DRAFT.
        for name, word, refusal in [("   ", "SIGN", NAME_REFUSAL),
                                    ("José Núñez", "SIGNED", WORD_REFUSAL),
                                    ("José Núñez", "sign", WORD_REFUSAL)]:
            _sign(at, name, word)
            assert [w.value for w in at.tabs[1].warning] == [refusal]
            assert _signoff_section(at)[2] == _DRAFT_CHIP
            assert "cs_signed_off" not in at.session_state
            assert "status" not in _audit_rows(at)[0]
            assert _download_labels(at)[-1] == "Download the DRAFT PDF"

        # Signed: FINAL, the signed line, read only, the FINAL PDF
        _sign(at, "José Núñez", " SIGN ")
        signed = at.session_state["cs_signed_off"]
        when = when_text(signed.signed_at)
        line = "Signed off by José Núñez on {}.".format(when)
        assert _signoff_section(at) == [
            "#### Sign-off", _SIGN_MEANS, "{}&nbsp; {}".format(_FINAL_CHIP, line)]
        assert not at.tabs[1].warning
        assert not _boxes(at) and not _signoff_buttons(at)
        assert [c.value for c in at.tabs[1].caption].count(SCOPE_NOTE) == 1
        assert _download_labels(at) == ["Download anomalies CSV", "Download the FINAL PDF"]
        final = _extract_pdf_text(downloads(at, "Download the FINAL PDF")).splitlines()
        assert final[3:5] == ["FINAL: signed off by José Núñez on {}".format(when),
                              "AI TRIAGE ASSESSMENT"]
        assert "DRAFT: not signed off" not in final

        # The run's session audit row carries the sign-off and the flagged
        # accounts shown at sign-off, and the session audit list shows it.
        (row,) = _audit_rows(at)
        assert {k: row[k] for k in ("status", "signed_off_by", "signed_off_at")} == {
            "status": "final", "signed_off_by": "José Núñez",
            "signed_off_at": signed.signed_at.isoformat()}
        assert row["acknowledged_at_sign_off"] == _DEFAULT_FLAGS
        assert row["flagged_accounts"] == [
            "Marketing Spend", "IT Infrastructure", "Travel & Ent"]
        (audit,) = [e for e in at.main.expander if e.label == "Session audit (1 run)"]
        assert "{}&nbsp; {}".format(_FINAL_CHIP, line) in [m.value for m in audit.markdown]

    def test_5b_a_second_signoff_is_refused(self, stand_in):
        at = _triaged(vol="Low")
        _sign(at)
        signed = at.session_state["cs_signed_off"]
        result = at.session_state["cs_signed_for"]
        # The page offers no second Sign off.
        assert not _signoff_buttons(at) and not _boxes(at)
        # And the rule refuses one: the same result, signed again while a
        # sign-off for it stands (a second click that lands after the first).
        at.select_slider(key="vol_level").set_value("Very low").run()
        at.select_slider(key="vol_level").set_value("Low").run()
        at.text_input(key="cs_sign").input("SIGN").run()
        at.session_state["cs_signed_off"] = signed
        at.session_state["cs_signed_for"] = result
        at.button(key="cs_signoff").click().run()
        assert not at.exception
        assert [w.value for w in at.tabs[1].warning] == [ALREADY_SIGNED]
        assert at.session_state["cs_signed_off"] == signed
        (row,) = _audit_rows(at)
        assert row["signed_off_at"] == signed.signed_at.isoformat()
        assert _signoff_section(at)[2].startswith(_FINAL_CHIP)

    def test_5b_a_new_triage_is_a_new_draft(self, stand_in, downloads):
        at = _triaged()
        _sign(at, "A. Reviewer")
        first = dict(_audit_rows(at)[0])
        at.button(key="run_triage").click().run()      # the same flags, a new run
        assert not at.exception
        section = _signoff_section(at)
        assert section[2] == _DRAFT_CHIP and _numbered(section[4]) == _DEFAULT_FLAGS
        # The name stays filled in; SIGN must be typed again.
        assert _boxes(at) == [("cs_reviewer", "A. Reviewer"), ("cs_sign", "")]
        assert "cs_signed_off" not in at.session_state
        assert "DRAFT: not signed off" in _extract_pdf_text(
            downloads(at, "Download the DRAFT PDF"))
        # Run 1 keeps its sign-off; run 2 has none until it is signed.
        run2, run1 = _audit_rows(at)
        assert run1 == first and "status" not in run2
        _sign(at, "A. Reviewer")
        run2, run1 = _audit_rows(at)
        assert run2["signed_off_by"] == "A. Reviewer" and run1 == first
        assert run2["signed_off_at"] != run1["signed_off_at"]

    def test_5b_a_new_triage_clears_a_typed_sign_word(self, stand_in):
        at = _triaged()
        at.text_input(key="cs_reviewer").input("A. Reviewer").run()
        at.text_input(key="cs_sign").input("SIGN").run()
        at.button(key="run_triage").click().run()
        assert _boxes(at) == [("cs_reviewer", "A. Reviewer"), ("cs_sign", "")]

    def test_5b_thresholds_that_change_the_flags_clear_the_triage(self, stand_in):
        at = _triaged()
        _sign(at, "A. Reviewer")
        # The page clears a triage whose flagged set changed, and the section
        # goes with it: no stale triage is ever on screen.
        at.select_slider(key="mat_level").set_value("High").run()
        assert not at.exception
        assert "The flagged set changed. Run AI triage again to interpret the " \
               "current accounts." in [i.value for i in at.info]
        assert "#### Sign-off" not in [m.value for m in at.markdown]
        assert _download_labels(at) == []
        # Back to Medium: the triage was cleared, so still nothing to sign.
        at.select_slider(key="mat_level").set_value("Medium").run()
        assert "#### Sign-off" not in [m.value for m in at.markdown]
        # A new triage is a new DRAFT with the name kept.
        at.button(key="run_triage").click().run()
        assert _signoff_section(at)[2] == _DRAFT_CHIP
        assert _boxes(at) == [("cs_reviewer", "A. Reviewer"), ("cs_sign", "")]
        run2, run1 = _audit_rows(at)
        assert run1["status"] == "final" and "status" not in run2

    def test_5b_thresholds_that_keep_the_flags_are_stale(self, stand_in, downloads):
        at = _triaged(vol="Low")
        _sign(at, "A. Reviewer")
        assert _audit_rows(at)[0]["acknowledged_at_sign_off"] == _LOW_VOL_FLAGS
        # Volatility Very low flags the same accounts for the same reasons, so
        # the triage stays; the thresholds it was run at no longer stand.
        at.select_slider(key="vol_level").set_value("Very low").run()
        assert not at.exception
        assert any(m.value.startswith('<div class="sc-narrative">') for m in at.tabs[1].markdown)
        section = _signoff_section(at)
        assert section[2] == _DRAFT_CHIP and _numbered(section[4]) == _LOW_VOL_FLAGS
        assert _boxes(at) == [("cs_reviewer", "A. Reviewer"), ("cs_sign", "")]
        (button,) = _signoff_buttons(at)
        assert button.disabled
        captions = [c.value for c in at.tabs[1].caption]
        assert captions.index(STALE_CAPTION) + 1 == captions.index(SCOPE_NOTE)
        draft = _extract_pdf_text(downloads(at, "Download the DRAFT PDF"))
        assert "DRAFT: not signed off" in draft
        assert "Modified z-score cutoff: 8.0." in draft     # Very low; the run was at 5.5
        # Back to the thresholds of the run: a DRAFT that can be signed again
        at.select_slider(key="vol_level").set_value("Low").run()
        assert _signoff_section(at)[2] == _DRAFT_CHIP
        (button,) = _signoff_buttons(at)
        assert not button.disabled
        assert STALE_CAPTION not in [c.value for c in at.caption]
        first = _audit_rows(at)[0]["signed_off_at"]
        _sign(at, "A. Reviewer")
        (row,) = _audit_rows(at)        # the same run: its sign-off is replaced
        assert row["status"] == "final" and row["signed_off_at"] != first
        assert _signoff_section(at)[2].startswith(_FINAL_CHIP)

    def test_5b_a_key_change_on_its_own_keeps_the_signoff(self, stand_in):
        # The key is not an input of the result: only a new run is.
        at = _triaged()
        _sign(at)
        at.sidebar.text_input(key="byo_api_key").input(
            "sk-ant-api03-" + "y" * 90).run()
        assert not at.exception
        assert _signoff_section(at)[2].startswith(_FINAL_CHIP)
        assert _download_labels(at)[-1] == "Download the FINAL PDF"
        at.button(key="run_triage").click().run()
        assert _signoff_section(at)[2] == _DRAFT_CHIP
        assert _boxes(at) == [("cs_reviewer", "José Núñez"), ("cs_sign", "")]

    def test_5b_names_print_safely(self, stand_in, downloads):
        at = _triaged()
        typed = "Zoë 李 <b>R&D</b>\x07" + "x" * 100
        _sign(at, typed)
        signed = at.session_state["cs_signed_off"]
        assert signed.name == clean_name(typed) and len(signed.name) == NAME_LIMIT
        assert signed.name.startswith("Zoë 李 <b>R&D</b>xxx")
        # On screen as typed, escaped; in the PDF as its fonts print it, never
        # as markup
        assert _signoff_section(at)[2] == "{}&nbsp; {}".format(
            _FINAL_CHIP, html.escape(signed.line))
        final = "".join(_extract_pdf_text(downloads(at, "Download the FINAL PDF")).split())
        status = "FINAL: signed off by {} on {}".format(
            pdf_name(signed.name), when_text(signed.signed_at))
        assert status.startswith("FINAL: signed off by Zoë ? <b>R&D</b>xxx")
        assert "".join(status.split()) in final
        assert _audit_rows(at)[0]["signed_off_by"] == signed.name

    def test_5b_outstanding_lines_use_the_tables_words(self):
        at = _page("views/close_scan.py")
        at.select_slider(key="mat_level").set_value("High").run()
        at.select_slider(key="vol_level").set_value("High").run()
        table = next(d.value for d in at.dataframe if "Reason" in d.value.columns)
        reasons = dict(zip(table["Account"], table["Reason"]))
        flagged, _ = detect_anomalies(
            load_close(CLOSE_FILE), load_history(HISTORY_FILE),
            *effective_thresholds("High", "High"))
        lines = outstanding_lines(flagged)
        assert lines == [
            "COGS: material vs budget, forecast.",
            "Marketing Spend: material vs prior, budget, forecast; unusual for account.",
            "IT Infrastructure: material vs prior, budget, forecast; stable account moved.",
            "Travel & Ent: material vs prior, budget, forecast; unusual for account.",
            "Utilities: unusual for account.",
        ]
        for f, line in zip(flagged, lines):
            assert reasons[f["account"]] == "; ".join(flag_reasons(f))
            assert line.startswith("{}: {}".format(f["account"], reasons[f["account"]]))
        # "Nothing is outstanding." needs a triage with no flags, which this
        # page never shows: with nothing flagged there is nothing to triage.
        assert outstanding_lines([]) == []

    def test_5b_cli_and_example_pdfs_have_no_status_line(self, flagged, tmp_dirs):
        capture = _sample_capture()
        text = _extract_pdf_text(build_pdf_bytes(
            capture["narrative"], flagged, capture["triage"], 1191, 930)).splitlines()
        assert text[3] == "AI TRIAGE ASSESSMENT"
        from src.step4_output_writer import write_pdf
        path = write_pdf(capture["narrative"], flagged, capture["triage"], 1191, 930)
        cli = _extract_pdf_text(path.read_bytes())
        assert cli.splitlines()[3] == "AI TRIAGE ASSESSMENT"
        for words in ("DRAFT", "FINAL", "signed off"):
            assert words not in cli and words not in "\n".join(text)

    def test_5b_wording(self):
        root = Path(__file__).resolve().parent.parent
        at = _page()
        values = [m.value for m in at.main.markdown]
        what = values[values.index("#### What it does") + 1]
        promise = values[values.index("#### The governance promise") + 1]
        assert what.splitlines()[-1] == (
            "- After a live triage, produces a review PDF and an anomalies CSV for "
            "the controller. The PDF is a DRAFT until a named reviewer signs it.")
        assert promise.splitlines()[-1] == (
            "- **The reviewer signs.** Nothing is final until a named reviewer reads "
            "it and signs.")
        built = " ".join(m.value for m in _page("views/how_its_built.py").markdown)
        assert ("This live demo stops at a typed sign-off kept for the session: a "
                "named reviewer types SIGN, and the PDF turns from DRAFT to FINAL. It "
                "still does not store or lock anything, so no data is retained.") in built
        stale = ["stops at flagging for review", "no sign-off step", "nobody approves",
                 "A person reviews", "nothing here is signed", "Download review PDF"]
        for path in [root / "README.md", *(root / "streamlit_app").rglob("*.py"),
                     *(root / "src").glob("*.py")]:
            text = path.read_text(encoding="utf-8")
            assert not [s for s in stale if s in text], path.name
        # The lifecycle diagram keeps the production design (approve, lock).
        lifecycle = (root / "streamlit_app" / "lib" / "governance_lifecycle.py").read_text(
            encoding="utf-8")
        assert "YOU APPROVE" in lifecycle


# =============================================================================
# Step 6a: the page files in views/, every address pinned
# =============================================================================

# Every page address on main before Step 6a (96f0adb), as the sidebar links
# them: Home is the root, and these four are published in posts and READMEs.
PINNED_ADDRESSES = ["how_it_works", "close_scan", "your_own_data", "how_its_built"]


def _declared_pages():
    """The argument for each st.Page that Home declares, in order: a view
    file's name or Home's function, and the keywords it passes."""
    import streamlit as st
    declared = []
    real_page = st.Page

    def spy_page(page, **kwargs):
        declared.append((getattr(page, "__name__", Path(str(page)).name), kwargs))
        return real_page(page, **kwargs)

    with patch("streamlit.Page", spy_page):
        at = _app_test()
    assert not at.exception
    return declared


class TestStep6Views:
    """A folder named pages/ beside Home.py switches on Streamlit's older
    multipage mode, which can run a page requested first without Home (file
    names in the sidebar). The views live in views/, and each address is
    pinned, so moving or renaming a file cannot move a published link."""

    def test_6a_no_folder_named_pages(self):
        app = APP_HOME.parent
        assert [p for p in app.rglob("pages") if p.is_dir()] == []
        assert sorted(p.name for p in (app / "views").glob("*.py")) == [
            "close_scan.py", "how_it_works.py", "how_its_built.py", "your_own_data.py"]

    def test_6a_every_address_is_pinned(self):
        declared = _declared_pages()
        assert [(name, kwargs.get("url_path")) for name, kwargs in declared] == [
            ("how_it_works.py", "how_it_works"),
            ("close_scan.py", "close_scan"),
            ("your_own_data.py", "your_own_data"),
            ("how_its_built.py", "how_its_built"),
            ("_home_page", None),
        ]
        pages, _ = _navigation()
        assert [p.url_path for p in pages] == [""] + PINNED_ADDRESSES


# =============================================================================
# Step 6b: the PDF button inside the Sign-off section, its file name in UTC
# =============================================================================

from streamlit.testing.v1.element_tree import Block

_TEXT_TYPES = ("markdown", "caption", "divider", "error", "warning", "info", "success")
_PDF_FAILED = ("The review PDF could not be built. Something went wrong running "
               "that request. Please try again.")


def _flow(block):
    """Everything under a block in the order the page draws it, as (type,
    text): the text of a markdown, a caption or a message, or a widget's
    label."""
    out = []
    for index in sorted(block.children):
        node = block.children[index]
        if isinstance(node, Block):
            out += _flow(node)
        elif node.type in _TEXT_TYPES:
            out.append((node.type, node.value))
        else:
            out.append((node.type, getattr(node, "label", None)))
    return out


def _flow_split(at):
    """Run it yourself as drawn: what comes before the Sign-off heading, and
    the section from its heading on (the section closes the tab)."""
    flow = _flow(at.tabs[1])
    start = flow.index(("markdown", "#### Sign-off"))
    return flow[:start], flow[start:]


def _draft_top(flags=_DEFAULT_FLAGS):
    """A DRAFT section down to its Sign off button."""
    return [("markdown", "#### Sign-off"), ("markdown", _SIGN_MEANS),
            ("markdown", _DRAFT_CHIP), ("markdown", "Outstanding before you sign:"),
            ("markdown", "\n".join("{}. {}".format(i, line)
                                   for i, line in enumerate(flags, start=1))),
            ("text_input", "Reviewer name"), ("text_input", WORD_LABEL),
            ("button", "Sign off")]


class TestStep6bSignOffPdf:
    """The review PDF downloads from inside the Sign-off section, in Project
    4's order, and its label follows the sign-off; the anomalies CSV stays in
    the downloads row above it (6b1). The PDF's file name is stamped in UTC,
    as the signed line is (6b2)."""

    def test_6b1_pdf_button_in_signoff_follows_the_state(self, stand_in):
        at = _triaged(vol="Low")
        before, section = _flow_split(at)
        # The CSV stays in the downloads row: after the triage, before the section.
        assert [t for k, t in before if k == "download_button"] == ["Download anomalies CSV"]
        assert before[-2:] == [("download_button", "Download anomalies CSV"),
                               ("markdown", "---")]
        # DRAFT: chip, outstanding list, Reviewer name, Type SIGN, Sign off,
        # the scope note, then the PDF
        assert section == _draft_top(_LOW_VOL_FLAGS) + [
            ("caption", SCOPE_NOTE), ("download_button", "Download the DRAFT PDF")]

        # A refusal comes after the scope note, before the PDF.
        _sign(at, "A. Reviewer", "sign")
        assert _flow_split(at)[1] == _draft_top(_LOW_VOL_FLAGS) + [
            ("caption", SCOPE_NOTE), ("warning", WORD_REFUSAL),
            ("download_button", "Download the DRAFT PDF")]

        # FINAL: the chip and signed line, the PDF, then the scope note
        _sign(at, "A. Reviewer")
        signed = at.session_state["cs_signed_off"]
        result = at.session_state["cs_signed_for"]
        final_line = ("markdown", "{}&nbsp; {}".format(_FINAL_CHIP, signed.line))
        before, section = _flow_split(at)
        assert [t for k, t in before if k == "download_button"] == ["Download anomalies CSV"]
        assert section == [
            ("markdown", "#### Sign-off"), ("markdown", _SIGN_MEANS), final_line,
            ("download_button", "Download the FINAL PDF"), ("caption", SCOPE_NOTE)]

        # Stale: a DRAFT again, its caption before the scope note, then the PDF
        at.select_slider(key="vol_level").set_value("Very low").run()
        assert _flow_split(at)[1] == _draft_top(_LOW_VOL_FLAGS) + [
            ("caption", STALE_CAPTION), ("caption", SCOPE_NOTE),
            ("download_button", "Download the DRAFT PDF")]

        # A second sign-off of the same result is refused: the refusal comes
        # after the signed line, before the PDF (as in test_5b_a_second_signoff).
        at.select_slider(key="vol_level").set_value("Low").run()
        at.text_input(key="cs_sign").input("SIGN").run()
        at.session_state["cs_signed_off"] = signed
        at.session_state["cs_signed_for"] = result
        at.button(key="cs_signoff").click().run()
        assert not at.exception
        assert _flow_split(at)[1] == [
            ("markdown", "#### Sign-off"), ("markdown", _SIGN_MEANS), final_line,
            ("warning", ALREADY_SIGNED), ("download_button", "Download the FINAL PDF"),
            ("caption", SCOPE_NOTE)]

    def test_6b1_pdf_failure_message_takes_the_buttons_place(self, stand_in):
        with patch("src.step4_output_writer.build_pdf_bytes",
                   side_effect=ValueError("paraparser: syntax error")):
            at = _triaged()
            before, section = _flow_split(at)
            assert [t for k, t in before if k == "download_button"] == [
                "Download anomalies CSV"]
            assert ("error", _PDF_FAILED) not in before
            assert section == _draft_top() + [
                ("caption", SCOPE_NOTE), ("error", _PDF_FAILED)]
            _sign(at, "A. Reviewer")
            assert _flow_split(at)[1][3:] == [("error", _PDF_FAILED),
                                              ("caption", SCOPE_NOTE)]

    def test_6b2_pdf_file_name_in_utc(self, stand_in, monkeypatch):
        import datetime as clock
        from streamlit.runtime.memory_media_file_storage import MemoryMediaFileStorage

        moment = clock.datetime(2026, 10, 29, 21, 30, 5, tzinfo=clock.timezone.utc)

        class ServerClock(clock.datetime):
            """The clock fixed at 21:30:05 UTC on 29 October 2026, on a server
            whose own zone is UTC+10, where it is 07:30:05 on the 30th."""

            @classmethod
            def now(cls, tz=None):
                if tz is None:
                    return moment.astimezone(
                        clock.timezone(clock.timedelta(hours=10))).replace(tzinfo=None)
                return moment.astimezone(tz)

        names = {}
        load = MemoryMediaFileStorage.load_and_get_id

        def _load(self, data, mimetype, kind, filename=None):
            file_id = load(self, data, mimetype, kind, filename)
            names[file_id] = filename
            return file_id

        at = _triaged()
        # One more run of the page on the fixed clock: the page reads the
        # time as it draws the downloads. Every module it imports is loaded
        # by now, so only the page's own import sees the fixed clock.
        monkeypatch.setattr(MemoryMediaFileStorage, "load_and_get_id", _load)
        with monkeypatch.context() as fixed:
            fixed.setattr(clock, "datetime", ServerClock)
            at.run()
        assert not at.exception
        files = {d.proto.label: names[d.proto.url.rsplit("/", 1)[-1].split(".")[0]]
                 for d in at.get("download_button")}
        # The CSV shares the stamp, so it is in UTC too.
        assert files == {
            "Download anomalies CSV": "anomalies_2026-10-29_21-30-05.csv",
            "Download the DRAFT PDF": "close_review_2026-10-29_21-30-05.pdf",
        }


# =============================================================================
# Step 6 wording: the governance flow ends at YOU SIGN OFF
# =============================================================================

import xml.etree.ElementTree as ET

from streamlit_app.lib.governance_flow import governance_flow_svg
from streamlit_app.lib.governance_lifecycle import governance_lifecycle_svg

_SVG = "{http://www.w3.org/2000/svg}"


def _diagram(svg):
    """A diagram's accessible name, its line breaks read as spaces, and the
    lines of each stage box in order: title, content lines, foot."""
    root = ET.fromstring(svg.strip())
    boxes = []
    for group in root.iter(_SVG + "g"):
        rect = group.find(_SVG + "rect")
        if rect is not None and rect.get("height") == "128":
            boxes.append([t.text for t in group.findall(_SVG + "text")])
    return " ".join(root.get("aria-label").split()), boxes


class TestStep6Wording:
    """The governance flow ends where Projects 2 and 4 end it, at YOU SIGN
    OFF: you judge every flagged account, then sign off. The lifecycle still
    starts at YOU REVIEW, before YOU APPROVE."""

    def test_6_governance_flow_ends_at_you_sign_off(self):
        svg = governance_flow_svg()
        name, boxes = _diagram(svg)
        assert [box[0] for box in boxes] == [
            "YOU LOAD THE CLOSE", "PYTHON DETECTS", "FLAGGED BY RULE",
            "THE MODEL TRIAGES", "YOU SIGN OFF"]
        _, *words, foot = boxes[-1]
        assert " ".join(words) == "You judge every flagged account, then sign off"
        assert foot == "the person"
        assert "YOU REVIEW" not in svg
        assert name == (
            "How a run is governed: you load the month-end close, Python runs the "
            "materiality and z-score tests, accounts that breach the thresholds are "
            "flagged by rule, the model triages the flagged set, and you review every "
            "flagged account, then sign off.")
        assert name.endswith("then sign off.")
        _, lifecycle = _diagram(governance_lifecycle_svg())
        assert [box[0] for box in lifecycle[:2]] == ["YOU REVIEW", "YOU APPROVE"]

    def test_6_the_prose_under_the_flow_ends_at_the_sign_off(self):
        # The paragraph under the governance flow and the In one line note
        # describe the flow, which ends at YOU SIGN OFF: the prose says the
        # same, as the page renders it
        at = _page("views/how_its_built.py")
        (paragraph,) = [m.value for m in at.main.markdown
                        if m.value.startswith("You load the month-end close;")]
        (line,) = [i.value for i in at.info if i.value.startswith("In one line:")]
        assert [paragraph, line] == [
            "You load the month-end close; Python runs the materiality and "
            "volatility tests against the thresholds you set and flags the "
            "outliers; the model reads the flagged accounts and triages each one "
            "with a priority, an assessment, and a hypothesis; and you review "
            "every flagged account, then sign off. The model never sees the "
            "unflagged accounts, never decides what is suspicious, and cannot "
            "remove a flag or suppress an account from the review.",
            "In one line: the model reads the flagged accounts and writes the "
            "triage; Python detects every anomaly and flags what needs a human; "
            "you review what is flagged, then sign off. The intelligence is at the "
            "edges, and the detection in the middle is deterministic and checkable."]
