"""
Focused tests for search-result card rendering (frontend/components.py).
=======================================================================

These exercise the PURE HTML builder `components._result_card_html` and the
related result constants, so they need no Streamlit runtime and no backend.

They lock in the guarantees that matter for result presentation:
  * the returned order is preserved (cards render in list order),
  * ranks render with a visible numeric "#N" and restrained top-3 tiers,
  * long titles are rendered in full (no intentional truncation),
  * existing metadata values pass through unchanged (exact formatting),
  * absent optional fields produce NO fake placeholder or zero,
  * the zero-result message is neutral (not an error),
  * every ranking method renders,
  * no internal backend URL or development-stage wording leaks into a card.
"""

from __future__ import annotations

import components
import utils


def _item(rank, pid, title, score, confidence):
    return {"rank": rank, "product_id": pid, "title": title,
            "score": score, "confidence": confidence}


# --------------------------------------------------------------------------
# Order preservation
# --------------------------------------------------------------------------
def test_result_order_is_preserved():
    results = [
        _item(1, "B001", "Alpha", 8.7421, 0.9998),
        _item(2, "B002", "Beta", 3.2100, 0.7100),
        _item(3, "B003", "Gamma", 1.0500, 0.5400),
    ]
    html = "".join(components._result_card_html(it, "LTR") for it in results)
    # The three product ids appear in exactly the given order.
    assert html.index("B001") < html.index("B002") < html.index("B003")
    # And the rank markers follow the same order.
    assert html.index("#1") < html.index("#2") < html.index("#3")


# --------------------------------------------------------------------------
# Rank rendering
# --------------------------------------------------------------------------
def test_rank_number_always_visible_and_tiered():
    assert '#1' in components._result_card_html(_item(1, "B", "t", 1.0, 0.5), "LTR")
    # Top three get restrained premium tier classes.
    assert 'rank-badge r1' in components._result_card_html(_item(1, "B", "t", 1.0, 0.5), "LTR")
    assert 'rank-badge r2' in components._result_card_html(_item(2, "B", "t", 1.0, 0.5), "LTR")
    assert 'rank-badge r3' in components._result_card_html(_item(3, "B", "t", 1.0, 0.5), "LTR")


def test_rank_four_onward_is_consistent_no_tier():
    html = components._result_card_html(_item(4, "B", "t", 1.0, 0.5), "LTR")
    assert '#4' in html
    # No premium tier class for #4 -- it uses the base badge style.
    assert 'rank-badge r' not in html
    assert 'class="rank-badge"' in html


# --------------------------------------------------------------------------
# Long titles: rendered in full, not truncated
# --------------------------------------------------------------------------
def test_long_title_not_truncated():
    long_title = ("Wireless Noise Cancelling Over-Ear Bluetooth Headphones with "
                  "40-Hour Battery Life, Hi-Res Audio, and Multipoint Pairing "
                  "for Travel, Office, and Home Use")
    html = components._result_card_html(_item(1, "B", long_title, 1.0, 0.9), "LTR")
    # The full title text is present verbatim (no ellipsis inserted by us).
    assert long_title in html
    assert "…" not in html


def test_title_is_html_escaped():
    html = components._result_card_html(_item(1, "B", "A & B <x>", 1.0, 0.9), "LTR")
    assert "A &amp; B &lt;x&gt;" in html
    assert "<x>" not in html


# --------------------------------------------------------------------------
# Metadata values pass through unchanged
# --------------------------------------------------------------------------
def test_metadata_values_unchanged():
    html = components._result_card_html(_item(1, "B07XYZ1234", "t", 8.7421, 0.589), "LTR")
    assert "B07XYZ1234" in html               # product id verbatim
    assert "8.7421" in html                    # raw score, 4dp, unchanged
    assert "58.9%" in html                     # confidence %, 1dp, unchanged
    # Labels are explicit and unambiguous.
    assert "Raw score" in html
    assert "Confidence" in html
    assert "Method" in html
    assert ">LTR<" in html


# --------------------------------------------------------------------------
# Absent optional fields -> no fabricated placeholder
# --------------------------------------------------------------------------
def test_absent_score_omits_raw_score_badge():
    item = {"rank": 1, "product_id": "B", "title": "t", "confidence": 0.5}
    html = components._result_card_html(item, "TF-IDF")
    assert "Raw score" not in html            # no score badge at all
    # No fake 0.0000 injected.
    assert "0.0000" not in html


def test_absent_confidence_omits_confidence_block():
    item = {"rank": 1, "product_id": "B", "title": "t", "score": 4.2}
    html = components._result_card_html(item, "TF-IDF")
    assert "Confidence" not in html
    assert "conf-bar" not in html
    assert "0.0%" not in html                 # no fake zero-percent bar


def test_details_expander_fields_only_present_ones():
    # The details dict shown in the expander includes only real fields (+method).
    item = {"rank": 2, "product_id": "B", "title": "t", "score": 1.5, "confidence": 0.6}
    keys = {k for k in components._DETAIL_FIELDS if k in item}
    assert keys == {"rank", "product_id", "title", "score", "confidence"}
    # A component-score field is NOT part of the contract, so it's never shown.
    assert "cross_encoder_score" not in components._DETAIL_FIELDS


# --------------------------------------------------------------------------
# Confidence is not implied to be a calibrated probability
# --------------------------------------------------------------------------
def test_confidence_bar_has_within_result_caveat():
    html = components._result_card_html(_item(1, "B", "t", 1.0, 0.9), "LTR")
    assert "within" in html.lower()
    assert "not a calibrated probability" in html.lower()


# --------------------------------------------------------------------------
# Zero-result state
# --------------------------------------------------------------------------
def test_zero_result_message_is_neutral():
    msg = components.NO_RESULTS_MESSAGE
    assert "No matching products were found" in msg
    assert "broader query" in msg
    # Neutral wording -- must not imply a failure/error.
    for bad in ("error", "failed", "exception"):
        assert bad not in msg.lower()


# --------------------------------------------------------------------------
# Every ranking method renders
# --------------------------------------------------------------------------
def test_all_ranking_methods_render():
    for label in utils.METHOD_LABELS:
        html = components._result_card_html(_item(1, "B", "t", 1.0, 0.9), label)
        assert f">{label}<" in html


# --------------------------------------------------------------------------
# No leaked internals / dev wording in a card
# --------------------------------------------------------------------------
def test_card_has_no_internal_url_or_dev_wording():
    html = components._result_card_html(_item(1, "B", "t", 1.0, 0.9), "LTR")
    for leak in ("127.0.0.1", ":8000", "localhost", "uvicorn", "Week"):
        assert leak not in html


# --------------------------------------------------------------------------
# "Why was this ranked?" explainability
# --------------------------------------------------------------------------
def test_why_section_present_and_collapsible():
    html = components._result_card_html(_item(1, "B", "t", 1.0, 0.9), "LTR")
    assert "Why was this ranked?" in html
    assert "<details" in html and "<summary>" in html


def test_why_stages_accurate_for_ltr():
    html = components._result_card_html(_item(1, "B", "t", 1.0, 0.9), "LTR")
    assert "candidate generation" in html.lower()
    assert "Cross Encoder" in html
    assert "Learning-to-Rank" in html


def test_why_stages_do_not_overclaim_for_lexical_method():
    # TF-IDF is sparse-only: it must NOT claim cross-encoder or LTR stages.
    html = components._result_card_html(_item(1, "B", "t", 1.0, 0.9), "TF-IDF")
    assert "sparse" in html.lower()
    assert "Cross Encoder" not in html
    assert "Learning-to-Rank" not in html


def test_why_includes_confidence_and_caveat_when_present():
    html = components._result_card_html(_item(1, "B", "t", 1.0, 0.589), "LTR")
    assert "Confidence: 58.9%" in html                  # same precision as the badge
    assert "NOT a calibrated probability" in html


def test_why_omits_confidence_line_when_absent():
    item = {"rank": 1, "product_id": "B", "title": "t", "score": 4.2}
    html = components._result_card_html(item, "BM25")
    assert "Why was this ranked?" in html               # stages still explained
    assert "Confidence:" not in html                    # no fabricated confidence


# --------------------------------------------------------------------------
# Component scores: shown only if genuinely present, never invented
# --------------------------------------------------------------------------
def test_component_scores_absent_by_default():
    # The current API returns no component scores -> none rendered, none faked.
    html = components._result_card_html(_item(1, "B", "t", 1.0, 0.9), "LTR")
    for label in ("BM25 score", "Dense score", "Cross Encoder score", "LTR score"):
        assert label not in html


def test_component_scores_shown_only_when_present():
    item = {"rank": 1, "product_id": "B", "title": "t", "score": 1.0,
            "confidence": 0.9, "bm25_score": 3.14, "cross_encoder_score": 0.77}
    html = components._result_card_html(item, "LTR")
    assert "BM25 score: 3.1400" in html
    assert "Cross Encoder score: 0.7700" in html
    assert "LTR score" not in html                      # not provided -> not shown


# --------------------------------------------------------------------------
# "What is Confidence?" hover tooltip
# --------------------------------------------------------------------------
def test_what_is_confidence_tooltip_present():
    html = components._result_card_html(_item(1, "B", "t", 1.0, 0.9), "LTR")
    assert "What is Confidence?" in html
    assert "should not be interpreted as an absolute probability of relevance" in html


# --------------------------------------------------------------------------
# Pipeline flow visual
# --------------------------------------------------------------------------
def test_pipeline_flow_stage_order():
    labels = [label for _, label in components._PIPELINE_STAGES]
    assert labels == ["Query", "Sparse Retrieval", "Dense Retrieval",
                      "Hybrid Merge", "Cross Encoder", "Learning-to-Rank",
                      "Final Results"]
