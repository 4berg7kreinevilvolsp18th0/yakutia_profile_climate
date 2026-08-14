from scripts.select_inversion_gold_candidates import select_candidates


def test_select_candidates_prefers_disagreements():
    rows = [
        {
            "profile_id": "a",
            "inversion_detected_v2": "True",
            "inversion_candidate_v2": "True",
            "n_inversion_layers_v3": "0",
            "pattern_v3": "NONE",
        },
        {
            "profile_id": "b",
            "inversion_detected_v2": "False",
            "inversion_candidate_v2": "False",
            "n_inversion_layers_v3": "3",
            "pattern_v3": "MULTI",
        },
        {
            "profile_id": "c",
            "inversion_detected_v2": "True",
            "inversion_candidate_v2": "True",
            "n_inversion_layers_v3": "1",
            "pattern_v3": "G",
        },
    ]
    picked = select_candidates(rows, limit=10)
    ids = [r["profile_id"] for r in picked]
    assert "a" in ids
    assert "b" in ids
    assert "c" not in ids
