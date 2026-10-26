from agent.agent import _compute_grounded_confidence

def test_compute_grounded_confidence():
    # Test empty
    conf, _ = _compute_grounded_confidence([], [])
    assert conf == 0.2
    
    # Test no match
    conf, _ = _compute_grounded_confidence(["No badge found"], ["get_alternative_docs"])
    assert conf == 0.3
    
    # Test exact match
    conf, _ = _compute_grounded_confidence(["Badge: FSSC 22000\nSome data"], ["get_alternative_docs"])
    assert conf == 0.85
    
    # Test semantic match
    conf, _ = _compute_grounded_confidence(["Badge: FSSC 22000"], ["semantic_docs_search"])
    assert conf == 0.65
    
    # Test exact + semantic corroboration
    conf, _ = _compute_grounded_confidence(["Badge: FSSC 22000", "Badge: FSSC 22000"], ["get_alternative_docs", "semantic_docs_search"])
    assert conf == 0.95
