from tools.alternative_docs_tool import _normalize_key, _clean_text_list

def test_normalize_key():
    assert _normalize_key("FSSC 22000") == "fssc_22000"
    assert _normalize_key("  USDA Organic  ") == "usda_organic"
    assert _normalize_key("ISO-9001:2015") == "iso_9001_2015"
    assert _normalize_key("") == ""
    assert _normalize_key(None) == ""

def test_clean_text_list():
    assert _clean_text_list([" a ", "b", None, ""]) == ["a", "b"]
    assert _clean_text_list([]) == []
    assert _clean_text_list(None) == []
