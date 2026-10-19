from fundx.search import fuzzy_search

NAMES = [
    "Parag Parikh Flexi Cap Fund - Direct Plan - Growth",
    "HDFC Flexi Cap Fund - Growth Option - Direct Plan",
    "UTI Nifty 50 Index Fund - Direct Plan - Growth",
    "Nippon India Small Cap Fund - Direct Plan - Growth",
    "Axis Midcap Fund - Direct Plan - Growth",
]


def test_exact_substring_first():
    assert fuzzy_search("Parag Parikh", NAMES)[0] == NAMES[0]


def test_typos_still_match():
    assert fuzzy_search("parag parikj flexi", NAMES)[0] == NAMES[0]
    assert fuzzy_search("nipon smal cap", NAMES)[0] == NAMES[3]


def test_word_order_and_partial_words():
    assert fuzzy_search("flexi hdfc", NAMES)[0] == NAMES[1]
    assert fuzzy_search("mid", NAMES) == [NAMES[4]]


def test_no_match_returns_empty():
    assert fuzzy_search("zzzzqqq", NAMES) == []


def test_empty_query_returns_all():
    assert fuzzy_search("  ", NAMES) == NAMES


def test_noise_words_alone_do_not_match_everything():
    assert fuzzy_search("fund", NAMES) == sorted(NAMES)  # plain substring hit everywhere
    assert fuzzy_search("direct growth", NAMES) == []
