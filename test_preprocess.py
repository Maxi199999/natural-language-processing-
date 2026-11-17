from preprocess import clean_text, preprocess_text


def test_url_removed():
    assert "http" not in clean_text("Voir https://ex.fr/film")


def test_negation_kept():
    assert "pas" in preprocess_text("Ce n'est pas bon")


def test_repetitions():
    assert clean_text("Super!!!!!") == "Super!"


def test_html_mentions_hashtags_spaces():
    assert clean_text("<br />Trop   bien @marie #cinema<br />") == "Trop bien"


def test_numbers_not_collapsed():
    assert clean_text("Sorti en 2000") == "Sorti en 2000"


def test_curly_elision_negation_kept():
    # n’ (curly apostrophe) must be recognised as the negation ne
    assert "ne" in preprocess_text("Je n’aime pas ce film")


def test_stop_words_removed_and_lemmatized():
    assert preprocess_text("Les acteurs") == ["acteur"]


def test_options_off():
    assert preprocess_text("Les films", lemmas=False, stop=False) == ["les", "films"]
