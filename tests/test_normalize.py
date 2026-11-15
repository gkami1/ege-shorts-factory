import pytest

from factory.speech import numbers as nb
from factory.speech.normalize import display_words, stress_format, to_speech, tokenize


def pairs(text):
    return [(t.display, t.spoken.replace("+", "")) for t in tokenize(text)]


# ── words → digits ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("words,expected", [
    ("восемьдесят", "80"),
    ("десяти", "10"),
    ("двумстам сорока трём", "243"),
    ("две тысячи двадцать пять", "2025"),
    ("тысяча", "1000"),
    ("ноль целых двадцать пять сотых", "0,25"),
    ("две целых пять десятых", "2,5"),
    ("три четвёртых", "3/4"),
    ("одна вторая", "1/2"),
    ("сто", "100"),
])
def test_parse_words(words, expected):
    toks = words.split()
    got = nb.parse_words(toks, 0)
    assert got == (expected, len(toks))


def test_parse_words_stops_at_punctuation_and_order():
    assert nb.parse_words("пять, два".split(), 0) == ("5", 1)
    assert nb.parse_words("двадцать двенадцать".split(), 0) == ("20", 1)
    assert nb.parse_words("один из".split(), 0) is None          # «один из способов» stays a word
    assert nb.parse_words("один".split(), 0, prev="Ответ:") == ("1", 1)


# ── digits → words ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("num,unit,expected", [
    ("2", "с", "две секунды"),
    ("1", "м/с", "один метр в секунду"),
    ("14", "м/с", "четырнадцать метров в секунду"),
    ("21", "кг", "двадцать один килограмм"),
    ("2,5", "м", "две целых пять десятых метра"),
    ("5", "%", "пять процентов"),
    ("1", "Н", "один ньютон"),
])
def test_units(num, unit, expected):
    assert nb.spell_with_unit(num, unit) == expected


def test_spell_number():
    assert nb.spell_number("-0,25")[0] == "минус ноль целых двадцать пять сотых"
    assert nb.spell_number("80")[0] == "восемьдесят"
    assert nb.spell_fraction(3, 4) == "три четвёртых"
    assert nb.spell_fraction(21, 100) == "двадцать одна сотая"


# ── narration ─────────────────────────────────────────────────────────────────────

def test_number_words_become_digits_in_captions():
    assert display_words("Ответ: восемьдесят!") == ["Ответ:", "80!"]
    assert display_words("Эту задачу из ЕГЭ решают за пять секунд. Смотри.") == [
        "Эту", "задачу", "из", "ЕГЭ", "решают", "за", "5", "секунд.", "Смотри."]
    # the voice keeps the words, with their case
    assert to_speech("Два в степени икс равно десяти.").replace("+", "") == "Два в степени икс равно десяти."


def test_dash_kept_in_captions_and_paused_in_speech():
    assert display_words("Десять на восемь — восемьдесят.") == ["10", "на", "8 —", "80."]
    assert to_speech("Десять на восемь — восемьдесят.") == "Десять на восемь, восемьдесят."


def test_math_reading():
    assert pairs("$2^{x+3}$") == [("2", "два"), ("в степени", "в степени"), ("x", "икс"), ("+", "плюс"), ("3", "три")]
    assert to_speech("$2^x = 10$.").replace("+", "") == "два в степени икс равно десять."
    assert to_speech("$10\\cdot 8 = 80$").replace("+", "") == "десять умножить на восемь равно восемьдесят"
    assert to_speech("$3t^2$").replace("+", "") == "три тэ в квадрате"
    assert to_speech("$x(t)$").replace("+", "") == "икс от тэ"
    assert to_speech("$\\frac{3}{4}$") == "три четвёртых"
    assert to_speech("$\\sqrt{16}$").replace("+", "") == "корень из шестнадцати"
    assert to_speech("$\\log_2 8$").replace("+", "") == "логарифм восьми по основанию два"
    assert to_speech("$2{,}5$").replace("+", "") == "две целых пять десятых"


def test_units_after_formula():
    assert pairs("в момент $t = 2$ с?")[-1] == ("2 с?", "две секунды?")
    assert pairs("Ответ: 14 м/с.")[-1] == ("14 м/с.", "четырнадцать метров в секунду.")
    assert pairs("на 25%")[-1] == ("25%", "двадцать пять процентов")


def test_brackets_glue_to_neighbours():
    disp = display_words("$(x+1)^2$")
    assert disp[0] == "(x" and disp[2] == "1)"


def test_stress_marks_and_formats():
    s = to_speech("Смотрим на показатель.")
    assert "показ+атель" in s
    assert stress_format("показ+атель", "acute") == "показа́тель"
    assert stress_format("показ+атель", "strip") == "показатель"
    assert stress_format("показ+атель", "plus") == "показ+атель"


def test_token_words_skip_punctuation():
    toks = tokenize("восемь — восемьдесят")
    assert [w for t in toks for w in t.words] == ["восемь,", "восемьдесят"]


def test_qa_canonical_matches_digits_and_letters():
    from factory.speech import qa

    ref = qa.canonical("Два в степени икс равно десяти.")
    hyp = qa.canonical_hyp(["Два", "в", "степени", "X", "равно", "10."])
    assert qa.wer(ref, hyp) == 0
    assert qa.wer(["a", "b"], ["a"]) == 0.5


def test_merge_number_words_in_transcript():
    from factory.speech.qa import merge_number_words

    hyp = [("Ответ", 0.0, 0.3), ("восемьдесят", 0.4, 0.9), ("пять", 0.9, 1.2)]
    assert merge_number_words(hyp) == [("Ответ", 0.0, 0.3), ("85", 0.4, 1.2)]


def test_latin_words_in_text_are_spelled():
    assert pairs("на ось Ox.")[-1] == ("Ox.", "о икс.")


def test_whisper_cyrillic_x_counts_as_iks():
    from factory.speech import qa

    assert qa.score("два в степени икс", ["2", "в", "степени", "х"]) == 0
