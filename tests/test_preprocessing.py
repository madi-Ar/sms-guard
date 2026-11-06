from ml.preprocessing import normalize_sms


def test_empty_input():
    assert normalize_sms("") == ""
    assert normalize_sms(" \t\n\x00 ") == ""


def test_url():
    assert normalize_sms("https://example.com/path?q=123") == "<url>"
    assert normalize_sms("Сайт example.kz!") == "сайт <url> !"


def test_mixed_case_and_nfkc():
    assert normalize_sms("ＨｅＬＬｏ WORLD！") == "hello world !"


def test_cyrillic():
    assert normalize_sms("ПРИВЕТ, мир?!") == "привет, мир ? !"


def test_kazakh_letters():
    assert normalize_sms("Ә Ғ Қ Ң Ө Ұ Ү Һ І") == "ә ғ қ ң ө ұ ү һ і"


def test_money():
    assert normalize_sms("₸500, 1 000 тг, 2 500 руб., $1,200.50 и £25") == (
        "<money> , <money> , <money> , <money> и <money>"
    )


def test_phone():
    assert normalize_sms("Звоните +7 (701) 123-45-67 или 8 701 123 45 67.") == (
        "звоните <phone> или <phone> ."
    )


def test_uk_phone_from_uci_format():
    assert normalize_sms("Call 08718729758 now") == "call <phone> now"
    assert normalize_sms("Call 07700 900123 now") == "call <phone> now"


def test_emoji():
    assert normalize_sms("Привет 😊!") == "привет 😊 !"


def test_email_and_numbers():
    assert normalize_sms("Пишите User+7@example.com, код 1234.") == (
        "пишите <email> , код <num> ."
    )


def test_controls_and_whitespace():
    assert normalize_sms("  Сло\u200bво\t\nдругое\x00  ") == "слово другое"


def test_stop_words_are_optional():
    assert normalize_sms("Я и ты!") == "я и ты !"
    assert normalize_sms("Я и ты!", stop_words=frozenset({"я", "и"})) == "ты !"


def test_url_preserves_sentence_punctuation():
    assert normalize_sms("Смотри https://example.com/test?q=1?!") == "смотри <url> ? !"


def test_currency_wins_over_phone_like_digits():
    assert normalize_sms("Сумма 1234567890 тг") == "сумма <money>"
