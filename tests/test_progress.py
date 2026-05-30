from lib import progress


def test_format_elapsed_uses_seconds_below_one_minute():
    assert progress._format_elapsed(0) == "0s"
    assert progress._format_elapsed(59.9) == "59s"


def test_format_elapsed_uses_minutes_at_one_minute():
    assert progress._format_elapsed(60) == "1m 00s"
    assert progress._format_elapsed(3599) == "59m 59s"


def test_format_elapsed_uses_hours_at_one_hour():
    assert progress._format_elapsed(3600) == "1h 00m 00s"
    assert progress._format_elapsed(3661) == "1h 01m 01s"