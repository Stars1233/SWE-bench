"""Issue references may be written as "#123" or as a full URL (#540)."""

from swebench.collect.utils import build_issues_pattern

CLONE = "https://github.com/owner/repo.git"


def _refs(text, clone_url=CLONE):
    return build_issues_pattern(clone_url).findall(text)


def test_hash_form_still_matches():
    assert _refs("fixes #123") == [("fixes", "123")]


def test_full_url_form_matches():
    assert _refs("fixes https://github.com/owner/repo/issues/123") == [("fixes", "123")]


def test_both_forms_in_one_body():
    text = "closes #7 and fixes https://github.com/owner/repo/issues/8"
    assert _refs(text) == [("closes", "7"), ("fixes", "8")]


def test_without_clone_url_only_hash_form():
    assert _refs("fixes #5", clone_url=None) == [("fixes", "5")]
    assert _refs("fixes https://github.com/owner/repo/issues/5", clone_url=None) == []


def test_url_dots_are_escaped_not_wildcards():
    """A lookalike host must not match: the escaped dots are literal."""
    assert _refs("fixes https://githubXcom/owner/repo/issues/9") == []


def test_clone_url_without_git_suffix():
    assert _refs(
        "fixes https://github.com/owner/repo/issues/4",
        clone_url="https://github.com/owner/repo",
    ) == [("fixes", "4")]


def test_other_repo_url_is_not_matched():
    assert _refs("fixes https://github.com/other/proj/issues/1") == []
