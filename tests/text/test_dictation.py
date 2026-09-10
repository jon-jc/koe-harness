"""Spacing, vocabulary, and reasoning-block stripping.

The three pieces adapted from OpenWhispr (MIT). Each one is a pure function
over text, which is the reason they were worth taking: the useful part of that
project's dictation workflow is a handful of rules about characters, and rules
about characters port cleanly into a codebase that shares none of its runtime.
"""

from __future__ import annotations

import pytest

from koe.text.script import Language
from koe.text.spacing import join, join_all, join_tokens, needs_space, with_trailing_space
from koe.text.thinking import has_open_thinking, strip_thinking
from koe.text.vocabulary import EMPTY, Entry, Vocabulary

# --------------------------------------------------------------------------
# spacing
# --------------------------------------------------------------------------


def test_english_fragments_get_a_space() -> None:
    assert join("the roadmap", "ships in March") == "the roadmap ships in March"


def test_japanese_fragments_do_not() -> None:
    """京都 に 行きます is not a typo to a Japanese reader; it is broken text."""
    assert join("京都", "に行きます") == "京都に行きます"


def test_the_seam_is_decided_by_both_sides_not_just_the_left() -> None:
    """The case a one-sided rule cannot get right.

    Japanese business speech code-switches constantly. Looking only at the left
    fragment sees Latin, adds a space, and produces "Q3 の売上".
    """
    assert join("Q3", "の売上") == "Q3の売上"
    assert join("そのKPI", "review") == "そのKPI review"


def test_cjk_punctuation_counts_as_japanese() -> None:
    """A trailing space after です。 is as wrong as one after です."""
    assert join("承知しました。", "明日送ります") == "承知しました。明日送ります"
    assert not needs_space("「はい」", "そうです")


def test_hangul_still_takes_spaces() -> None:
    """Korean separates words; lumping it in with Han and kana would break it."""
    assert join("안녕하세요", "반갑습니다") == "안녕하세요 반갑습니다"


def test_no_space_before_closing_punctuation() -> None:
    """A speaker who pauses before ", and then we ship" gets two utterances."""
    assert join("we shipped it", ", finally") == "we shipped it, finally"
    assert join("done", ".") == "done."


def test_existing_whitespace_is_not_doubled() -> None:
    assert join("hello ", "world") == "hello world"
    assert join("hello", " world") == "hello world"


def test_empty_fragments_pass_through() -> None:
    assert join("", "hello") == "hello"
    assert join("hello", "") == "hello"
    assert join_all([]) == ""


def test_join_all_folds_a_session() -> None:
    assert (
        join_all(["本日の議題は", "第三四半期の", "売上です。"])
        == "本日の議題は第三四半期の売上です。"
    )
    assert join_all(["one", "two", "three"]) == "one two three"


def test_the_one_sided_rule_is_still_available_for_pasting() -> None:
    """Pasting at a cursor in another app has no right-hand side to consult."""
    assert with_trailing_space("hello") == "hello "
    assert with_trailing_space("こんにちは") == "こんにちは"
    assert with_trailing_space("hello ") == "hello "
    assert with_trailing_space("") == ""


def test_rejoining_tokens_keeps_a_space_between_two_latin_words() -> None:
    """Found by running a vocabulary correction through the real pipeline.

    MeCab emits "KPI" and "dashboard" as separate tokens with no record that a
    space stood between them, and a per-language rule that joins Japanese with
    "" produced "KPIdashboard".
    """
    assert join_tokens(["KPI", "dashboard", "は"], Language.JA) == "KPI dashboardは"


def test_rejoining_tokens_does_not_invent_a_space_around_a_digit() -> None:
    """MeCab also splits "Q3", out of a string that never had a space in it.
    The exception is anchored on letters because a digit is not a word."""
    assert join_tokens(["Q", "3", "の", "売上"], Language.JA) == "Q3の売上"


def test_rejoining_english_tokens_is_unchanged() -> None:
    """The English tokenizer emits words that had spaces between them, so the
    reconstruction is exact and there is nothing to be clever about."""
    assert join_tokens(["today's", "agenda"], Language.EN) == "today's agenda"


def test_rejoining_japanese_tokens_stays_tight() -> None:
    assert join_tokens(["本日", "の", "議題"], Language.JA) == "本日の議題"


def test_character_tokens_are_joined_tightly_whatever_script_they_are() -> None:
    """Caught by CI's no-MeCab matrix leg, which the local run does not cover.

    Without MeCab the Japanese fallback is one token per character, so "two
    Latin tokens meet" becomes true between every pair of letters inside a
    single word, and the word-level rule renders "KPI dashboard" as
    "K P I d a s h b o a r d". The segmentation threw the spaces away and left
    no boundary that means anything, so there is nothing to reconstruct from
    and joining tightly is the honest answer.
    """
    characters = list("KPIdashboardは")
    assert join_tokens(characters, Language.JA, word_level=False) == "KPIdashboardは"


# --------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------


def test_a_correction_is_applied() -> None:
    vocab = Vocabulary.of([("Coe", "koe")])
    assert vocab.apply("the Coe harness") == "the koe harness"


def test_a_latin_rule_respects_word_boundaries() -> None:
    """Otherwise a rule for "koe" fires inside "invoke"."""
    vocab = Vocabulary.of([("koe", "koe-harness")])
    assert vocab.apply("invoke the thing") == "invoke the thing"
    assert vocab.apply("koe runs") == "koe-harness runs"


def test_a_japanese_rule_matches_as_a_substring() -> None:
    r"""\b is defined on \w, and between two kanji there is no boundary to find,
    so a boundary-anchored Japanese rule matches nothing at all."""
    vocab = Vocabulary.of([("経営会議", "取締役会")])
    assert vocab.apply("明日の経営会議について") == "明日の取締役会について"


def test_rules_do_not_cascade() -> None:
    """Sequential replacement makes the result depend on storage order."""
    vocab = Vocabulary.of([("A", "B"), ("B", "C")])
    assert vocab.apply("A and B") == "B and C"


def test_the_longest_rule_wins() -> None:
    """A bare "KPI" would otherwise match the prefix and strand the rest."""
    vocab = Vocabulary.of([("KPI", "指標"), ("KPI dashboard", "KPIダッシュボード")])
    assert vocab.apply("the KPI dashboard") == "the KPIダッシュボード"


def test_matching_is_case_insensitive_but_the_replacement_is_literal() -> None:
    vocab = Vocabulary.of([("kpi", "KPI")])
    assert vocab.apply("the KPI and the kpi") == "the KPI and the KPI"


def test_a_term_without_a_written_form_rewrites_nothing() -> None:
    """It is a hint for the recognizer, not a correction."""
    vocab = Vocabulary.of(["Yamamoto"])
    assert vocab.apply("Yamamoto san") == "Yamamoto san"
    assert vocab.terms == ("Yamamoto",)


def test_terms_bias_toward_the_written_form() -> None:
    """Biasing toward the misrecognition would ask for the very mistake the
    entry exists to fix."""
    vocab = Vocabulary.of([("Coe", "koe"), "Yamamoto"])
    assert vocab.terms == ("koe", "Yamamoto")
    assert vocab.prompt_hint() == "koe, Yamamoto"


def test_the_import_format_takes_lines_commas_and_arrows() -> None:
    vocab = Vocabulary.parse(
        """
        # a comment
        Coe => koe
        山本, 田中
        KPI -> 指標
        """
    )
    assert [(e.heard, e.written) for e in vocab] == [
        ("Coe", "koe"),
        ("山本", ""),
        ("田中", ""),
        ("KPI", "指標"),
    ]


def test_a_comma_inside_a_correction_is_not_a_separator() -> None:
    """Only arrow-free lines are split on commas."""
    vocab = Vocabulary.parse("hello => hi, there")
    assert len(vocab) == 1
    assert vocab.entries[0].written == "hi, there"


def test_duplicates_are_dropped() -> None:
    vocab = Vocabulary.parse("koe\nKOE\nkoe => koe")
    assert len(vocab) == 1


def test_the_import_format_round_trips() -> None:
    source = "Coe => koe\n山本\nKPI => 指標"
    assert Vocabulary.parse(source).to_text() == source


def test_an_empty_vocabulary_is_a_pass_through() -> None:
    assert EMPTY.apply("anything at all") == "anything at all"
    assert EMPTY.terms == ()
    assert len(EMPTY) == 0


def test_an_entry_that_rewrites_to_itself_is_not_a_correction() -> None:
    assert not Entry(heard="koe", written="koe").corrects


# --------------------------------------------------------------------------
# reasoning blocks
# --------------------------------------------------------------------------


def test_a_closed_block_is_removed() -> None:
    assert strip_thinking("<think>hmm, maybe</think>The answer is 4.") == "The answer is 4."


def test_nested_blocks_do_not_strand_a_tag() -> None:
    """A single lazy match runs from the outer open tag to the inner close and
    leaves the outer </think> behind as visible text."""
    assert strip_thinking("<think>a<think>b</think>c</think>done") == "done"


def test_an_unterminated_block_is_removed_while_streaming() -> None:
    """Otherwise the user watches the model think."""
    assert strip_thinking("Here goes.<think>still working") == "Here goes."


def test_an_orphan_closing_tag_is_dropped() -> None:
    assert strip_thinking("answer</think>") == "answer"


def test_the_thinking_spelling_is_accepted_too() -> None:
    assert strip_thinking("<thinking>x</thinking>done") == "done"


def test_text_without_tags_is_untouched() -> None:
    assert strip_thinking("just an answer") == "just an answer"
    assert strip_thinking("") == ""


@pytest.mark.parametrize(
    ("text", "open_"),
    [
        ("<think>working", True),
        ("<think>done</think>answer", False),
        ("<think>a</think><think>b", True),
        ("plain text", False),
    ],
)
def test_an_open_block_is_detectable_for_a_thinking_indicator(text: str, open_: bool) -> None:
    """A streaming UI needs this to say "thinking" rather than render an empty
    bubble, which is indistinguishable from a stalled request."""
    assert has_open_thinking(text) is open_
