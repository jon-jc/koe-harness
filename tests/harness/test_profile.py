"""Declarative composition: profiles, patch layers, and applying them."""

from __future__ import annotations

from pathlib import Path

import pytest

from koe.api.app import Services
from koe.config import Settings
from koe.harness.profile import (
    PluginRow,
    Profile,
    ProfileError,
    apply_profile,
    load,
    parse,
    parse_patch,
)

BASE = """
[[plugins]]
id = "terminal"

[[plugins]]
id = "user-vocabulary"
[plugins.config]
path = "/etc/koe/vocabulary.txt"
"""


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------


def test_a_row_defaults_its_name_to_its_id() -> None:
    """They are usually the same and repeating it is noise."""
    profile = parse('[[plugins]]\nid = "terminal"\n')
    assert profile.rows[0].name == "terminal"


def test_id_and_name_can_differ() -> None:
    """Two rows may mount the same plugin with different configuration, and a
    patch has to be able to name one of them."""
    profile = parse('[[plugins]]\nid = "fast"\nname = "user-vocabulary"\n')
    assert profile.rows[0].id == "fast"
    assert profile.rows[0].name == "user-vocabulary"


def test_a_row_without_an_id_is_refused() -> None:
    with pytest.raises(ProfileError, match="needs an id"):
        parse('[[plugins]]\nname = "terminal"\n')


def test_two_rows_with_one_id_are_refused() -> None:
    """It makes every patch that names it ambiguous, which is a composition
    whose meaning depends on iteration order."""
    with pytest.raises(ProfileError, match="duplicate"):
        parse('[[plugins]]\nid = "a"\n\n[[plugins]]\nid = "a"\n')


def test_malformed_toml_says_so() -> None:
    with pytest.raises(ProfileError, match="not valid TOML"):
        parse("[[plugins]\nid = ")


def test_an_empty_document_is_an_empty_profile() -> None:
    assert len(parse("")) == 0


# --------------------------------------------------------------------------
# patching
# --------------------------------------------------------------------------


def test_a_patch_addresses_a_row_by_id() -> None:
    profile = parse(BASE).patch([{"op": "update", "id": "terminal", "disabled": True}])
    assert profile.get("terminal").disabled is True
    assert profile.get("user-vocabulary").disabled is False


def test_patching_returns_a_new_profile() -> None:
    """A layer that half-applied and then failed would leave a composition
    nobody wrote."""
    original = parse(BASE)
    patched = original.patch([{"op": "update", "id": "terminal", "disabled": True}])

    assert original.get("terminal").disabled is False
    assert patched.get("terminal").disabled is True


def test_the_last_layer_wins_per_row() -> None:
    """What lets layers compose: koe ships a base, a deployment adds a patch,
    the user adds another, and none has to know what the others contain."""
    profile = parse(BASE)
    profile = profile.patch([{"op": "update", "id": "terminal", "disabled": True}])
    profile = profile.patch([{"op": "update", "id": "terminal", "disabled": False}])
    assert profile.get("terminal").disabled is False


def test_a_patch_replaces_config_rather_than_merging_into_it() -> None:
    """Merging means a user who sets one field inherits every other field from
    a layer they cannot see, so the effective config is a computation nobody
    has written down."""
    profile = parse(BASE).patch(
        [{"op": "update", "id": "user-vocabulary", "config": {"limit": 10}}]
    )
    assert profile.get("user-vocabulary").config == {"limit": 10}
    assert "path" not in profile.get("user-vocabulary").config


def test_omitting_config_leaves_it_alone() -> None:
    profile = parse(BASE).patch([{"op": "update", "id": "user-vocabulary", "disabled": True}])
    assert profile.get("user-vocabulary").config == {"path": "/etc/koe/vocabulary.txt"}


def test_inserting_a_row_that_exists_replaces_it() -> None:
    """How a later layer takes ownership of a row without having to know
    whether an earlier one created it."""
    profile = parse(BASE).patch([{"op": "insert", "id": "terminal", "disabled": True}])
    assert len([row for row in profile if row.id == "terminal"]) == 1
    assert profile.get("terminal").disabled is True


def test_remove_drops_a_row() -> None:
    profile = parse(BASE).patch([{"op": "remove", "id": "terminal"}])
    assert profile.get("terminal") is None
    assert profile.get("user-vocabulary") is not None


def test_removing_a_row_that_is_not_there_is_not_an_error() -> None:
    """A layer that removes something an earlier layer did not add should not
    have to know that."""
    assert len(parse(BASE).patch([{"op": "remove", "id": "nothing"}])) == 2


def test_updating_a_row_that_is_not_there_is_an_error() -> None:
    """Unlike remove: an update names fields to change on a row it believes
    exists, and silently doing nothing would hide the typo."""
    with pytest.raises(ProfileError, match="unknown row"):
        parse(BASE).patch([{"op": "update", "id": "nothing", "disabled": True}])


def test_an_unknown_operation_is_refused() -> None:
    with pytest.raises(ProfileError, match="unknown patch operation"):
        parse(BASE).patch([{"op": "frobnicate", "id": "terminal"}])


def test_a_patch_file_parses() -> None:
    operations = parse_patch('[[patch]]\nop = "update"\nid = "terminal"\ndisabled = true\n')
    assert operations == [{"op": "update", "id": "terminal", "disabled": True}]


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def test_a_missing_profile_is_an_empty_one(tmp_path: Path) -> None:
    """koe runs its built-in composition, and a profile only ever adjusts it,
    so "no file" and "a file that changes nothing" behave the same."""
    assert len(load(tmp_path / "nothing.toml")) == 0


def test_patch_files_apply_in_order(tmp_path: Path) -> None:
    (tmp_path / "profile.toml").write_text(BASE, encoding="utf-8")
    (tmp_path / "one.toml").write_text(
        '[[patch]]\nop = "update"\nid = "terminal"\ndisabled = true\n', encoding="utf-8"
    )
    (tmp_path / "two.toml").write_text(
        '[[patch]]\nop = "update"\nid = "terminal"\ndisabled = false\n', encoding="utf-8"
    )

    profile = load(
        tmp_path / "profile.toml", patches=[tmp_path / "one.toml", tmp_path / "two.toml"]
    )
    assert profile.get("terminal").disabled is False


def test_a_missing_patch_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "profile.toml").write_text(BASE, encoding="utf-8")
    profile = load(tmp_path / "profile.toml", patches=[tmp_path / "absent.toml"])
    assert len(profile) == 2


# --------------------------------------------------------------------------
# applying
# --------------------------------------------------------------------------


def services() -> Services:
    return Services.default(Settings(environment="local"))


def test_a_profile_can_disable_a_built_in_plugin() -> None:
    """The point of the whole file: a koe without the terminal should not be a
    fork."""
    subject = services()
    assert subject.plugins.record("terminal").enabled

    apply_profile(subject.plugins, parse('[[plugins]]\nid = "terminal"\ndisabled = true\n'))

    assert subject.plugins.record("terminal").enabled is False


def test_a_profile_can_configure_a_plugin() -> None:
    subject = services()
    apply_profile(
        subject.plugins,
        parse('[[plugins]]\nid = "user-vocabulary"\n[plugins.config]\npath = "/tmp/v.txt"\n'),
    )
    assert subject.plugins.record("user-vocabulary").config == {"path": "/tmp/v.txt"}


def test_a_row_naming_a_plugin_this_build_lacks_is_reported_not_raised() -> None:
    """A profile written for a koe with an extra plugin should still start the
    koe you have, minus that plugin, and say so. Failing would make one stale
    row a startup failure."""
    subject = services()
    changes = apply_profile(subject.plugins, parse('[[plugins]]\nid = "imaginary"\n'))
    assert any("no plugin named" in change for change in changes)


def test_applying_reports_only_what_changed() -> None:
    subject = services()
    # A row that agrees with the current state changes nothing.
    changes = apply_profile(subject.plugins, parse('[[plugins]]\nid = "terminal"\n'))
    assert changes == []


def test_a_profile_survives_a_round_trip_through_its_own_dict() -> None:
    profile = parse(BASE)
    rebuilt = Profile(rows=tuple(PluginRow(**row) for row in profile.to_dict()["plugins"]))
    assert rebuilt.to_dict() == profile.to_dict()
