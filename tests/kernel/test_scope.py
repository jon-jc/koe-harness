"""Scope ownership: order, idempotence, and failure containment."""

from __future__ import annotations

import pytest

from koe.kernel import DisposalError, Scope, ScopeDisposed


def test_teardowns_run_in_reverse_registration_order() -> None:
    order: list[str] = []
    scope = Scope("root")
    scope.collect("first", lambda: order.append("first"))
    scope.collect("second", lambda: order.append("second"))
    scope.collect("third", lambda: order.append("third"))

    scope.dispose()

    # teardown mirrors construction, so a resource is never freed before
    # something built on top of it
    assert order == ["third", "second", "first"]


def test_children_are_disposed_before_parent() -> None:
    order: list[str] = []
    root = Scope("root")
    root.collect("root-teardown", lambda: order.append("root"))
    child = root.fork("child")
    child.collect("child-teardown", lambda: order.append("child"))
    grandchild = child.fork("grandchild")
    grandchild.collect("grandchild-teardown", lambda: order.append("grandchild"))

    root.dispose()

    assert order == ["grandchild", "child", "root"]


def test_disposal_is_idempotent() -> None:
    calls: list[int] = []
    scope = Scope("root")
    scope.collect("once", lambda: calls.append(1))

    scope.dispose()
    scope.dispose()
    scope.dispose()

    assert calls == [1]


def test_one_failing_teardown_does_not_strand_the_others() -> None:
    """A leaked socket in one plugin must not leak its siblings' sockets."""
    freed: list[str] = []
    scope = Scope("session")
    scope.collect("close-socket", lambda: freed.append("socket"))
    scope.collect("explode", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    scope.collect("close-decoder", lambda: freed.append("decoder"))

    with pytest.raises(DisposalError) as excinfo:
        scope.dispose()

    assert freed == ["decoder", "socket"]
    assert len(excinfo.value.failures) == 1
    assert excinfo.value.failures[0][0] == "explode"


def test_failures_aggregate_across_the_scope_tree() -> None:
    root = Scope("root")
    child = root.fork("child")
    child.collect("child-fail", lambda: (_ for _ in ()).throw(ValueError("child")))
    root.collect("root-fail", lambda: (_ for _ in ()).throw(ValueError("root")))

    with pytest.raises(DisposalError) as excinfo:
        root.dispose()

    labels = {label for label, _ in excinfo.value.failures}
    assert labels == {"child-fail", "root-fail"}


def test_disposed_scope_rejects_further_work() -> None:
    scope = Scope("root")
    scope.dispose()

    with pytest.raises(ScopeDisposed):
        scope.collect("late", lambda: None)
    with pytest.raises(ScopeDisposed):
        scope.fork("late-child")


def test_path_reflects_tree_position() -> None:
    root = Scope("root")
    assert root.fork("session").fork("asr").path == "root.session.asr"


def test_context_manager_disposes_on_exit() -> None:
    freed: list[str] = []
    with Scope("root") as scope:
        scope.collect("cleanup", lambda: freed.append("done"))
        assert scope.active
    assert freed == ["done"]
