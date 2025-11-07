"""Tests for the Environment class."""

import pytest

from east.runtime.environment import Environment, EnvironmentError


class TestEnvironmentBasics:
    """Test basic environment operations."""

    def test_empty_environment(self):
        """Test empty environment."""
        env = Environment()
        assert env.parent is None
        assert env.bindings == {}

    def test_bind_immutable_variable(self):
        """Test binding an immutable variable."""
        env = Environment()
        env.bind("x", 42, mutable=False)
        assert env.lookup("x") == 42

    def test_bind_mutable_variable(self):
        """Test binding a mutable variable."""
        env = Environment()
        env.bind("x", 42, mutable=True)
        assert env.lookup("x") == 42

    def test_rebind_raises_error(self):
        """Test that rebinding in same scope raises error."""
        env = Environment()
        env.bind("x", 42)
        with pytest.raises(EnvironmentError, match="already bound"):
            env.bind("x", 100)

    def test_lookup_undefined_variable(self):
        """Test lookup of undefined variable raises error."""
        env = Environment()
        with pytest.raises(EnvironmentError, match="not found"):
            env.lookup("x")


class TestEnvironmentAssignment:
    """Test variable assignment (mutation)."""

    def test_assign_mutable_variable(self):
        """Test assigning to mutable variable."""
        env = Environment()
        env.bind("x", 42, mutable=True)
        env.assign("x", 100)
        assert env.lookup("x") == 100

    def test_assign_immutable_variable_raises_error(self):
        """Test assigning to immutable variable raises error."""
        env = Environment()
        env.bind("x", 42, mutable=False)
        with pytest.raises(EnvironmentError, match="not mutable"):
            env.assign("x", 100)

    def test_assign_undefined_variable_raises_error(self):
        """Test assigning to undefined variable raises error."""
        env = Environment()
        with pytest.raises(EnvironmentError, match="not found"):
            env.assign("x", 100)

    def test_assign_mutable_variable_multiple_times(self):
        """Test multiple assignments to mutable variable."""
        env = Environment()
        env.bind("x", 42, mutable=True)
        env.assign("x", 100)
        assert env.lookup("x") == 100
        env.assign("x", 200)
        assert env.lookup("x") == 200


class TestEnvironmentNesting:
    """Test nested scopes."""

    def test_extend_creates_child_scope(self):
        """Test that extend creates a child scope."""
        parent = Environment()
        parent.bind("x", 42)
        child = parent.extend()
        assert child.parent is parent
        assert child.lookup("x") == 42

    def test_child_can_shadow_parent(self):
        """Test that child scope can shadow parent variable."""
        parent = Environment()
        parent.bind("x", 42)
        child = parent.extend()
        child.bind("x", 100)
        assert parent.lookup("x") == 42
        assert child.lookup("x") == 100

    def test_lookup_traverses_parent_chain(self):
        """Test that lookup traverses parent chain."""
        grandparent = Environment()
        grandparent.bind("x", 1)
        parent = grandparent.extend()
        parent.bind("y", 2)
        child = parent.extend()
        child.bind("z", 3)

        assert child.lookup("z") == 3
        assert child.lookup("y") == 2
        assert child.lookup("x") == 1

    def test_bind_in_child_does_not_affect_parent(self):
        """Test that binding in child doesn't affect parent."""
        parent = Environment()
        child = parent.extend()
        child.bind("x", 42)

        assert child.lookup("x") == 42
        with pytest.raises(EnvironmentError, match="not found"):
            parent.lookup("x")

    def test_assign_in_child_updates_parent_variable(self):
        """Test that assignment in child updates parent variable."""
        parent = Environment()
        parent.bind("x", 42, mutable=True)
        child = parent.extend()
        child.assign("x", 100)

        assert parent.lookup("x") == 100
        assert child.lookup("x") == 100

    def test_assign_shadowed_variable_updates_child_only(self):
        """Test that assigning shadowed variable updates child only."""
        parent = Environment()
        parent.bind("x", 42, mutable=True)
        child = parent.extend()
        child.bind("x", 100, mutable=True)

        child.assign("x", 200)
        assert parent.lookup("x") == 42
        assert child.lookup("x") == 200

    def test_cannot_assign_to_immutable_parent_from_child(self):
        """Test that child cannot assign to immutable parent variable."""
        parent = Environment()
        parent.bind("x", 42, mutable=False)
        child = parent.extend()

        with pytest.raises(EnvironmentError, match="not mutable"):
            child.assign("x", 100)

    def test_deep_nesting(self):
        """Test deeply nested scopes."""
        env = Environment()
        env.bind("x", 0)

        # Create 10 levels of nesting
        for i in range(1, 11):
            env = env.extend()
            env.bind(f"var{i}", i)

        # Lookup should work at any level
        assert env.lookup("x") == 0
        for i in range(1, 11):
            assert env.lookup(f"var{i}") == i


class TestEnvironmentEdgeCases:
    """Test edge cases."""

    def test_bind_various_types(self):
        """Test binding various Python types."""
        env = Environment()
        env.bind("null", None)
        env.bind("bool", True)
        env.bind("int", 42)
        env.bind("float", 3.14)
        env.bind("str", "hello")
        env.bind("list", [1, 2, 3])
        env.bind("dict", {"a": 1})

        assert env.lookup("null") is None
        assert env.lookup("bool") is True
        assert env.lookup("int") == 42
        assert env.lookup("float") == 3.14
        assert env.lookup("str") == "hello"
        assert env.lookup("list") == [1, 2, 3]
        assert env.lookup("dict") == {"a": 1}

    def test_variable_names_can_be_keywords(self):
        """Test that variable names can be Python keywords."""
        env = Environment()
        # These are Python keywords but valid East identifiers
        env.bind("class", 1)
        env.bind("def", 2)
        env.bind("if", 3)

        assert env.lookup("class") == 1
        assert env.lookup("def") == 2
        assert env.lookup("if") == 3

    def test_mutable_binding_remains_mutable_after_assign(self):
        """Test that mutable binding stays mutable."""
        env = Environment()
        env.bind("x", 1, mutable=True)
        env.assign("x", 2)
        env.assign("x", 3)
        assert env.lookup("x") == 3
