"""Tests for linked_memory_ids encode/decode helpers and entity-linking paths.

Covers:
- _encode_linked_ids: deterministic JSON serialization for scalar-only metadata
- _decode_linked_ids: backward-compatible deserialization (str, list, None, malformed)
- Entity upsert with encoded linked_memory_ids
- Entity cleanup (_remove_memory_from_entity_store)
- Entity boost computation (_compute_entity_boosts)
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from mem0.memory.main import _decode_linked_ids, _encode_linked_ids


# ---------------------------------------------------------------------------
# Unit tests for _encode_linked_ids
# ---------------------------------------------------------------------------


class TestEncodeLinkedIds:
    def test_encode_list(self):
        result = _encode_linked_ids(["id1", "id2"])
        assert result == '["id1", "id2"]'
        assert isinstance(result, str)

    def test_encode_empty_list(self):
        result = _encode_linked_ids([])
        assert result == "[]"

    def test_encode_single_item(self):
        result = _encode_linked_ids(["abc-123"])
        assert result == '["abc-123"]'

    def test_encode_set_is_sorted(self):
        result = _encode_linked_ids({"z", "a", "m"})
        decoded = json.loads(result)
        assert decoded == ["a", "m", "z"]

    def test_encode_produces_scalar_string(self):
        """The whole point: output must be a plain str, not a list."""
        result = _encode_linked_ids(["id1", "id2", "id3"])
        assert isinstance(result, str)
        # Should be valid JSON
        parsed = json.loads(result)
        assert isinstance(parsed, list)

    def test_encode_preserves_order_for_list(self):
        result = _encode_linked_ids(["c", "a", "b"])
        assert json.loads(result) == ["c", "a", "b"]


# ---------------------------------------------------------------------------
# Unit tests for _decode_linked_ids
# ---------------------------------------------------------------------------


class TestDecodeLinkedIds:
    def test_decode_json_string(self):
        encoded = '["id1", "id2"]'
        assert _decode_linked_ids(encoded) == ["id1", "id2"]

    def test_decode_empty_json_array(self):
        assert _decode_linked_ids("[]") == []

    def test_decode_passthrough_list(self):
        """Backward compat: if value is already a list, return it as-is."""
        assert _decode_linked_ids(["id1", "id2"]) == ["id1", "id2"]

    def test_decode_empty_list(self):
        assert _decode_linked_ids([]) == []

    def test_decode_none(self):
        assert _decode_linked_ids(None) == []

    def test_decode_empty_string(self):
        assert _decode_linked_ids("") == []

    def test_decode_malformed_json(self):
        """Malformed JSON degrades to single-element list if non-empty."""
        result = _decode_linked_ids("{not valid json")
        assert result == ["{not valid json"]

    def test_decode_plain_string_id(self):
        """A plain string (not JSON array) is treated as a single ID."""
        result = _decode_linked_ids("some-uuid-value")
        assert result == ["some-uuid-value"]

    def test_decode_json_non_list(self):
        """JSON that decodes to a non-list (e.g. dict) falls back gracefully."""
        result = _decode_linked_ids('{"key": "val"}')
        # Not a list, so falls through to [value] branch
        assert result == ['{"key": "val"}']

    def test_decode_integer_fallback(self):
        """Non-str, non-list input returns empty list."""
        assert _decode_linked_ids(42) == []

    def test_roundtrip(self):
        """Encode then decode should be identity."""
        original = ["mem-001", "mem-002", "mem-003"]
        assert _decode_linked_ids(_encode_linked_ids(original)) == original

    def test_roundtrip_set(self):
        """Set encode -> decode produces sorted list."""
        original = {"z-id", "a-id", "m-id"}
        result = _decode_linked_ids(_encode_linked_ids(original))
        assert result == sorted(original)


# ---------------------------------------------------------------------------
# Integration-style tests for entity linking paths
# ---------------------------------------------------------------------------


class MockMatch:
    """Simulates a vector store search/list result row."""

    def __init__(self, match_id, payload, score=0.99):
        self.id = match_id
        self.payload = payload
        self.score = score


def _make_memory_instance():
    """Create a Memory instance with mocked internals for entity-linking tests."""
    from mem0 import Memory

    with patch.object(Memory, "__init__", return_value=None):
        m = Memory()
    m.embedding_model = MagicMock()
    m.embedding_model.embed = MagicMock(return_value=[0.1, 0.2, 0.3])
    m._entity_store = MagicMock()
    return m


class TestEntityUpsert:
    """Tests _upsert_entity writes scalar metadata."""

    @pytest.fixture
    def memory_instance(self):
        yield _make_memory_instance()

    def test_upsert_new_entity_writes_scalar(self, memory_instance):
        """When no matching entity exists, insert payload must have str linked_memory_ids."""
        memory_instance.entity_store.search = MagicMock(return_value=[])

        memory_instance._upsert_entity(
            entity_text="John Doe",
            entity_type="person",
            memory_id="mem-001",
            filters={"user_id": "u1"},
        )

        call_args = memory_instance.entity_store.insert.call_args
        payload = call_args[1]["payloads"][0] if "payloads" in (call_args[1] or {}) else call_args[0][2][0] if len(call_args[0]) > 2 else call_args[1].get("payloads", [{}])[0]
        linked = payload["linked_memory_ids"]
        assert isinstance(linked, str), f"Expected str metadata, got {type(linked)}"
        assert json.loads(linked) == ["mem-001"]

    def test_upsert_existing_entity_appends_and_stays_scalar(self, memory_instance):
        """When a matching entity exists, update must keep linked_memory_ids as str."""
        existing_payload = {
            "data": "John Doe",
            "entity_type": "person",
            "linked_memory_ids": _encode_linked_ids(["mem-001"]),
            "user_id": "u1",
        }
        existing_match = MockMatch("ent-1", existing_payload, score=0.99)
        memory_instance.entity_store.search = MagicMock(return_value=[existing_match])

        memory_instance._upsert_entity(
            entity_text="John Doe",
            entity_type="person",
            memory_id="mem-002",
            filters={"user_id": "u1"},
        )

        call_args = memory_instance.entity_store.update.call_args
        updated_payload = call_args[1].get("payload") or call_args[0][2]
        linked = updated_payload["linked_memory_ids"]
        assert isinstance(linked, str)
        assert json.loads(linked) == ["mem-001", "mem-002"]

    def test_upsert_existing_entity_with_legacy_list_payload(self, memory_instance):
        """Backward compat: existing entity has list payload (pre-fix), update still works."""
        existing_payload = {
            "data": "John Doe",
            "entity_type": "person",
            "linked_memory_ids": ["mem-001"],  # legacy list format
            "user_id": "u1",
        }
        existing_match = MockMatch("ent-1", existing_payload, score=0.99)
        memory_instance.entity_store.search = MagicMock(return_value=[existing_match])

        memory_instance._upsert_entity(
            entity_text="John Doe",
            entity_type="person",
            memory_id="mem-002",
            filters={"user_id": "u1"},
        )

        call_args = memory_instance.entity_store.update.call_args
        updated_payload = call_args[1].get("payload") or call_args[0][2]
        linked = updated_payload["linked_memory_ids"]
        assert isinstance(linked, str), "After update, legacy list should be converted to str"
        assert json.loads(linked) == ["mem-001", "mem-002"]

    def test_upsert_no_duplicate_ids(self, memory_instance):
        """If memory_id already present, it should not be duplicated."""
        existing_payload = {
            "data": "John Doe",
            "entity_type": "person",
            "linked_memory_ids": _encode_linked_ids(["mem-001"]),
            "user_id": "u1",
        }
        existing_match = MockMatch("ent-1", existing_payload, score=0.99)
        memory_instance.entity_store.search = MagicMock(return_value=[existing_match])

        memory_instance._upsert_entity(
            entity_text="John Doe",
            entity_type="person",
            memory_id="mem-001",  # same id
            filters={"user_id": "u1"},
        )

        # Should not call update since id is already present
        memory_instance.entity_store.update.assert_not_called()


class TestRemoveMemoryFromEntityStore:
    """Tests _remove_memory_from_entity_store cleanup path."""

    @pytest.fixture
    def memory_instance(self):
        yield _make_memory_instance()

    def test_remove_last_id_deletes_entity(self, memory_instance):
        """When removing the only linked memory, the entity record should be deleted."""
        row = MockMatch("ent-1", {
            "data": "Alice",
            "linked_memory_ids": _encode_linked_ids(["mem-001"]),
        })
        memory_instance.entity_store.list = MagicMock(return_value=[[row]])

        memory_instance._remove_memory_from_entity_store("mem-001", {"user_id": "u1"})

        memory_instance.entity_store.delete.assert_called_once_with(vector_id="ent-1")

    def test_remove_one_of_many_updates_entity(self, memory_instance):
        """When removing one of several linked memories, entity is updated (not deleted)."""
        row = MockMatch("ent-1", {
            "data": "Alice",
            "linked_memory_ids": _encode_linked_ids(["mem-001", "mem-002", "mem-003"]),
        })
        memory_instance.entity_store.list = MagicMock(return_value=[[row]])

        memory_instance._remove_memory_from_entity_store("mem-002", {"user_id": "u1"})

        memory_instance.entity_store.delete.assert_not_called()
        call_args = memory_instance.entity_store.update.call_args
        updated_payload = call_args[1].get("payload") or call_args[0][2]
        linked = updated_payload["linked_memory_ids"]
        assert isinstance(linked, str)
        assert json.loads(linked) == ["mem-001", "mem-003"]

    def test_remove_with_legacy_list_payload(self, memory_instance):
        """Backward compat: entity has legacy list payload, cleanup still works."""
        row = MockMatch("ent-1", {
            "data": "Alice",
            "linked_memory_ids": ["mem-001", "mem-002"],  # legacy
        })
        memory_instance.entity_store.list = MagicMock(return_value=[[row]])

        memory_instance._remove_memory_from_entity_store("mem-001", {"user_id": "u1"})

        call_args = memory_instance.entity_store.update.call_args
        updated_payload = call_args[1].get("payload") or call_args[0][2]
        linked = updated_payload["linked_memory_ids"]
        assert isinstance(linked, str), "Cleanup should convert legacy list to str"
        assert json.loads(linked) == ["mem-002"]

    def test_remove_nonexistent_id_is_noop(self, memory_instance):
        """Removing an id that doesn't exist in any entity is a no-op."""
        row = MockMatch("ent-1", {
            "data": "Alice",
            "linked_memory_ids": _encode_linked_ids(["mem-001"]),
        })
        memory_instance.entity_store.list = MagicMock(return_value=[[row]])

        memory_instance._remove_memory_from_entity_store("mem-999", {"user_id": "u1"})

        memory_instance.entity_store.delete.assert_not_called()
        memory_instance.entity_store.update.assert_not_called()


class TestComputeEntityBoosts:
    """Tests _compute_entity_boosts with encoded payloads."""

    @pytest.fixture
    def memory_instance(self):
        yield _make_memory_instance()

    def test_boost_with_encoded_payload(self, memory_instance):
        """Entity boost reads encoded linked_memory_ids correctly."""
        match = MockMatch("ent-1", {
            "data": "Alice",
            "linked_memory_ids": _encode_linked_ids(["mem-001", "mem-002"]),
        }, score=0.9)
        memory_instance.entity_store.search = MagicMock(return_value=[match])

        boosts = memory_instance._compute_entity_boosts(
            query_entities=[("person", "Alice")],
            filters={"user_id": "u1"},
        )

        assert "mem-001" in boosts
        assert "mem-002" in boosts
        assert all(isinstance(v, float) and v > 0 for v in boosts.values())

    def test_boost_with_legacy_list_payload(self, memory_instance):
        """Backward compat: entity with legacy list payload still produces boosts."""
        match = MockMatch("ent-1", {
            "data": "Alice",
            "linked_memory_ids": ["mem-001"],  # legacy
        }, score=0.9)
        memory_instance.entity_store.search = MagicMock(return_value=[match])

        boosts = memory_instance._compute_entity_boosts(
            query_entities=[("person", "Alice")],
            filters={"user_id": "u1"},
        )

        assert "mem-001" in boosts

    def test_boost_with_empty_linked_ids_skipped(self, memory_instance):
        """Entities with empty linked_memory_ids are skipped."""
        match = MockMatch("ent-1", {
            "data": "Alice",
            "linked_memory_ids": _encode_linked_ids([]),
        }, score=0.9)
        memory_instance.entity_store.search = MagicMock(return_value=[match])

        boosts = memory_instance._compute_entity_boosts(
            query_entities=[("person", "Alice")],
            filters={"user_id": "u1"},
        )

        assert boosts == {}

    def test_boost_low_similarity_ignored(self, memory_instance):
        """Entities with similarity < 0.5 should be ignored."""
        match = MockMatch("ent-1", {
            "data": "Alice",
            "linked_memory_ids": _encode_linked_ids(["mem-001"]),
        }, score=0.3)
        memory_instance.entity_store.search = MagicMock(return_value=[match])

        boosts = memory_instance._compute_entity_boosts(
            query_entities=[("person", "Alice")],
            filters={"user_id": "u1"},
        )

        assert boosts == {}

    def test_boost_attenuation_many_linked(self, memory_instance):
        """Entities linking many memories get attenuated boost."""
        many_ids = [f"mem-{i:04d}" for i in range(100)]
        match_many = MockMatch("ent-1", {
            "data": "Alice",
            "linked_memory_ids": _encode_linked_ids(many_ids),
        }, score=0.9)
        few_ids = ["mem-single"]
        match_few = MockMatch("ent-2", {
            "data": "Bob",
            "linked_memory_ids": _encode_linked_ids(few_ids),
        }, score=0.9)
        memory_instance.entity_store.search = MagicMock(return_value=[match_many, match_few])

        boosts = memory_instance._compute_entity_boosts(
            query_entities=[("person", "Alice")],
            filters={"user_id": "u1"},
        )

        # Bob's single-linked memory should have higher boost than Alice's individual ones
        bob_boost = boosts.get("mem-single", 0)
        alice_boost_avg = sum(boosts.get(mid, 0) for mid in many_ids[:5]) / 5
        assert bob_boost > alice_boost_avg, "Single-linked entity should have higher per-memory boost"
