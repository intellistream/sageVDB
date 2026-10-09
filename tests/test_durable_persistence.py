from __future__ import annotations

import json
from pathlib import Path

import pytest

from sagevdb.persistence import PersistentRecord, PersistentSageVDB, stable_content_hash


class FakeDatabase:
    def __init__(self) -> None:
        self.items: dict[int, dict[str, object]] = {}
        self.next_id = 0
        self.build_count = 0

    def add(self, vector):
        vector_id = self.next_id
        self.next_id += 1
        self.items[vector_id] = {"vector": list(vector), "metadata": {}}
        return vector_id

    def set_metadata(self, vector_id, metadata):
        self.items[vector_id]["metadata"] = dict(metadata)

    def remove(self, vector_id):
        return self.items.pop(vector_id, None) is not None

    def build_index(self):
        self.build_count += 1

    def save(self, base_path):
        base = Path(base_path)
        payload = {"items": self.items, "next_id": self.next_id}
        base.with_suffix(".vectors").write_text(json.dumps(payload), encoding="utf-8")
        base.with_suffix(".metadata").write_text("metadata", encoding="utf-8")
        base.with_suffix(".config").write_text("config", encoding="utf-8")

    def load(self, base_path):
        payload = json.loads(Path(base_path).with_suffix(".vectors").read_text(encoding="utf-8"))
        self.items = {int(key): value for key, value in payload["items"].items()}
        self.next_id = int(payload["next_id"])


def record(key: str, content: str, *, title: str | None = None) -> PersistentRecord:
    return PersistentRecord(
        key=key,
        content_hash=stable_content_hash(content),
        metadata={"title": title or key},
    )


def test_restart_loads_without_embedding_unchanged_records(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def embed(records):
        calls.append([item.key for item in records])
        return [[float(len(item.key)), 1.0] for item in records]

    first = PersistentSageVDB(tmp_path, FakeDatabase, compatibility={"model": "demo", "dim": 2})
    assert first.open().loaded is False
    result = first.sync([record("a", "alpha"), record("b", "beta")], embed)
    assert result.embedded == 2
    assert calls == [["a", "b"]]

    second = PersistentSageVDB(tmp_path, FakeDatabase, compatibility={"model": "demo", "dim": 2})
    opened = second.open()
    assert opened.loaded is True
    calls.clear()
    result = second.sync([record("a", "alpha"), record("b", "beta")], embed)
    assert result.committed is False
    assert result.embedded == 0
    assert calls == []


def test_incremental_sync_embeds_only_added_and_changed_records(tmp_path: Path) -> None:
    store = PersistentSageVDB(tmp_path, FakeDatabase, compatibility={"model": "demo"})
    store.open()
    store.sync(
        [record("a", "alpha"), record("b", "beta")],
        lambda records: [[1.0, float(index)] for index, _ in enumerate(records)],
    )

    embedded: list[str] = []

    def embed(records):
        embedded.extend(item.key for item in records)
        return [[2.0, float(index)] for index, _ in enumerate(records)]

    result = store.sync([record("b", "beta-2"), record("c", "gamma")], embed)
    assert result.added == 1
    assert result.updated == 1
    assert result.deleted == 1
    assert result.embedded == 2
    assert embedded == ["c", "b"]
    assert set(store.vector_ids) == {"b", "c"}


def test_metadata_only_change_does_not_reembed(tmp_path: Path) -> None:
    store = PersistentSageVDB(tmp_path, FakeDatabase, compatibility={"model": "demo"})
    store.open()
    store.sync([record("a", "alpha", title="old")], lambda records: [[1.0, 2.0]])

    result = store.sync(
        [record("a", "alpha", title="new")],
        lambda records: pytest.fail("metadata-only sync must not embed"),
    )
    assert result.committed is True
    assert result.embedded == 0


def test_corrupt_current_recovers_previous_generation(tmp_path: Path) -> None:
    store = PersistentSageVDB(
        tmp_path,
        FakeDatabase,
        compatibility={"model": "demo"},
        keep_generations=3,
    )
    store.open()
    first = store.sync([record("a", "alpha")], lambda records: [[1.0, 0.0]])
    second = store.sync([record("a", "alpha-2")], lambda records: [[2.0, 0.0]])
    assert first.generation != second.generation
    current_dir = tmp_path / f"generation-{second.generation}"
    (current_dir / "index.vectors").write_text("corrupt", encoding="utf-8")

    recovered = PersistentSageVDB(
        tmp_path,
        FakeDatabase,
        compatibility={"model": "demo"},
        keep_generations=3,
    )
    opened = recovered.open()
    assert opened.loaded is True
    assert opened.recovered is True
    assert opened.generation == first.generation


def test_incompatible_fingerprint_requires_rebuild(tmp_path: Path) -> None:
    original = PersistentSageVDB(tmp_path, FakeDatabase, compatibility={"model": "v1"})
    original.open()
    original.sync([record("a", "alpha")], lambda records: [[1.0, 0.0]])

    incompatible = PersistentSageVDB(tmp_path, FakeDatabase, compatibility={"model": "v2"})
    opened = incompatible.open()
    assert opened.loaded is False
    assert "incompatible index fingerprint" in opened.reason


def test_embedding_failure_preserves_previous_generation(tmp_path: Path) -> None:
    store = PersistentSageVDB(tmp_path, FakeDatabase, compatibility={"model": "demo"})
    store.open()
    initial = store.sync([record("a", "alpha")], lambda records: [[1.0, 0.0]])

    with pytest.raises(RuntimeError, match="embedding failed"):
        store.sync(
            [record("a", "changed")],
            lambda records: (_ for _ in ()).throw(RuntimeError("embedding failed")),
        )

    assert store.generation == initial.generation
    restarted = PersistentSageVDB(tmp_path, FakeDatabase, compatibility={"model": "demo"})
    assert restarted.open().generation == initial.generation
