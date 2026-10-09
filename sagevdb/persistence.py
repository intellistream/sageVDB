"""Durable, generation-based persistence for SageVDB indexes.

The native ``SageVDB.save`` API writes several sidecar files.  This module
turns those files into an atomic index generation with a checksummed manifest,
an atomic ``CURRENT`` pointer, last-known-good recovery, and document-level
incremental synchronization.

Applications provide stable record keys and content hashes.  Embeddings are
requested only for records that are new or whose content hash changed.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

MANIFEST_SCHEMA_VERSION = 1
_MANIFEST_NAME = "manifest.json"
_CURRENT_NAME = "CURRENT"
_DATABASE_BASENAME = "index"
_RESERVED_KEY = "__sagevdb_record_key__"
_RESERVED_HASH = "__sagevdb_content_hash__"


class PersistenceError(RuntimeError):
    """Raised when no compatible, valid persistent generation can be used."""


@dataclass(frozen=True)
class PersistentRecord:
    """A stable logical record tracked by a persistent SageVDB index."""

    key: str
    content_hash: str
    metadata: Mapping[str, str]


@dataclass(frozen=True)
class OpenResult:
    """Outcome of opening an existing persistent index."""

    loaded: bool
    recovered: bool
    generation: str | None
    reason: str


@dataclass(frozen=True)
class SyncResult:
    """Incremental synchronization summary."""

    added: int
    updated: int
    deleted: int
    embedded: int
    generation: str | None
    committed: bool


def stable_content_hash(*parts: str) -> str:
    """Return an unambiguous SHA-256 digest for ordered text parts."""

    digest = hashlib.sha256()
    for part in parts:
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


class PersistentSageVDB:
    """Own the durable lifecycle of one SageVDB database.

    ``database_factory`` must return a fresh database exposing ``save``,
    ``load``, ``add``, ``remove``, ``set_metadata``, and ``build_index``.
    Keeping construction injectable lets the lifecycle work with native
    SageVDB today and other SageVDB backends once they implement that contract.
    """

    def __init__(
        self,
        root: str | Path,
        database_factory: Callable[[], Any],
        *,
        compatibility: Mapping[str, Any],
        keep_generations: int = 2,
    ) -> None:
        if keep_generations < 2:
            raise ValueError("keep_generations must be at least 2")
        self.root = Path(root)
        self._database_factory = database_factory
        self._compatibility = _canonical_json_value(dict(compatibility))
        self._keep_generations = keep_generations
        self._database = database_factory()
        self._records: dict[str, dict[str, Any]] = {}
        self._generation: str | None = None
        self._loaded = False
        self._thread_lock = threading.RLock()

    @property
    def database(self) -> Any:
        return self._database

    @property
    def generation(self) -> str | None:
        return self._generation

    @property
    def vector_ids(self) -> dict[str, int]:
        return {key: int(value["vector_id"]) for key, value in self._records.items()}

    def open(self) -> OpenResult:
        """Open CURRENT or recover the newest compatible valid generation."""

        with self._thread_lock, self._file_lock(exclusive=False):
            candidates = self._candidate_generations()
            current = self._read_current()
            errors: list[str] = []
            for generation in candidates:
                try:
                    manifest = self._validate_generation(generation)
                    database = self._database_factory()
                    database.load(str(self._generation_dir(generation) / _DATABASE_BASENAME))
                    self._database = database
                    self._records = {
                        str(key): dict(value)
                        for key, value in dict(manifest.get("records", {})).items()
                    }
                    self._generation = generation
                    self._loaded = True
                    recovered = current is not None and generation != current
                    return OpenResult(
                        loaded=True,
                        recovered=recovered,
                        generation=generation,
                        reason="recovered_last_known_good" if recovered else "loaded_current",
                    )
                except (OSError, ValueError, TypeError, RuntimeError, PersistenceError) as exc:
                    errors.append(f"{generation}: {exc}")

            self._database = self._database_factory()
            self._records = {}
            self._generation = None
            self._loaded = True
            reason = "no_generation" if not candidates else "no_compatible_valid_generation"
            if errors:
                reason = f"{reason}: {'; '.join(errors)}"
            return OpenResult(False, False, None, reason)

    def sync(
        self,
        records: Sequence[PersistentRecord],
        vector_provider: Callable[[Sequence[PersistentRecord]], Sequence[Sequence[float]]],
    ) -> SyncResult:
        """Synchronize logical records and atomically commit a new generation.

        The provider is invoked once, before mutation, and only for added or
        changed records.  If embedding fails, the in-memory and durable indexes
        remain on the previous generation.
        """

        with self._thread_lock:
            if not self._loaded:
                self.open()

        with self._thread_lock, self._file_lock(exclusive=True):
            current = self._read_current()
            if current is not None and current != self._generation:
                try:
                    manifest = self._validate_generation(current)
                except PersistenceError:
                    if self._generation is not None:
                        raise
                else:
                    database = self._database_factory()
                    database.load(str(self._generation_dir(current) / _DATABASE_BASENAME))
                    self._database = database
                    self._records = {
                        str(key): dict(value)
                        for key, value in dict(manifest.get("records", {})).items()
                    }
                    self._generation = current

            desired: dict[str, PersistentRecord] = {}
            for record in records:
                if not record.key:
                    raise ValueError("record key must not be empty")
                if not record.content_hash:
                    raise ValueError(f"content_hash must not be empty for {record.key!r}")
                if record.key in desired:
                    raise ValueError(f"duplicate record key: {record.key!r}")
                desired[record.key] = record

            existing_keys = set(self._records)
            desired_keys = set(desired)
            deleted_keys = sorted(existing_keys - desired_keys)
            added_keys = sorted(desired_keys - existing_keys)
            updated_keys = sorted(
                key
                for key in existing_keys & desired_keys
                if str(self._records[key].get("content_hash")) != desired[key].content_hash
            )
            embed_keys = added_keys + updated_keys
            to_embed = [desired[key] for key in embed_keys]
            vectors = list(vector_provider(to_embed)) if to_embed else []
            if len(vectors) != len(to_embed):
                raise ValueError(
                    f"vector_provider returned {len(vectors)} vectors for {len(to_embed)} records"
                )

            metadata_only_changed = any(
                _canonical_json_value(dict(desired[key].metadata))
                != _canonical_json_value(dict(self._records[key].get("metadata", {})))
                for key in (existing_keys & desired_keys) - set(updated_keys)
            )
            needs_commit = bool(embed_keys or deleted_keys or metadata_only_changed)
            if not needs_commit and self._generation is not None:
                return SyncResult(0, 0, 0, 0, self._generation, False)

            previous_generation = self._generation
            try:
                for key in deleted_keys + updated_keys:
                    vector_id = int(self._records[key]["vector_id"])
                    if not self._database.remove(vector_id):
                        raise PersistenceError(
                            f"database refused to remove vector {vector_id} for record {key!r}"
                        )
                    self._records.pop(key, None)

                vector_by_key = dict(zip(embed_keys, vectors))
                for key in embed_keys:
                    record = desired[key]
                    metadata = {str(k): str(v) for k, v in record.metadata.items()}
                    metadata[_RESERVED_KEY] = record.key
                    metadata[_RESERVED_HASH] = record.content_hash
                    vector_id = int(self._database.add(list(vector_by_key[key])))
                    self._database.set_metadata(vector_id, metadata)
                    self._records[key] = {
                        "content_hash": record.content_hash,
                        "vector_id": vector_id,
                        "metadata": _canonical_json_value(dict(record.metadata)),
                    }

                # Metadata-only changes retain the vector and update metadata in place.
                for key in sorted((existing_keys & desired_keys) - set(updated_keys)):
                    record = desired[key]
                    old_metadata = _canonical_json_value(
                        dict(self._records[key].get("metadata", {}))
                    )
                    new_metadata = _canonical_json_value(dict(record.metadata))
                    if old_metadata == new_metadata:
                        continue
                    vector_id = int(self._records[key]["vector_id"])
                    metadata = {str(k): str(v) for k, v in record.metadata.items()}
                    metadata[_RESERVED_KEY] = record.key
                    metadata[_RESERVED_HASH] = record.content_hash
                    self._database.set_metadata(vector_id, metadata)
                    self._records[key]["metadata"] = new_metadata

                self._database.build_index()
                generation = self._commit_generation()
            except Exception:
                self._restore_generation(previous_generation)
                raise

            return SyncResult(
                added=len(added_keys),
                updated=len(updated_keys),
                deleted=len(deleted_keys),
                embedded=len(embed_keys),
                generation=generation,
                committed=True,
            )

    def _commit_generation(self) -> str:
        self.root.mkdir(parents=True, exist_ok=True)
        generation = f"{time.time_ns():020d}-{uuid.uuid4().hex}"
        staging = self.root / f".staging-{generation}"
        final = self._generation_dir(generation)
        staging.mkdir(mode=0o700)
        try:
            database_base = staging / _DATABASE_BASENAME
            self._database.save(str(database_base))
            files = self._collect_files(staging)
            manifest = {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "generation": generation,
                "compatibility": self._compatibility,
                "records": self._records,
                "files": files,
            }
            manifest_path = staging / _MANIFEST_NAME
            _write_json_fsynced(manifest_path, manifest)
            _fsync_directory(staging)
            os.replace(staging, final)
            _fsync_directory(self.root)
            self._replace_current(generation)
            self._generation = generation
            self._loaded = True
            self._prune_generations()
            return generation
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def _restore_generation(self, generation: str | None) -> None:
        self._database = self._database_factory()
        self._records = {}
        self._generation = None
        if generation is None:
            return
        manifest = self._validate_generation(generation)
        self._database.load(str(self._generation_dir(generation) / _DATABASE_BASENAME))
        self._records = {
            str(key): dict(value) for key, value in dict(manifest.get("records", {})).items()
        }
        self._generation = generation

    def _validate_generation(self, generation: str) -> dict[str, Any]:
        if not generation or Path(generation).name != generation:
            raise PersistenceError("invalid generation name")
        directory = self._generation_dir(generation)
        manifest_path = directory / _MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise PersistenceError("unsupported manifest schema")
        if manifest.get("generation") != generation:
            raise PersistenceError("generation name does not match manifest")
        if _canonical_json_value(manifest.get("compatibility")) != self._compatibility:
            raise PersistenceError("incompatible index fingerprint")
        files = manifest.get("files")
        if not isinstance(files, dict) or not files:
            raise PersistenceError("manifest contains no database files")
        for relative_name, expected in files.items():
            path = directory / str(relative_name)
            if path.parent != directory or not path.is_file():
                raise PersistenceError(f"missing database file: {relative_name}")
            if int(expected.get("size", -1)) != path.stat().st_size:
                raise PersistenceError(f"size mismatch: {relative_name}")
            if str(expected.get("sha256")) != _sha256_file(path):
                raise PersistenceError(f"checksum mismatch: {relative_name}")
        records = manifest.get("records")
        if not isinstance(records, dict):
            raise PersistenceError("manifest records must be an object")
        return manifest

    def _collect_files(self, directory: Path) -> dict[str, dict[str, Any]]:
        files: dict[str, dict[str, Any]] = {}
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.name == _MANIFEST_NAME:
                continue
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
            files[path.name] = {"size": path.stat().st_size, "sha256": _sha256_file(path)}
        if not files:
            raise PersistenceError("SageVDB save produced no files")
        return files

    def _candidate_generations(self) -> list[str]:
        if not self.root.exists():
            return []
        current = self._read_current()
        generations = sorted(
            (
                path.name[len("generation-") :]
                for path in self.root.iterdir()
                if path.is_dir() and path.name.startswith("generation-")
            ),
            reverse=True,
        )
        if current in generations:
            generations.remove(current)
            generations.insert(0, current)
        return generations

    def _read_current(self) -> str | None:
        path = self.root / _CURRENT_NAME
        if not path.is_file():
            return None
        value = path.read_text(encoding="utf-8").strip()
        return value or None

    def _replace_current(self, generation: str) -> None:
        temporary = self.root / f".{_CURRENT_NAME}-{uuid.uuid4().hex}"
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(f"{generation}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.root / _CURRENT_NAME)
        _fsync_directory(self.root)

    def _generation_dir(self, generation: str) -> Path:
        return self.root / f"generation-{generation}"

    def _prune_generations(self) -> None:
        keep = set(self._candidate_generations()[: self._keep_generations])
        for path in self.root.iterdir():
            if not path.is_dir() or not path.name.startswith("generation-"):
                continue
            generation = path.name[len("generation-") :]
            if generation not in keep:
                shutil.rmtree(path)

    @contextmanager
    def _file_lock(self, *, exclusive: bool) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        lock_path = self.root / ".lock"
        with lock_path.open("a+b") as handle:
            try:
                import fcntl
            except ImportError:
                # SageVDB production deployments are Linux.  Keep the API
                # usable on platforms without fcntl while retaining thread safety.
                yield
                return
            operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(handle.fileno(), operation)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _canonical_json_value(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _write_json_fsynced(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
