"""Test suite for Letta (MemGPT) to Memanto OKF Migration Adapter.

Verifies:
1. Valid conversion of archival memory passages into OKF documents.
2. Core memory persona/human blocks mapped to OKF metadata tags and documents.
3. Recall memory dialogue converted to contextual OKF records.
4. Clean handling of missing, empty, or malformed fields and corrupt files.
5. Directory and JSON bundle export fidelity and full round-trip compatibility
   with Memanto's native `load_okf_bundle` and `map_okf`.
6. Command-line interface (CLI) behavior and exit codes.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

# Ensure the adapter directory is in sys.path
_ADAPTER_DIR = Path(__file__).resolve().parent
if str(_ADAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(_ADAPTER_DIR))

from adapter import (  # noqa: E402
    OKF_VERSION,
    LettaOKFAdapter,
    MigrationError,
    _sanitize_tags,
    _slugify,
    convert_letta_to_okf_bundle,
    extract_core_metadata_tags,
    load_letta_export,
    main,
    normalize_timestamp,
)
from memanto.cli.migrate.mappers import map_okf
from memanto.cli.migrate.okf_loader import load_okf_bundle

FIXTURES_PATH = _ADAPTER_DIR / "fixtures" / "sample_letta_export.json"


@pytest.fixture
def sample_export_data() -> dict[str, Any]:
    """Load the realistic sample Letta export fixture."""
    return load_letta_export(FIXTURES_PATH)


class TestLettaSampleFixture:
    """Verifies conversion of the realistic Letta export fixture."""

    def test_fixture_file_exists_and_loads(self, sample_export_data: dict[str, Any]):
        assert isinstance(sample_export_data, dict)
        assert sample_export_data.get("agent", {}).get("name") == "devops-samantha"
        assert len(sample_export_data.get("passages", [])) == 4
        assert len(sample_export_data.get("recall_memory", [])) == 3

    def test_full_bundle_conversion(self, sample_export_data: dict[str, Any]):
        adapter = LettaOKFAdapter(sample_export_data)
        bundle = adapter.to_json_bundle()

        assert bundle["okf_version"] == OKF_VERSION
        assert bundle["source"] == "letta"
        assert bundle["agent"]["name"] == "devops-samantha"

        # Base tags should have extracted agent and user from core memory
        core_tags = bundle["core_memory_tags"]
        assert "source:letta" in core_tags
        assert "agent:devops-samantha" in core_tags
        assert "user:alex-rivera" in core_tags
        assert "persona:samantha" in core_tags
        assert "core:human" in core_tags
        assert "core:persona" in core_tags

        summary = bundle["summary"]
        assert summary["archival_count"] == 4
        assert summary["core_memory_blocks"] == 3  # human, persona, guidelines
        assert summary["recall_count"] == 3
        assert summary["total_documents"] == 10

        per_type = summary["per_type_counts"]
        assert per_type["observation"] == 4
        assert per_type["instruction"] == 1  # persona
        assert per_type["preference"] == 1   # human
        assert per_type["fact"] == 1         # guidelines
        assert per_type["context"] == 3      # recall messages


class TestArchivalMemoryConversion:
    """Verifies archival memory passage mapping into OKF observation documents."""

    def test_archival_passage_fields(self, sample_export_data: dict[str, Any]):
        adapter = LettaOKFAdapter(sample_export_data)
        docs = [d for d in adapter.convert() if d.doc_type == "observation"]

        assert len(docs) == 4
        first_doc = docs[0]

        assert first_doc.doc_type == "observation"
        assert "kubernetes" in first_doc.tags
        assert "deployments" in first_doc.tags
        assert "archival" in first_doc.tags
        assert "source:letta" in first_doc.tags
        assert "agent:devops-samantha" in first_doc.tags

        # Title generated from first line
        assert "Kubernetes zero-downtime rollouts" in first_doc.title
        # Content includes body and [Supporting data]
        doc_dict = first_doc.to_dict()
        assert "maxSurge=25%" in doc_dict["content"]
        assert "[Supporting data]" in doc_dict["content"]
        assert "Source: letta:" in doc_dict["content"]

        # Provenance in x_memanto
        x_mem = doc_dict["x_memanto"]
        assert x_mem["source"] == "letta"
        assert x_mem["provenance"] == "imported"
        assert x_mem["confidence"] == 0.8
        assert x_mem["metadata"]["source"] == "infrastructure-runbook"
        assert x_mem["metadata"]["author"] == "alex.rivera"

    def test_alternative_passage_fields(self):
        data = {
            "passages": [
                {
                    "content": "Alternative field name for passage text",
                    "createdAt": "2026-05-01T12:00:00Z",
                }
            ]
        }
        adapter = LettaOKFAdapter(data)
        docs = adapter.convert()
        assert len(docs) == 1
        assert docs[0].body == "Alternative field name for passage text"
        assert docs[0].timestamp == "2026-05-01T12:00:00Z"
        assert docs[0].doc_id == "archival-passage-0001"

    def test_passages_nested_in_archival_memory_key(self):
        data = {
            "archival_memory": [
                {"id": "arch-1", "text": "Archival memory text via archival_memory key"}
            ]
        }
        adapter = LettaOKFAdapter(data)
        docs = adapter.convert()
        assert len(docs) == 1
        assert docs[0].body == "Archival memory text via archival_memory key"
        assert docs[0].source_ref == "arch-1"

    def test_passages_nested_in_by_agent_dict(self):
        data = {
            "passages_by_agent": {
                "agent-1": [
                    {"id": "p-1", "text": "Passage from agent 1"},
                ],
                "agent-2": [
                    {"id": "p-2", "text": "Passage from agent 2"},
                ],
            }
        }
        adapter = LettaOKFAdapter(data)
        docs = adapter.convert()
        assert len(docs) == 2
        assert {d.source_ref for d in docs} == {"p-1", "p-2"}


class TestCoreMemoryMapping:
    """Verifies core memory blocks (human, persona, custom) mapping and tag propagation."""

    def test_structured_core_memory_blocks(self, sample_export_data: dict[str, Any]):
        adapter = LettaOKFAdapter(sample_export_data)
        core_docs = [d for d in adapter.convert() if "core_memory" in d.tags]

        assert len(core_docs) == 3

        # Persona -> instruction
        persona_doc = next(d for d in core_docs if d.doc_id == "core-persona")
        assert persona_doc.doc_type == "instruction"
        assert persona_doc.confidence == 1.0
        assert "I am Samantha" in persona_doc.body
        assert "persona" in persona_doc.tags
        assert "instruction" in persona_doc.tags

        # Human -> preference
        human_doc = next(d for d in core_docs if d.doc_id == "core-human")
        assert human_doc.doc_type == "preference"
        assert human_doc.confidence == 1.0
        assert "Alex Rivera" in human_doc.body
        assert "human" in human_doc.tags
        assert "user_profile" in human_doc.tags

        # Guidelines -> fact
        guidelines_doc = next(d for d in core_docs if d.doc_id == "core-guidelines")
        assert guidelines_doc.doc_type == "fact"
        assert "Never run destructive database migrations" in guidelines_doc.body
        assert "block:guidelines" in guidelines_doc.tags

    def test_direct_key_core_memory(self):
        """Test Letta exports with direct 'human' and 'persona' strings instead of 'blocks'."""
        data = {
            "agent": {"name": "researcher-bot"},
            "core_memory": {
                "human": "The user is Alice, a genomics researcher.",
                "persona": "I am Mendel, an assistant specialized in bioinformatics.",
            },
        }
        adapter = LettaOKFAdapter(data)
        docs = adapter.convert()

        assert len(docs) == 2
        persona_doc = next(d for d in docs if d.doc_id == "core-persona")
        assert persona_doc.doc_type == "instruction"
        assert "Mendel" in persona_doc.body

        human_doc = next(d for d in docs if d.doc_id == "core-human")
        assert human_doc.doc_type == "preference"
        assert "Alice" in human_doc.body

        # Check tag extraction
        tags = extract_core_metadata_tags(data["core_memory"], data["agent"])
        assert "agent:researcher-bot" in tags
        assert "user:alice" in tags
        assert "persona:mendel" in tags
        assert "core:human" in tags
        assert "core:persona" in tags

    def test_core_tags_propagated_to_all_documents(self, sample_export_data: dict[str, Any]):
        adapter = LettaOKFAdapter(sample_export_data)
        docs = adapter.convert()

        # Every single document must inherit the core metadata tags
        for doc in docs:
            assert "source:letta" in doc.tags
            assert "agent:devops-samantha" in doc.tags
            assert "user:alex-rivera" in doc.tags
            assert "persona:samantha" in doc.tags


class TestRecallMemoryConversion:
    """Verifies recall memory conversation logs converted to OKF context documents."""

    def test_recall_messages_conversion(self, sample_export_data: dict[str, Any]):
        adapter = LettaOKFAdapter(sample_export_data)
        recall_docs = [d for d in adapter.convert() if d.doc_type == "context"]

        assert len(recall_docs) == 3

        # First message: user turn
        msg0 = recall_docs[0]
        assert msg0.doc_id == "recall-msg-001"
        assert msg0.title.startswith("[USER]")
        assert "role:user" in msg0.tags
        assert "recall" in msg0.tags
        assert msg0.metadata["role"] == "user"

        # Second message: assistant turn
        msg1 = recall_docs[1]
        assert msg1.doc_id == "recall-msg-002"
        assert msg1.title.startswith("[ASSISTANT]")
        assert "role:assistant" in msg1.tags
        assert "helm rollback" in msg1.body

    def test_recall_with_messages_key(self):
        data = {
            "messages": [
                {"role": "system", "text": "System bootstrap prompt."},
                {"role": "user", "text": "Hello there!"},
            ]
        }
        adapter = LettaOKFAdapter(data)
        docs = adapter.convert()
        assert len(docs) == 2
        assert docs[0].doc_type == "context"
        assert docs[0].title.startswith("[SYSTEM]")
        assert "role:system" in docs[0].tags


class TestResilienceAndMalformedInputs:
    """Verifies robust handling of missing, empty, or malformed data."""

    def test_empty_export_dict(self):
        adapter = LettaOKFAdapter({})
        docs = adapter.convert()
        assert docs == []
        bundle = adapter.to_json_bundle()
        assert bundle["summary"]["total_documents"] == 0

    def test_non_dict_export_raises_migration_error(self):
        with pytest.raises(MigrationError, match="must be a valid JSON dictionary"):
            LettaOKFAdapter(["not", "a", "dict"])  # type: ignore

    def test_missing_or_empty_text_passages_are_skipped(self):
        data = {
            "passages": [
                {"id": "p1", "text": ""},
                {"id": "p2", "text": "   "},
                {"id": "p3"},  # no text key
                {"id": "p4", "text": "Valid passage content."},
            ]
        }
        adapter = LettaOKFAdapter(data)
        docs = adapter.convert()
        assert len(docs) == 1
        assert docs[0].source_ref == "p4"

    def test_missing_or_empty_recall_messages_are_skipped(self):
        data = {
            "recall_memory": [
                {"id": "m1", "text": ""},
                {"id": "m2", "text": "   "},
                {"id": "m3", "role": "user"},  # no text
                {"id": "m4", "role": "user", "text": "Valid message."},
            ]
        }
        adapter = LettaOKFAdapter(data)
        docs = adapter.convert()
        assert len(docs) == 1
        assert docs[0].source_ref == "m4"

    def test_corrupt_or_missing_file_raises(self, tmp_path: Path):
        non_existent = tmp_path / "does_not_exist.json"
        with pytest.raises(FileNotFoundError):
            load_letta_export(non_existent)

        corrupt_file = tmp_path / "corrupt.json"
        corrupt_file.write_text("{bad json: 123", encoding="utf-8")
        with pytest.raises(MigrationError, match="Corrupt JSON"):
            load_letta_export(corrupt_file)

        array_file = tmp_path / "array.json"
        array_file.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(MigrationError, match="must contain a root JSON object"):
            load_letta_export(array_file)

    def test_timestamp_normalization(self):
        # Epoch seconds
        assert normalize_timestamp(1750000000) == "2025-06-15T15:06:40Z"
        # ISO string with offset
        assert normalize_timestamp("2026-07-02T10:15:30+00:00") == "2026-07-02T10:15:30Z"
        assert normalize_timestamp("2026-07-02T12:15:30+02:00") == "2026-07-02T10:15:30Z"
        # ISO string with Z
        assert normalize_timestamp("2026-07-02T10:15:30Z") == "2026-07-02T10:15:30Z"
        # Invalid timestamps should return None safely
        assert normalize_timestamp(None) is None
        assert normalize_timestamp("") is None
        assert normalize_timestamp("invalid-date-format") is None
        assert normalize_timestamp(-9999999999999999) is None

    def test_sanitize_tags(self):
        assert _sanitize_tags(None) == []
        assert _sanitize_tags("") == []
        assert _sanitize_tags(["tag1", "tag2", "tag1", None, ""]) == ["tag1", "tag2"]
        assert _sanitize_tags("alpha, beta, gamma , alpha") == ["alpha", "beta", "gamma"]
        assert _sanitize_tags({"s1", "s2"}) in (["s1", "s2"], ["s2", "s1"])
        assert _sanitize_tags(123) == ["123"]

    def test_slugify(self):
        assert _slugify("Hello World!") == "hello-world"
        assert _slugify("devops-samantha") == "devops-samantha"
        assert _slugify("  Special @#% Characters!  ") == "special-characters"
        assert _slugify("") == "item"


class TestBundleAndDirectoryExport:
    """Verifies directory writing, JSON manifest generation, and Memanto import compatibility."""

    def test_export_to_directory_structure(self, sample_export_data: dict[str, Any], tmp_path: Path):
        output_dir = tmp_path / "letta_okf_bundle"
        adapter = LettaOKFAdapter(sample_export_data)
        bundle_manifest = adapter.export_to_directory(output_dir)

        assert output_dir.exists()
        assert (output_dir / "bundle.json").exists()
        assert (output_dir / "index.md").exists()

        memories_dir = output_dir / "memories"
        assert memories_dir.exists()
        assert (memories_dir / "observation").exists()
        assert (memories_dir / "instruction").exists()
        assert (memories_dir / "preference").exists()
        assert (memories_dir / "fact").exists()
        assert (memories_dir / "context").exists()

        # Check total markdown files
        md_files = list(memories_dir.rglob("*.md"))
        assert len(md_files) == 10

        # Check frontmatter of an exported markdown file
        sample_md = list((memories_dir / "observation").glob("*.md"))[0]
        content = sample_md.read_text(encoding="utf-8")
        assert content.startswith("---\n")
        parts = content.split("---\n", 2)
        frontmatter = yaml.safe_load(parts[1])
        assert frontmatter["type"] == "observation"
        assert "x_memanto" in frontmatter
        assert frontmatter["x_memanto"]["source"] == "letta"

    def test_round_trip_with_memanto_okf_loader_and_mapper(
        self, sample_export_data: dict[str, Any], tmp_path: Path
    ):
        """Assert that Memanto's native load_okf_bundle and map_okf ingest the exported bundle seamlessly."""
        output_dir = tmp_path / "letta_imported_bundle"
        adapter = LettaOKFAdapter(sample_export_data)
        adapter.export_to_directory(output_dir)

        # Ingest using Memanto's official loader
        loaded = load_okf_bundle(output_dir)
        assert "memories" in loaded
        entries = loaded["memories"]
        assert len(entries) == 10

        # Ingest into Memanto mapper
        mapped = map_okf(loaded)
        assert len(mapped) == 10

        types = {m["type"] for m in mapped}
        assert types == {"observation", "instruction", "preference", "fact", "context"}

        # Verify x_memanto provenance survived round trip
        first_entry = mapped[0]
        assert "title" in first_entry
        assert "content" in first_entry
        assert "tags" in first_entry
        assert "source:letta" in first_entry["tags"]

    def test_convert_letta_to_okf_bundle_high_level_api(
        self, sample_export_data: dict[str, Any], tmp_path: Path
    ):
        source_json = tmp_path / "sample.json"
        source_json.write_text(json.dumps(sample_export_data), encoding="utf-8")

        # 1. Output to .json file
        out_json = tmp_path / "out_bundle.json"
        bundle = convert_letta_to_okf_bundle(source_json, output_path=out_json)
        assert bundle["okf_version"] == OKF_VERSION
        assert out_json.exists()
        loaded_json = json.loads(out_json.read_text(encoding="utf-8"))
        assert loaded_json["summary"]["total_documents"] == 10

        # 2. Output to directory
        out_dir = tmp_path / "out_dir_bundle"
        convert_letta_to_okf_bundle(source_json, output_path=out_dir)
        assert (out_dir / "bundle.json").exists()
        assert (out_dir / "index.md").exists()


class TestCLI:
    """Verifies command line interface execution and arguments."""

    def test_cli_dry_run(self, capsys: pytest.CaptureFixture):
        ret = main_cli_invoke([str(FIXTURES_PATH), "--dry-run"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "Letta (MemGPT) -> Memanto OKF Migration" in captured.out
        assert "Total documents:   10" in captured.out
        assert "Dry-run complete. No files written." in captured.out

    def test_cli_output_json_file(self, tmp_path: Path):
        target_json = tmp_path / "cli_output.json"
        ret = main_cli_invoke([str(FIXTURES_PATH), "--out", str(target_json)])
        assert ret == 0
        assert target_json.exists()
        data = json.loads(target_json.read_text(encoding="utf-8"))
        assert data["summary"]["total_documents"] == 10

    def test_cli_output_directory(self, tmp_path: Path):
        target_dir = tmp_path / "cli_output_dir"
        ret = main_cli_invoke([str(FIXTURES_PATH), "--out", str(target_dir)])
        assert ret == 0
        assert (target_dir / "bundle.json").exists()
        assert (target_dir / "memories").exists()

    def test_cli_missing_source_file(self, capsys: pytest.CaptureFixture):
        ret = main_cli_invoke(["non_existent_file.json"])
        assert ret == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err


def main_cli_invoke(args: list[str]) -> int:
    """Helper to invoke adapter CLI main function with custom args."""
    orig_argv = sys.argv
    try:
        sys.argv = ["adapter.py", *args]
        return main()
    finally:
        sys.argv = orig_argv
