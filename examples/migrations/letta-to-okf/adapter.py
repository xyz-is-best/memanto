#!/usr/bin/env python3
"""Letta (MemGPT) to Memanto OKF (Open Knowledge Format) Migration Adapter.

Converts Letta agent memory exports — comprising Core Memory (human & persona
blocks), Archival Memory (passages), and Recall Memory (interaction history) —
into deterministic, portable OKF v0.2 JSON bundles and Markdown directories.

Design Principles:
- Complete Fidelity: Preserves all canonical source truth while separating derived
  embeddings/vectors.
- Core Context Propagation: Persona and human core memory blocks are extracted into
  rich metadata tags propagated across all descendant memory documents.
- Resilient Ingestion: Gracefully handles missing, partial, or malformed fields
  without crashing or silently dropping valid records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


class MigrationError(ValueError):
    """Raised when source export data is fundamentally unparseable."""


OKF_VERSION = "0.2"
DEFAULT_CONFIDENCE = 0.8
MAX_PORTABLE_BODY_CHARS = 9000
MAX_TITLE_CHARS = 90


def _slugify(value: str) -> str:
    """Create a safe, deterministic filename slug from a string."""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^\w\s-]", "", text).strip().lower()
    slug = re.sub(r"[-\s]+", "-", slug)
    return slug[:60] or "item"


def normalize_timestamp(value: Any) -> str | None:
    """Normalize epoch, ISO 8601 string, or datetime to UTC ISO string with 'Z' suffix."""
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)) and value > 0:
        try:
            dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
            return dt.isoformat().replace("+00:00", "Z")
        except (ValueError, OSError, OverflowError):
            return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None

    if dt.tzinfo is None or dt.utcoffset() is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def _sanitize_tags(raw: Any) -> list[str]:
    """Clean, flatten, and deduplicate tags into a list of strings."""
    if raw in (None, ""):
        return []
    items: list[str] = []
    if isinstance(raw, (list, tuple, set)):
        for item in raw:
            if item not in (None, ""):
                items.append(str(item).strip())
    elif isinstance(raw, str):
        items.extend(p.strip() for p in raw.split(","))
    else:
        items.append(str(raw).strip())

    cleaned = []
    seen: set[str] = set()
    for tag in items:
        if tag and tag not in seen:
            seen.add(tag)
            cleaned.append(tag)
    return cleaned


def extract_core_metadata_tags(core_memory: Any, agent: dict[str, Any]) -> list[str]:
    """Extract contextual tags from agent info and core memory blocks (human & persona)."""
    tags: list[str] = ["source:letta"]

    agent_name = agent.get("name")
    if agent_name and isinstance(agent_name, str):
        tags.append(f"agent:{_slugify(agent_name)}")

    if not isinstance(core_memory, dict):
        return tags

    # Extract user identity from human block if present
    human_val = ""
    if "human" in core_memory and isinstance(core_memory["human"], str):
        human_val = core_memory["human"]
    elif "blocks" in core_memory and isinstance(core_memory["blocks"], list):
        for b in core_memory["blocks"]:
            if isinstance(b, dict) and b.get("name") == "human":
                human_val = str(b.get("value") or "")
                break

    if human_val:
        name_match = re.search(
            r"(?:name is|user is|user:)\s*([A-Za-z0-9_-]+(?:\s+[A-Za-z0-9_-]+)?)",
            human_val,
            re.IGNORECASE,
        )
        if name_match:
            tags.append(f"user:{_slugify(name_match.group(1))}")
        tags.append("core:human")

    # Extract persona identity from persona block if present
    persona_val = ""
    if "persona" in core_memory and isinstance(core_memory["persona"], str):
        persona_val = core_memory["persona"]
    elif "blocks" in core_memory and isinstance(core_memory["blocks"], list):
        for b in core_memory["blocks"]:
            if isinstance(b, dict) and b.get("name") == "persona":
                persona_val = str(b.get("value") or "")
                break

    if persona_val:
        p_match = re.search(
            r"(?:I am|persona is|agent is|name is)\s+([A-Za-z0-9_-]+(?:\s+[A-Za-z0-9_-]+)?)",
            persona_val,
            re.IGNORECASE,
        )
        if p_match:
            tags.append(f"persona:{_slugify(p_match.group(1))}")
        tags.append("core:persona")

    return list(dict.fromkeys(tags))


def _format_supporting_data(items: list[tuple[str, Any]]) -> str:
    """Format standard supporting data footer."""
    lines: list[str] = []
    for label, val in items:
        if val in (None, "", [], {}):
            continue
        if isinstance(val, (list, tuple)):
            text = ", ".join(str(x) for x in val if x not in (None, ""))
        elif isinstance(val, dict):
            text = "; ".join(f"{k}={v}" for k, v in val.items() if v not in (None, ""))
        else:
            text = str(val)
        if text:
            if len(text) > 200:
                text = text[:197] + "..."
            lines.append(f"- {label}: {text}")

    if not lines:
        return ""
    return "\n\n---\n[Supporting data]\n" + "\n".join(lines)


@dataclass
class OKFDocument:
    """Represents a single Open Knowledge Format memory document."""

    doc_id: str
    doc_type: str
    title: str
    body: str
    tags: list[str]
    timestamp: str | None
    source_ref: str | None
    metadata: dict[str, Any] = field(default_factory=dict)
    confidence: float = DEFAULT_CONFIDENCE

    def to_dict(self) -> dict[str, Any]:
        """Convert to standard OKF document dictionary."""
        footer_items = [
            ("Source", f"letta:{self.source_ref}" if self.source_ref else "letta"),
            ("Tags", self.tags),
        ]
        if self.metadata:
            footer_items.append(("Source metadata", self.metadata))
        if self.timestamp:
            footer_items.append(("Source created_at", self.timestamp))

        content_with_footer = self.body + _format_supporting_data(footer_items)
        if len(content_with_footer) > MAX_PORTABLE_BODY_CHARS:
            content_with_footer = content_with_footer[: MAX_PORTABLE_BODY_CHARS - 3] + "..."

        doc: dict[str, Any] = {
            "id": self.doc_id,
            "type": self.doc_type,
            "title": self.title,
            "content": content_with_footer,
            "tags": self.tags,
            "timestamp": self.timestamp,
            "x_memanto": {
                "id": self.doc_id,
                "type": self.doc_type,
                "confidence": self.confidence,
                "source": "letta",
                "source_ref": self.source_ref,
                "provenance": "imported",
                "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "metadata": self.metadata,
            },
        }
        return doc

    def to_markdown(self) -> str:
        """Render to standard OKF Markdown document with YAML frontmatter."""
        doc_dict = self.to_dict()
        frontmatter = {
            "type": self.doc_type,
            "title": self.title,
            "tags": self.tags,
        }
        if self.timestamp:
            frontmatter["timestamp"] = self.timestamp
        frontmatter["x_memanto"] = doc_dict["x_memanto"]

        front = yaml.safe_dump(
            frontmatter,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        ).strip()

        return f"---\n{front}\n---\n\n{doc_dict['content']}\n"


class LettaOKFAdapter:
    """Engine for converting Letta agent export data to OKF bundles."""

    def __init__(self, export_data: dict[str, Any]):
        if not isinstance(export_data, dict):
            raise MigrationError("Export data must be a valid JSON dictionary")
        self.raw = export_data
        self.agent = export_data.get("agent") if isinstance(export_data.get("agent"), dict) else {}
        self.core_memory = export_data.get("core_memory") or export_data.get("memory") or {}
        self.passages = self._extract_passages()
        self.recall = self._extract_recall()

    def _extract_passages(self) -> list[dict[str, Any]]:
        """Extract passages from root or nested structures."""
        passages = self.raw.get("passages")
        if isinstance(passages, list):
            return [p for p in passages if isinstance(p, dict)]
        archival = self.raw.get("archival_memory")
        if isinstance(archival, list):
            return [p for p in archival if isinstance(p, dict)]
        by_agent = self.raw.get("passages_by_agent")
        if isinstance(by_agent, dict):
            collected: list[dict[str, Any]] = []
            for agent_passages in by_agent.values():
                if isinstance(agent_passages, list):
                    collected.extend(p for p in agent_passages if isinstance(p, dict))
            return collected
        return []

    def _extract_recall(self) -> list[dict[str, Any]]:
        """Extract recall memory conversation records."""
        recall = self.raw.get("recall_memory")
        if isinstance(recall, list):
            return [m for m in recall if isinstance(m, dict)]
        messages = self.raw.get("messages")
        if isinstance(messages, list):
            return [m for m in messages if isinstance(m, dict)]
        return []

    def convert(self) -> list[OKFDocument]:
        """Convert all export sections into OKFDocument instances."""
        documents: list[OKFDocument] = []
        base_core_tags = extract_core_metadata_tags(self.core_memory, self.agent)

        # 1. Convert Core Memory Blocks
        documents.extend(self._convert_core_memory(base_core_tags))

        # 2. Convert Archival Memory Passages
        documents.extend(self._convert_archival(base_core_tags))

        # 3. Convert Recall Memory Messages
        documents.extend(self._convert_recall(base_core_tags))

        return documents

    def _convert_core_memory(self, base_tags: list[str]) -> list[OKFDocument]:
        """Convert core memory persona, human, and custom blocks into documents."""
        docs: list[OKFDocument] = []
        blocks: list[dict[str, str]] = []

        if isinstance(self.core_memory, dict):
            # Check structured 'blocks' array
            raw_blocks = self.core_memory.get("blocks")
            if isinstance(raw_blocks, list):
                for b in raw_blocks:
                    if isinstance(b, dict) and b.get("value"):
                        blocks.append({
                            "name": str(b.get("name") or "core_block"),
                            "label": str(b.get("label") or b.get("name") or "Block"),
                            "value": str(b.get("value")).strip(),
                        })

            # Check direct 'human' and 'persona' keys if not in blocks
            existing_names = {b["name"] for b in blocks}
            if "human" in self.core_memory and "human" not in existing_names:
                val = str(self.core_memory["human"] or "").strip()
                if val:
                    blocks.append({"name": "human", "label": "User Profile", "value": val})
            if "persona" in self.core_memory and "persona" not in existing_names:
                val = str(self.core_memory["persona"] or "").strip()
                if val:
                    blocks.append({"name": "persona", "label": "Agent Persona", "value": val})

        for block in blocks:
            name = block["name"]
            val = block["value"]
            if not val:
                continue

            if name == "persona":
                doc_type = "instruction"
                title = "Letta Core Memory: Agent Persona"
                block_tags = ["core_memory", "persona", "instruction"]
            elif name == "human":
                doc_type = "preference"
                title = "Letta Core Memory: User Profile"
                block_tags = ["core_memory", "human", "user_profile"]
            else:
                doc_type = "fact"
                title = f"Letta Core Memory: {block['label']}"
                block_tags = ["core_memory", f"block:{_slugify(name)}"]

            merged_tags = list(dict.fromkeys([*base_tags, *block_tags]))
            doc_id = f"core-{_slugify(name)}"
            docs.append(
                OKFDocument(
                    doc_id=doc_id,
                    doc_type=doc_type,
                    title=title,
                    body=val,
                    tags=merged_tags,
                    timestamp=normalize_timestamp(self.agent.get("created_at") or self.raw.get("exported_at")),
                    source_ref=f"core_memory:{name}",
                    metadata={"block_name": name, "label": block["label"]},
                    confidence=1.0,
                )
            )

        return docs

    def _convert_archival(self, base_tags: list[str]) -> list[OKFDocument]:
        """Convert archival memory passages into OKF documents."""
        docs: list[OKFDocument] = []

        for idx, passage in enumerate(self.passages, start=1):
            text = (passage.get("text") or passage.get("content") or "").strip()
            if not text:
                continue

            passage_id = passage.get("id") or f"passage-{idx:04d}"
            # Construct a clean human title from the first line
            first_line = text.splitlines()[0].strip()
            title = first_line[:MAX_TITLE_CHARS] if len(first_line) > 10 else f"Letta passage {passage_id[:12]}"

            # Tags integration
            source_tags = _sanitize_tags(passage.get("tags"))
            combined_tags = list(dict.fromkeys([*base_tags, *source_tags, "archival"]))

            timestamp = normalize_timestamp(passage.get("created_at") or passage.get("createdAt"))
            meta = passage.get("metadata") if isinstance(passage.get("metadata"), dict) else {}

            docs.append(
                OKFDocument(
                    doc_id=f"archival-{_slugify(str(passage_id))}",
                    doc_type="observation",
                    title=title,
                    body=text,
                    tags=combined_tags,
                    timestamp=timestamp,
                    source_ref=str(passage_id),
                    metadata=meta,
                    confidence=DEFAULT_CONFIDENCE,
                )
            )

        return docs

    def _convert_recall(self, base_tags: list[str]) -> list[OKFDocument]:
        """Convert recall memory conversation interaction logs."""
        docs: list[OKFDocument] = []

        for idx, msg in enumerate(self.recall, start=1):
            text = (msg.get("text") or msg.get("content") or "").strip()
            if not text:
                continue

            msg_id = msg.get("id") or f"msg-{idx:04d}"
            role = str(msg.get("role") or "unknown").strip().lower()

            first_line = text.splitlines()[0].strip()
            title = f"[{role.upper()}] {first_line[:MAX_TITLE_CHARS]}"

            role_tag = f"role:{role}" if role else "role:unknown"
            combined_tags = list(dict.fromkeys([*base_tags, "recall", role_tag]))

            timestamp = normalize_timestamp(msg.get("created_at") or msg.get("createdAt") or msg.get("timestamp"))

            docs.append(
                OKFDocument(
                    doc_id=f"recall-{_slugify(str(msg_id))}",
                    doc_type="context",
                    title=title,
                    body=text,
                    tags=combined_tags,
                    timestamp=timestamp,
                    source_ref=str(msg_id),
                    metadata={"role": role},
                    confidence=DEFAULT_CONFIDENCE,
                )
            )

        return docs

    def to_json_bundle(self) -> dict[str, Any]:
        """Generate a complete, serializable OKF JSON bundle."""
        docs = self.convert()
        type_counts = Counter(d.doc_type for d in docs)
        base_core_tags = extract_core_metadata_tags(self.core_memory, self.agent)

        bundle: dict[str, Any] = {
            "okf_version": OKF_VERSION,
            "source": "letta",
            "exported_at": normalize_timestamp(self.raw.get("exported_at"))
            or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "agent": {
                "id": self.agent.get("id"),
                "name": self.agent.get("name"),
                "description": self.agent.get("description"),
            },
            "core_memory_tags": base_core_tags,
            "summary": {
                "total_documents": len(docs),
                "archival_count": len(self.passages),
                "core_memory_blocks": len([d for d in docs if "core_memory" in d.tags]),
                "recall_count": len(self.recall),
                "per_type_counts": dict(sorted(type_counts.items())),
            },
            "documents": [d.to_dict() for d in docs],
        }
        return bundle

    def export_to_directory(self, output_dir: Path | str) -> dict[str, Any]:
        """Write the complete OKF bundle directory to disk (Markdown + JSON)."""
        out_path = Path(output_dir)
        staging = Path(tempfile.mkdtemp(prefix=f".{out_path.name}.tmp-", dir=str(out_path.parent if out_path.parent.exists() else ".")))

        try:
            bundle_json = self.to_json_bundle()
            docs = self.convert()

            # Write bundle.json manifest
            (staging / "bundle.json").write_text(json.dumps(bundle_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

            # Write memory markdown files grouped by type
            memories_dir = staging / "memories"
            memories_dir.mkdir(parents=True, exist_ok=True)

            type_counts: Counter[str] = Counter()
            for doc in docs:
                type_dir = memories_dir / doc.doc_type
                type_dir.mkdir(parents=True, exist_ok=True)
                file_path = type_dir / f"{doc.doc_id}.md"
                file_path.write_text(doc.to_markdown(), encoding="utf-8")
                type_counts[doc.doc_type] += 1

            # Write root index.md
            index_lines = [
                "---",
                f'okf_version: "{OKF_VERSION}"',
                "source: letta",
                "---",
                "",
                "# Letta Memory — Portable OKF Bundle",
                "",
                f"Generated from Letta export for agent `{self.agent.get('name') or self.agent.get('id') or 'unknown'}`.",
                "",
                "## Memory Inventory",
                "",
            ]
            for m_type in sorted(type_counts):
                index_lines.append(f"- [{m_type}](memories/{m_type}/) — {type_counts[m_type]} document(s)")
            index_lines.append("")
            (staging / "index.md").write_text("\n".join(index_lines), encoding="utf-8")

            if out_path.exists():
                shutil.rmtree(out_path)
            os.replace(staging, out_path)

        finally:
            if staging.exists():
                shutil.rmtree(staging)

        return bundle_json


def load_letta_export(path: Path | str) -> dict[str, Any]:
    """Load and validate Letta export JSON file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Letta export file not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MigrationError(f"Corrupt JSON in Letta export: {exc}") from exc

    if not isinstance(data, dict):
        raise MigrationError("Letta export must contain a root JSON object")
    return data


def convert_letta_to_okf_bundle(
    source_path: Path | str,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    """High-level Python API to convert a Letta export file to an OKF bundle."""
    data = load_letta_export(source_path)
    adapter = LettaOKFAdapter(data)
    bundle = adapter.to_json_bundle()

    if output_path:
        out = Path(output_path)
        if out.suffix.lower() == ".json":
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(bundle, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        else:
            adapter.export_to_directory(out)

    return bundle


def main() -> int:
    """CLI entrypoint for letta-to-okf adapter."""
    parser = argparse.ArgumentParser(
        description="Migrate Letta (MemGPT) agent memory exports into Memanto OKF bundles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("source", type=Path, help="Path to Letta export JSON file")
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        default=None,
        help="Destination path (directory for full OKF bundle, or .json file for bundle JSON)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform conversion and display summary metrics without writing to disk",
    )
    args = parser.parse_args()

    try:
        data = load_letta_export(args.source)
    except (FileNotFoundError, MigrationError) as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    adapter = LettaOKFAdapter(data)
    bundle = adapter.to_json_bundle()
    summary = bundle["summary"]

    print("=" * 60)
    print("Letta (MemGPT) -> Memanto OKF Migration")
    print("=" * 60)
    print(f"Source file:       {args.source}")
    print(f"Agent:             {bundle['agent'].get('name') or bundle['agent'].get('id') or 'N/A'}")
    print(f"Core tags:         {', '.join(bundle['core_memory_tags'])}")
    print(f"Total documents:   {summary['total_documents']}")
    print(f"  - Archival:      {summary['archival_count']}")
    print(f"  - Core blocks:   {summary['core_memory_blocks']}")
    print(f"  - Recall msgs:   {summary['recall_count']}")
    print("-" * 60)
    print("Documents per type:")
    for doc_type, count in summary["per_type_counts"].items():
        print(f"  {doc_type:<16} {count}")
    print("=" * 60)

    if args.dry_run:
        print("\nDry-run complete. No files written.")
        return 0

    if args.out:
        if args.out.suffix.lower() == ".json":
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(bundle, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"\nWritten OKF JSON bundle to: {args.out}")
        else:
            adapter.export_to_directory(args.out)
            print(f"\nWritten OKF directory bundle to: {args.out}")
    else:
        # Default: print JSON to stdout
        print("\n" + json.dumps(bundle, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
