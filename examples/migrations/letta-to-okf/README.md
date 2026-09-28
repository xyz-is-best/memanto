# Letta (MemGPT) → Memanto OKF Migration Adapter

A production migration adapter and CLI tool that converts **Letta** (formerly **MemGPT**) stateful agent memory exports into portable, deterministic **Open Knowledge Format (OKF v0.2)** bundles.

---

## Architecture: Multi-Tiered Memory Mapping

Letta's memory architecture partitions state across three distinct tiers:
1. **Core Memory**: In-context persistent blocks defining the agent persona, human user profile, and operational constraints.
2. **Archival Memory**: Out-of-context, long-term semantic passages searchable by vector/text embeddings.
3. **Recall Memory**: Conversational interaction history (raw user, assistant, and system dialogue turns).

This adapter maps each tier into Memanto's canonical OKF v0.2 document types while preserving provenance, metadata, and cross-cutting contextual tags.

| Letta Source Tier | Memanto OKF Type | Canonical Confidence | Extracted Tags / Frontmatter | OKF Body & Metadata |
|---|---|---|---|---|
| **Core Memory (`persona`)** | `instruction` | `1.0` | `core_memory`, `persona`, `instruction`, `persona:<slug>` | Agent persona instructions; `source_ref: "core_memory:persona"` |
| **Core Memory (`human`)** | `preference` | `1.0` | `core_memory`, `human`, `user_profile`, `user:<slug>` | User identity, preferences, and technical stack profile |
| **Core Memory (custom blocks)** | `fact` | `1.0` | `core_memory`, `block:<slug>` | Custom guidelines, architectural constraints, security rules |
| **Archival Memory (`passages`)** | `observation` | `0.8` | `archival`, `<passage.tags>`, `agent:<name>`, `user:<name>` | Full passage text, author, cluster, runbook section metadata |
| **Recall Memory (`messages`)** | `context` | `0.8` | `recall`, `role:<role>`, `agent:<name>` | Dialogue turns formatted with `[ROLE]` headers and timestamps |

### Contextual Tag Propagation

To ensure isolated retrieval retains its operational context in downstream systems, core memory identities (e.g., `agent:devops-samantha`, `user:alex-rivera`, `persona:samantha`, `core:human`, `core:persona`) are automatically propagated to all descendant memory documents in the bundle.

---

## Quickstart & CLI Usage

### 1. Dry Run (Inspect without writing)

Analyze a Letta export and inspect document counts, type breakdowns, and extracted tags:

```bash
python adapter.py fixtures/sample_letta_export.json --dry-run
```

**Output:**
```text
============================================================
Letta (MemGPT) → Memanto OKF Migration
============================================================
Source file:       fixtures/sample_letta_export.json
Agent:             devops-samantha
Core tags:         source:letta, agent:devops-samantha, user:alex-rivera, core:human, persona:samantha, core:persona
Total documents:   10
  - Archival:      4
  - Core blocks:   3
  - Recall msgs:   3
------------------------------------------------------------
Documents per type:
  context          3
  fact             1
  instruction      1
  observation      4
  preference       1
============================================================

Dry-run complete. No files written.
```

### 2. Export to OKF Directory Bundle

Generate a full directory bundle containing `index.md`, `bundle.json`, and partitioned Markdown files under `memories/<type>/`:

```bash
python adapter.py fixtures/sample_letta_export.json --out ./okf_bundle
```

Generated directory layout:
```text
okf_bundle/
├── bundle.json
├── index.md
└── memories/
    ├── context/
    │   ├── recall-msg-001.md
    │   ├── recall-msg-002.md
    │   └── recall-msg-003.md
    ├── fact/
    │   └── core-guidelines.md
    ├── instruction/
    │   └── core-persona.md
    ├── observation/
    │   ├── archival-passage-8f2c3d4e-1a2b-4c5d-6e7f-8a9b0c1d2e3f.md
    │   ├── archival-passage-9a3b4c5d-2b3c-4d5e-7f8a-9b0c1d2e3f4a.md
    │   ├── archival-passage-0b1c2d3e-3c4d-5e6f-8a9b-0c1d2e3f4a5b.md
    │   └── archival-passage-1c2d3e4f-4d5e-6f7a-9b0c-1d2e3f4a5b6c.md
    └── preference/
        └── core-human.md
```

### 3. Export to Single JSON Bundle

Generate a single standalone JSON manifest compatible with Memanto's batch API:

```bash
python adapter.py fixtures/sample_letta_export.json --out letta_bundle.json
```

### 4. Direct Ingestion into Memanto

Import the generated bundle directly into a Memanto agent:

```bash
memanto migrate okf ./okf_bundle --agent <my-agent-id>
```

---

## Python API Usage

You can also invoke the adapter programmatically inside your migration scripts:

```python
from adapter import LettaOKFAdapter, convert_letta_to_okf_bundle

# High-level conversion to directory
convert_letta_to_okf_bundle(
    source_path="fixtures/sample_letta_export.json",
    output_path="./output/letta_okf_bundle"
)

# Programmatic access to OKF document objects
with open("fixtures/sample_letta_export.json", "r", encoding="utf-8") as f:
    import json
    data = json.load(f)

adapter = LettaOKFAdapter(data)
bundle = adapter.to_json_bundle()
print(f"Migrated {bundle['summary']['total_documents']} documents.")
```

---

## Document Frontmatter & Provenance Contract

Every generated `.md` file includes standard OKF YAML frontmatter along with Memanto's `x_memanto` extension block:

```markdown
---
type: observation
title: "For Kubernetes zero-downtime rollouts, configure maxSurge=25% and maxUnavailable=0"
tags:
  - source:letta
  - agent:devops-samantha
  - user:alex-rivera
  - core:human
  - persona:samantha
  - core:persona
  - kubernetes
  - deployments
  - zero-downtime
  - reliability
  - archival
timestamp: "2026-07-02T10:15:30Z"
x_memanto:
  id: "archival-passage-8f2c3d4e-1a2b-4c5d-6e7f-8a9b0c1d2e3f"
  type: observation
  confidence: 0.8
  source: letta
  source_ref: "passage-8f2c3d4e-1a2b-4c5d-6e7f-8a9b0c1d2e3f"
  provenance: imported
  updated_at: "2026-09-27T11:24:00Z"
  metadata:
    source: infrastructure-runbook
    author: alex.rivera
    verified: true
    section: "2.4"
---

For Kubernetes zero-downtime rollouts, configure maxSurge=25% and maxUnavailable=0 in Deployment specs. Ensure readiness probes have at least a 10s initial delay and 5s timeout to avoid dropping active in-flight requests during pod lifecycle transitions.

---
[Supporting data]
- Source: letta:passage-8f2c3d4e-1a2b-4c5d-6e7f-8a9b0c1d2e3f
- Tags: source:letta, agent:devops-samantha, user:alex-rivera, core:human, persona:samantha, core:persona, kubernetes, deployments, zero-downtime, reliability, archival
- Source metadata: source=infrastructure-runbook; author=alex.rivera; verified=True; section=2.4
- Source created_at: 2026-07-02T10:15:30Z
```

---

## Testing & Verification

The adapter includes a comprehensive test suite covering all memory tiers, edge cases, malformed payloads, and Memanto loader round-trips.

Run tests via the virtual environment pytest:

```bash
# Run adapter-specific tests
pytest examples/migrations/letta-to-okf/test_letta_to_okf.py -v

# Or run via central test runner
pytest tests/test_letta_to_okf.py -v
```

### Verified Test Cases (26/26 Passing)
- **Archival Memory Mapping**: Passage extraction, UTC timestamp normalization, slugified IDs, custom metadata propagation.
- **Core Memory Mapping**: Structured `blocks` vs direct key exports, persona -> `instruction`, human -> `preference`, guidelines -> `fact`.
- **Recall Memory Mapping**: Dialogue turns to `context` documents with role tags (`role:user`, `role:assistant`).
- **Resilience & Fault Tolerance**: Missing core memory, empty passages, whitespace-only texts, corrupt JSON, invalid timestamps.
- **Memanto Native Interoperability**: Round-trip verification using Memanto's official `load_okf_bundle` and `map_okf`.
