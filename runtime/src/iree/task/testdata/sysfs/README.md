# Sysfs CPU Topology Test Data

This directory contains snapshots of `/sys/devices/system/{cpu,node}/`
structures used to test `topology_sysfs.c` without live hardware.

**Corpus policy (provenance first):** see ops `process/CORPUS.md`. Fixtures
grow from an **authoritative corpus**, not a forest of unexplained synthetic
trees. Every checked-in tree is classified A0 / A1 / A2 below. **No orphan
A2** — each mutation names its parent and the exact edit. Cases that cannot
be derived from a real dump or documented kernel shape are **deferred (P2:
needs capture)**, not invented.

**A1 authority:** ticket `ORACLE_TOPOLOGY.md` (issue #24761 reporter paste).

Archives only are checked in (extracted dirs are gitignored):

```bash
tar xzf arm64_pixel6_tensor.tar.gz
tar xzf x86_hybrid_sparse_clusters.tar.gz
```

## Corpus lineage

```
A0 arm64_pixel6_tensor          (upstream capture, #22455 / capture_sysfs)
  │
A1 x86_hybrid_sparse_clusters   (issue #24761 reporter cluster_id paste)
  ├── A2a x86_no_numa_single_package   (from A1: delete node/)
  └── A2b x86_missing_cluster_id       (from A1: delete cluster_id files)
```

| Class | Name | Provenance | Notes |
|-------|------|------------|-------|
| **A0** | `arm64_pixel6_tensor` | Upstream capture (Pixel 6 / Tensor GS101; #22455 era `capture_sysfs`) | Real dump; no `node/`; dense clusters 0–2 |
| **A1** | `x86_hybrid_sparse_clusters` | **Issue-derived** — full-fidelity map from ticket `ORACLE_TOPOLOGY.md` §2 ([#24761](https://github.com/iree-org/iree/issues/24761) paste) | 24 CPUs; exact 10 cluster_ids `0,8,…,72`; SMT/E layout from paste; `node0`+`package_id=0` are **harness scaffolding** (not reporter dumps — see oracle §1/§3) |
| **A2a** | `x86_no_numa_single_package` | Mutation of **A1**: remove `node/` | Package-fallback path when NUMA sysfs absent |
| **A2b** | `x86_missing_cluster_id` | Mutation of **A1**: remove all `topology/cluster_id` | Affinity falls back to `physical_package_id` (pre-5.16 / missing ABI); **keeps single NUMA** |

### Deferred (P2 — needs authoritative capture; not checked in)

| Case | Why not invented |
|------|------------------|
| Dual-socket / dual-NUMA | No real dump in-repo; would be orphan synthetic |
| Sparse kernel NUMA `online=0,2` | Needs live multi-node capture |
| AMD SNC (1 package, multi-NUMA) | Needs live capture |
| Multi-package without NUMA | Dual-socket dump required |
| Empty cpulist / uncovered CPU / partial cpu dirs | Speculative corners; capture or kernel-doc repro first |

## A1 oracle detail

Full verbatim paste + MUST/MAY rules: ticket
`tickets/iree-org-iree#24761/ORACLE_TOPOLOGY.md`.

Summary (MUST match):

| Metric | Value |
|--------|-------|
| Logical CPUs | **24** (`cpu0`–`cpu23`) |
| Unique `cluster_id` | **10** — `0,8,16,24,32,40,48,56,64,72` |
| Benchmark numbers | Documentation only (422/422 broken; 63.9/709 diagnostic) — **not** gtest wall-time gates |

**Expected (DESIGN + oracle invariants — not invented toy counts):**

- `query_node_count()` == **1** (scaffolded single NUMA/package), **≠** 10 unique clusters
- Physical-core span covers all scheduling domains (8P+8E from reporter taxonomy → 16 groups with SMT scaffolding)
- `ideal_thread_affinity.group` still carries sparse issue cluster ids (incl. ≥64)
- Do **not** assert wall/CPU ms in unit tests

## A0 Pixel detail

**Hardware:** Google Pixel 6 (Tensor GS101), ARM64 big.LITTLE
**Expected:** one package node (not 3 clusters as nodes); affinity groups 0/1/2; capacity filtering as before.

## Capturing new data

```bash
./capture_sysfs.sh my_system_name
COPYFILE_DISABLE=1 tar czf my_system_name.tar.gz --exclude='._*' my_system_name/
```

Prefer live multi-socket / hybrid captures for future A0/A1 entries. Only check
in small interesting trees; large x86 dumps stay local.

## Testing

Compile with sysfs topology (`IREE_ENABLE_CPUINFO=OFF` on Linux) and run
`topology_sysfs_test`, or use `IREE_SYSFS_ROOT` / `iree_sysfs_set_root_path_for_testing`.

Validate:

1. Node id follows NUMA `node*/cpulist` (else package), **never** raw `cluster_id`
2. Sparse / large cluster ids remain affinity hints only
3. Mutations preserve relative invariants of the parent corpus
