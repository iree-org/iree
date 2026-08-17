# Sysfs CPU Topology Test Data

This directory contains snapshots of `/sys/devices/system/{cpu,node}/`
structures used to test `topology_sysfs.c` without live hardware.

**Corpus policy:** see ops `process/CORPUS.md`. Two layers:

1. **Authority corpus (A0 / A1 / A2 mutations)** — grow from real dumps /
   issue paste. Never invent a fake machine and call it authority.
2. **Property test doubles (P-\*)** — minimal synthetic trees that are
   **inputs to the mapping function `f()`**, not captured machines. They
   discriminate DESIGN invariants (dense NUMA remap, package-only-when-no-node,
   empty cpulist, degrade, …). Labeled honestly; **deferred capture ≠ deferred
   property tests**.

Archives only are checked in (extracted dirs are gitignored):

```bash
tar xzf arm64_pixel6_tensor.tar.gz
tar xzf x86_hybrid_sparse_clusters.tar.gz
tar xzf prop_dual_numa.tar.gz   # etc.
```

## Authority corpus lineage

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

## Property test doubles (P-\*) — synthetic input to `f()`, not a machine

Each tree is a **test double** for a DESIGN discriminator. Do **not** cite these
as dual-socket / SNC hardware evidence. Live multi-node dumps remain welcome as
future **A0** entries.

| ID | Fixture | What it proves (invariant) |
|----|---------|----------------------------|
| **P1** | `prop_dual_numa` | Two NUMA cpulists → dense nodes `0,1`; filter by node keeps membership |
| **P2** | `prop_sparse_kernel_numa` | `node/online=0,2` → dense ordinals; raw kernel id must not be used as dense bit |
| **P3** | `prop_numa_over_package` | WITH `node/` → must **not** collapse to single package even if `package_id=0` |
| **P4** | `prop_package_multi` | No `node/` → multi-package dense fallback (package-only-when-no-node) |
| **P5** | `prop_empty_cpulist` | Empty cpulist on an online node: counted, maps no CPUs |
| **P6** | `prop_uncovered_cpu` | Unmapped CPUs kept when dense-node map fails (documented degrade) |
| **P7** | `prop_numa_no_package` | NUMA path works without `physical_package_id` files |

Also covered on **authority** trees (not separate P fixtures):

- A1: `cluster_id` only in `affinity.group`; `node_count ≠ |unique clusters|`; ids ≥64 stay affinity (mask safety)
- A2a / A0: no `node/` → package dense fallback
- A2b: missing `cluster_id` → affinity falls back to package

### Still deferred as **authority** (needs capture) — not as property tests

Dual-socket / SNC / sparse-`online` **live dumps** for A0 fidelity. Property
coverage for those *mapping* behaviors is already in P1–P3 above.

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

Prefer live multi-socket / hybrid captures for future **A0/A1** entries. Property
doubles stay minimal; do not grow a forest of unlabeled synthetics.

## Testing

Compile with sysfs topology (`IREE_ENABLE_CPUINFO=OFF` on Linux) and run
`topology_sysfs_test`, or use `IREE_SYSFS_ROOT` / `iree_sysfs_set_root_path_for_testing`.

Validate:

1. Node id follows NUMA `node*/cpulist` (else package), **never** raw `cluster_id`
2. Sparse / large cluster ids remain affinity hints only
3. Authority mutations preserve relative invariants of the parent corpus
4. P-\* asserts check mapping properties of the synthetic tree, not HW claims
