# WidenInnerTiledReduction — Design Discussion

## What the pass does

For VDMFMA ops (AMD sparse trick emulating M=8 GEMMs), each K-iteration inside a
reduction loop expands ACC from `vector<2xf32>` to `vector<4xf32>`, runs SMFMAC,
then collapses back. This wastes ~2048 instructions per thread per workgroup tile.

The fix: hoist expand before the loop and collapse after, keeping ACC in native
`vector<4xf32>` form throughout. `buildVDMFMAOps` already has `accIsNative` support.

## Pipeline position

```
1. Pack to Intrinsics    → inner_tiled created (undistributed tensors)
2. Tile Reduction        → scf.for created, inner_tiled inside loop
3. Distribute to Lanes   → per-thread distributed types (vector<2xf32> ACC)
4. WidenInnerTiledReduction  ← THIS PASS (before unroll)
5. Unroll to Intrinsics  → outer dims → individual inner_tiled ops (K-step chaining)
6. Lower to Hardware     → inner_tiled → SMFMAC/WMMA via buildUnderlyingOperations
```

## Current implementation

The pass runs before `UnrollToIntrinsics`. At this point there is exactly one
`inner_tiled` op per `scf.for` reduction loop. The pass:

1. Walks `scf.for` ops looking for VDMFMA `inner_tiled` candidates
2. Expands the init arg before the loop (`vector.interleave` with zeros)
3. Creates a new `scf.for` with the widened init arg
4. Merges the old body, rebuilds the `inner_tiled` with `promotedAcc=true` semantics
5. Collapses the result after the loop (`vector.deinterleave` + pairwise add)

After widening, `UnrollToIntrinsics` creates K-step chains that carry native ACC.
`buildVDMFMAOps` sees `accIsNative=true` and skips expand/collapse entirely.

## Key infrastructure

- `promotedAcc` flag on `InnerTiledSemanticsAttr` — tells the verifier to accept
  the native ACC type (`vector<4xf32>` instead of `vector<2xf32>`)
- `getTileTypes` override — returns the native type when `promotedAcc=true`
- `accIsNative` in `buildVDMFMAOps` — skips expand/collapse when ACC is already native
- `expandAccumulator` / `collapseAccumulator` — shared utilities (also used by
  `buildVDMFMAOps` for the non-hoisted case)

## Design alternatives explored

### 1. TypeSwitch approach (current + WMMA extension)

Extend `WidenInnerTiledReduction` with a `TypeSwitch` on the `kind` attribute to
handle multiple intrinsics (VDMFMA, WMMA f16/bf16 from PR #23806):

```cpp
struct AccWidenInfo {
  VectorType nativeAccType;
  function_ref<Value(OpBuilder &, Location, Value)> collapse;
};

static std::optional<AccWidenInfo> getAccWidenInfo(Attribute kind, VectorType accType) {
  return TypeSwitch<Attribute, std::optional<AccWidenInfo>>(kind)
    .Case<VirtualMMAAttr>([&](auto vmma) -> std::optional<AccWidenInfo> {
      if (!isVDMFMAIntrinsic(vmma.getIntrinsic())) return std::nullopt;
      return AccWidenInfo{VectorType::get({4}, elemTy), collapseAccumulator};
    })
    .Case<MMAAttr>([&](auto mma) -> std::optional<AccWidenInfo> {
      if (!isWMMAR3F16(mma.getIntrinsic())) return std::nullopt;
      return AccWidenInfo{VectorType::get({16}, elemTy), collapseWMMAAccumulator};
    })
    .Default(std::nullopt);
}
```

**Pros**: Simple, ships now, handles both consumers.
**Cons**: Each new intrinsic requires changes in 4 places (TypeSwitch, collapse
function, `promotedAcc`/`getTileTypes` override, `accIsNative` in lowering).

### 2. New ops: `unfinalize_accumulator` / `finalize_accumulator`

Introduce a pair of ops that bracket hardware operations requiring non-standard
accumulator types. The conversion logic lives in region bodies:

```mlir
%native = iree_gpu.unfinalize_accumulator %semantic "vdmfma" {
  ^bb(%arg: vector<2xf32>):
    %zero = arith.constant dense<0.0> : vector<2xf32>
    %result = vector.interleave %arg, %zero
    yield %result
} : vector<2xf32> -> vector<4xf32>

%hw_result = sparse_mfma ... + %native

%semantic = iree_gpu.finalize_accumulator %hw_result "vdmfma" {
  ^bb(%arg: vector<4xf32>):
    %e, %o = vector.deinterleave %arg
    %sum = arith.addf %e, %o
    yield %sum
} : vector<4xf32> -> vector<2xf32>
```

Three cleanup rules:
1. **Cancel**: `finalize(unfinalize(x))` → `x` (matching tags) — a fold
2. **Loop hoist**: move unfinalize before loop, finalize after — a pass
3. **Materialize**: inline surviving region bodies — trivial

**Pros**: Adding a new intrinsic = just emit the ops with the right body.
No TypeSwitch, no `promotedAcc`, no `accIsNative` threading.
**Cons**: 2 new ops + 3 patterns. More infrastructure for 2 consumers.

### 3. Tagged `scf.execute_region` (lighter-weight variant of #2)

Instead of new ops, wrap expand/collapse in `scf.execute_region` with a
discardable tag attribute and `noInline=true` (prevents canonicalization from
inlining before cleanup patterns run):

```mlir
%native = scf.execute_region {iree_gpu.acc_tag = "vdmfma_expand", no_inline} -> vector<4xf32> {
  %zero = arith.constant dense<0.0> : vector<2xf32>
  %result = vector.interleave %acc, %zero
  scf.yield %result
}
```

Same three rules as #2, but matching on tagged execute_regions instead of
dedicated ops. Materialization is free (just clear `noInline`, canonicalization
inlines the region).

**Pros**: No new ops needed. Same generic cleanup.
**Cons**: Tag is a discardable attribute, not part of formal op semantics.

### 4. Emit at `LowerInnerTiledPattern` time (after unroll)

Emit tagged execute_regions (or new ops) inside `buildVDMFMAOps` / `createMmaOp`
during lowering, then cancel + hoist afterward.

**Rejected because**: After unroll, the composite accumulator is sliced per N-tile
with `extract_strided_slice` / `insert_strided_slice`. The loop hoist would need to
see through this extract/insert noise — a dataflow analysis, not a simple pattern
match. Running before unroll avoids this entirely.

### 5. Emit execute_regions before unroll, hoist generically

Split the TypeSwitch approach into two passes:
- **Emission pass** (knows about intrinsics): wraps VDMFMA/WMMA `inner_tiled` ACC
  in tagged execute_regions
- **Hoist pass** (generic): matches tagged execute_regions around loop boundaries,
  restructures the loop

Adding a new intrinsic: change only the emission pass.

**Status**: Viable evolution of the current approach. The hoist pass becomes
intrinsic-agnostic. Still needs `promotedAcc` / `getTileTypes` for the inner_tiled
verifier to accept the native ACC type after hoisting.

## Second consumer: WMMA f16/bf16 (PR #23806)

PR #23806 fixes the RDNA3 WMMA f16/bf16 accumulator layout. It introduces the
same interleave/deinterleave pattern but with a different collapse mode:

| | VDMFMA | WMMA f16 |
|---|---|---|
| Expand | `interleave(acc, zero)` | `interleave(acc, zero)` |
| Collapse | `deinterleave` → evens + odds (pairwise sum) | `deinterleave` → just take evens |
| Width | 2 → 4 → 2 | 8 → 16 → 8 |
| Why | SMFMAC accumulates into both even and odd slots | WMMA only writes even slots |

## Why not setType() on the existing ForOp?

`scf::ForOp::verifyRegions` checks that init arg types, block arg types, yield
operand types, and result types all match. `OpResult` doesn't have a proper
`setType()` in the rewriter API. While you could technically mutate types directly,
it creates transient invalid IR states and bypasses the rewriter's change tracking.
The new-ForOp + mergeBlocks pattern is the established idiom
(cf. `ForOpCanonicalizationPass.cpp:251`).

## Why not fold into PropagateReshapesByExpansion?

Completely different transformations at different abstraction levels.
`PropagateReshapesByExpansion` operates on tensor-level IR
(`tensor.collapse_shape`, `scf.forall`) early in the pipeline.
`WidenInnerTiledReduction` operates on distributed vector-level IR
(`inner_tiled` with VDMFMA semantics inside `scf.for`) late in the pipeline.
They share no ops, types, patterns, or pipeline stage.

## Open questions

- Should we invest in the execute_region / new-op framework now (approach #3/#5),
  or land the TypeSwitch approach (#1) and refactor when a third consumer arrives?
- If we go with the emit-then-hoist split (#5), can the emission be folded into
  an existing pass (e.g., distribution) rather than being a standalone pass?
- Can we eliminate the `promotedAcc` flag entirely if the execute_region / new op
  itself acts as the type boundary for the verifier?
