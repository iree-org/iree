# Direct-Load Tiling Heuristics Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** When `iree-llvmgpu-use-direct-load` is enabled, use larger MN tile seeds (2x multiplier) and validate DMA linearization alignment as a soft constraint, targeting CDNA4 (gfx950).

**Architecture:** Thread the `useDirectLoad` boolean through seed selection in ConfigUtils.cpp to produce 2x `bestMNTileCountPerSubgroup`. Add a `dmaLinearizationAlignment` optional parameter to `deduceMMASchedule()` in GPUHeuristics that enforces tile alignment as a soft constraint (try with alignment first, fall back to without).

**Tech Stack:** MLIR/C++17, IREE compiler infrastructure, lit tests.

---

### Task 1: Add `useDirectLoad` parameter to seed functions in ConfigUtils.cpp

**Files:**
- Modify: `compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.cpp:252-333`

**Step 1: Add `useDirectLoad` to `getGemmHeuristicSeeds`**

At line 252, change the signature from:
```cpp
static std::optional<GPUMMAHeuristicSeeds>
getGemmHeuristicSeeds(GemmSize gemmSize, int64_t inBitWidth, bool scaled) {
```
to:
```cpp
static std::optional<GPUMMAHeuristicSeeds>
getGemmHeuristicSeeds(GemmSize gemmSize, int64_t inBitWidth, bool scaled,
                      bool useDirectLoad) {
```

After each `return GPUMMAHeuristicSeeds(...)` statement (there are 5 of them: lines 256, 263, 270, 277, 284), add a multiplier. The cleanest way: after computing the seeds in each case, apply the multiplier before returning. Replace each case body to store in a local, multiply, then return:

```cpp
case GemmSize::LargeGemm: {
    // ... existing seed construction ...
    GPUMMAHeuristicSeeds seeds(...);
    if (useDirectLoad)
      seeds.bestMNTileCountPerSubgroup *= 2;
    return seeds;
  }
```

Apply the same pattern to all 5 return paths (SmallGemm, MediumGemm non-scaled, MediumGemm scaled, LargeGemm non-scaled, LargeGemm scaled).

**Step 2: Add `useDirectLoad` to `getConvolutionHeuristicSeeds`**

At line 295, change:
```cpp
static std::optional<GPUMMAHeuristicSeeds>
getConvolutionHeuristicSeeds(GemmSize gemmSize, int64_t inBitWidth) {
```
to:
```cpp
static std::optional<GPUMMAHeuristicSeeds>
getConvolutionHeuristicSeeds(GemmSize gemmSize, int64_t inBitWidth,
                             bool useDirectLoad) {
```

Apply the same `if (useDirectLoad) seeds.bestMNTileCountPerSubgroup *= 2;` pattern to all 3 return paths.

**Step 3: Add `useDirectLoad` to `getContractionHeuristicSeeds`**

At line 324, change:
```cpp
static std::optional<GPUMMAHeuristicSeeds>
getContractionHeuristicSeeds(GPUMatmulShapeType problem, bool isGemm,
                             bool scaled) {
```
to:
```cpp
static std::optional<GPUMMAHeuristicSeeds>
getContractionHeuristicSeeds(GPUMatmulShapeType problem, bool isGemm,
                             bool scaled, bool useDirectLoad) {
```

Update the two call sites inside it:
```cpp
  if (isGemm) {
    return getGemmHeuristicSeeds(gemmSize, inBitWidth, scaled, useDirectLoad);
  }
  return getConvolutionHeuristicSeeds(gemmSize, inBitWidth, useDirectLoad);
```

**Step 4: Thread `useDirectLoad` through `getMmaScheduleFromProblemAndTarget`**

At line 340, change:
```cpp
static std::optional<GPUMMASchedule> getMmaScheduleFromProblemAndTarget(
    IREE::GPU::TargetAttr target, GPUMatmulShapeType problem, Location loc,
    bool transposedLhs, bool transposedRhs, bool isGemm,
    bool mustBeAligned = true, bool doCPromotion = false, bool scaled = false,
    int64_t splitReductionTripCnt = 0) {
```
to:
```cpp
static std::optional<GPUMMASchedule> getMmaScheduleFromProblemAndTarget(
    IREE::GPU::TargetAttr target, GPUMatmulShapeType problem, Location loc,
    bool transposedLhs, bool transposedRhs, bool isGemm,
    bool mustBeAligned = true, bool doCPromotion = false, bool scaled = false,
    int64_t splitReductionTripCnt = 0, bool useDirectLoad = false) {
```

Update the call at line 425-426:
```cpp
  std::optional<GPUMMAHeuristicSeeds> maybeSeeds =
      getContractionHeuristicSeeds(problem, isGemm, scaled, useDirectLoad);
```

**Step 5: Thread `useDirectLoad` through `getMatmulOrIGEMMLoweringConfigAndWorkgroupSize`**

The function at line 651 already receives `useDirectLoad`. Update its call to `getMmaScheduleFromProblemAndTarget` at line 841 and line 851 to pass `useDirectLoad`:

```cpp
  std::optional<GPUMMASchedule> schedule = getMmaScheduleFromProblemAndTarget(
      target, problem, loc, transposedLhs, transposedRhs, isGemm,
      /*mustBeAligned=*/true, doCPromotion, scaled, splitReductionTripCnt,
      useDirectLoad);
```

And the unaligned fallback at line 851:
```cpp
    schedule = getMmaScheduleFromProblemAndTarget(
        target, problem, loc, transposedLhs, transposedRhs, isGemm,
        mustBeAligned, doCPromotionUnaligned, scaled, splitReductionTripCnt,
        useDirectLoad);
```

**Step 6: Build and verify compilation**

Run: `ninja -C ~/build iree-opt 2>&1 | tail -20`
Expected: Compiles without errors.

**Step 7: Commit**

```bash
git add compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.cpp
git commit -m "[Codegen] Thread useDirectLoad through tiling seed selection.

When useDirectLoad is true, multiply bestMNTileCountPerSubgroup by 2x
to take advantage of freed VGPR budget from DMA-based global loads."
```

---

### Task 2: Add `dmaLinearizationAlignment` parameter to `deduceMMASchedule`

**Files:**
- Modify: `compiler/src/iree/compiler/Codegen/Common/GPU/GPUHeuristics.h:154-160`
- Modify: `compiler/src/iree/compiler/Codegen/Common/GPU/GPUHeuristics.cpp:664-732`

**Step 1: Update the declaration in GPUHeuristics.h**

At line 154, change:
```cpp
FailureOr<GPUMMASchedule> deduceMMASchedule(
    const GPUMatmulShapeType &problem, ArrayRef<GPUIntrinsicType> intrinsics,
    const GPUMMAHeuristicSeeds &seeds, int64_t sharedMemLimitInBytes,
    int64_t subgroupSize, std::optional<int64_t> cuCount, Location loc,
    bool transposedLhs = false, bool transposedRhs = false,
    bool canUpcastAcc = false, bool mustBeAligned = true,
    bool doCPromotion = false, int64_t splitReductionTripCnt = 0);
```
to:
```cpp
FailureOr<GPUMMASchedule> deduceMMASchedule(
    const GPUMatmulShapeType &problem, ArrayRef<GPUIntrinsicType> intrinsics,
    const GPUMMAHeuristicSeeds &seeds, int64_t sharedMemLimitInBytes,
    int64_t subgroupSize, std::optional<int64_t> cuCount, Location loc,
    bool transposedLhs = false, bool transposedRhs = false,
    bool canUpcastAcc = false, bool mustBeAligned = true,
    bool doCPromotion = false, int64_t splitReductionTripCnt = 0,
    std::optional<int64_t> dmaLinearizationAlignment = std::nullopt);
```

**Step 2: Update the definition in GPUHeuristics.cpp**

At line 664, update the signature to match the declaration (add the new parameter).

**Step 3: Implement two-pass schedule deduction**

Replace the return statement at line 729:
```cpp
    return fitScheduleInSharedMemory(schedule, isValidSchedule);
```

With the two-pass logic:
```cpp
    // If DMA linearization alignment is requested, first try to find a
    // schedule that satisfies both LDS budget and DMA alignment.
    if (dmaLinearizationAlignment) {
      auto isValidWithDMA = [&](const GPUMMASchedule &schedule) -> bool {
        if (!isValidSchedule(schedule))
          return false;
        int64_t tileM = schedule.getTotalMSize() *
                        schedule.getTotalMTileSize() *
                        schedule.getTotalMSubgroupCount();
        int64_t tileN = schedule.getTotalNSize() *
                        schedule.getTotalNTileSize() *
                        schedule.getTotalNSubgroupCount();
        int64_t tileK =
            schedule.getTotalKSize() * schedule.getTotalKTileSize();
        int64_t alignment = *dmaLinearizationAlignment;
        bool lhsAligned = (tileM * tileK) % alignment == 0;
        bool rhsAligned = (tileN * tileK) % alignment == 0;
        return lhsAligned && rhsAligned;
      };
      FailureOr<GPUMMASchedule> alignedSchedule =
          fitScheduleInSharedMemory(schedule, isValidWithDMA);
      if (succeeded(alignedSchedule))
        return alignedSchedule;
      LDBG() << "DMA-aligned schedule not found, falling back to "
                "non-aligned schedule";
    }
    // Fallback: LDS budget check only (no DMA alignment).
    return fitScheduleInSharedMemory(schedule, isValidSchedule);
```

**Step 4: Build and verify compilation**

Run: `ninja -C ~/build iree-opt 2>&1 | tail -20`
Expected: Compiles without errors.

**Step 5: Commit**

```bash
git add compiler/src/iree/compiler/Codegen/Common/GPU/GPUHeuristics.h \
        compiler/src/iree/compiler/Codegen/Common/GPU/GPUHeuristics.cpp
git commit -m "[Codegen] Add DMA linearization alignment to deduceMMASchedule.

Two-pass approach: first try to find a schedule satisfying both LDS
budget and DMA alignment, fall back to LDS-only if alignment cannot
be satisfied."
```

---

### Task 3: Compute and pass `dmaLinearizationAlignment` from ConfigUtils.cpp

**Files:**
- Modify: `compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.cpp:430-442`

**Step 1: Compute alignment and pass to `deduceMMASchedule`**

In `getMmaScheduleFromProblemAndTarget`, after the seeds are computed (around line 428), add alignment computation and pass it through:

```cpp
  GPUMMAHeuristicSeeds seeds = maybeSeeds.value();

  // Compute DMA linearization alignment when using direct load.
  // The alignment ensures that LDS tile sizes are divisible by
  // subgroupSize * (128 / elementBits), enabling optimal 128-bit DMA
  // transfers.
  std::optional<int64_t> dmaLinearizationAlignment = std::nullopt;
  if (useDirectLoad) {
    int64_t elementBits = problem.aType.getIntOrFloatBitWidth();
    dmaLinearizationAlignment = targetSubgroupSize * (128 / elementBits);
  }

  int64_t maxSharedMemoryBytes = target.getWgp().getMaxWorkgroupMemoryBytes();
```

Then update the `deduceMMASchedule` call at line 438 to pass it:

```cpp
  std::optional<GPUMMASchedule> schedule = deduceMMASchedule(
      problem, intrinsics, seeds, maxSharedMemoryBytes, targetSubgroupSize,
      wgpCount, loc, transposedLhs, transposedRhs, /*canUpcastAcc=*/false,
      /*mustBeAligned=*/mustBeAligned, doCPromotion, splitReductionTripCnt,
      dmaLinearizationAlignment);
```

**Step 2: Build and verify compilation**

Run: `ninja -C ~/build iree-opt 2>&1 | tail -20`
Expected: Compiles without errors.

**Step 3: Commit**

```bash
git add compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.cpp
git commit -m "[Codegen] Compute DMA linearization alignment for direct-load path.

Pass the alignment to deduceMMASchedule so tile sizes are validated
against the optimal DMA transfer size (subgroupSize * 128/elemBits)."
```

---

### Task 4: Update `deduceMMASchedule` call sites in KernelConfig.cpp

**Files:**
- Modify: `compiler/src/iree/compiler/Codegen/LLVMGPU/KernelConfig.cpp`

**Step 1: Verify no changes needed**

The `deduceMMASchedule` call sites in KernelConfig.cpp (lines 319-328 for convolution, lines 583-590 for matmul vector distribution) use the old pipeline paths, not the TileAndFuse path. These paths don't use direct load. Since the new parameter has a default value of `std::nullopt`, these call sites compile without changes. Verify this.

Run: `ninja -C ~/build iree-opt 2>&1 | tail -20`
Expected: Compiles without errors.

**Step 2: Also verify SPIRV KernelConfig.cpp**

The call site in `compiler/src/iree/compiler/Codegen/SPIRV/KernelConfig.cpp:965` also uses the default. No changes needed.

**Step 3: Commit (skip if no changes)**

No commit needed if all call sites use defaults.

---

### Task 5: Add lit test for gfx950 direct-load tiling

**Files:**
- Create: `compiler/src/iree/compiler/Codegen/LLVMGPU/test/ROCDL/config_tile_and_fuse_direct_load_gfx950.mlir`

**Step 1: Write the test**

Create a test that verifies the direct-load path produces larger tile sizes than
the non-direct-load path. Use `--iree-llvmgpu-use-direct-load` flag with
`--iree-gpu-test-target=gfx950`.

```mlir
// RUN: iree-opt --mlir-print-local-scope --split-input-file --iree-gpu-test-target=gfx950 \
// RUN:   --iree-llvmgpu-use-direct-load \
// RUN:   --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s \
// RUN:   | FileCheck %s

// Large f16 matmul — should get larger tiles with direct load.
func.func @matmul_f16_direct_load(
    %arg0: tensor<4096x4096xf16>,
    %arg1: tensor<4096x4096xf16>,
    %arg2: tensor<4096x4096xf32>) -> tensor<4096x4096xf32> {
  // CHECK-LABEL: func.func @matmul_f16_direct_load
  // CHECK: lowering_config = #iree_gpu.lowering_config
  // CHECK-SAME: mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>
  // CHECK-SAME: promote_operands = [0, 1]
  // CHECK-SAME: promotion_types = [#iree_gpu.use_global_load_dma, #iree_gpu.use_global_load_dma]
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<4096x4096xf16>, tensor<4096x4096xf16>)
                      outs(%arg2 : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
  return %0 : tensor<4096x4096xf32>
}
```

Note: The exact CHECK lines for workgroup/subgroup/reduction tile sizes will need
to be filled in after running the test once to see the actual output. Run:

```bash
~/build/tools/iree-opt --mlir-print-local-scope --split-input-file \
  --iree-gpu-test-target=gfx950 --iree-llvmgpu-use-direct-load \
  --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" \
  <test_file>
```

Capture the output and update the CHECK lines to match.

**Step 2: Add to CMakeLists.txt and BUILD.bazel**

Add the new test file to
`compiler/src/iree/compiler/Codegen/LLVMGPU/test/ROCDL/CMakeLists.txt` and
`compiler/src/iree/compiler/Codegen/LLVMGPU/test/ROCDL/BUILD.bazel`.

**Step 3: Run the test**

Run: `ctest -R config_tile_and_fuse_direct_load_gfx950 --test-dir ~/build -V`
Expected: PASS

**Step 4: Commit**

```bash
git add compiler/src/iree/compiler/Codegen/LLVMGPU/test/ROCDL/config_tile_and_fuse_direct_load_gfx950.mlir \
        compiler/src/iree/compiler/Codegen/LLVMGPU/test/ROCDL/CMakeLists.txt \
        compiler/src/iree/compiler/Codegen/LLVMGPU/test/ROCDL/BUILD.bazel
git commit -m "[Codegen] Add lit test for direct-load tiling heuristics on gfx950."
```

---

### Task 6: Verify existing tests still pass

**Step 1: Run existing gfx950 config tests**

Run: `ctest -R "config_tile_and_fuse_gfx950|config_vector_distribute_gfx950" --test-dir ~/build -V`
Expected: All PASS (no regressions — existing tests don't use `--iree-llvmgpu-use-direct-load`)

**Step 2: Run existing gfx942 config tests**

Run: `ctest -R "config_matmul" --test-dir ~/build -V`
Expected: All PASS (gfx942 tests are unaffected by changes)

**Step 3: Run GPU heuristics-related tests**

Run: `ctest -R "gpu" --test-dir ~/build -V 2>&1 | grep -E "PASS|FAIL"`
Expected: All PASS
