# Direct-Load Tiling Heuristics for CDNA4

## Problem

When `iree-enable-direct-load` is active, the DMA engine handles global-to-LDS
transfers instead of using VGPR-based loads. This frees significant VGPR budget
(no global load staging registers needed), but the current tiling heuristics
use identical seeds regardless of whether direct load is enabled. The result is
under-utilizing the available VGPR headroom for larger MN tiles.

## Target

CDNA4 (gfx950) only:
- Subgroup size: 64
- DMA sizes: [32, 128] bits
- LDS: 160KB
- VGPR: 512 x 32-bit registers
- LDS banks: 64

## Approach: Conservative Multiplier

Minimal-change approach: multiply the MN tile seed by 2x when direct load is
enabled, add DMA alignment validation as a soft constraint, and let the existing
`fitScheduleInSharedMemory()` handle shrinking.

## Design

### 1. Seed Multiplier

**File:** `ConfigUtils.cpp` — `getGemmHeuristicSeeds()` /
`getContractionHeuristicSeeds()`

Add `bool useDirectLoad` parameter. When true, multiply
`bestMNTileCountPerSubgroup` by 2x. All other seeds remain unchanged.

Example for large gemm (f16):
- Current: `bestMNTileCountPerSubgroup = 16`
- Direct-load: `bestMNTileCountPerSubgroup = 32`

Rationale: VGPRs previously used for global load staging are freed, allowing
more accumulator tiles per subgroup.

### 2. DMA Alignment Validation (Soft Constraint)

**Files:** `GPUHeuristics.h` / `GPUHeuristics.cpp` — `deduceMMASchedule()`

Add `std::optional<int64_t> dmaLinearizationAlignment` parameter.

The alignment value is computed as:
```
dmaLinearizationAlignment = subgroupSize * (128 / elementBits)
```
- f16: 64 * 8 = 512 elements
- f32: 64 * 4 = 256 elements
- f8:  64 * 16 = 1024 elements

The constraint applies to LDS destination tile sizes:
```
lhsTileElements = tileM * tileK
rhsTileElements = tileN * tileK
lhsTileElements % dmaLinearizationAlignment == 0
rhsTileElements % dmaLinearizationAlignment == 0
```

Two-pass approach:
1. First attempt: `fitScheduleInSharedMemory()` with LDS budget AND DMA
   alignment checks.
2. Fallback: if first attempt fails, re-run with LDS budget check only.
   Produces a valid schedule that uses more DMA instructions.

### 3. Prefetch Stages

No changes. Current behavior preserved:
- Prefetch enabled: `prefetchNumStages = 2` (fixed)
- Prefetch disabled: `prefetchNumStages = 0`

The 2x seed multiplier combined with LDS budget check in
`fitScheduleInSharedMemory()` naturally accounts for double-buffering LDS cost.

## File Change List

| File | Change |
|------|--------|
| `ConfigUtils.cpp` | Add `useDirectLoad` param to seed functions; multiply MN seed by 2x; compute `dmaLinearizationAlignment` and pass to `deduceMMASchedule()` |
| `GPUHeuristics.h` | Add `dmaLinearizationAlignment` param to `deduceMMASchedule()` |
| `GPUHeuristics.cpp` | Two-pass schedule deduction: strict (with DMA alignment), fallback (without) |
| `KernelConfig.cpp` | Update call sites to pass new parameter (default `std::nullopt`) |

## Non-Goals

- Explicit VGPR pressure modeling (use empirical seeds instead)
- Dynamic prefetch stage selection
- Changes to MFMA intrinsic selection or subgroup counts
- Support for non-CDNA4 targets (can be added later)
