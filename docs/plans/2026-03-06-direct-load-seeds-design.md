# Direct-Load Heuristic Seeds for gfx950

## Goal

Plumb `useDirectLoad` into the GPU heuristic seed selection path so that
gfx950 with direct-load enabled can use independently tunable seeds.
Seeds start as duplicates of existing values.

## Approach

Add `useDirectLoad` flag to the existing `getGemmHeuristicSeeds()` function.
Inside each `GemmSize` case, branch on the flag to return direct-load-specific
seeds (initially identical to the current values, marked with TODO for tuning).

## Data Flow

```
setMatmulLoweringConfig(useDirectLoad)          // already has it
  getMatmulOrIGEMMLoweringConfigAndWorkgroupSize(useDirectLoad)  // already has it
    getMmaScheduleFromProblemAndTarget()         // NEW: add useDirectLoad param
      getContractionHeuristicSeeds()            // NEW: add useDirectLoad param
        getGemmHeuristicSeeds()                 // NEW: add useDirectLoad param
```

## gfx950 Gating

In `getMmaScheduleFromProblemAndTarget()`, which already has the `target` attr,
compute whether to use direct-load seeds:

```cpp
bool isGfx950DirectLoad = useDirectLoad && target.getArch() == "gfx950";
```

Pass this single bool down to `getContractionHeuristicSeeds()` and
`getGemmHeuristicSeeds()`.

## Seed Function Change

In `getGemmHeuristicSeeds()`, for each `GemmSize` case, add a branch:

```cpp
case GemmSize::LargeGemm:
case GemmSize::VeryLargeGemm:
  if (useDirectLoad) {
    // TODO: Tune seeds for direct-load on gfx950.
    return GPUMMAHeuristicSeeds({4, 16, 2, <derived>});
  }
  if (scaled) { ... }
  return GPUMMAHeuristicSeeds({4, 16, 2, <derived>});
```

## Test

Add a lit test in `config_tile_and_fuse_gfx950.mlir` that runs with
`--iree-llvmgpu-use-direct-load=true` and verifies the selected config.

## Files Changed

- `ConfigUtils.cpp` — seed functions + `getMmaScheduleFromProblemAndTarget()`
- `ConfigUtils.h` — updated signatures
- `config_tile_and_fuse_gfx950.mlir` — new test case
