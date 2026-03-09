# SPIR-V JIT Register Spilling Repro

Demonstrates the ~20x performance gap between IREE's AOT (HSACO) and JIT
(SPIR-V) paths for f32 matmul 2048x2048x2048 on gfx1201 (RDNA4).

## Root Cause

The ROCm comgr JIT pipeline runs AMDGPU codegen at **-O0** (or equivalent)
for SPIR-V inputs. This causes massive register spilling (1229 scratch ops)
and no instruction scheduling (0 dual-issue ops). The same IR compiled at
-O2 produces 0 scratch spills regardless of whether buffer instructions are
present.

## Files

- `matmul_f32_buffer.ll` — IREE AOT output with buffer instructions (AS 7)
- `matmul_f32_global.ll` — SPIR-V round-trip output (reverse-translated by
  `amd-llvm-spirv -r`), uses `global_load` instead of `buffer_load`

## Reproducing

Requires `llc` with AMDGPU target and ROCm's `clang-22`.

```bash
# 1. Compile AOT version (baseline — 0 scratch spills):
llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1201 -filetype=asm \
    matmul_f32_buffer.ll -o buffer.s

# 2. Compile global version at -O2 (also 0 scratch spills!):
llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1201 -filetype=asm \
    matmul_f32_global.ll -o global_O2.s

# 3. Compile global version at -O0 (reproduces JIT spilling):
clang -target amdgcn-amd-amdhsa -mcpu=gfx1201 -S -O0 \
    matmul_f32_global.ll -o global_O0.s

# Compare:
echo "buffer -O2:  scratch=$(grep -c scratch_ buffer.s)"
echo "global -O2:  scratch=$(grep -c scratch_ global_O2.s)"
echo "global -O0:  scratch=$(grep -c scratch_ global_O0.s)"
# Expected: 0, 0, ~1229
```

## ISA Comparison

| Metric | buffer -O2 | global -O2 | global -O0 (JIT) |
|--------|-----------|-----------|-------------------|
| `scratch_*` (spill) | 0 | 0 | **1229** |
| `buffer_load` | 136 | 0 | 0 |
| `global_load` | 0 | 136 | 160 |
| `v_dual_*` | 97 | 163 | **0** |
| `s_wait*` | 121 | 74 | **672** |

The `-O0` column matches the actual JIT output (extracted from comgr's
`hip_code_object.spv.bc.o` via `llvm-objdump`).

## Key Insight

The lack of buffer instructions is **not** the primary cause of the 20x
slowdown. At `-O2`, the global-load version produces comparable code
(0 spills, even more dual-issue ops than the buffer version). The entire
performance gap comes from comgr running codegen at `-O0`.
