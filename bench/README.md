# AOT (HSACO) vs JIT (SPIR-V) Benchmarks

Compares IREE's native AOT path against the `amdgcnspirv` SPIR-V JIT path on
AMD Radeon AI PRO R9700 (gfx1201, RDNA4).

## Results

| Kernel | HSACO (AOT) | SPIR-V (JIT) | Slowdown |
|--------|-------------|--------------|----------|
| matmul 2048³ f32 | 2.95 ms (5.82 TF) | 57.8 ms (0.30 TF) | 19.7x |
| matmul 4096³ f32 | 24.4 ms (5.63 TF) | 629 ms (0.22 TF) | 25.8x |
| matmul 1024³ f32 tb | 1.55 ms (1.39 TF) | 7.31 ms (0.29 TF) | 4.7x |
| add 2048² f32 | 0.067 ms | 0.067 ms | 1.0x |
| add 4096² f32 | 0.440 ms | 0.429 ms | 1.0x |

Bandwidth-bound kernels (elementwise add) show no difference. Compute-bound
matmul shows 5–26x slowdown depending on size.

## ISA Analysis: f32 Matmul 2048³

| Instruction | AOT | JIT | Delta |
|-------------|-----|-----|-------|
| `buffer_load` | 136 | 0 | -136 |
| `global_load` | 0 | 160 | +160 |
| `scratch_*` (spill) | 0 | 1229 | +1229 |
| `v_fmac` (FMA) | 512 | 512 | 0 |
| `v_dual_*` (dual-issue) | 194 | 0 | -194 |
| `s_wait*` (stalls) | 121 | 672 | +551 |

The JIT path loses buffer instructions (SPIR-V has no representation for AMDGPU
address space 7) and suffers massive register spilling. However, compiling the
same reverse-translated IR at `-O2` produces 0 scratch spills and comparable
dual-issue counts — the spilling is caused by comgr running AMDGPU codegen at
`-O0` (or equivalent), not by the loss of buffer instructions.

See `spirv-repros/spill_repro/` for a standalone reproduction.

## Usage

```bash
python3 bench/compare.py matmul 2048 2048 2048 --dtype f32 --reps 5
python3 bench/compare.py matmul 2048 2048 2048 --dtype f32 --transpose-b --reps 5
python3 bench/compare.py matmul 2048 2048 2048 --dtype f32 --isa-only --dump-isa
python3 bench/compare.py file bench/matmul_f32.mlir --reps 5
```
