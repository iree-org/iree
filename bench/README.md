# AOT (HSACO) vs JIT (SPIR-V) Benchmarks

Compares IREE's native AOT path against the `amdgcnspirv` SPIR-V JIT path on
AMD Radeon AI PRO R9700 (gfx1201, RDNA4).

## Results (with comgr -O3 fix)

| Kernel | HSACO (AOT) | SPIR-V (JIT) | Ratio |
|--------|-------------|--------------|-------|
| matmul 2048³ f32 | 2.94 ms | 2.75 ms | 0.94x |
| add 2048² f32 | 0.067 ms | 0.067 ms | 1.0x |
| add 4096² f32 | 0.440 ms | 0.429 ms | 1.0x |

The JIT path now matches or slightly outperforms AOT. The previous 20x slowdown
was caused by comgr defaulting to `-O0` codegen when no `@llvm.cmdline` global
was present in the SPIR-V module. The fix embeds `-O3` via `@llvm.cmdline` in
the `ROCDLPrepareForSPIRVPass`.

## Usage

```bash
python3 bench/compare.py matmul 2048 2048 2048 --dtype f32 --reps 5
python3 bench/compare.py matmul 2048 2048 2048 --dtype f32 --transpose-b --reps 5
python3 bench/compare.py matmul 2048 2048 2048 --dtype f32 --isa-only --dump-isa
python3 bench/compare.py file bench/matmul_f32.mlir --reps 5
```
