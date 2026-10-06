# MLIR Microkernel Template Guide

## Data-Tiled Kernels

### `pingpong_dt_medium_f4E2M1FN`
```bash
python mlir_ukernel_gen.py iree_uk_amdgpu_dt_scaled_matmul_f4E2M1FN.mlir.in \
  -D INTRINSIC=MFMA_SCALE_F32_16x16x128_B32 \
     INTRINSICS_M=4 INTRINSICS_N=8 INTRINSICS_K=2 SUBGROUPS_M=2 SUBGROUPS_N=2 ARCH=gfx950 \
  -o generated/iree_uk_amdgpu_dt_scaled_matmul_f4E2M1FN.mlir
```

### `pingpong_dt_large_f16`
```bash
python mlir_ukernel_gen.py iree_uk_amdgpu_dt_matmul_large.mlir.in \
  -D ELEM_TYPE=f16 INTRINSIC=MFMA_F32_16x16x16_F16 \
     INTRINSICS_M=8 INTRINSICS_N=4 INTRINSICS_K=1 SUBGROUPS_M=2 SUBGROUPS_N=4 ARCH=gfx942 \
     SIZE_MIN_0=512 SIZE_DIV_0=64 SIZE_MIN_1=32832 SIZE_DIV_1=64 SIZE_MIN_2=512 SIZE_DIV_2=64 \
  -o generated/iree_uk_amdgpu_dt_matmul_f16.mlir
```

### `pingpong_dt_large_f8E4M3FNUZ`
```bash
python mlir_ukernel_gen.py iree_uk_amdgpu_dt_matmul_large.mlir.in \
  -D ELEM_TYPE=f8E4M3FNUZ INTRINSIC=MFMA_F32_16x16x32_F8E4M3FNUZ \
     INTRINSICS_M=8 INTRINSICS_N=4 INTRINSICS_K=1 SUBGROUPS_M=2 SUBGROUPS_N=4 ARCH=gfx942 \
     SIZE_MIN_0=64 SIZE_MIN_1=2048 SIZE_MAX_1=8192 \
  -o generated/iree_uk_amdgpu_dt_matmul_f8E4M3FNUZ_large.mlir
```

### `pingpong_dt_medium_f8E4M3FNUZ`
```bash
python mlir_ukernel_gen.py iree_uk_amdgpu_dt_matmul_medium.mlir.in \
  -D ELEM_TYPE=f8E4M3FNUZ INTRINSIC=MFMA_F32_16x16x32_F8E4M3FNUZ \
     INTRINSICS_M=8 INTRINSICS_N=2 INTRINSICS_K=2 SUBGROUPS_M=1 SUBGROUPS_N=8 ARCH=gfx942 \
     SIZE_MIN_0=32 \
  -o generated/iree_uk_amdgpu_dt_matmul_f8E4M3FNUZ_medium.mlir
```

### `pingpong_dt_large_f8E4M3FN`
```bash
python mlir_ukernel_gen.py iree_uk_amdgpu_dt_matmul_large.mlir.in \
  -D ELEM_TYPE=f8E4M3FN INTRINSIC=MFMA_F32_16x16x32_F8E4M3FN \
     INTRINSICS_M=8 INTRINSICS_N=4 INTRINSICS_K=1 SUBGROUPS_M=2 SUBGROUPS_N=4 ARCH=gfx950 \
     SIZE_MIN_0=64 SIZE_MIN_1=2048 SIZE_MAX_1=8192 \
  -o generated/iree_uk_amdgpu_dt_matmul_f8E4M3FN_large.mlir
```

### `pingpong_dt_medium_f8E4M3FN`
```bash
python mlir_ukernel_gen.py iree_uk_amdgpu_dt_matmul_medium.mlir.in \
  -D ELEM_TYPE=f8E4M3FN INTRINSIC=MFMA_F32_16x16x32_F8E4M3FN \
     INTRINSICS_M=8 INTRINSICS_N=2 INTRINSICS_K=2 SUBGROUPS_M=1 SUBGROUPS_N=8 ARCH=gfx950 \
     SIZE_MIN_0=32 \
  -o generated/iree_uk_amdgpu_dt_matmul_f8E4M3FN_medium.mlir
```

---

## Non-Data-Tiled Kernels

### `pingpong_large_f16`
```bash
python mlir_ukernel_gen.py iree_uk_amdgpu_matmul_large.mlir.in \
  -D ELEM_TYPE=f16 INTRINSIC=MFMA_F32_16x16x16_F16 \
     INTRINSICS_M=8 INTRINSICS_N=4 INTRINSICS_K=1 \
     SUBGROUPS_M=2 SUBGROUPS_N=4 ARCH=gfx942 \
  -o generated/pingpong_large_f16.mlir
```

### `pingpong_medium_f16_expanded`
```bash
python mlir_ukernel_gen.py iree_uk_amdgpu_matmul_medium.mlir.in \
  -D ELEM_TYPE=f16 INTRINSIC=MFMA_F32_16x16x16_F16 \
     INTRINSICS_M=4 INTRINSICS_N=4 INTRINSICS_K=2 \
     SUBGROUPS_M=2 SUBGROUPS_N=4 ARCH=gfx942 \
  -o generated/pingpong_medium_f16_expanded.mlir
```

### `pingpong_large_bf16`
```bash
python mlir_ukernel_gen.py iree_uk_amdgpu_matmul_large.mlir.in \
  -D ELEM_TYPE=bf16 INTRINSIC=MFMA_F32_16x16x16_BF16 \
     INTRINSICS_M=8 INTRINSICS_N=4 INTRINSICS_K=1 \
     SUBGROUPS_M=2 SUBGROUPS_N=4 ARCH=gfx942 \
  -o generated/pingpong_large_bf16.mlir
```

### `pingpong_medium_bf16_expanded`
```bash
python mlir_ukernel_gen.py iree_uk_amdgpu_matmul_medium.mlir.in \
  -D ELEM_TYPE=bf16 INTRINSIC=MFMA_F32_16x16x16_BF16 \
     INTRINSICS_M=4 INTRINSICS_N=4 INTRINSICS_K=2 \
     SUBGROUPS_M=2 SUBGROUPS_N=4 ARCH=gfx942 \
  -o generated/pingpong_medium_bf16_expanded.mlir
```

### `pingpong_medium_f8E4M3FNUZ_expanded`
```bash
python mlir_ukernel_gen.py iree_uk_amdgpu_matmul_medium.mlir.in \
  -D ELEM_TYPE=f8E4M3FNUZ INTRINSIC=MFMA_F32_16x16x32_F8E4M3FNUZ \
     INTRINSICS_M=4 INTRINSICS_N=4 INTRINSICS_K=2 \
     SUBGROUPS_M=2 SUBGROUPS_N=4 ARCH=gfx942 \
  -o generated/pingpong_medium_f8E4M3FNUZ_expanded.mlir
```

### `pingpong_large_f8E4M3FNUZ_expanded`
```bash
python mlir_ukernel_gen.py iree_uk_amdgpu_matmul_large.mlir.in \
  -D ELEM_TYPE=f8E4M3FNUZ INTRINSIC=MFMA_F32_16x16x32_F8E4M3FNUZ \
     INTRINSICS_M=8 INTRINSICS_N=4 INTRINSICS_K=1 \
     SUBGROUPS_M=2 SUBGROUPS_N=4 ARCH=gfx942 \
  -o generated/pingpong_large_f8E4M3FNUZ_expanded.mlir
```

### `pingpong_medium_f8E4M3FN_expanded`
```bash
python mlir_ukernel_gen.py iree_uk_amdgpu_matmul_medium.mlir.in \
  -D ELEM_TYPE=f8E4M3FN INTRINSIC=MFMA_F32_16x16x32_F8E4M3FN \
     INTRINSICS_M=4 INTRINSICS_N=4 INTRINSICS_K=2 \
     SUBGROUPS_M=2 SUBGROUPS_N=4 ARCH=gfx942 \
  -o generated/pingpong_medium_f8E4M3FN_expanded.mlir
```

### `pingpong_large_f8E4M3FN_expanded`
```bash
python mlir_ukernel_gen.py iree_uk_amdgpu_matmul_large.mlir.in \
  -D ELEM_TYPE=f8E4M3FN INTRINSIC=MFMA_F32_16x16x32_F8E4M3FN \
     INTRINSICS_M=8 INTRINSICS_N=4 INTRINSICS_K=1 \
     SUBGROUPS_M=2 SUBGROUPS_N=4 ARCH=gfx942 \
  -o generated/pingpong_large_f8E4M3FN_expanded.mlir
```
