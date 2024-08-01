# Single-tile swizzling code samples for AMDGPU MFMA intrinsics.

The tests in this directory demonstrate how a `tensor.expand_shape` and a `linalg.transpose` can perform the tile-swizzling
to bring a mmt4d tile into the layout expected by AMDGPU MFMA kernels.

Some of these kernels consist of a single MFMA intrinsic, so their tile layout equals the intrinsic's layout as described in
https://github.com/iree-org/iree/blob/cddcd5b2eac99a0f6407bab3347e4c61fc6f3cb7/compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.cpp#L520
These are only stepping stones: a real optimized kernel will put several such tiles size by size along each dimension, and in some cases interleave them. The examples in this directory are concerned with those resulting kernel tiles. The intrinsic tiles are only a detail.

# MFMA kernels for `f32` matmuls

These kernels all use the same MFMA intrinsic: `mfma_f32_16x16x4f32`. To get familiar with what this intrinsic does, look at its [description](https://gist.github.com/bjacob/359882de84157eb7486af14f525fde07#file-mfma-txt-L1-L60) by `amd_matrix_instruction_calculator` and read its [CPU-threads model](https://github.com/bjacob/hip-matmul/blob/9ceb3c89763d9127573f49a3cf2000936bec3bc4/mfma_on_cpu_threads.cc#L28-L52).


## Single-intrinsic kernel

In this paragraph we write MLIR code to pack tiles into the layouts required (for optimal performance) by the `mfma_f32_16x16x4f32` intrinsic. Here is a sample kernel that would consume such tiles:

https://github.com/bjacob/hip-matmul/blob/9ceb3c89763d9127573f49a3cf2000936bec3bc4/matmul.hip#L567

The swizzling code (for the LHS ("A-matrix") and accumulator ("C-matrix)) for this kernel is in this test file:

[swizzle_1x1_mfma_f32_16x16x4f32.mlir](swizzle_1x1_mfma_f32_16x16x4f32.mlir)

Here is the swizzling code for the A-matrix tile:

```mlir
  %expand = tensor.expand_shape %src [[0], [1]] output_shape [16, 4] : tensor<16x4xf32> into tensor<16x4xf32>
  %transpose_init = tensor.empty() : tensor<4x16xf32>
  %transpose = linalg.transpose
    ins(%expand : tensor<16x4xf32>)
    outs(%transpose_init : tensor<4x16xf32>)
    permutation = [1, 0]
```

Here is the swizzling code for the C-matrix tile:

```mlir
  %expand = tensor.expand_shape %src [[0, 1], [2]] output_shape [4, 4, 16] : tensor<16x16xf32> into tensor<4x4x16xf32>
  %transpose_init = tensor.empty() : tensor<4x16x4xf32>
  %transpose = linalg.transpose
    ins(%expand : tensor<4x4x16xf32>)
    outs(%transpose_init : tensor<4x16x4xf32>)
    permutation = [0, 2, 1]
```

In this naive kernel consisting of a single intrinsic, the shapes and layout are just that of the intrinsic itself.

## 4-intrinsics kernel with 4x unroll in the K-dimension, interleaved.

The next kernel that we consider is a variation on the previous one where the K-dimension has been unrolled by 4x and
the resulting 4 intrinsic-tiles are interleaved. The rationale for this design is that each intrinsic reads 32-bit matrix elements from A and B, but the target favors reading 128 bits at once, so with this 4x-unrolling and interleaved layout, we can do more efficient 128-bit loads, and then we do 4 intrinsics at once.

Here is a sample kernel implementing this idea:

https://github.com/bjacob/hip-matmul/blob/9ceb3c89763d9127573f49a3cf2000936bec3bc4/matmul.hip#L612

The swizzling code (for the LHS ("A-matrix") and accumulator ("C-matrix)) for this kernel is in this test file:

[swizzle_1x1_mfma_f32_16x16x4f32_Kx4.mlir](swizzle_1x1_mfma_f32_16x16x4f32_Kx4.mlir)

Here is the swizzling code for the A-matrix tile:

```mlir
  %expand = tensor.expand_shape %src [[0], [1, 2]] output_shape [16, 4, 4] : tensor<16x16xf32> into tensor<16x4x4xf32>
  %transpose_init = tensor.empty() : tensor<4x16x4xf32>
  %transpose = linalg.transpose
    ins(%expand : tensor<16x4x4xf32>)
    outs(%transpose_init : tensor<4x16x4xf32>)
    permutation = [2, 0, 1]
```

Here is the swizzling code for the C-matrix tile (that one is unchanged from the previous kernel):

```mlir
  %expand = tensor.expand_shape %src [[0, 1], [2]] output_shape [4, 4, 16] : tensor<16x16xf32> into tensor<4x4x16xf32>
  %transpose_init = tensor.empty() : tensor<4x16x4xf32>
  %transpose = linalg.transpose
    ins(%expand : tensor<4x4x16xf32>)
    outs(%transpose_init : tensor<4x16x4xf32>)
    permutation = [0, 2, 1]
```

## 2x2 (MxN) expansion of the previous kernel

The next kernel that we consider is a variation on the previous one where we further unroll by 2x in each of the M and N
dimensions. The idea of unrolling in the M and N dimensions is not MFMA-specific, it is a common trait of matmul kernel design
on any target: by widening the kernel in the M and N dimensions, we achieve more reuse of the data loaded from the A and B
matrices, resulting in higher arithmetic intensity.

Here is a sample kernel implementing this idea:

https://github.com/bjacob/hip-matmul/blob/9ceb3c89763d9127573f49a3cf2000936bec3bc4/matmul.hip#L1288

In the above link, the kernel is templatized in template parameters MS and NS. To achieve the 2x2 unrolling, one would instantiate with MS=2, NS=2.

In practice, the template parameters tend to be MS=8, NS=8:

https://github.com/bjacob/hip-matmul/blob/9ceb3c89763d9127573f49a3cf2000936bec3bc4/matmul.hip#L2419

We just chose 2x2 in this example to avoid making it too heavy.

The swizzling code (for the LHS ("A-matrix") and accumulator ("C-matrix)) for this kernel is in this test file:

[swizzle_2x2_mfma_f32_16x16x4f32_Kx4.mlir](swizzle_2x2_mfma_f32_16x16x4f32_Kx4.mlir)

Here is the swizzling code for the A-matrix tile:

```mlir
  %expand = tensor.expand_shape %src [[0, 1], [2, 3]] output_shape [2, 16, 4, 4] : tensor<32x16xf32> into tensor<2x16x4x4xf32>
  %transpose_init = tensor.empty() : tensor<2x4x16x4xf32>
  %transpose = linalg.transpose
    ins(%expand : tensor<2x16x4x4xf32>)
    outs(%transpose_init : tensor<2x4x16x4xf32>)
    permutation = [0, 3, 1, 2]
```

Here is the swizzling code for the C-matrix tile:

```mlir
  %expand = tensor.expand_shape %src [[0, 1, 2], [3, 4]] output_shape [2, 4, 4, 2, 16] : tensor<32x32xf32> into tensor<2x4x4x2x16xf32>
  %transpose_init = tensor.empty() : tensor<2x2x4x16x4xf32>
  %transpose = linalg.transpose
    ins(%expand : tensor<2x4x4x2x16xf32>)
    outs(%transpose_init : tensor<2x2x4x16x4xf32>)
    permutation = [0, 3, 1, 4, 2]
```

## 8x8 (MxN) expansion of the previous kernel

As mentioned in the previous section, 2x2 was only given as an example, and the
actual kernel should use 8x8 (as [linked above](https://github.com/bjacob/hip-matmul/blob/9ceb3c89763d9127573f49a3cf2000936bec3bc4/matmul.hip#L2419
)). The e2e test for that would be heavy, but we can
at least give the swizzling code as a variant of the one in the previous paragraph.
The 2's become 8's, the reassociation and permutation indices stay the same.

Here is the swizzling code for the A-matrix tile:

```mlir
  %expand = tensor.expand_shape %src [[0, 1], [2, 3]] output_shape [8, 16, 4, 4] : tensor<128x16xf32> into tensor<8x16x4x4xf32>
  %transpose_init = tensor.empty() : tensor<8x4x16x4xf32>
  %transpose = linalg.transpose
    ins(%expand : tensor<8x16x4x4xf32>)
    outs(%transpose_init : tensor<8x4x16x4xf32>)
    permutation = [0, 3, 1, 2]
```

Here is the swizzling code for the C-matrix tile:

```mlir
  %expand = tensor.expand_shape %src [[0, 1, 2], [3, 4]] output_shape [8, 4, 4, 8, 16] : tensor<128x128xf32> into tensor<8x4x4x8x16xf32>
  %transpose_init = tensor.empty() : tensor<8x8x4x16x4xf32>
  %transpose = linalg.transpose
    ins(%expand : tensor<8x4x4x8x16xf32>)
    outs(%transpose_init : tensor<8x8x4x16x4xf32>)
    permutation = [0, 3, 1, 4, 2]
```

# MFMA kernels for `f16` matmuls accumulating into `f32`

These kernels all use the same MFMA intrinsic: `mfma_f32_16x16x16_f16`. To get familiar with what this intrinsic does, look at its [description](https://gist.github.com/bjacob/359882de84157eb7486af14f525fde07#file-mfma-txt-L63-L123) by `amd_matrix_instruction_calculator`.

We are not going to start all over again from a naive single-intrinsic kernel. For that, refer to the above `f32` section. Here we just right away to near the end.

## 2x2 (MxN) kernel with Kx2 interleaved

This is analogous to what we studied above for the `f32` case. Here, the intrinsic reads vectors of 4 `f16` values, so it reads 64 bits from the A-matrix. Since our target architecture favors 128-bit loads, we unroll that by 2x, and interleave the layouts, like above. We also unroll along the M and N dimensions by 2x2, without interleaving. Here, like above, 2x2 is only given as an example and the actual kernel (next paragraph) will be 8x8.


Here is a sample kernel implementing this idea:

https://github.com/bjacob/hip-matmul/blob/9ceb3c89763d9127573f49a3cf2000936bec3bc4/matmul.hip#L1907

The swizzling code (for the LHS ("A-matrix") and accumulator ("C-matrix)) for this kernel is in this test file:

[swizzle_2x2_mfma_f32_16x16x16f16_Kx2.mlir](swizzle_2x2_mfma_f32_16x16x16f16_Kx2.mlir)

Here is the swizzling code for the A-matrix tile:

```mlir
  %expand = tensor.expand_shape %src [[0, 1], [2, 3, 4]] output_shape [2, 16, 2, 4, 4] : tensor<32x32xf16> into tensor<2x16x2x4x4xf16>
  %transpose_init = tensor.empty() : tensor<2x4x16x2x4xf16>
  %transpose = linalg.transpose
    ins(%expand : tensor<2x16x2x4x4xf16>)
    outs(%transpose_init : tensor<2x4x16x2x4xf16>)
    permutation = [0, 3, 1, 2, 4]
```

Here is the swizzling code for the C-matrix tile (it's still the same as in the `f32` kernel):

```mlir
  %expand = tensor.expand_shape %src [[0, 1, 2], [3, 4]] output_shape [2, 4, 4, 2, 16] : tensor<32x32xf32> into tensor<2x4x4x2x16xf32>
  %transpose_init = tensor.empty() : tensor<2x2x4x16x4xf32>
  %transpose = linalg.transpose
    ins(%expand : tensor<2x4x4x2x16xf32>)
    outs(%transpose_init : tensor<2x2x4x16x4xf32>)
    permutation = [0, 3, 1, 4, 2]
```

## 8x8 (MxN) expansion of the previous kernel

As mentioned above, in practice a real kernel will have 8x8 rather than 2x2,
like here:

https://github.com/bjacob/hip-matmul/blob/9ceb3c89763d9127573f49a3cf2000936bec3bc4/matmul.hip#L2418

The e2e test for that would be heavy, but we can
at least give the swizzling code as a variant of the one in the previous paragraph.
The 2's (for the M/N dimensions, not K) become 8's, the reassociation and permutation indices stay the same.

Here is the swizzling code for the A-matrix tile:

```mlir
  %expand = tensor.expand_shape %src [[0, 1], [2, 3, 4]] output_shape [8, 16, 2, 4, 4] : tensor<128x32xf16> into tensor<8x16x2x4x4xf16>
  %transpose_init = tensor.empty() : tensor<8x4x16x2x4xf16>
  %transpose = linalg.transpose
    ins(%expand : tensor<8x16x2x4x4xf16>)
    outs(%transpose_init : tensor<8x4x16x2x4xf16>)
    permutation = [0, 3, 1, 2, 4]
```

Here is the swizzling code for the C-matrix tile (it's still the same as in the `f32` kernel):

```mlir
  %expand = tensor.expand_shape %src [[0, 1, 2], [3, 4]] output_shape [8, 4, 4, 8, 16] : tensor<128x128xf32> into tensor<8x4x4x8x16xf32>
  %transpose_init = tensor.empty() : tensor<8x8x4x16x4xf32>
  %transpose = linalg.transpose
    ins(%expand : tensor<8x4x4x8x16xf32>)
    outs(%transpose_init : tensor<8x8x4x16x4xf32>)
    permutation = [0, 3, 1, 4, 2]
```

# MFMA kernels for `i8` matmuls accumulating into `i32`

These kernels all use the same MFMA intrinsic: `mfma_i32_16x16x32_i8` (there is another competing intrinsic `mfma_i32_32x32x16_i8` which is thought to be more power-efficient but lower peak; it will likely enter the picture as a later refinement once we have basic things working; for now let us focus on using `mfma_i32_16x16x32_i8` only).

To get familiar with what this `mfma_i32_16x16x32_i8` intrinsic does, look at its [description](https://gist.github.com/bjacob/359882de84157eb7486af14f525fde07#file-mfma-txt-L126-L186) by `amd_matrix_instruction_calculator`.

We are not going to start all over again from a naive single-intrinsic kernel. For that, refer to the above `f32` section. Here we just right away to near the end.

## 2x2 (MxN) kernel with Kx2 interleaved

This is analogous to what we studied above for the `f32` case. Here, the intrinsic reads vectors of 8 `i8` values (weirdly using `i64` as "vector of `8xi8`"), so it reads 64 bits from the A-matrix. Since our target architecture favors 128-bit loads, we unroll that by 2x, and interleave the layouts, like above. We also unroll along the M and N dimensions by 2x2, without interleaving. Here, like above, 2x2 is only given as an example and the actual kernel (next paragraph) will be 8x8.

Here is a sample kernel implementing this idea:

https://github.com/bjacob/hip-matmul/blob/9ceb3c89763d9127573f49a3cf2000936bec3bc4/matmul.hip#L2191

The swizzling code (for the LHS ("A-matrix") and accumulator ("C-matrix)) for this kernel is in this test file:

[swizzle_2x2_mfma_i32_16x16x32i8_Kx2.mlir](swizzle_2x2_mfma_i32_16x16x32i8_Kx2.mlir)

Here is the swizzling code for the A-matrix tile:

```mlir
  %expand = tensor.expand_shape %src [[0, 1], [2, 3, 4]] output_shape [2, 16, 2, 4, 8] : tensor<32x64xi8> into tensor<2x16x2x4x8xi8>
  %transpose_init = tensor.empty() : tensor<2x4x16x2x8xi8>
  %transpose = linalg.transpose
    ins(%expand : tensor<2x16x2x4x8xi8>)
    outs(%transpose_init : tensor<2x4x16x2x8xi8>)
    permutation = [0, 3, 1, 2, 4]
```

Here is the swizzling code for the C-matrix tile (it's still the same as in the `f32` kernel, just with `i32` instead of `f32`):

```mlir
  %expand = tensor.expand_shape %src [[0, 1, 2], [3, 4]] output_shape [2, 4, 4, 2, 16] : tensor<32x32xi32> into tensor<2x4x4x2x16xi32>
  %transpose_init = tensor.empty() : tensor<2x2x4x16x4xi32>
  %transpose = linalg.transpose
    ins(%expand : tensor<2x4x4x2x16xi32>)
    outs(%transpose_init : tensor<2x2x4x16x4xi32>)
    permutation = [0, 3, 1, 4, 2]
```

## 8x8 (MxN) expansion of the previous kernel

As mentioned above, in practice a real kernel will have 8x8 rather than 2x2,
like here:

https://github.com/bjacob/hip-matmul/blob/9ceb3c89763d9127573f49a3cf2000936bec3bc4/matmul.hip#L2418

The e2e test for that would be heavy, but we can
at least give the swizzling code as a variant of the one in the previous paragraph.
The 2's (for the M/N dimensions, not K) become 8's, the reassociation and permutation indices stay the same.

Here is the swizzling code for the A-matrix tile:

```mlir
  %expand = tensor.expand_shape %src [[0, 1], [2, 3, 4]] output_shape [8, 16, 2, 4, 8] : tensor<128x64xi8> into tensor<8x16x2x4x8xi8>
  %transpose_init = tensor.empty() : tensor<8x4x16x2x8xi8>
  %transpose = linalg.transpose
    ins(%expand : tensor<8x16x2x4x8xi8>)
    outs(%transpose_init : tensor<8x4x16x2x8xi8>)
    permutation = [0, 3, 1, 2, 4]
```

Here is the swizzling code for the C-matrix tile (it's still the same as in the `f32` kernel, just with `i32` instead of `f32`):

```mlir
  %expand = tensor.expand_shape %src [[0, 1, 2], [3, 4]] output_shape [8, 4, 4, 8, 16] : tensor<128x128xi32> into tensor<8x4x4x8x16xi32>
  %transpose_init = tensor.empty() : tensor<8x8x4x16x4xi32>
  %transpose = linalg.transpose
    ins(%expand : tensor<8x4x4x8x16xi32>)
    outs(%transpose_init : tensor<8x8x4x16x4xi32>)
    permutation = [0, 3, 1, 4, 2]
```

