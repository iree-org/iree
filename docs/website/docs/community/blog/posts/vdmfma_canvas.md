---
date: 2026-05-28
authors:
  - efric
categories:
  - Performance
tags:
  - GPU
  - Codegen
---

# Virtual Dense MFMAs for Skinny GEMM

When we have a GEMM `A * B = C`, and it is the situation that `A` has a
small number of rows and many columns, we classify this problem as a skinny
GEMM. The decode phase of LLM inference is a common sight of this problem: a
small batch of tokens multiplies against a large weight matrix. Skinny GEMMs are
less convenient for modern GPU architectures than their non-skinny cousins. One
reason is because modern GPUs take advantage of matrix core units which offer
instructions that are specifically designed for matrix multiplication and
operate on fixed tile sizes, and skinny GEMMs are too small to utilize them to
their intended size.

On AMDGPUs and in particular on the MI3XX Instinct (CDNA) series, these
instructions are known as MFMA instructions; for example,
`V_MFMA_F32_16x16x16_F16`. One useful part of the name is the `MxNxK`
tile shape, where `M` is the number of rows of the left hand matrix, `N` is the
number of columns of the right hand matrix, and `K` is the shared dimension of
both.

<!-- more -->

For the ordinary dense GEMM MFMA path available to AMDGPU CDNA series, the
relevant 16-bit and 8-bit MFMAs have at least 16 rows in M. Consider M=8, which
is larger than the path we take in IREE for GEMV-like problems, but evidently
smaller than 16. The previous codegen path in IREE handled this by padding the
workgroup `M` tile to 16 and
using the ordinary dense MFMA configuration. The IR snippet below shows this
directly: the logical `M=8` operation is configured with `padding = [16, ...]`,
a dense `mma_layout`, and a `workgroup` tile of 16 rows.

<!-- markdownlint-disable -->
<details><summary>IR with padding</summary>

```mlir
%10 = linalg.generic {
  indexing_maps = [
    affine_map<(d0, d1, d2) -> (d0, d2)>,
    affine_map<(d0, d1, d2) -> (d1, d2)>,
    affine_map<(d0, d1, d2) -> (d0, d1)>
  ],
  iterator_types = ["parallel", "parallel", "reduction"]
} ins(%6, %7 : tensor<8x16384xf16>, tensor<13312x16384xf16>)
  outs(%9 : tensor<8x13312xf32>)
  attrs = {
    lowering_config = #iree_gpu.lowering_config<{
      mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
      padding = [16, 64, 128],
      promote_operands = [0, 1],
      reduction = [0, 0, 8],
      subgroup = [1, 2, 0],
      workgroup = [16, 64, 0]
    }>
  } {
  ...
}
```

</details>
<!-- markdownlint-restore -->

Padding is simple and robust, but we would be wasting cycles on rows that are
not present in the original matrix. The question is whether we can use the 16
physical rows of the hardware instruction more carefully.

## Removing Padding with Sparse MFMA

AMD sparse MFMA instructions, `V_SMFMAC`, are matrix-core accumulate
instructions for a 4:2 structured-sparse `A` matrix and a dense `B` matrix. The
old `D` value is the accumulator, and the encoded third source is sparse index
metadata, not a separate `C` matrix operand. The `A` operand is sparse along
`K`: in each group of four `K` positions, the sparse index metadata tells the
instruction which two positions are present.

On CDNA3/gfx942, the relevant sparse instruction has the same physical `16x16`
output tile but twice the K-depth of the corresponding dense 16x16 MFMA. For
F16/BF16, dense `V_MFMA_F32_16X16X16_F16` and sparse
`V_SMFMAC_F32_16X16X32_F16` are both 16-cycle instructions on gfx942. For
8-bit inputs, the analogous 16-cycle sparse instruction is `16x16x64`.

The idea, described in the Hugging Face
[MI300 kernel article](https://huggingface.co/blog/mi300kernels), is to make
two sparse rows represent one dense row. One lane selects positions
`{0, 1}` in each group of four. Its paired lane selects positions `{2, 3}`.
Together, the two lanes cover the dense `K` positions for one logical row. The
benefit, in addition to removing padding, is that a 16-cycle sparse
instruction covers twice the logical `K` depth of the corresponding dense
16-cycle F16/BF16 MFMA.

![Using sparsity for skinny inputs](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/mi300kernels/sparsity_trick.png)

Figure: "Using sparsity for skinny inputs" from
[Creating custom kernels for the AMD MI300](https://huggingface.co/blog/mi300kernels).

After the sparse MFMA, the four-element native accumulator contains pairs of
partial sums for the same logical rows. The lowering adds those pairs together,
so the result again has the normal dense `M=8` meaning.

## Original HuggingFace Approach

On the standard path for processing data enroute to MFMA instructions, we go
through global memory -> LDS/Shared memory -> Registers -> MFMA instruction.*
In the original Hugging Face skinny GEMM kernel, data from matrix `A` is shuffled
on the way into LDS. The shuffle is necessary to meet the semantics of using the
sparse trick. If we were to use even lanes to select positions `{0,1}` and odd
lanes to select positions `{2,3}`, then for a load with 8 contiguous elements along
`K`:

```text
K0 K1 K2 K3 K4 K5 K6 K7
```

We would want even lanes to hold:

```text
K0 K1 K4 K5
```

and odd lanes to hold

```text
K2 K3 K6 K7
```

In other words, the data loaded from LDS looks exactly like:

```text
lane 0: K0 K1 _  _  K4 K5 _  _
lane 1: _  _  K2 K3 _  _  K6 K7
```

Together (as an even/odd pair), and across all threads in the subgroup, these
precisely reconstruct the original dense rows. Following the loop around the
inner K tile, these partials are then reduced to yield the full dense result.

??? note "* Shared-memory hierarchy note"

    This path is a simplified storyline. The actual shared-memory hierarchy has
    more detail than is useful for the VDMFMA discussion; refer to the AMDGPU ISA
    documentation for the full memory hierarchy and instruction-level behavior.

## Design and Implementation in IREE

The HF kernel makes `A` sparse-trick "friendly" before we read it from shared
memory. If IREE wanted to materialize that shuffled `A` form as a
compiler-owned tensor or storage layout, the natural existing mechanism would be
data tiling: attach an encoding, carry the encoded tensor type through the
producer/consumer boundary, and materialize the layout change with
packing/unpacking or other physical layout operations when needed. That is the
model described in IREE's [data-tiling path](data-tiling-walkthrough.md). In the
GPU data-tiling path, encoded contractions reach
`#iree_gpu.data_tiled_mma_layout` on `iree_codegen.inner_tiled`.

Instead, we take advantage of "virtual MMAs" in IREE. Virtual MMAs in IREE
represent a lowering which is intended to match real MFMAs in the same way but
are otherwise composed of or are a modification of ordinary MFMAs.
`#iree_gpu.virtual_mma_layout` is an MMA/inner-tile descriptor: it supplies the
semantic tile shape, distributed fragment layout, and target lowering, while the
promoted/shared-memory layouts remain dense and ordinary. Using VDMFMAs, we do
not need to write a sparse-trick friendly `A` tile to LDS and we do not make the
promoted shared-memory layout carry the sparse trick. Instead, the
subgroup-level MMA lowering keeps `A` as the dense per-lane fragment loaded from
LDS, slices that fragment into native sparse-MFMA chunks, and performs a
per-lane shuffle of the `B` matrix register data so the dense `K` values line up
with the positions selected by the sparse metadata. Choosing to shuffle `B` in
registers keeps this part local to the virtual MMA; shuffling `A` into LDS would
also need a matching promotion/read layout for that operand. The final assembly
forms those fragments with `ds_read2_b64` LDS reads, which incidentally loads
twice as much data from LDS as the HF kernel. For CDNA3 F16/BF16, the virtual
`8x16x64` operation lowers to two native `16x16x32` SMFMAC instructions because
each native sparse instruction covers `K=32`.

In doing so, we give flexibility and keep the sparse trick from becoming a
skinny-only tensor layout. The current selector still uses it conservatively,
only when the problem's total
`M` fits in the virtual `M=8` tile and total `K` is divisible by the VDMFMA
selection tile. But the abstraction is an `8`-row virtual MMA, not an encoded
storage format for an entire matmul. A future selector could tile a larger
multiple-of-8 `M` problem into VDMFMA-sized pieces without making producers
materialize a fake sparse `A` tensor.

Concretely, IREE represents VDMFMA as a virtual MMA layout:

```mlir
#iree_gpu.virtual_mma_layout<VDMFMA_F32_8x16x64x2_F16>
```

Read this as a dense `8x16x64` virtual operation with F16 inputs and F32
accumulation. The trailing `x2` says that, on the
CDNA3 F16 path, the virtual operation lowers to two native sparse MFMA
instructions along `K`.

At the virtual interface, each lane sees dense fragments:

```text
A   : vector<8xf16>
B   : vector<16xf16>
Acc : vector<2xf32>
```

The native sparse instruction wants a different physical view:

```text
A           : vector<4xf16>
B           : vector<8xf16>
Acc/D       : vector<4xf32>
SparseIndex : vector<4xi8>
```

VDMFMA is the adapter between these two views. It expands the accumulator,
chooses sparse metadata from lane parity, slices `A`, interleaves the per-lane
`B` register fragment, issues the sparse MFMAs, and collapses the accumulator
back to the dense virtual shape.

A native F16 sparse MFMA consumes four F16 `A` values per lane. The VDMFMA F16
fragment has eight, so the lowering emits two native sparse MFMAs. Each one
uses a different half of `A` and a different per-lane shuffle of `B`.

For one lane pair, the two instructions can be visualized as follows. The `K`
numbering below is the numbering in the dense per-lane fragment after
distribution. `--` marks `A` positions that are implied zero for that physical
sparse row. The non-zero `A` samples are packed, and sparse index metadata maps
them back to positions within each `K` group of four.

```text
                         first smfmac                         second smfmac
sparse A positions       0   1   2   3 | 0   1   2    3       0   1   2    3 | 0   1   2    3
L0, selector 0x44        K0  K1  --  --| K2  K3  --   --      K4  K5  --   --| K6  K7  --   --
L1, selector 0xEE        --  --  K8  K9| --  --  K10  K11     --  --  K12  K13| --  --  K14  K15
B after shuffle          B0  B1  B8  B9| B2  B3  B10  B11     B4  B5  B12  B13| B6  B7  B14  B15
```

The corresponding shuffle indices in the lowering are:

```text
first smfmac  B shuffle: [0, 1, 8, 9, 2, 3, 10, 11]
second smfmac B shuffle: [4, 5, 12, 13, 6, 7, 14, 15]
```

Within each physical sparse group of four, the even lane contributes the values
that the sparse selector places in positions `{0, 1}`, while the odd lane
contributes the values placed in positions `{2, 3}`. The `B` operand has to be
interleaved in that same physical order. Otherwise the dense `K` positions
covered by the two lanes would be multiplied by the wrong `B` values.

The lowering is:

```text
[c0, c1] -> [c0, 0, c1, 0]

sparse_index = (lane_id & 1) ? 0xEE : 0x44

smfmac(A[0:4], shuffle(B, [0, 1, 8, 9, 2, 3, 10, 11]))
smfmac(A[4:8], shuffle(B, [4, 5, 12, 13, 6, 7, 14, 15]))

[d0, d1, d2, d3] -> [d0 + d1, d2 + d3]
```

The accumulator conversions are wrapped in `util.hoistable_conversion`. In
IREE, this marks temporary marshaling between the layout used by `inner_tiled`
and the layout expected by the target intrinsic, so matching conversions can be
moved out of loops or canceled when the surrounding IR permits it. For VDMFMA,
that marshaling expands the logical two-element accumulator into the
four-element SMFMAC form before the sparse MFMA chain, then collapses the native
accumulator back by summing lane-pair partials.

## Reading the Single-Subgroup Layout

The virtual MMA layout uses `MMASingleSubgroupLayout`, so it is worth unpacking
the terminology.

A single-subgroup layout describes how one operand of one subgroup-level matrix
operation is distributed across lanes. For each semantic operand dimension, such
as `M`, `N`, or `K`, it has:

* `outer`: repetitions outside the lane decomposition;
* `thread`: the logical thread grid over that dimension;
* `tstrides`: the lane-id stride for each cross-thread dimension;
* `element`: the contiguous logical element tile for one thread-grid position,
  before any `physicalLanesPerThread` split.

For each dimension, `outer[i] * thread[i] * element[i]` is the semantic tile
size. For the F16 VDMFMA LHS, IREE uses:

```text
outer    = {1, 1}
thread   = {8, 4}
tstrides = {2, 16}
element  = {1, 16}
```

The semantic dimensions are `M` and `K`, so this is an `8x64` LHS tile:
`1 * 8 * 1 = 8` rows and `1 * 4 * 16 = 64` reduction elements.

For the LHS layout, the product of `thread` is 32, while the VDMFMA subgroup has
64 lanes. With `tstrides = {2, 16}`,
lanes `2p` and `2p+1` share the same logical M/K thread-grid coordinates. We then
then split the divisible element dimension, K, so lane `2p` receives the lower
8 elements of the 16-wide `K` element tile and lane `2p+1` receives the upper 8.
The RHS and accumulator layouts use 64 logical thread positions and are not split
in the same way.

This is the layout-side part that gives VDMFMA the "virtual dense" behavior: the
compiler still distributes a dense `8x64` LHS tile, but the physical lanes are
grouped so that each even/odd lane pair owns the two dense halves that the sparse
instruction trick will reinterpret. The `{0, 1}` versus `{2, 3}` sparse-position
meaning is introduced later by the MMA lowering: lane-parity sparse metadata,
`B` interleaving, sparse MFMA emission, and accumulator collapse.

## Selecting VDMFMA

VDMFMA is not selected for every matmul. IREE has multiple codegen pipelines,
and the one which is relevant for the shape of skinny GEMMs belongs to
TileAndFuse. TileAndFuse derives VDMFMA candidates from the target's concrete
MFMA capabilities. On the CDNA3 F16 path, the
virtual `VDMFMA_F32_8x16x64x2_F16` candidate is derived from
`MFMA_F32_16x16x16_F16`.

The selector is intentionally narrow today: it only considers VDMFMA when the
problem's total `M` is at most 8, and when the total `K` is divisible by the
VDMFMA K tile size used by the selector. For `VDMFMA_F32_8x16x64x2_F16`, that
selection tile is `2 * 64 = 128` elements along `K`.

There is one tuning detail that is easy to miss. Since sparse MFMAs have twice
the K-depth as dense MFMAs, the compute phase is shorter than the padded dense
MFMA sequence it replaces.
In a software-pipelined loop, that can reduce the amount of compute available
to hide the next tile's memory latency. The final selection change scales the
reduction tile count by the virtual intrinsic's K unroll factor to compensate
for the shorter compute phase.

With VDMFMA selected for the same shape, the new IR excerpt
has no `M=16` padding. The workgroup `M` tile is 8, and the MMA kind is the
virtual layout.

<!-- markdownlint-disable -->
<details><summary>IR with VDMFMA</summary>

```mlir
%10 = linalg.generic {
  indexing_maps = [
    affine_map<(d0, d1, d2) -> (d0, d2)>,
    affine_map<(d0, d1, d2) -> (d1, d2)>,
    affine_map<(d0, d1, d2) -> (d0, d1)>
  ],
  iterator_types = ["parallel", "parallel", "reduction"]
} ins(%6, %7 : tensor<8x16384xf16>, tensor<13312x16384xf16>)
  outs(%9 : tensor<8x13312xf32>)
  attrs = {
    lowering_config = #iree_gpu.lowering_config<{
      mma_kind =
        #iree_gpu.virtual_mma_layout<VDMFMA_F32_8x16x64x2_F16>,
      promote_operands = [0, 1],
      reduction = [0, 0, 4],
      subgroup = [1, 2, 0],
      workgroup = [8, 64, 0]
    }>
  } {
  ...
}
```

</details>
<!-- markdownlint-restore -->

## Performance

The first end-to-end 16-bit selection change reported the following numbers on
CDNA3, compared with the padded dense baseline:

| Shape | VDMFMA | Baseline | Improvement |
| --- | ---: | ---: | ---: |
| `f16_8x13312x16384` | 189 us | 206 us | +8.3% |
| `f16_8x13312x8192` | 117 us | 116 us | - |
| `f16_8x2304x16384` | 133 us | 138 us | +3.6% |
| `f16_8x2304x8192` | 103 us | 110 us | +6.4% |
| `f16_8x6656x16384` | 127 us | 130 us | +2.3% |
| `f16_8x6656x8192` | 102 us | 109 us | +6.4% |

## Conclusion

VDMFMA is a small compiler abstraction around a target-specific instruction
mapping. At the IR boundary, it is a dense `8x16xK` virtual MMA. The generated
code for the F16 kernel above uses paired `ds_read2_b64` LDS reads to form
dense per-lane fragments; the virtual MMA lowering then uses lane parity, sparse
selectors, `B` register interleaving, and accumulator folding to invoke AMD
sparse MFMA instructions. At configuration time, it is selected only for skinny
shapes where the total `M` fits within the virtual `M=8` tile and total `K`
is divisible by the VDMFMA selection tile. The result is an end-to-end
adaptation of a hand-written HIP optimization into IREE's AMDGPU codegen
pipeline.
