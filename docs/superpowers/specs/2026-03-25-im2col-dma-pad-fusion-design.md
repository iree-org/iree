# Im2col DMA Pad Fusion Design

## Goal

Support `padding`/`padding_conv` configs with im2col DMA by fusing
`tensor.pad` into the gather DMA path, analogous to how copy DMA already
handles pad fusion.

## Problem

When the auto-selected config includes `padding`/`padding_conv`, the im2col
input is padded (e.g., `tensor<1x10x10x512xf16>` → `tensor<1x10x18x512xf16>`).
The im2col→gather conversion collapses the padded tensor and computes indices
against the padded shape. After bufferization, the padded tensor materializes
in `#gpu.address_space<private>` (it must write padding values somewhere).
`amdgpu.gather_to_lds` then fails because it requires `fat_raw_buffer` source.

## Background: How Copy DMA Handles Padding

In `createDMAInForall<CopyOp>` (line 443), when the copy's input traces to a
`tensor.pad`:
1. Uses `pad.getSource()` as DMA source (bypasses pad materialization)
2. Validates: low padding = 0, pad value = 0, row size DWORD-aligned
3. Computes `in_bounds` per dimension from pad's low/high values
4. The `AMDGPULowerCoalescedDMAToGatherLDS` pass handles OOB: for non-outermost
   OOB dims, forces outermost index to `sourceShape[0]`, pushing linearized
   offset past buffer end. `fat_raw_buffer` hardware returns zero for OOB reads.

## Approach: Pad Fusion in `ConvertGatherToCoalescedDMA`

Fuse `tensor.pad` into the gather DMA path by rewriting the source and indices
in `ConvertGatherToCoalescedDMA::matchAndRewrite` (GPUConvertToCoalescedDMA.cpp,
line 683).

### Where in the Code

The im2col gather path goes through `ConvertGatherToCoalescedDMA` (not
`createDMAInForall<GatherOp>`). At line 785-790, after thread-level tiling:

```cpp
Value source = tiledGatherOp.getSource();
// If source comes from an extract_slice, get its source (from warp-level).
if (auto extractOp = source.getDefiningOp<tensor::ExtractSliceOp>()) {
  source = extractOp.getSource();
}
```

The `source` here is the warp-level source. It is an `extract_slice` from the
collapsed padded tensor. We need to trace further to find the pad and replace
the source.

The indices at this point are **1-D index tensors** (e.g.,
`tensor<8xindex>`), not vectors. They are passed directly to
`CoalescedGatherDMAOp::create` as tensor operands and stay as tensors through
bufferization. The index rewriting must therefore operate on the index tensor
(via `linalg.generic`), not on vectors.

### IR Before (Current Failure)

```mlir
%padded = tensor.pad %input low[0,0,0,0] high[0,0,8,0] {
  tensor.yield %zero : f16
} : tensor<1x10x10x512xf16> to tensor<1x10x18x512xf16>

%collapsed = tensor.collapse_shape %padded [[0,1,2],[3]]
  : tensor<1x10x18x512xf16> into tensor<180x512xf16>

// Indices linearized with padded strides: basis [1, 10, 18]
%indices = linalg.generic { ... } -> tensor<64xindex>

// After warp + thread tiling:
%warp_src = tensor.extract_slice %collapsed[0, %k_off] [180, 128] [1, 1]
//   ^-- thread tiling peels another extract_slice from this
%thread_src = tensor.extract_slice %warp_src[...] [180, 128] [1, 1]
//   ^-- ConvertGatherToCoalescedDMA traces back to %warp_src (line 788)

%dma = iree_gpu.coalesced_gather_dma %warp_src[%idx_slice] into %dest ...
```

### IR After (Proposed Fix)

```mlir
// Trace: warp_src (extract_slice) → collapsed (collapse_shape) → padded (pad) → %input
%raw_collapsed = tensor.collapse_shape %input [[0,1,2],[3]]
  : tensor<1x10x10x512xf16> into tensor<100x512xf16>

%raw_src = tensor.extract_slice %raw_collapsed[0, %k_off] [100, 128] [1, 1]

// Rewrite indices tensor: delinearize with padded shape, re-linearize with
// unpadded shape, clamp OOB
%new_indices = linalg.generic { ... rewrite logic ... } -> tensor<8xindex>

%dma = iree_gpu.coalesced_gather_dma %raw_src[%new_indices] into %dest
  lane(%lane) in_bounds [false, true] ...
```

### Index Rewriting

The indices are a 1-D index tensor. Each element is a linearized position in
the padded collapsed outer dimension. We rewrite via a `linalg.generic` that
wraps each index:

Given:
- Padded outer shape: `[d0_p, d1_p, ..., dn_p]` (e.g., `[1, 10, 18]`)
- Unpadded outer shape: `[d0_u, d1_u, ..., dn_u]` (e.g., `[1, 10, 10]`)
  - May have dynamic dimensions

For each index value `idx` in the tensor:

```
// 1. Delinearize with padded strides
coords[i] = (idx / padded_stride[i]) % padded_dim[i]

// 2. Check OOB: any coord exceeds unpadded dim?
is_oob = false
for each dim i:
  is_oob |= (coords[i] >= unpadded_dim[i])

// 3. Re-linearize with unpadded strides
new_idx = sum(coords[i] * unpadded_stride[i] for all i)

// 4. Clamp OOB to past-end index
new_idx = select(is_oob, unpadded_outer_size, new_idx)
```

**Implementation**: Create a new `linalg.generic` that reads the original
index tensor and writes rewritten indices. The body performs the
delinearize→check→re-linearize→clamp logic using `arith` ops. For dynamic
unpadded dimensions, use `tensor.dim` on the pad's source to get SSA values.

The explicit OOB clamp to `unpadded_outer_size` is necessary because
delinearize→re-linearize is not injective when shapes differ — an OOB
coordinate could alias a valid position in the flat unpadded buffer.

### Source Replacement

When tracing from `source` (the warp-level `extract_slice`):

1. Get the `extract_slice`'s source: should be the collapsed padded tensor
2. Get the `collapse_shape`'s source: should be the `tensor.pad` result
3. Get `pad.getSource()`: the raw unpadded tensor (from dispatch.tensor.load)

Rebuild:
1. `collapse_shape(pad.getSource(), same_reassociation)` → raw collapsed
2. `extract_slice(raw_collapsed, same_offsets, [unpadded_outer_dim, K_tile], same_strides)`

The `extract_slice` sizes change: the outer dimension shrinks from padded to
unpadded size. Inner dimension (K tile) stays the same since padding only
affects the outer group (see constraint below).

### Pad Constraints

Same as copy path, plus an additional constraint:

- Low padding must be all zeros
- Pad value must be zero (float or integer)
- Source row size (innermost dim) must be DWORD-aligned (4 bytes)
- **Padding must only affect the outer reassociation group** (the group
  corresponding to collapsed dim 0). If `padding_conv` also pads the channel
  dimension (inner group), bail out. This is checked by verifying that
  `pad.getMixedHighPad()` is zero for all dimensions in the inner
  reassociation group.

This constraint ensures `in_bounds = [false, true]` is correct: dim 0
(gathered, outer group) can be OOB, dim 1 (contiguous, inner/channel group)
is always in-bounds.

### Testing

1. **Unit test**: Add test case to `gpu_convert_to_coalesced_dma.mlir` with
   gather whose source traces through collapse_shape → pad → raw buffer.
   Verify `coalesced_gather_dma` with `in_bounds [false, true]` and that
   indices are rewritten.
2. **AMDGPU lowering test**: Add test case to
   `amdgpu_lower_coalesced_dma_to_gather_lds.mlir` combining indices and
   `in_bounds` together (currently tested separately but never combined).
3. **Pipeline test**: Add test case to `pipeline_im2col_dma_gfx950.mlir`
   with `padding`/`padding_conv` config, verify `amdgpu.gather_to_lds`.
4. **Existing tests**: All existing DMA tests must continue to pass.

## Files to Modify

1. `compiler/src/iree/compiler/Codegen/Common/GPU/GPUConvertToCoalescedDMA.cpp`
   - Add pad fusion logic in `ConvertGatherToCoalescedDMA::matchAndRewrite`
   - Add helper to build index-rewriting `linalg.generic`
   - Add helper to trace gather source through extract_slice → collapse → pad
2. `compiler/src/iree/compiler/Codegen/Common/GPU/test/gpu_convert_to_coalesced_dma.mlir`
   - Add gather + pad fusion test case
3. `compiler/src/iree/compiler/Codegen/Common/GPU/test/amdgpu_lower_coalesced_dma_to_gather_lds.mlir`
   - Add indices + in_bounds combined test case
4. `compiler/src/iree/compiler/Codegen/LLVMGPU/test/ROCDL/pipeline_im2col_dma_gfx950.mlir`
   - Add padded conv pipeline test case

## Risks

- **Index aliasing**: Incorrect OOB handling could silently produce wrong
  results. The explicit clamp to `unpadded_outer_size` mitigates this.
- **Dynamic shapes**: `linalg.generic` body with dynamic dim values adds
  complexity but is straightforward (capture via block args or closure).
- **Non-zero pad values**: Not supported in v1 (same as copy path). Could be
  added later with select-based fixup.
- **Channel padding**: Explicitly bailed out. Could be supported later by
  also rewriting the contiguous dimension access.
