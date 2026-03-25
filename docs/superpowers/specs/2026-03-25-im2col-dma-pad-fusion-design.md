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

## Approach: Pad Fusion in Gather DMA Conversion

Fuse `tensor.pad` into the gather DMA path by rewriting the source and indices
in `createDMAInForall<GatherOp>`.

### IR Before (Current Failure)

```mlir
%padded = tensor.pad %input low[0,0,0,0] high[0,0,8,0] {
  tensor.yield %zero : f16
} : tensor<1x10x10x512xf16> to tensor<1x10x18x512xf16>

%collapsed = tensor.collapse_shape %padded [[0,1,2],[3]]
  : tensor<1x10x18x512xf16> into tensor<180x512xf16>

// Indices linearized with padded strides: basis [1, 10, 18]
%indices = linalg.generic { ... } -> tensor<64xindex>

// After tiling:
%src_slice = tensor.extract_slice %collapsed[0, %k_off] [180, 128] [1, 1]
%dma = iree_gpu.coalesced_gather_dma %src_slice[%idx_slice] into %dest ...
```

### IR After (Proposed Fix)

```mlir
// Trace through: extract_slice → collapse_shape → pad → raw source
%raw_collapsed = tensor.collapse_shape %input [[0,1,2],[3]]
  : tensor<1x10x10x512xf16> into tensor<100x512xf16>

%raw_slice = tensor.extract_slice %raw_collapsed[0, %k_off] [100, 128] [1, 1]

// Rewrite indices: delinearize with padded shape, re-linearize with unpadded
%new_indices = <rewritten indices vector>

%dma = iree_gpu.coalesced_gather_dma %raw_slice[%new_indices] into %dest
  lane(%lane) in_bounds [false, true] ...
```

### Index Rewriting

The indices are a vector of i32 values (after `vector.transfer_read` +
`arith.index_cast`). Each index is a linearized position in the padded
collapsed outer dimension. We need to convert to the unpadded coordinate space.

Given:
- Padded outer shape: `[d0_p, d1_p, ..., dn_p]` (e.g., `[1, 10, 18]`)
- Unpadded outer shape: `[d0_u, d1_u, ..., dn_u]` (e.g., `[1, 10, 10]`)
  - May have dynamic dimensions

For each index value `idx`:
1. Delinearize with padded strides:
   `coords[i] = (idx / padded_stride[i]) % padded_dim[i]`
2. Re-linearize with unpadded strides:
   `new_idx = coords[0] * unpadded_stride[0] + ... + coords[n]`

For OOB positions (where a coord exceeds the unpadded dim), the re-linearized
index will naturally exceed the unpadded buffer's outer dimension. Combined with
`in_bounds = [false, ...]`, the existing OOB mechanism in
`AMDGPULowerCoalescedDMAToGatherLDS` forces `fat_raw_buffer` to return zero.

However, the re-linearized index for OOB positions might happen to alias a
valid position in the flat buffer (delinearize → re-linearize is not
injective when shapes differ). To guarantee correct OOB detection, we should
explicitly clamp: if any coord is OOB, set the index to `unpadded_outer_size`
(guaranteed past buffer end).

**Implementation using vector ops:**
```
// All ops are on vector<Nxi32> where N = indices vector length
// padded_strides and unpadded_strides are computed as constants/values

// 1. Delinearize: extract coordinates from padded linear index
for each dim i (outermost to innermost, excluding last channel dim):
  coord_i = (idx / padded_stride_i) % padded_dim_i

// 2. Check OOB: any coord exceeds unpadded dim?
is_oob = false
for each dim i:
  is_oob |= (coord_i >= unpadded_dim_i)

// 3. Re-linearize with unpadded strides
new_idx = sum(coord_i * unpadded_stride_i for all i)

// 4. Clamp OOB to past-end index
new_idx = select(is_oob, unpadded_outer_size, new_idx)
```

For dynamic unpadded dimensions, `unpadded_dim_i` and `unpadded_stride_i` are
SSA values (from `tensor.dim` on the pad's source), splatted to vector for
element-wise ops.

### Where to Insert

In `createDMAInForall<GatherOp>` (GPUConvertToCoalescedDMA.cpp, ~line 506),
after `source = innerOp.getSource()`:

1. Trace `source` through `extract_slice` → `collapse_shape` → `pad`
2. If found and pad constraints are met (low=0, pad_value=0):
   - Replace `source` with `extract_slice` of `collapse_shape(pad.getSource())`
   - After indices are converted to i32 vector (~line 543), insert index
     rewriting ops
   - Set `inBoundsVec = {false, true}` (dim 0 OOB, dim 1 in-bounds)
3. If not found or constraints fail, fall through to existing behavior

### Pad Constraints

Same as copy path:
- Low padding must be all zeros (no low padding)
- Pad value must be zero (float or integer)
- Source row size must be DWORD-aligned (4 bytes)

### Collapse Shape Handling

The `collapse_shape` between pad and the gather source groups spatial dims
together. We need the reassociation indices to know which input dims map to
the collapsed outer dimension. This is available from the `CollapseShapeOp`.

For the outer dimension group `[0, 1, ..., k]`:
- Padded dims: `pad.getResult().getType().getShape()[0..k]`
- Unpadded dims: `pad.getSource().getType().getShape()[0..k]` (may be dynamic)

### Testing

1. **Unit test**: Add test case to `gpu_convert_to_coalesced_dma.mlir` with
   im2col + pad input, verify `coalesced_gather_dma` with `in_bounds [false, true]`
   and rewritten indices
2. **Pipeline test**: Update `pipeline_im2col_dma_gfx950.mlir` to add a test
   case with `padding`/`padding_conv` config, verify `amdgpu.gather_to_lds`
3. **Existing tests**: All existing DMA tests must continue to pass

## Files to Modify

1. `compiler/src/iree/compiler/Codegen/Common/GPU/GPUConvertToCoalescedDMA.cpp`
   - Add pad fusion logic in `createDMAInForall<GatherOp>` branch
   - Add helper to rewrite indices vector
2. `compiler/src/iree/compiler/Codegen/Common/GPU/test/gpu_convert_to_coalesced_dma.mlir`
   - Add im2col + pad test case
3. `compiler/src/iree/compiler/Codegen/LLVMGPU/test/ROCDL/pipeline_im2col_dma_gfx950.mlir`
   - Add padded conv pipeline test case

## Risks

- **Index aliasing**: Incorrect OOB handling could silently produce wrong
  results. The explicit clamp to `unpadded_outer_size` mitigates this.
- **Dynamic shapes**: Vector ops with dynamic splats add complexity but are
  straightforward.
- **Non-zero pad values**: Not supported in v1 (same as copy path). Could be
  added later with select-based fixup.
