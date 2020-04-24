---
layout: default
permalink: HLOOpCoverage
title: XLA HLO Operation Coverage
nav_order: 6
---

# HLO Op Coverage
{: .no_toc }
There are four backend [targets](https://github.com/google/iree/tree/master/iree/compiler/Dialect/HAL/Target) in IREE:

- vmla
- llvm-ir
- vulkan (direct path)
- vulkan (structured ops path)

> Note
> {: .label .label-blue }
> IREE currently has two compilation paths for Vulkan,
> shown as above. The direct path lowers XLA HLOs to SPIR-V in one step; the
> structured ops path goes multiple steps in a progressive way. The plan is to
> deprecate the direct path soon.)

The table shows the supported XLA HLO ops on each backend.

op | vmla | vulkan (direct path) | vulkan (structured ops path) | llvm-ir
:-: | :-: | :-: | :-: | :-:
abs | ✓ | ✓ | ✓ | ✓
add | ✓ | ✓ | ✓ | ✓
batch_norm_inference | ✓ | ✗ | ✓ | ✓
broadcast | ✓ | ✓ | ✗ | ✗
broadcast_in_dim | ✓ | ✓ | ✓ | ✓
clamp | ✓ | ✗ | ✗ | ✗
compare | ✓ | ✓ | ✗ | ✓
concatenate | ✓ | ✓ | ✗ | ✗
constant | ✓ | ✓ | ✓ | ✓
convert | ✓ | ✓ | ✗ | ✗
convolution | ✓ | ✗ | ✗ | ✓
cosine | ✓ | ✓ | ✓ | ✓
dot | ✓ | ✓ | ✗ | ✓
dot_general | ✓ | ✗ | ✗ | ✗
exponential | ✓ | ✓ | ✓ | ✓
floor | ✓ | ✓ | ✗ | ✗
gather | ✓ | ✓ | ✗ | ✗
gather_concat | ✓ | ✓ | ✗ | ✗
gemm | ✓ | ✓ | ✓ | ✓
gemm_large | ✓ | ✓ | ✓ | ✓
log | ✓ | ✓ | ✓ | ✓
maximum | ✓ | ✓ | ✓ | ✓
minimum | ✓ | ✓ | ✓ | ✓
multiply | ✓ | ✓ | ✗ | ✓
negate | ✓ | ✗ | ✗ | ✓
pad | ✓ | ✓ | ✓ | ✓
reduce | ✓ | ✓ | ✗ | ✓
reduce_window | ✓ | ✗ | ✗ | ✓
remainder | ✓ | ✓ | ✓ | ✓
reshape | ✓ | ✓ | ✗ | ✓
reverse | ✓ | ✓ | ✗ | ✗
rsqrt | ✓ | ✓ | ✓ | ✓
select | ✓ | ✓ | ✓ | ✓
sine | ✓ | ✓ | ✗ | ✗
slice | ✓ | ✓ | ✗ | ✗
sqrt | ✓ | ✓ | ✓ | ✓
torch_select | ✓ | ✗ | ✗ | ✗
transpose | ✓ | ✓ | ✓ | ✓
while | ✓ | ✓ | ✗ | ✓