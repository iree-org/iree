---
layout: default
permalink: HLOOpCoverage
title: XLA HLO Operation Coverage
nav_order: 6
---

# HLO Op Coverage
{: .no_toc }
There are three backend [targets](https://github.com/google/iree/tree/master/iree/compiler/Dialect/HAL/Target) in IREE:

- vmla
- llvm-ir
- vulkan-spirv

The table shows the supported XLA HLO ops on each backend.

op | vmla | vulkan-spirv | llvm-ir
:-: | :-: | :-: | :-:
abs | ✓ | ✓ | ✓
add | ✓ | ✓ | ✓
batch_norm_inference | ✓ | ✓ | ✓
broadcast | ✓ | ✓ | ✓
broadcast_add | ✓ | ✓ | ✗
broadcast_in_dim | ✓ | ✓ | ✓
clamp | ✓ | ✗ | ✓
compare | ✓ | ✓ | ✓
concatenate | ✓ | ✗ | ✗
constant | ✓ | ✓ | ✓
convert | ✓ | ✗ | ✗
convolution | ✓ | ✗ | ✓
cosine | ✓ | ✓ | ✓
divide | ✓ | ✓ | ✓
dot | ✓ | ✓ | ✓
dot_general | ✓ | ✗ | ✗
exponential | ✓ | ✓ | ✓
floor | ✓ | ✗ | ✗
gather | ✓ | ✗ | ✗
gather_concat | ✓ | ✗ | ✗
gemm | ✓ | ✓ | ✓
gemm_large | ✓ | ✓ | ✓
log | ✓ | ✓ | ✓
maximum | ✓ | ✓ | ✓
minimum | ✓ | ✓ | ✓
multiply | ✓ | ✓ | ✓
negate | ✓ | ✓ | ✓
pad | ✓ | ✓ | ✓
reduce | ✓ | ✓ | ✓
reduce_window | ✓ | ✗ | ✓
remainder | ✓ | ✓ | ✓
reshape | ✓ | ✓ | ✓
reverse | ✓ | ✓ | ✓
rsqrt | ✓ | ✓ | ✓
select | ✓ | ✓ | ✓
sine | ✓ | ✓ | ✓
slice | ✓ | ✗ | ✗
sqrt | ✓ | ✓ | ✓
subtract | ✓ | ✓ | ✓
tanh | ✓ | ✓ | ✗
torch_index_select | ✓ | ✓ | ✓
transpose | ✓ | ✓ | ✓
while | ✓ | ✗ | ✓