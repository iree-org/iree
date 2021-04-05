---
layout: default
permalink: xla-op-coverage
title: "XLA HLO Op Coverage"
nav_order: 6
js_files: 
- js/add_classes.js
---

# XLA HLO Op Coverage
{: .no_toc }
There are three backend [targets](https://github.com/google/iree/tree/main/iree/compiler/Dialect/HAL/Target) in IREE:

- vmla
- dylib-llvm-aot
- vulkan-spirv

The table shows the supported XLA HLO ops on each backend. It is auto-generated
from IREE's test status.

op | vmla | vulkan-spirv | dylib-llvm-aot
:-: | :-: | :-: | :-:
abs | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
add | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
batch_norm_inference | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
broadcast | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
broadcast_add | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
broadcast_in_dim | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
clamp | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
compare | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
concatenate | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
constant | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
convert | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
convolution | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
cosine | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
divide | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
dot | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
dot_general | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
exponential | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
exponential_minus_one | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
fft | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
finite | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
floor | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
gather | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
iota | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
log | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
log_plus_one | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
maximum | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
minimum | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
multiply | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
negate | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
pad | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
reduce | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
reduce_window | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
remainder | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
reshape | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
reverse | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
round | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
rsqrt | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
select | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
sine | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
slice | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
sort | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
sqrt | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
subtract | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
tanh | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
torch_index_select | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
transpose | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
while | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>