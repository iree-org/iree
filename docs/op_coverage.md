# Op Coverage

There are four backend
[targets](https://github.com/google/iree/tree/master/iree/compiler/Dialect/HAL/Target)
in IREE:

-   vmla
-   llvm-ir
-   vulkan (direct path)
-   vulkan (structured ops path)

IREE has two path for Vulkan backend, one is using direct path (XLA -> SPIR-V),
and another is using structured ops path (XLA -> Linalg -> ... -> SPIR-V).

The table shows the supported XLA HLO ops on each backend.

| op                   | vmla | vulkan (direct | vulkan (structured | llvm-ir |
:                      :      : path)          : ops path)          :         :
| :------------------: | :--: | :------------: | :----------------: | :-----: |
| abs                  | ✓    | ✓              | ✓                  | ✓       |
| add                  | ✓    | ✓              | ✓                  | ✓       |
| batch_norm_inference | ✓    | ✗              | ✓                  | ✓       |
| broadcast            | ✓    | ✓              | ✗                  | ✗       |
| broadcast_in_dim     | ✓    | ✓              | ✓                  | ✓       |
| compare              | ✓    | ✓              | ✗                  | ✓       |
| concatenate          | ✓    | ✓              | ✗                  | ✗       |
| constant             | ✓    | ✓              | ✓                  | ✓       |
| conv                 | ✓    | ✗              | ✗                  | ✓       |
| convert_int          | ✓    | ✓              | ✗                  | ✗       |
| cos                  | ✓    | ✓              | ✓                  | ✓       |
| dot                  | ✓    | ✓              | ✗                  | ✓       |
| dot_general          | ✓    | ✗              | ✗                  | ✗       |
| exp                  | ✓    | ✓              | ✓                  | ✓       |
| floor                | ✓    | ✓              | ✗                  | ✗       |
| gather               | ✓    | ✓              | ✗                  | ✗       |
| gather_concat        | ✓    | ✓              | ✗                  | ✗       |
| gemm                 | ✓    | ✓              | ✓                  | ✓       |
| gemm_large           | ✓    | ✓              | ✓                  | ✓       |
| log                  | ✓    | ✓              | ✓                  | ✓       |
| max_float            | ✓    | ✓              | ✓                  | ✓       |
| max_int              | ✓    | ✓              | ✓                  | ✓       |
| min_float            | ✓    | ✓              | ✓                  | ✓       |
| min_int              | ✓    | ✓              | ✓                  | ✓       |
| multiply             | ✓    | ✓              | ✗                  | ✓       |
| negate               | ✓    | ✗              | ✗                  | ✓       |
| pad                  | ✓    | ✓              | ✗                  | ✗       |
| reduce_float         | ✓    | ✓              | ✗                  | ✓       |
| reduce_int           | ✓    | ✓              | ✗                  | ✓       |
| reduce_window        | ✓    | ✗              | ✗                  | ✓       |
| rem                  | ✓    | ✓              | ✓                  | ✓       |
| reshape              | ✓    | ✓              | ✗                  | ✓       |
| reshape_adddims      | ✓    | ✓              | ✓                  | ✓       |
| reshape_dropdims     | ✓    | ✓              | ✓                  | ✓       |
| reverse              | ✓    | ✓              | ✗                  | ✗       |
| rsqrt                | ✓    | ✓              | ✓                  | ✓       |
| select               | ✓    | ✓              | ✓                  | ✓       |
| sin                  | ✓    | ✓              | ✗                  | ✗       |
| slice                | ✓    | ✓              | ✗                  | ✗       |
| sqrt                 | ✓    | ✓              | ✓                  | ✓       |
| while                | ✓    | ✓              | ✗                  | ✓       |
