# Custom CPU Dispatch Functions and Rewrites for Dynamically-linked Plugins

See the [plugin README](/samples/custom_dispatch/cpu/plugin/README.md) for an
more information on plugins.

`mlp_plugin/` demonstrates how to define external device functions
that can be dispatched from within IREE programs via simple function calls
(similar to `plugin/`). `mlp_plugin` also demonstrates how `iree-compile`
supports various languages to preform rewrites.

In contrast with `plugin/example.mlir`, `mlp_plugin/mlp.mlir` does not
use any external function calls, but is instead composed of only builtin
mlir ops. iree-compile uses a rewrite pattern to match and rewrite the input
MLIR, replacing matched ops with a call to `mlp_external()` defined in
[mlp_plugin.c](/samples/custom_dispatch/cpu/mlp_plugin/mlp_plugin.c).

## Rewrite Languages

### 1. [Transform Dialect](https://mlir.llvm.org/docs/Dialects/Transform/)

See example in [mlp.mlir](/samples/custom_dispatch/cpu/mlp_plugin/mlp.mlir)

### 2. [PDL](https://mlir.llvm.org/docs/Dialects/PDLOps/)

See example in [mlp_linalg_spec.pdl.mlir](/samples/custom_dispatch/cpu/mlp_plugin/mlp_linalg_spec.pdl.mlir)

### 3. [PDLL](https://mlir.llvm.org/docs/PDLL/)

See example in [mlp_linalg_spec.pdll](/samples/custom_dispatch/cpu/mlp_plugin/pdll/mlp_linalg_spec.pdll)
