# Custom CPU Dispatch Functions and Rewrites for Dynamically-linked Plugins

See the [plugin README](/samples/custom_dispatch/cpu/plugin/README.md) for an
overview of this approach. This sample is derived from the

`mlp_plugin/` demonstrates how to define external device functions
that can be dispatched from within IREE programs via simple function calls 
(similar to `plugin/`).
`mlp_plugin` also demonstrates how either 
[PDL](https://mlir.llvm.org/docs/Dialects/PDLOps/) or
[PDLL](https://mlir.llvm.org/docs/PDLL/) can be used to perform custom pattern writing to inject calls to custom CPU dispatch functions

In contrast with `plugin/example.mlir`, `mlp_plugin/mlp.mlir` does not
use any external function calls, but is instead composed of only builtin
mlir ops. iree-compile uses a rewrite pattern (either PDL or PDLL) to match and rewrite the input MLIR, replacing matched ops with a call to `mlp_external()` defined in 
[mlp_plugin.c](/samples/custom_dispatch/cpu/mlp_plugin/mlp_plugin.c).

