This directory should be the only one in IREE that pulls a dependency on `tf`
dialect and related dialects (with the exception of a small selection of "safe"
XLA IR).

## Tools

### Development Tools

*   `iree-tf-opt` : MLIR Opt tool with TensorFlow and IREE passes/dialects
    linked in
*   `iree-tf-translate` : Equivalent to `mlir-tf-translate` tool in TensorFlow,
    with IREE passes/dialects linked in

### Production Tools

#### iree-tf-import

`iree-tf-import` provides a single entry-point for compiling TensorFlow saved
models to "IREE Input Dialects" that can be fed to `iree-translate` or
`iree-opt` and operated on further.

##### Usage:

```shell
iree-tf-import /path/to/saved_model_v2
# Optional args: --tf-savedmodel-exported-names=subset,of,exported,names

iree-tf-import /path/to/saved_model_v1 --tf-import-type=savedmodel_v1
# Optional args:
#   --tf-savedmodel-exported-names=subset,of,exported,names
#   --tf-savedmodel-tags=serving
```
