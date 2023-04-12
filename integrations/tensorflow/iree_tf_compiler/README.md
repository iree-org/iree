This directory should be the only one in IREE that pulls a dependency on `tf`
dialect and related dialects.

# Tools

## Production Tools

### iree-import-tf

`iree-import-tf` provides a single entry-point for compiling TensorFlow saved
models to "IREE Input Dialects" that can be fed to `iree-compile` or
`iree-opt` and operated on further.

#### Usage

```shell
iree-import-tf /path/to/saved_model_v2
# Optional args: --tf-savedmodel-exported-names=subset,of,exported,names

iree-import-tf /path/to/saved_model_v1 --tf-import-type=savedmodel_v1
# Optional args:
#   --tf-savedmodel-exported-names=subset,of,exported,names
#   --tf-savedmodel-tags=serving
```
