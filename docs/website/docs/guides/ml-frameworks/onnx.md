---
hide:
  - tags
tags:
  - ONNX
  - Python
  - PyTorch
icon: simple/onnx
status: new
---

# ONNX support

!!! caution "Caution - under development"

    Support for a broad set of [ONNX operators](https://onnx.ai/onnx/operators/)
    and [data types](https://onnx.ai/onnx/intro/concepts.html#supported-types)
    is an active investment area. See the
    [ONNX Op Support tracking issue](https://github.com/nod-ai/SHARK-ModelDev/issues/215)
    for the latest status.

## :octicons-book-16: Overview

Machine learning models using the
[Open Neural Network Exchange (ONNX)](https://onnx.ai/) format can be deployed
using the IREE compiler and runtime:

``` mermaid
graph LR
  accTitle: ONNX to runtime deployment workflow overview
  accDescr {
    Programs start as ONNX protobufs.
    Programs are imported into MLIR using iree-import-onnx.
    The IREE compiler uses the imported MLIR.
    Compiled programs are used by the runtime.
  }

  A["ONNX\n(protobuf)"]
  B["MLIR\n(torch-mlir)"]
  C[IREE compiler]
  D[Runtime deployment]

  A -- iree-import-onnx --> B
  B --> C
  C --> D
```

## :octicons-download-16: Prerequisites

1. Install ONNX:

    ``` shell
    python -m pip install onnx
    ```

2. Install IREE packages, either by
    [building from source](../../building-from-source/getting-started.md#python-bindings)
    or from pip:

    === "Stable releases"

        Stable release packages are
        [published to PyPI](https://pypi.org/user/google-iree-pypi-deploy/).

        ``` shell
        python -m pip install \
          iree-compiler[onnx] \
          iree-runtime
        ```

    === ":material-alert: Nightly releases"

        Nightly releases are published on
        [GitHub releases](https://github.com/iree-org/iree/releases).

        ``` shell
        python -m pip install \
          --find-links https://iree.dev/pip-release-links.html \
          --upgrade \
          iree-compiler[onnx] \
          iree-runtime
        ```

## :octicons-rocket-16: Quickstart

1. Start with a `.onnx` protobuf file, such as a model from
   <https://github.com/onnx/models>.

2. Convert the `.onnx` file into MLIR using the `iree-import-onnx` tool:

    ```shell
    iree-import-onnx [model.onnx] -o [model.mlir]
    ```

    This tool produces a MLIR file with the help of the
    [torch-mlir](https://github.com/llvm/torch-mlir) project.

3. Once imported, the standard set of tools and APIs available for any of
   IREE's [deployment configurations](../deployment-configurations/index.md) and
   [API bindings](../../reference/bindings/index.md) can be used:

    ```shell
    iree-compile \
      model.mlir \
      --iree-hal-target-backends=llvm-cpu \
      -o model_cpu.vmfb

    iree-run-module \
      --module=model_cpu.vmfb \
      --device=local-task \
      --function=... \
      --input=... \
      ...
    ```

## :octicons-code-16: Samples

| Code samples |  |
| -- | -- |
Generated op tests | [iree-test-suites `onnx_ops`](https://github.com/iree-org/iree-test-suites/tree/main/onnx_ops)
Public model tests | [iree-test-suites `onnx_models`](https://github.com/iree-org/iree-test-suites/tree/main/onnx_models)
Curated op and model tests | SHARK-TestSuite [`e2eshark/onnx`](https://github.com/nod-ai/SHARK-TestSuite/tree/main/e2eshark/onnx) and [`alt_e2eshark/onnx_tests`](https://github.com/nod-ai/SHARK-TestSuite/tree/main/alt_e2eshark/onnx_tests)
Importer tests | [torch-mlir `test/python/onnx_importer`](https://github.com/llvm/torch-mlir/tree/main/test/python/onnx_importer)

## :octicons-question-16: Troubleshooting

### Failed to legalize operation that was explicitly marked illegal

If you see an error compiling a converted .mlir file like this:

```console
$ iree-compile model.mlir --iree-hal-target-backends=llvm-cpu -o model.vmfb

model.mlir:507:12: error: failed to legalize operation 'torch.operator' that was explicitly marked illegal
    %503 = torch.operator "onnx.Identity"(%arg0) : (!torch.vtensor<[?],si64>) -> !torch.vtensor<[?],si64>
           ^
```

There are several possible scenarios:

1. The operator is not implemented, or the implementation is missing a case.
   Search for a matching issue in one of these places:
     * <https://github.com/llvm/torch-mlir/issues>
     * <https://github.com/nod-ai/SHARK-ModelDev/issues>
2. The operator is implemented but only for a more recent ONNX version. You can
   try upgrading your .onnx file using the
   [ONNX Version Converter](https://github.com/onnx/onnx/blob/main/docs/VersionConverter.md):

    ```python title="convert_onnx_model.py"
    import onnx
    original_model = onnx.load_model("model.onnx")
    converted_model = onnx.version_converter.convert_version(original_model, 17)
    onnx.save(converted_model, "model_17.onnx")
    ```

    and then attempting the convert -> compile again:

    ```shell
    iree-import-onnx model_17.onnx -o model_17.mlir
    iree-compile model_17.mlir ...
    ```
