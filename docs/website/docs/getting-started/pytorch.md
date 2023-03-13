# PyTorch Integration

IREE supports compiling and running PyTorch programs represented as
`nn.Module` [classes](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
as well as models defined using [`functorch`](https://pytorch.org/functorch/).

## Prerequisites

Install IREE pip packages, either from pip or by
[building from source](../building-from-source/python-bindings-and-importers.md):

```shell
pip install \
  iree-compiler \
  iree-runtime
```

Install [`torch-mlir`](https://github.com/llvm/torch-mlir), necessary for
compiling PyTorch models to a format IREE is able to execute:

```shell
pip install -f https://llvm.github.io/torch-mlir/package-index/ torch-mlir
```

A special `iree_torch` package makes it easy to compile PyTorch programs and
run them on IREE:

```shell
pip install git+https://github.com/iree-org/iree-torch.git
```

## Running a model

Going from a loaded PyTorch model to one that's executing on IREE happens in
four steps:

1. Compile the model to [MLIR](https://mlir.llvm.org)
2. Compile the MLIR to IREE VM flatbuffer
3. Load the VM flatbuffer into IREE
4. Execute the model via IREE

!!! note
    In the following steps, we'll be borrowing the model from
    [this BERT colab](https://github.com/iree-org/iree-torch/blob/main/examples/bert.ipynb)
    and assuming it is available as `model`.

### Compile the model to MLIR

First, we need to trace and compile our model to MLIR:

```python
model = # ... the model we're compiling
example_input = # ... an input to the model with the expected shape and dtype
mlir = torch_mlir.compile(
    model,
    example_input,
    output_type=torch_mlir.OutputType.LINALG_ON_TENSORS,
    use_tracing=True)
```

The full list of available output types can be found
[here](https://github.com/llvm/torch-mlir/blob/6403c0e56f0e93e231df1c8d3dc78df7dd721b80/python/torch_mlir/__init__.py#L19)
and includes linalg on tensors, mhlo, and tosa.

### Compile the MLIR to an IREE VM flatbuffer

Next, we compile the resulting MLIR to IREE's deployable file format:

```python
iree_backend = "llvm-cpu"
iree_vmfb = iree_torch.compile_to_vmfb(mlir, iree_backend)
```

Here we have a choice of backend we want to target. See the
[Deployment Configurations](https://openxla.github.io/iree/deployment-configurations/)
section of this site for a full list of targets and configurations.

The generated flatbuffer can now be serialized and stored for another time or
loaded and executed immediately.

### Load the VM flatbuffer into IREE

Next, we load the flatbuffer into the IREE runtime. `iree_torch` provides a
convenience method for loading this flatbuffer from Python:

```python
invoker = iree_torch.load_vmfb(iree_vmfb, iree_backend)
```

### Execute the model via IREE

Finally, we can execute the loaded model:

```python
result = invoker.forward(example_input)
```

## Training

Training with PyTorch in IREE is supported via `functorch`. The steps for
loading the model into IREE, once defined, are nearly identical to the above
example.

You can find a full end-to-end example of defining a basic regression model,
training with it, and running inference on it
[here](https://github.com/iree-org/iree-torch/blob/main/examples/regression.py).

## Native / On-device Training

A small (~100-250KB), self-contained binary can be built for deploying to
resource-constrained environments. An example illustrating this can be found in
[this example](https://github.com/iree-org/iree-torch/tree/main/examples/native_training).
This binary runs a model without a Python interpreter.

## Samples

| Colab notebooks |  |
| -- | -- |
Inference on BERT | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iree-org/iree-torch/blob/main/examples/bert.ipynb)

| Example scripts |
| -- |
| [Basic Inference and Training Example](https://github.com/iree-org/iree-torch/blob/main/examples/regression.py) |
| [Native On-device Training Example](https://github.com/iree-org/iree-torch/tree/main/examples/native_training) |
