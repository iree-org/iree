# Embedded Example

This example shows how to

1. Build a PyTorch functional model for training
2. Import that model into IREE's compiler
3. Compile that model to an IREE VM bytecode module
4. Load the compiled module using IREE's high level runtime C API into a
   lightweight program
5. Train the loaded model

This example was built with the goal of allowing you to be able to build it
outside this repo in your own project with minimal changes.

The weights for the model are stored in the program itself and updated in
memory. This can be modified to be stored however you see fit.

## Running the Example

Install `iree-torch` and other dependencies necessary for this example.
[iree-torch](git@github.com:iree-org/iree-torch.git) provides a number of
convenient wrappers around `torch-mlir` and `iree` compilation:

```shell
pip install -f https://iree-org.github.io/iree/pip-release-links.html iree-compiler
pip install -f https://llvm.github.io/torch-mlir/package-index/ torch-mlir
pip install git+https://github.com/iree-org/iree-torch.git
pip install scikit-learn
```

Update submodules in this repo:

```shell
(cd $(git rev-parse --show-toplevel) && git submodule update --init)
```

Build the IREE runtime:

```shell
(cd $(git rev-parse --show-toplevel) && cmake -GNinja -B /tmp/iree-build-runtime/ .)
cmake --build /tmp/iree-build-runtime/ --target iree_runtime_unified
```

Make sure you're in this example's directory:

```shell
cd $(git rev-parse --show-toplevel)/samples/embedded
```

Build the embedded example:

```shell
make
```

Generate the IREE VM bytecode for the model:

```shell
python embedded.py
```

Run the embedded model:

```shell
./embedded embedded.vmfb
```