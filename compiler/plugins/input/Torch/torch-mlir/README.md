# torch-mlir vendor build

We vendor the torch-mlir source code via the ${TORCH_MLIR_ROOT_DIR}
CMake variable. Because the build system isn't really aligned with
what we want (just the dialects/API), we provide a parallel CMake
build that takes care of just what we want.

We may ultimately fork this further in-tree but are taking this
half step in order to retain optionality.
