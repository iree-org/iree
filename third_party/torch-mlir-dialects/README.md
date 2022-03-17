# Torch-MLIR Dialects Project

Sources for torch-mlir's public dialects (containing ops/types/attributes that
are unique to Torch-MLIR at the moment)

This project is intended to be used via LLVM's external projects setup:

* `-DLLVM_EXTERNAL_PROJECTS=torch-mlir-dialects`
* `-DLLVM_EXTERNAL_TORCH_MLIR_DIALECTS_SOURCE_DIR={this_directory}`

It depends on the `mlir` project.
