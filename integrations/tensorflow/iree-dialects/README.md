# IREE Dialects Project

Sources for IREE's public dialects (containing ops/types/attributes that are
unique to IREE and can appear in compiler inputs).

This project is intended to be used via LLVM's external projects setup:

* `-DLLVM_EXTERNAL_PROJECTS=iree-dialects`
* `-DLLVM_EXTERNAL_IREE_DIALECTS_SOURCE_DIR={this_directory}`

It depends on the `mlir` project.
