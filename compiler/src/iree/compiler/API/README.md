# IREE Compiler Backend

This is a top-level project for building public facing API packages that
combine all dependent MLIR-based projects along with IREE's compiler backend
API.

It exports artifacts:

* `iree-compiler-backend` Python wheel and source distributions, providing
  the `iree.compiler_backend` Python packages.
* Compiler C-API source and binary tarballs (future), with some CLI tools.
