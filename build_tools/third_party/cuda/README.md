# CUDA Toolkit Configuration

This directory contains:

* CMake support for auto-downloading and finding the CUDA toolkit.
* Targets in an "iree_cuda" namespace that encapsulate CUDA for the rest
  of the codebase.
* Bazel overlay.

Note that the amount of CUDA that IREE itself depends on is minimal and
encapsulated to targets in this directory (versus being loose leaf across the
codebase).
