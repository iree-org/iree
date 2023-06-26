# Alternate LLVM dependency management

The default build instructions for IREE do an in-tree build of LLVM and all
LLVM sub-projects in a single cmake session. While this has the fewest moving
parts and allows edit-compile-run across the entire project, it is not the
only way to proceed.

This directory contains helpers and scripts to build-your-own LLVM and have
IREE use that. LLVM is a complicated dependency to take, and not all
possible installation modalities are supported. However, for those that are,
we will attempt to document here and test in the CI.

## Baseline byo_llvm.sh script

The baseline `byo_llvm.sh` script builds the stack of:

* LLVM (with libLLVM.so)
* Clang (bundled with the LLVM build)
* LLD (bundled with the LLVM build)
* MLIR (as a standalone installation that depends on LLVM)
* IREE (depending on all of the above)

This split is likely the most advanced configuration possible and represents
a common use case for hardware enablement:

* Often there is an LLVM installation for a part with proprietary backend, etc.
* IREE is tightly coupled to MLIR, which can drift from the installed LLVM
  so long as there are not LLVM API breaks.
* libLLVM.so is built and linked against.

Note that when built in this configuration, the resulting *installed* IREE
can only be used with `LD_LIBRARY_PATH` set appropriately (or if built against
a system LLVM). Per usual CMake policy, binaries in the build tree will
always be hard-coded to the path on the build machine and do not need this.

Always make sure that there is not the same version of LLVM's shared library
on your library path in a way that will cause it to take precedence.

### Usage:

Note that full CMake command lines are logged so that you can create your
own scripting as needed.

```
./build_tools/llvm/byo_llvm.sh build_llvm
./build_tools/llvm/byo_llvm.sh build_mlir
./build_tools/llvm/byo_llvm.sh build_iree
```

Tests can be run with:

```
./build_tools/llvm/byo_llvm.sh test_iree
```
