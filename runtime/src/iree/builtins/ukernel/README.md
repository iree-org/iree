# Microkernels library

## Walk-through presentation

Here is a walk-through of how ukernels are built and used in IREE:
https://gist.github.com/bjacob/2c160c102cee33562826c730945b49f2

## What is a microkernel?

A microkernel (abbreviated as "ukernel") is a function that can be used as a lowering for a MLIR arithmetic operation. Specifically, in the IREE Codegen dialect we define a `ukernel.generic` operation which takes a "ukernel function name" attribute and gets lowered to a `func.call` op calling the specified function. This directory is where we build the functions that can be used for that purpose.

## What can ukernels do?

Ukernels can:
* Perform arithmetic, and access (read and write) memory buffers passed to them as pointer and strides arguments.

Ukernels cannot:
* Use any library, not even the C standard library.
  * In particular, ukernels can't allocate memory. Any buffer needs to be passed to it by the caller.
  * Ukernels can't even #include C standard library headers. Depending on the toolchain/platform, even stdint.h can bring in OS dependencies.
* Specialize for or interface with the operating system in any way.
  * If a ukernel needs information that would typically come from the OS, such as CPU identification details, that information needs to be passed to them as an argument, moving the problem to the caller.
  * Ukernels are built once for each target architecture, not for each target platform. Different platforms (e.g. Windows vs Linux) need to be able to share the exact same ukernel code.
* Have side effects, besides writing to destination buffers.
* Have state, such as accessing globals.
* Be non-reentrant. Ukernels will be called concurrently on multiple threads.

## How are ukernels compiled?

Ukernels are typically built in two different ways:
1. Ukernels are compiled to LLVM bitcode using the `iree_bitcode_library` function for CPU ukernels, and analogous functions for GPU ukernels. The resulting `.bc` bitcode files are then embedded as static data in the IREE compiler. This works in exactly the same way in the CMake and Bazel builds.
    * The IREE compiler also allows passing external ukernel bitcode files, allowing to use externally built ukernels.
2. Ukernels are also built as a normal library using the native toolchain. This is mostly used for local development, testing and benchmarking. This part is only fully implemented with CMake with all the architecture-specific code paths, while the Bazel build only has a minimal stub with architecture-specific code paths left out.
    * There is only one way in which this native-toolchain build of ukernels is actually used in IREE: with the VMVX back-end, ukernels are supported by linking this native ukernel code into the VMVX module, which is part of the IREE runtime.

## Unit-tests and microbenchmarks

The `tools/` directory contains unit-tests and microbenchmarks for ukernels. It allows developing ukernels within this directory, as a self-contained C project.
