# Experimental MetaProgramming + MLIR JIT

The purpose of this directory is to demonstrate C++ metaprogramming features
that are available in MLIR core. At a high-level, metaprogramming can be
interpreted as "programming with a level of indirection".

This experimental `mlir::ModelBuilder` is used to generate operations using the
[Linalg Dialect](https://mlir.llvm.org/docs/Dialects/Linalg/).

This exposes the `mlir::edsc::ValueHandle` and `mlir::edsc::StructuredIndexed`
classes and other functionality into an `mlir::ModelBuilder`. to build an MLIR
function for a whole model (in the case of the [TestMNISTJit](TestMNISTJit.cpp)
example, a `3-MLP`).

The MLIR function can be compiled using MLIR passes and progressively lowered to
LLVMIR. The `mlir::ModelRunner` then kicks in and produces an executable version
in memory that can be called directly, by name, from the main function.

This allows writing a `3-MLP` test in C++, allocate some buffers, JIT-compile
and run it end-to-end, as the test demonstrates.

Note that for the moment `mlir::ModelBuilder` does not perform any type of
advanced transformations and basically only lowers LLVM, where LLVM tools (`opt`
and `llc`) pick it up and apply their optimizations.

At this time this is only intended for prototyping and experimenting on vanilla
Linux CPU targets.
