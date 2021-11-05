libmmt4d
========

A micro library containing variants of the
[`linalg.mmt4d`](https://mlir.llvm.org/docs/Dialects/Linalg/#linalgmmt4d-mlirlinalgmmt4dop)
op specialized to various hardware architectures. This is designed to be linked
in to the executable binaries produced by the IREE compiler but can also be used
standalone for testing or static linkage into applications.

## Overview

The MMT4D op is defined in [`linalg/opdsl/core_named_ops.py`](https://github.com/llvm/llvm-project/blob/main/mlir/python/mlir/dialects/linalg/opdsl/ops/core_named_ops.py):
```py
@linalg_structured_op
def mmt4d(lhs=TensorDef(TV.LhsType, S.M, S.K, S.M0, S.K0),
          rhs=TensorDef(TV.RhsType, S.N, S.K, S.N0, S.K0),
          accum=TensorDef(TV.AccumType, S.M, S.N, S.M0, S.N0,
                                  output=True)):
  """Performs a matrix-matrix-transpose multiplication of two 4D inputs.
    Differences from linalg.matmul:
    * The right hand side is transposed, whence the 't' in 'mmt'.
    * The input and output tensors have a 4D shape instead of a 2D shape. They
      are interpreted as 2D matrices with one level of 2D tile subdivision,
      whence the 2+2=4 dimensions. The inner tile dimensions are identified with
      '0' suffixes below, for instance the LHS matrix shape (M, K, M0, K0) reads
      as: MxK tiles, each of shape M0xK0.
  """
  domain(D.m, D.n, D.k, D.m0, D.n0, D.k0)
  implements(ContractionOpInterface)
  accum[D.m, D.n, D.m0, D.n0] +=
      cast(TV.AccumType, lhs[D.m, D.k, D.m0, D.k0]) *
      cast(TV.AccumType, rhs[D.n, D.k, D.n0, D.k0])
```

In MLIR this op would appear as:
```mlir
linalg.mmt4d ins(%lhs, %rhs: memref<M1xK1xM0xK0xi8>, memref<N1xK1xN0xK0xi8>)
            outs(%dst: memref<M1xN1xM0xN0xi32>)
```

This library provides specializations for various M0xK0xN0 and data type
combinations declared in [`mmt4d.h`](mmt4d.h). For example,
`mmt4d_8x4x8_i8i8i32` implements the op for M0=8, K0=4, N0=8 and LHS/RHS of
`int8_t` and a destination of `int32_t`:
```c
MMT4D_EXPORT void mmt4d_8x4x8_i8i8i32(int k_size, const int8_t* lhs,
                                      const int8_t* rhs,
                                      int32_t* MMT4D_RESTRICT dst);
```

The `k_size` provided is K0 * K1 from the original 4D input dimensions.

The IREE compiler turns the `linalg.mmt4d` ops into calls to the functions:
```mlir
linalg.mmt4d ins(%lhs, %rhs: memref<?x?x8x4xi8>, memref<?x?x8x4xi8>)
            outs(%dst: memref<?x?x8x8xi32>)
```

->

```mlir
%c1 = arith.constant 1 : index
%c3 = arith.constant 3 : index
%k0 = arith.constant 4 : index
%k1 = memref.dim %lhs, %c1 : memref<?x?x8x4xi8>
%k_size = arith.muli %k0, %k1 : index
%k_size_i32 = arith.index_cast %k_size : index to i32
call @mmt4d_8x4x8_i8i8i32(%k_size_i32, %lhs, %rhs, %dst) :
    (i32, memref<?x?x8x4xi8>, memref<?x?x8x4xi8>, memref<?x?x8x8xi32>) -> ()
```

And then links the functions directly into the binary allowing for inlining.

## Bitcode Files

The IREE compiler embeds bitcode files and when producing executable libraries
will select one for linkage based on the specified target machine. As these
bitcode files can only be produced by a cross-compilation-enabled clang they are
built offline and checked into the repository. Future improvements to the
compiler could also allow for external files to be specified to avoid the need
to rebuild the compiler however for now this keeps things simple and hermetic.

The naming convention is `libmmt4d_[arch]_[features].bc`, corresponding to the
source files of `mmt4d_[arch].c` with the features specifying conditional target
CPU features such as extended instruction sets. When no special features are
required `generic` is used.

For example, the implementations for all ISA variants of AArch64 is found in
[`mmt4d_aarch64.c`](mmt4d_aarch64.c) and an implementation for the baseline ISA
is compiled into `libmmt4d_aarch64_generic.bc`. When the dot product
instructions are available (`-march=armv8.2-a+dotprod`) the more specialized
`libmmt4d_aarch64_dotprod.bc` bitcode file will be used.

### Updating Bitcode Files

The bitcode files need to be rebuilt whenever the source is modified, new
variants are added, or new architectures are targeted. The
[`bin/build.sh`](bin/build.sh) uses a compatible clang and llvm toolchain to
produce the files in the correct format and location.

Requirements:
* A modern version of Clang/LLVM (tested with 13)
* A build of llvm-as with all target architectures linked in

This script could use some usability improvements, but for now a common
invocation will look like:
```sh
LLVM_AS=/usr/bin/llvm-as \
CLANG=/usr/bin/clang-13 \
./iree/builtins/libmmt4d/bin/build.sh
```

If there are complaints that llvm-as does not support a target architecture then
the llvm-as included in the IREE CMake distribution should be built and provided
by way of the `IREE_BUILD_DIR`:
```sh
IREE_BUILD_DIR=../iree-build \
CLANG=/usr/bin/clang-13 \
./iree/builtins/libmmt4d/bin/build.sh
```

After this the newly updated/added bitcode files can be added to git.

### Compiler Bitcode Selection

The logic in the compiler for selecting which bitcode file to use is found in
[`iree/compiler/Dialect/HAL/Target/LLVM/Builtins/LibMMT4D.cpp`](/iree/compiler/Dialect/HAL/Target/LLVM/Builtins/LibMMT4D.cpp).
The `lookupMMT4DFile` function uses the `llvm::TargetMachine` to query the
architecture, CPU features, and other properties to choose the corresponding
bitcode file. If no matching bitcode file is found a fallback of the WebAssembly
generic implementation is used as its bitcode is generally portable. It's not
fast, though, and should only be used for correctness testing during bringup.

## Adding Variants

### Adding a Shape/Type Variant

New exported `mmt4d` functions for differing shapes and data types are added by
first declaring them in [`mmt4d.h`](mmt4d.h) and adding a `MMT4D_GENERIC`
line in [`mmt4d_generic.c`](mmt4d_generic.c). In addition for each architecture
where no specialized implementation will be provided a `MMT4D_GENERIC` should be
added to the respective architecture file.

Ergonomic improvements here would allow for weak linkage/overrides such that
new generic functions could be added without needing to add the generic versions
to each file, however that's TBD (C/C++ weak linkage in static libraries is...
not great).

### Adding an Architecture/ISA Bitcode File

First copy [`mmt4d_generic.c`](mmt4d_generic.c) and name it consistent with the
canonical LLVM architecture (the first part of the target triple, e.g. if you
pass `--target=aarch64-arm-none-eabi` to Clang you'd name it `aarch64`).

From there guard the new file with the architecture-specific preprocessor guards
and add the inverse to `mmt4d_generic.c` to prevent it from being used when the
source files are globbed.

Now as needed start replacing the `MMT4D_GENERIC` versions with specialized
implementations using intrinsics, inline assembly, or other magic.

Finally update the `LibMMT4D.cpp` file in the compiler to select the new bitcode
file based on the `llvm::TargetMachine`.

## Engineering Requirements

As this library is directly merged into the compiler-generated code there are
specific restrictions as to what can be used inherited from the IREE executable
requirements:

* No mutable globals/static variables or thread-local storage
* No syscalls
* No libc calls outside of builtins (like memset/memcpy) - _no mallocs_!

Though the primary usage of the library is through the precompiled bitcode files
that only need to work with Clang the library may also be built on other
toolchains such as GCC and MSVC (or older version of Clang). When standard
intrinsics are used this will generally not be a problem however inline assembly
may need compiler-specific variants or at least exclusions that fall back to
generic paths.

## Testing and Benchmarking

[`tools/mmt4d_test.cc`](tools/mmt4d_test.cc) provides a gtest runner that
compares the results of the optimized implementations for the target
architecture against a reference implementation for correctness.

[`tools/mmt4d_benchmark.c`](tools/mmt4d_benchmark.c) provides a benchmark suite
for the optimized implementations of the target architecture.

Both are compiled for the CMake target and can be used to develop
implementations without the need to rebuild/run the compiler.

## Future Work

Today the `mmt4d_` functions are all memory->memory. This makes them easy and
safe to integrate into the generated code and use from handwritten programs but
leaves a lot of performance on the table when it comes to more complex fusions.
Our intention is that future iterations will also expose an interface that
operates directly on ISA registers. This will look nearly identical to the
[CUDA WMMA primitives](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/)
used for NVIDIA's TensorCores or the
[SPIR-V Cooperative Matrix extension](https://github.com/KhronosGroup/SPIRV-Registry/blob/main/extensions/NV/SPV_NV_cooperative_matrix.asciidoc).

From `libmmt4d` we'll export methods like:
* `mmt4d_load_*`: loads registers from memory
* `mmt4d_mma_*`: performs the matmul-add on registers
* `mmt4d_store_*`: stores registers to memory

In addition, several element-wise arithmetic and type conversion operations will
be provided that operate on the registers directly:
* `mmt4d_add_*`
* `mmt4d_sub_*`
* `mmt4d_mul_*`
* `mmt4d_div_*`
* `mmt4d_neg_*`
* `mmt4d_convert_*`
* (others, as needed)

The IREE compiler will then produce these sequences such that as much work as
possible is performed in registers without needing to round-trip through memory.
All but the load and store operations will act on implicitly-defined registers
instead of passing them through ABI registers or memory. Since the library
bitcode is directly merged into the generated executable inlining is possible.

How the compiler produces these sequences will look similar to the [`iree/compiler/Codegen/SPIRV/SPIRVVectorToCooperativeOps.cpp`](iree/compiler/Codegen/SPIRV/SPIRVVectorToCooperativeOps.cpp)
pass that maps vector operations, with for example `vector.transfer_read` being
turned into `mmt4d_load_*` calls. This pass and how best (if possible) to share
code is still TBD. Given that the generic implementations won't be able to
control machine registers this mode is likely to be conditional based on target
architecture.
