IREE Microkernels Library: `libukernel`
=======================================

This library provides builtin microkernels to both the IREE VMVX module for
runtime linkage and the IREE compiler for ahead-of-time compilation. Each
deployment approach has tradeoffs and the intent with this library is to share
the same compiler passes/infrastructure for emitting the microkernel ops and
the same microkernel implementations.

## Runtime Linkage

For deployments targeting the IREE VM the compiler will produce .vmfb modules
that use the VMVX module (`iree/modules/vmvx/module.c`). The code in this
library is linked into the runtime VMVX module and called via the VM FFI:
```
                     +------------+      +---------+      +================+
                     | input.mlir | ---> | codegen | ---> |  iree-compile  |
                     +------------+      +---------+      +================+
                                                                  |
                                                                  v
+-----------+      +------------+      +--------------+    +--------------+
| mmt4d_*.c | ---> | C compiler | ---> | libukernel.a |    | .vmfb module |
+-----------+      +------------+      +--------------+    +--------------+
                                             |                    |
                                             v                    v
                                 +-------------------+      +============+
                                 | iree/modules/vmvx | ---> | VM context |
                                 +-------------------+      +============+
```

As the definition of the VMVX ops in the compiler have to match the ones in the
runtime the interface is difficult to change without breaking binary
compatibility. Because of this the exported VMVX methods are intended to be
generic, stable, and consistent across platforms. The microkernels in this
library are insulated by the VMVX module layer which can perform versioning and
provide fallbacks as needed.

## Ahead-of-time Linkage

For deployments using ahead-of-time compilation the library is compiled to
bitcode files that are loaded and linked while producing the generated code:
```
+-----------+      +-------+      +-------------------------------+
| mmt4d_*.c | ---> | clang | ---> |+--------------------------------+
+-----------+      +-------+      +| libukernel_[arch]_[variant].bc |
                                   +--------------------------------+
                                                  |||
                                                  vvv
      +------------+      +---------+      +================+
      | input.mlir | ---> | codegen | ---> |  iree-compile  |
      +------------+      +---------+      +================+
                                                   |
                      +----------------------------+
                      v                            v
         +------------------------+   +----------------------------+
         | static library (.o/.a) |   | dynamic library (.so/.dll) |
         +------------------------+   +----------------------------+
```

By linking the generated code together with the library bitcode the compiler can
perform intra-procedural optimization to efficiently cull unused code paths and
propagate known-constant values. The compiler outputs are hermetic and avoid
version skew between the compiler and the runtime.

## Bitcode Files

The IREE compiler embeds bitcode files and when producing executable libraries
will select one for linkage based on the specified target machine. As these
bitcode files can only be produced by a cross-compilation-enabled Clang they are
built offline and checked into the repository. Future improvements to the
compiler could also allow for external files to be specified to avoid the need
to rebuild the compiler however for now this keeps things simple and hermetic.

Usage is currently not wired up in the compiler but will look very similar to
the `iree/builtins/device/` approach.

## Engineering Requirements

As this library is directly merged into the compiler-generated code there are
specific restrictions as to what can be used inherited from the IREE executable
requirements:

* No mutable globals/static variables or thread-local storage
* No syscalls
* No libc calls outside of builtins (like memset/memcpy) - _no mallocs_!

Though precompiled bitcode files only need to work with Clang the library may
also be built on other toolchains such as GCC and MSVC (or older version of
Clang). When standard intrinsics are used this will generally not be a problem
however inline assembly may need compiler-specific variants or at least
exclusions that fall back to generic paths.

### Compile-time Configuration

Preprocessor statements used to control behavior must only use information known
when the bitcode files are being compiled. This means that if the bitcode file
being produced is for AArch64 it is safe to use the `__aarch64__` macro.
Information that is only available after the bitcode file is produced - such as
in the IREE compiler pipelines - must use link-time configuration.

### Link-time Configuration

As we are producing bitcode files we cannot rely on the C preprocessor for
changing behavior based on some information only known during linking. In other
cases we may want to specialize code paths based on knowledge about the context
in which the kernels are used. To provide this link-time modification ability
there is support for flags by way of `extern` globals. These globals are either
specified by the IREE compiler when linking the bitcode or by the hosting
application when linked statically.

For example, this flag can be specified by either passing a define when
compiling the library for standalone/VMVX use or using the
`overridePlatformGlobal` helper when emitting LLVM IR in the IREE compiler:
```c
#if defined(IREE_UK_PLATFORM_EXAMPLE_FLAG)
static const int iree_microkernels_platform_example_flag =
    IREE_UK_PLATFORM_EXAMPLE_FLAG;
#else
extern int iree_microkernels_platform_example_flag;
#endif  // IREE_UK_PLATFORM_EXAMPLE_FLAG
```

Any code may then use this flag to condition/control behavior:
```c
if (iree_microkernels_platform_example_flag >= 1) {
  // Do something special.
}
```

When linking libmicrokernels statically the flags can be provided by the hosting
application via compiler defines:
`-DIREE_UK_PLATFORM_EXAMPLE_FLAG=123`.

When producing bitcode the flags are left symbolic and the IREE compiler
provides their values:
```c++
overridePlatformGlobal(*bitcodeModule,
                       "iree_microkernels_platform_example_flag", 123u);
```

What flags are useful and how to handle cases where flags are arch-dependent are
still TBD.

## Testing and Benchmarking

[`tools/mmt4d_test.cc`](tools/mmt4d_test.cc) provides a gtest runner
that compares the results of the optimized implementations for the target
architecture against a reference implementation for correctness.

[`tools/mmt4d_benchmark.c`](tools/mmt4d_benchmark.c) provides a
benchmark suite for the optimized implementations of the target architecture.

Both are compiled for the CMake target and can be used to develop
implementations without the need to rebuild/run the compiler or produce full
compiled artifacts that operate in the runtime.
