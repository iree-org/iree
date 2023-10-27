# Custom CPU Dispatch Functions for Statically-linked Embedded ELFs

See the [custom_dispatch README](/samples/custom_dispatch/README.md) for an
overview of this approach.

This sample demonstrates how to define external device functions that can be
dispatched from within IREE programs via simple function calls. Here the
functions are declared in the MLIR executables, called as normal calls, and
then defined in a .c file that is cross-compiled for various architectures.
The compiler uses the attribute specifying which object files to link against
when performing its final linking and runs LTO to optimize across both the
generated portions and hand-authored portions.

### Work in Progress

The calling convention used for passing pointers is currently a mess as the MLIR
`memref` type is used to model buffer references and that expands to many
arguments. Future revisions will just pass the pointers instead.

Currently weak linkage is not available and external functions must always be
provided when referenced. In future versions fallback IR will allow for object
files to be specified only for certain platforms while allowing others to be
generated via normal codegen paths.

## Workflow

```
+-------------+               +---------------------+       +--------------+
| functions.c | -> clang -+-> | functions_aarch64.o | -+    | example.mlir |
+-------------+           |   +---------------------+  |    +--------------+
                          |   +---------------------+  |           v
                          +-> | functions_x86_64.o  | -+----> iree-compile
                              +---------------------+              v
                                                            +--------------+
                                                            | example.vmfb |
                                                            +--------------+
```

1. The user authors their functions in bare-metal C (no TLS, no threads, no
   malloc, etc). These functions can cover entire workgroups (and a dispatch can
   be a single workgroup so effectively just function calls) or be utilities
   used by the function for localized work (microkernels, data type conversion,
   etc). It's important to remember that parallelism scheduling is done
   _outside_ of the function via the workgroup count and multiple threads may be
   executing the function at any time.

```c
// NOTE: this will be simplified in the future:
//  void simple_mul_workgroup(
//    const float* restrict binding0, const float* restrict binding1,
//    float* restrict binding2, size_t dim, size_t tid);
void simple_mul_workgroup(
    const float* restrict binding0, const float* restrict binding0_aligned,
    size_t binding0_offset, size_t binding0_size, size_t binding0_stride,
    const float* restrict binding1, const float* restrict binding1_aligned,
    size_t binding1_offset, size_t binding1_size, size_t binding1_stride,
    float* restrict binding2, float* restrict binding2_aligned,
    size_t binding2_offset, size_t binding2_size, size_t binding2_stride,
    size_t dim, size_t tid) {
  size_t end = tid + 64;
  if (end > dim) end = dim;
  for (size_t i = tid; i < end; ++i) {
    binding2[i] = binding0[i] * binding1[i];
  }
}
```

2. Source files are compiled to object files with bare-metal settings. Each
   architecture the user is targeting will need its own object file(s).

```cmake
clang -target aarch64 ...[see CMakeLists.txt]... functions.c -o functions_aarch64.o
```

3. The user (or compiler transforms) adds calls to their functions by declaring
   them and marking them as statically-linked.

```mlir
func.func private @simple_mul_workgroup(
    %binding0: memref<?xf32>, %binding1: memref<?xf32>, %binding2: memref<?xf32>,
    %dim: index, %tid: index) attributes {hal.import.static}
...
func.call @simple_mul_workgroup(%memref0, %memref1, %memref2, %dim, %tid) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, index, index) -> ()
```

4. The user (or compiler transforms) annotates the executables with the objects
   to link against providing the function definitions.

```mlir
  stream.executable private @executable attributes {
    hal.executable.objects = #hal.executable.objects<{
      #aarch64_target = [
        #hal.executable.object<{path = "functions_aarch64.o"}>
      ]
    }>
```

5. The IREE compiler selects the appropriate object files for the target
   configuration and links them into the binaries it produces.

## Instructions

This presumes that `iree-compile` and `iree-run-module` have been installed or
built. [See here](https://iree.dev/building-from-source/getting-started/)
for instructions for CMake setup and building from source.

0. Ensure that `clang` is on your PATH:

    ```
    clang --version
    ```

1. Build the `iree-sample-deps` CMake target to compile
   [functions.c](./functions.c) to object files for aarch64 and x86_64:

    ```
    cmake --build ../iree-build/ --target iree-sample-deps
    ```

    In a user application this would be replaced with whatever build
    infrastructure the user has for compiling code to object files. No IREE
    compiler or runtime changes are required and the normal compiler install can
    be used. Note that specific flags are required when producing the object
    files.

2. Compile the [example module](./example_stream.mlir) to a .vmfb file and pass
   the path to the build directory so the .o files can be found:

    ```
    iree-compile \
        --iree-hal-executable-object-search-path=../iree-build/ \
        samples/custom_dispatch/cpu/embedded/example_stream.mlir \
        -o=/tmp/example.vmfb
    ```

    [example_stream.mlir](./example_stream.mlir) demonstrates a high-level
    approach without needing to specify too much information while
    [example_hal.mlir](./example_hal.mlir) shows the lower-level representation
    it gets expanded into.

3. Run the example program using the custom functions:

    ```
    iree-run-module \
        --device=local-sync \
        --function=mixed_invocation \
        --input=8xf32=2 \
        --input=8xf32=4 \
        --module=/tmp/example.vmfb
    ```
