# IREE "Static Library" sample

This sample shows how to:
1. Produce a static library and bytecode module with IREE's compiler
2. Compile the static library into a program using the `static_library_loader`
3. Run the demo with the module using functions exported by the static library

The model compiled into the static library exports a single function
`simple_mul` that returns the multiplication of two tensors:

```mlir
func @simple_mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32>
{
  %0 = "mhlo.multiply"(%arg0, %arg1) {name = "mul.1"} : (tensor<4xf32>,
    tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
```

## Background

IREE's `static_library_loader` allows applications to inject a set of static
libraries that can be resolved at runtime by name. This can be particularly
useful on "bare metal" or embedded systems running IREE that lack operating
systems or the ability to load shared libraries in binaries.

When static library output is enabled, `iree-translate` produces a separate
static library to compile into the target program. At runtime bytecode module
instructs the VM which static libraries to load exported functions from the
model.

## Instructions
_Note: run the following commands from IREE's github repo root._

1. Configure CMake for building the static library then demo. You'll need to set
the flags building samples, the compiler, and the `dylib-llvm-aot`
driver/backend. See
[here](https://google.github.io/iree/building-from-source/getting-started/)
for general instructions on building using CMake):

  ```shell
  cmake -B ../iree-build/
    -DIREE_BUILD_SAMPLES=ON \
    -DIREE_TARGET_BACKENDS_TO_BUILD=DYLIB-LLVM-AOT \
    -DIREE_HAL_DRIVERS_TO_BUILD=DYLIB \
    -DIREE_BUILD_COMPILER=ON \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo .
  ```

2. Build the `simple_mul` CMake target. This will run  `iree-translate` on
`simple_mul.mlir` (located in this directory) to produce a static library,
header file, and bytecode module:

  ```shell
  cmake --build ../iree-build/ --target iree_samples_static_library_simple_mul_c
  ```

3. Build the `static_library_demo` CMake target. Note: this assumes you've
generated the static library files and module in the build directory from step
2:

  ```shell
  cmake --build ../iree-build/ --target iree_samples_static_library_demo
  ```

4. Run the sample binary:

  ```shell
  ../iree-build/iree/samples/static_library/static_library_demo

  # Output: static_library_run passed
  ```
