# IREE "Static Library" sample

This sample shows how to:
1. Produce a static library and bytecode module with IREE's compiler
2. Compile the static library into a program using the `static_library_loader`
3. Run the demo with the module using functions exported by the static library

The model compiled into the static library exports a single function `simple_mul` that returns the multiplication of two tensors:

```mlir
func @simple_mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32>
    attributes { iree.module.export } {
  %0 = "mhlo.multiply"(%arg0, %arg1) {name = "mul.1"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
```

## Background

IREE's `static_library_loader` allows applications to inject a set of static libraries that can be resolved at runtime by name. This can be particularly useful on "bare metal" or embedded systems running IREE that lack operating systems or the ability to load shared libraries in binaries.

When static library output is enabled, `iree-translate` produces a separate static library to compile into the target program. At runtime bytecode module instructs the VM which static libraries to load exported functions from the model.

## Instructions
Note: run these commands from IREE's github root.

1. Configure CMake with the `IREE_BUILD_STATIC_LIBRARY_SAMPLES` then build the `iree_translate` CMake target (see
   [here](https://google.github.io/iree/building-from-source/getting-started/)
   for general instructions on building using CMake):

   ```shell
   cmake -B ../iree-build/ -IREE_BUILD_STATIC_LIBRARY_SAMPLES=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo .
   cmake --build ../iree-build/ --target iree_tools_iree-translate
   ```

2. Run `iree-translate` on  `simple_mul.mlir` (located in this directory) to produce the static library, header file, and bytecode module:

    ```shell
    ../iree-build/tools/iree-translate \
            iree/samples/static_library/simple_mul.mlir \
            -iree-input-type=mhlo \
            -iree-mlir-to-vm-bytecode-module \
            -iree-hal-target-backends=dylib-llvm-aot \
            -o iree/samples/static_library/static_library_module.vmfb \
            --iree-llvm-static-library-output-path=iree/samples/static_library/static_library_module.o
    ```

3. Compile the `static_library_demo` CMake target. Note: this assumes you've added the static library files and module in the demo directory from step 2:

    ```shell
    cmake --build ../iree-build/ --target iree_samples_static-library-demo
    ```

4. Run the sample binary

   ```shell
   ../iree-build/iree/samples/static_library/static_library_demo

   # Output: static_library_run passed
   ```
