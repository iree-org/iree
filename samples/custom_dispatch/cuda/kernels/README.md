# Custom CUDA Dispatch Kernels

See the [custom_dispatch README](/samples/custom_dispatch/README.md) for an
overview of this approach.

This sample demonstrates how to define external device functions that can be
dispatched from within IREE programs when following the IREE CUDA ABI. The user
authoring the kernels compiles their CUDA code to PTX blobs and can dispatch
functions within those blobs by declaring them in their IR.

### Work in Progress

Note that currently only entire kernel launches can be modeled and this prevents
IREE from performing optimizations it otherwise can. In the future PTX linking
will be implemented such that the external functions are referenced and linked
with the compiler-produced portions such that more information about the usage
of the dispatch can be used to specialize/prune the hand-authored kernel. Since
the IREE CUDA ABI is not version-stable this entire kernel approach may require
updating when taking new IREE versions while function-level linking would not.

Since today only entire kernels can be provided the user must specify an empty
executable (no `builtin.module` contents) and thus must provide objects for
all targets they are compiling for. When partial function linking is available
it'll be possible to provide fallback code as IR for when objects are not
available.

## Workflow

```
+------------+              +-------------------+       +--------------+
| kernels.cu | -> nvcc -+-> | kernels_sm_52.ptx | -+    | example.mlir |
+------------+          |   +-------------------+  |    +--------------+
                        |   +-------------------+  |           v
                        +-> | kernels_sm_80.ptx | -+----> iree-compile
                            +-------------------+              v
                                                        +--------------+
                                                        | example.vmfb |
                                                        +--------------+
```

1. The user authors their kernels in a .cu file.

```c
extern "C" __global__ void simple_mul(const float* __restrict__ binding0,
                                      const float* __restrict__ binding1,
                                      float* __restrict__ binding2, int dim) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < dim) binding2[tid] = binding0[tid] * binding1[tid];
}
```

2. Source files are compiled to PTX files via nvcc. Each architecture the user
   is targeting will need its own object file(s).

```cmake
nvcc ... (TODO, see CMakeLists.txt) -o kernels_sm_80.ptx
```

3. The user (or compiler transforms) declare the externally defined kernels and
   the target-to-objects map for each architecture. The layout specifies how
   the dispatch arguments are mapped to buffers and the workgroup size is the
   CUDA block size. The region on the export is used to compute the workgroup
   count (CUDA grid size) and can query the `%device` if runtime device
   information is needed.

```mlir
  hal.executable.source private @executable attributes {
    objects = #hal.executable.objects<{
      #nvptx_sm_52_target = [
        #hal.executable.object<{path = "kernels_sm_80.ptx"}>
      ]
    }>
    hal.executable.export public @simple_mul ordinal(0)
        layout(#hal.pipeline.layout<push_constants = 1, sets = [
          <0, bindings = [
              <0, storage_buffer, ReadOnly>,
              <1, storage_buffer, ReadOnly>,
              <2, storage_buffer>
          ]>
        ]>) attributes {workgroup_size = [64 : index, 1 : index, 1 : index]} {
    ^bb0(%device: !hal.device, %workload: index):
      %x = affine.apply affine_map<()[s0] -> (s0 ceildiv 64)>()[%workload]
      %c1 = arith.constant 1 : index
      hal.return %x, %c1, %c1 : index, index, index
    }
  }
```

4. The user (or compiler transforms) dispatches the kernel.

```mlir
  %0 = flow.dispatch @executable::@simple_mul[%dim](%dim_i32, %arg0, %arg1) :
      (i32, tensor<?xf32>{%dim}, tensor<?xf32>{%dim}) -> tensor<?xf32>{%dim}
```

5. The IREE compiler selects the appropriate object files for the target
   configuration and links them into the binaries it produces. Dispatches are
   automatically routed to the appropriate variant of those available at
   runtime.

## Instructions

This presumes that `iree-compile` and `iree-run-module` have been installed or
built. [See here](https://iree.dev/building-from-source/getting-started/)
for instructions for CMake setup and building from source.

0. Ensure that the [CUDA SDK](https://developer.nvidia.com/cuda-downloads) and `nvcc` is on your PATH:

    ```
    nvcc --version
    ```

1. Build the `iree-sample-deps` CMake target to compile the .cu to .ptx:

    ```
    cmake --build ../iree-build/ --target iree-sample-deps
    ```

    In a user application this would be replaced with whatever build
    infrastructure the user has for compiling kernels to PTX. No IREE
    compiler or runtime changes are required and the normal compiler install can
    be used.

2. Compile the [example module](./example.mlir) to a .vmfb file and pass the
   path to the build directory so the .spv files can be found:

    ```
    iree-compile \
        --iree-hal-executable-object-search-path=../iree-build/ \
        samples/custom_dispatch/cuda/kernels/example.mlir \
        -o=/tmp/example.vmfb
    ```

3. Run the example program using the custom kernels:

    ```
    iree-run-module \
        --device=cuda \
        --function=mixed_invocation \
        --input=8xf32=2 \
        --input=8xf32=4 \
        /tmp/example.vmfb
    ```
