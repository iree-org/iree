# Custom Vulkan/SPIR-V Dispatch Shaders

See the [custom_dispatch README](/samples/custom_dispatch/README.md) for an
overview of this approach.

This sample demonstrates how to define external device functions that can be
dispatched from within IREE programs when following the IREE Vulkan/SPIR-V ABI.
The user authoring the shaders compiles their GLSL/HLSL/etc code to SPIR-V blobs
and can dispatch functions within those blobs by declaring them in their IR.

### Work in Progress

Note that currently only entire dispatches can be modeled and this prevents
IREE from performing optimizations it otherwise can. In the future SPIR-V
linking will be implemented such that the external functions are referenced and
linked with the compiler-produced portions such that more information about the
usage of the dispatch can be used to specialize/prune the hand-authored
portions. Since the IREE Vulkan/SPIR-V ABI is not version-stable this entire
shader approach may require updating when taking new IREE versions while
function-level linking would not.

Since today only entire shaders can be provided the user must specify an empty
executable (no `builtin.module` contents) and thus must provide objects for
all targets they are compiling for. When partial function linking is available
it'll be possible to provide fallback code as IR for when objects are not
available.

## Workflow

```
                                                         +--------------+
                                                         | example.mlir |
                                                         +--------------+
+-----------------+             +----------------+              v
| simple_mul.glsl | -> glslc -> | simple_mul.spv | ------> iree-compile
+-----------------+             +----------------+              v
                                                         +--------------+
                                                         | example.vmfb |
                                                         +--------------+
```

1. The user authors their shaders in a .glsl (.hlsl/etc) file.

```c
#version 450
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0) readonly buffer Binding0 { float binding0[]; };
layout(set = 0, binding = 1) readonly buffer Binding1 { float binding1[]; };
layout(set = 0, binding = 2) buffer Binding2 { float binding2[]; };
layout(push_constant) uniform PushConstants { uint dim; };
void main() {
  uint tid = gl_GlobalInvocationID.x;
  if (tid < dim) binding2[tid] = binding0[tid] * binding1[tid];
}
```

2. Source files are compiled to SPIR-V files via glslc. Each architecture or
   set of extensions the user is targeting will need its own object file(s).

```cmake
glslc -fshader-stage=compute simple_mul.glsl -o simple_mul.spv
```

3. The user (or compiler transforms) declare the externally defined shaders and
   the target-to-objects map for each configuration. The layout specifies how
   the dispatch arguments are mapped to buffers. The region on the export is
   used to compute the workgroup count and can query the `%device` if runtime
   device information is needed.

```mlir
  hal.executable.source private @executable attributes {
    objects = #hal.executable.objects<{
      #spirv_target = [
        #hal.executable.object<{path = "simple_mul.spv"}>
      ]
    }>
    hal.executable.export public @simple_mul ordinal(0)
        layout(#hal.pipeline.layout<push_constants = 1, sets = [
          <0, bindings = [
              <0, storage_buffer, ReadOnly>,
              <1, storage_buffer, ReadOnly>,
              <2, storage_buffer>
          ]>
        ]>) {
    ^bb0(%device: !hal.device, %workload: index):
      %x = affine.apply affine_map<()[s0] -> (s0 ceildiv 64)>()[%workload]
      %c1 = arith.constant 1 : index
      hal.return %x, %c1, %c1 : index, index, index
    }
  }
```

4. The user (or compiler transforms) dispatches the shader.

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

0. Ensure that `glslc` is on your PATH (comes with the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home)):

    ```
    glslc --version
    ```

1. Build the `iree-sample-deps` CMake target to compile the GLSL to SPIR-V:

    ```
    cmake --build ../iree-build/ --target iree-sample-deps
    ```

    In a user application this would be replaced with whatever build
    infrastructure the user has for compiling shaders to SPIR-V. No IREE
    compiler or runtime changes are required and the normal compiler install can
    be used.

2. Compile the [example module](./example.mlir) to a .vmfb file and pass the
   path to the build directory so the .spv files can be found:

    ```
    iree-compile \
        --iree-hal-executable-object-search-path=../iree-build/ \
        samples/custom_dispatch/vulkan/shaders/example.mlir \
        -o=/tmp/example.vmfb
    ```

3. Run the example program using the custom shaders:

    ```
    iree-run-module \
        --device=vulkan \
        --function=mixed_invocation \
        --input=8xf32=2 \
        --input=8xf32=4 \
        /tmp/example.vmfb
    ```

## Custom Kernel Match and Replace Scripting Instructions

This is a flow for authoring custom dispatches externally alongside match and
replace logic that can be fed directly into a pre-built version of the compiler.

In addition to the above steps, when compiling the module, pass in both the
target module and the transform library implementing the matcher + kernel.  

    ```
    iree-compile \
        --iree-hal-executable-object-search-path=../iree-build/ \
        samples/custom_dispatch/vulkan/shaders/example_transform.mlir \
        --iree-preprocessing-transform-library=samples/custom_dispatch/vulkan/shaders/example_transform_spec.mlir \
        -o=/tmp/example.vmfb
    ```
