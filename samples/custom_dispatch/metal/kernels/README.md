# Custom Metal Dispatch Kernels

This sample mixes two hand-written kernels in [kernels.metal](./kernels.metal)
with an IREE-generated dispatch. Each HAL export names an MSL kernel and
declares its three-dimensional `workgroup_size` (threads per threadgroup).
The kernel receives its bindings in an argument buffer at `[[buffer(0)]]`,
where `[[id(N)]]` is the HAL binding ordinal. Dispatch constants are passed
as 32-bit values at `[[buffer(3)]]`.

## Run the sample

From the repository root on macOS, use a Metal-enabled `iree-compile` and
`iree-run-module`. Choose one of the two inputs below.

**MSL source.** [example_msl.mlir](./example_msl.mlir) uses the source directly.
On macOS, IREE compiles it to a metallib by default. With
`--iree-metal-compile-to-metallib=false`, or on other compilation hosts, IREE
embeds the source for Metal to compile when the module loads.

```sh
sample=samples/custom_dispatch/metal/kernels
iree-compile "$sample/example_msl.mlir" \
    --iree-hal-executable-object-search-path="$sample" -o /tmp/example.vmfb
```

**Precompiled library.** [example.mlir](./example.mlir) loads a metallib built
with the Xcode Metal toolchain. For an IREE build in `../iree-build` with
samples enabled, run:

```sh
cmake --build ../iree-build --target iree_samples_custom_dispatch_metal_kernels_metallib
iree-compile samples/custom_dispatch/metal/kernels/example.mlir \
    --iree-hal-executable-object-search-path=../iree-build -o /tmp/example.vmfb
```

Run either module. The result is a tensor of eight values, each equal to 96.

```sh
iree-run-module --device=metal --module=/tmp/example.vmfb \
    --function=mixed_invocation --input=8xf32=2 --input=8xf32=4
```
