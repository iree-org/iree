# WebGPU Hello World

This sample is the first bring-up checkpoint for the WebGPU target. It keeps the
scope intentionally small: compile one tensor program to a WebGPU VMFB, dump the
compiler-generated WGSL, and ask Dawn to validate that WGSL as a compute
pipeline.

The sample does not use `samples/simple_embedding`. That sample exercises a
synchronous embedding flow with blocking host transfers, which is the wrong
shape for the current inline WebGPU host because JavaScript promises cannot
settle while the wasm thread is blocked in C.

## Build the VMFB

Enable WebGPU SPIR-V compiler support in the Bazel configuration before
building the sample:

```sh
IREE_TARGET_BACKEND_WEBGPU_SPIRV=ON build_tools/bin/iree-bazel-configure
```

```sh
build_tools/bin/iree-bazel-build \
  //samples/webgpu/hello_world:hello_world_bytecode_module_webgpu
```

## Dump and Validate WGSL

The validator uses the `webgpu` Node package. Install it outside of the checkout
so npm does not create `package.json`, `package-lock.json`, or `node_modules/`
in the repository root:

```sh
npm install --prefix /tmp/iree-webgpu-validator webgpu
```

```sh
mkdir -p /tmp/iree-webgpu-hello-world

bazel-bin/tools/iree-compile \
  samples/webgpu/hello_world/hello_world.mlir \
  --iree-hal-target-device=webgpu \
  --iree-hal-dump-executable-binaries-to=/tmp/iree-webgpu-hello-world \
  -o=/tmp/iree-webgpu-hello-world/hello_world.vmfb

IREE_WEBGPU_PACKAGE_ROOT=/tmp/iree-webgpu-validator \
  node samples/webgpu/hello_world/validate_wgsl.mjs \
  /tmp/iree-webgpu-hello-world/*.wgsl
```
