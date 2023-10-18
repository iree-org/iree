// RUN: iree-compile %s \
// RUN:     --iree-hal-executable-object-search-path=$IREE_BINARY_DIR \
// RUN:     --iree-opt-extern-dispatch-pattern-module=%p/example_patterns.mlir | \
// RUN: iree-run-module \
// RUN:     --device=vulkan \
// RUN:     --module=- \
// RUN:     --function=mixed_invocation \
// RUN:     --input=8xf32=2 \
// RUN:     --input=8xf32=4 | \
// RUN: FileCheck %s

// The configuration used for executable compilation.
// This lets the compiler and runtime know the format and requirements of the
// executable binaries produced and multiple variants with differing formats
// and compilation options (architectures, etc) can be embedded for runtime
// selection.
#spirv_target = #hal.executable.target<"vulkan", "vulkan-spirv-fb", {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.3, [Shader, GroupNonUniform], [SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>,
    #spirv.resource_limits<max_compute_workgroup_size = [128, 128, 64], subgroup_size = 64>
  >
}>

// The target devices that the program will run on.
// These can come from compiler flags and multiple targets can be supported
// It's possible, for example, to support targeting multiple devices in the same
// compiled binary.
#vulkan_target = #hal.device.target<"vulkan", {
  executable_targets = [#spirv_target],
  // HACK: Vulkan target currently uses the legacy synchronous execution model.
  legacy_sync
}>

module @example attributes {hal.device.targets = [#vulkan_target]} {

  // Function demonstrating replacing a kernel with a hand-written implementation.
  // Invoke with:
  //  --device=vulkan
  //  --function=mixed_invocation
  //  --input=8xf32=2
  //  --input=8xf32=4
  // CHECK-LABEL: EXEC @mixed_invocation
  func.func @mixed_invocation(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    // Target to match and replace with a hand written dispatch
    %0 = arith.mulf %arg0, %arg1 : tensor<?xf32>

    // Code gen some other ops - these will interleave with the hand-authored
    // ones but naturally won't be able to fuse with them.
    %1 = arith.addf %0, %arg1 : tensor<?xf32>

    // CHECK: 8xf32=12 12 12 12 12 12 12 12
    return %1 : tensor<?xf32>
  }

}  // module
