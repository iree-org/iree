// RUN: iree-compile %s \
// RUN:     --iree-hal-executable-object-search-path=$IREE_BINARY_DIR \
// RUN:     --iree-preprocessing-transform-spec-filename=%p/example_transform_spec.mlir | \
// RUN: iree-run-module \
// RUN:     --device=vulkan \
// RUN:     --module=- \
// RUN:     --function=mixed_invocation \
// RUN:     --input=1x128xf32=4 \
// RUN:     --input=1x128xf32=3 | \
// RUN: FileCheck %s

// The configuration used for executable compilation.
// This lets the compiler and runtime know the format and requirements of the
// executable binaries produced and multiple variants with differing formats
// and compilation options (architectures, etc) can be embedded for runtime
// selection.
// HACK: Currently this must match EXACTLY with the executable target for the
// custom kernel. For things to be truly portable, we need to be able to compare
// executable configurations.
#spirv_target = #hal.executable.target<"vulkan", "vulkan-spirv-fb", {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.3, [Shader, GroupNonUniform, GroupNonUniformArithmetic, GroupNonUniformBallot],
                     [SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>,
    #spirv.resource_limits<max_compute_workgroup_size = [128, 128, 64], subgroup_size = 64>
  >
}>

// The target devices that the program will run on. We can compile and run with
// multiple targets, but this example is maintaining an implicit requirement
// that the custom kernel being spliced in is supported by the target device,
// hence we only support vulkan here. It is possible to hand author a custom
// kernel that supports multiple targets by specifying an object per-target, but
// that requires authoring the kernel for multiple targets.
#vulkan_target = #hal.device.target<"vulkan", {
  executable_targets = [#spirv_target],
  // HACK: Vulkan target currently uses the legacy synchronous execution model.
  legacy_sync
}>

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
module @example attributes {hal.device.targets = [#vulkan_target]} {

  // CHECK-LABEL: EXEC @mixed_invocation
  func.func @mixed_invocation(%arg0: tensor<1x128xf32>, %arg1: tensor<1x128xf32>) -> tensor<1xi64> {
    // Code gen some other ops - these will interleave with the matched and
    // replaced ones but naturally won't be able to fuse with them.
    %add = arith.addf %arg0, %arg1 : tensor<1x128xf32>

    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0xFF800000 : f32
    %1 = tensor.empty() : tensor<1xi64>
    %2 = linalg.fill ins(%c0_i64 : i64) outs(%1 : tensor<1xi64>) -> tensor<1xi64>
    %3 = tensor.empty() : tensor<1xf32>
    %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<1xf32>) -> tensor<1xf32>
    // Argmax that is the target for the custom kernel. Note that this operation
    // only has uses for a single result and takes a single input.
    %5:2 = linalg.generic {indexing_maps = [#map, #map1, #map1],
                           iterator_types = ["parallel", "reduction"]}
                           ins(%add : tensor<1x128xf32>)
                           outs(%4, %2 : tensor<1xf32>, tensor<1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_0: i64):
      %6 = linalg.index 1 : index
      %7 = arith.index_cast %6 : index to i64
      %8 = arith.maximumf %in, %out : f32
      %9 = arith.cmpf ogt, %in, %out : f32
      %10 = arith.select %9, %7, %out_0 : i64
      linalg.yield %8, %10 : f32, i64
    } -> (tensor<1xf32>, tensor<1xi64>)

    // CHECK: 1xi64=0
    return %5#1 : tensor<1xi64>
  }
}  // module
