// Compile and then run:
//
// RUN: iree-compile --iree-hal-target-backends=vmvx %s | iree-run-module --device=local-task --module=- --function=multiple_results --input=f32=-2 --input=f32=-3 | FileCheck %s --check-prefix=EXEC
// RUN: iree-compile --iree-hal-target-backends=llvm-cpu %s | iree-run-module --device=local-task --module=- --function=multiple_results --input=f32=-2 --input=f32=-3 | FileCheck %s --check-prefix=EXEC
// RUN: iree-compile --iree-hal-target-backends=vulkan-spirv %s | iree-run-module --device=vulkan --module=- --function=multiple_results --input=f32=-2 --input=f32=-3 | FileCheck %s --check-prefix=EXEC

// Compile but do not run:
//
// WebGPU fails at the moment. Flip the return code to success using 'not'
// RUN: not iree-compile --iree-hal-target-backends=webgpu --iree-stream-resource-alias-mutable-bindings=true %s 2>&1 | FileCheck %s --check-prefix=WEBGPU-ERRORS
// TODO(scotttodd): IREE_WEBGPU_DISABLE env var for this (enable only in CMake)

// EXEC-LABEL: EXEC @multiple_results
func.func @multiple_results(
    %input_0 : tensor<f32>,
    %input_1 : tensor<f32>
) -> (tensor<f32>, tensor<f32>) {
  %result_0 = math.absf %input_0 : tensor<f32>
  %result_1 = math.absf %input_1 : tensor<f32>
  return %result_0, %result_1 : tensor<f32>, tensor<f32>
}
// EXEC-NEXT: result[0]: hal.buffer_view
// EXEC-NEXT: f32=2
// EXEC-NEXT: result[1]: hal.buffer_view
// EXEC-NEXT: f32=3

// WEBGPU-ERRORS: Tint reported 1 error(s) for a SPIR-V program, see diagnostics:
// WEBGPU-ERRORS: error: entry point 'd0' references multiple variables that use the same resource binding @group(0), @binding(2)
