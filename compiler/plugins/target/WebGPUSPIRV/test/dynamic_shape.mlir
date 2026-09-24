// RUN: iree-compile --iree-hal-target-backends=webgpu-spirv %s -o %t.vmfb

// The dynamic dimension is passed to WGSL through the immediate address space.
func.func @add(%a: tensor<?xf32>, %b: tensor<?xf32>) -> tensor<?xf32> {
  %result = arith.addf %a, %b : tensor<?xf32>
  return %result : tensor<?xf32>
}
