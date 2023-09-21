// RUN: (iree-compile --iree-hal-target-backends=vmvx %s | \
// RUN:  iree-run-module --device=local-sync --module=- --function=echo \
// RUN:    --parameters=%p/parameters_a.safetensors \
// RUN:    --parameters=%p/parameters_b.safetensors \
// RUN:    --expected_output=4xi64=0,1,2,3 \
// RUN:    --expected_output=4xi64=4,5,6,7 \
// RUN:    --expected_output=8xi64=8,9,10,11,12,13,14,15 \
// RUN:    --expected_output=8xi64=16,17,18,19,20,21,22,23) | \
// RUN:  FileCheck %s
// CHECK: [SUCCESS]

// Simple named parameters with no scope. Parameter files are combined at
// runtime to allow for filesystem sharding while still providing a flat set of
// parameters in the compiler input.
util.global private @a0 = #stream.parameter.named<"a0"> : tensor<4xi64>
util.global private @a1 = #stream.parameter.named<"a1"> : tensor<4xi64>
util.global private @b0 = #stream.parameter.named<"b0"> : tensor<8xi64>
util.global private @b1 = #stream.parameter.named<"b1"> : tensor<8xi64>
func.func @echo() -> (tensor<4xi64>, tensor<4xi64>, tensor<8xi64>, tensor<8xi64>) {
  %a0 = util.global.load @a0 : tensor<4xi64>
  %a1 = util.global.load @a1 : tensor<4xi64>
  %b0 = util.global.load @b0 : tensor<8xi64>
  %b1 = util.global.load @b1 : tensor<8xi64>
  return %a0, %a1, %b0, %b1 : tensor<4xi64>, tensor<4xi64>, tensor<8xi64>, tensor<8xi64>
}
