// RUN: (iree-compile --iree-hal-target-backends=vmvx %s | \
// RUN:  iree-run-module --device=local-sync --module=- --function=echo \
// RUN:    --parameters=a=%p/parameters_a.safetensors \
// RUN:    --parameters=b=%p/parameters_b.safetensors \
// RUN:    --expected_output=4xi64=0,1,2,3 \
// RUN:    --expected_output=4xi64=4,5,6,7 \
// RUN:    --expected_output=8xi64=8,9,10,11,12,13,14,15 \
// RUN:    --expected_output=8xi64=16,17,18,19,20,21,22,23) | \
// RUN:  FileCheck %s
// CHECK: [SUCCESS]

// Parameters scoped to allow for separating parameters from multiple models or
// model stages in a compiled pipeline. It's possible to have multiple files
// provide content for a single scope but not to have a single file provide
// content for multiple scopes. Since parameter keys only need to be unique
// within a scope this test could use the same name for both scopes if needed.
util.global private @a0 = #stream.parameter.named<"a"::"a0"> : tensor<4xi64>
util.global private @a1 = #stream.parameter.named<"a"::"a1"> : tensor<4xi64>
util.global private @b0 = #stream.parameter.named<"b"::"b0"> : tensor<8xi64>
util.global private @b1 = #stream.parameter.named<"b"::"b1"> : tensor<8xi64>
func.func @echo() -> (tensor<4xi64>, tensor<4xi64>, tensor<8xi64>, tensor<8xi64>) {
  %a0 = util.global.load @a0 : tensor<4xi64>
  %a1 = util.global.load @a1 : tensor<4xi64>
  %b0 = util.global.load @b0 : tensor<8xi64>
  %b1 = util.global.load @b1 : tensor<8xi64>
  return %a0, %a1, %b0, %b1 : tensor<4xi64>, tensor<4xi64>, tensor<8xi64>, tensor<8xi64>
}
