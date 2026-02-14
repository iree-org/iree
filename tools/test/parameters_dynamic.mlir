// RUN: (iree-compile \
// RUN:    --iree-hal-target-device=local \
// RUN:    --iree-hal-local-target-device-backends=vmvx %s | \
// RUN:  iree-run-module --device=local-sync --module=- --function=echo \
// RUN:    --parameters=%p/parameters_dynamic.safetensors \
// RUN:    --expected_output=4xi64=24,28,32,36) | \
// RUN:  FileCheck %s
// CHECK: [SUCCESS]

// Dynamic parameter loading using util.string.format to construct keys inside
// a loop. Iterates param0..param3, loading each and accumulating a running sum.
// All inputs are constants so the entire loop should be hoisted into an
// initializer by HoistIntoGlobals.
func.func @echo() -> tensor<4xi64> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c0_i64 = arith.constant 0 : i64
  %init = arith.constant dense<0> : tensor<4xi64>
  %result = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %init) -> tensor<4xi64> {
    %key = util.string.format "param{}"(%i) : (index) -> !util.buffer
    %param = flow.parameter.load %key[%c0_i64] : tensor<4xi64>
    %sum = arith.addi %acc, %param : tensor<4xi64>
    scf.yield %sum : tensor<4xi64>
  }
  return %result : tensor<4xi64>
}
