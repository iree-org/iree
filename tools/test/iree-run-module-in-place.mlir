// RUN: iree-compile %s \
// RUN:    --iree-hal-target-device=local \
// RUN:    --iree-hal-local-target-device-backends=vmvx \
// RUN:    --compile-to=stream | \
// RUN: FileCheck %s --check-prefix=CHECK-IR

// Ensure no allocations made it into the IR - in this example the output should
// be placed in the %output storage and the required transient in
// %transient_storage.
// CHECK-IR-LABEL: util.func public @in_place_computation
// CHECK-IR-NOT: stream.resource.alloc
// CHECK-IR-NOT: stream.resource.alloca
// CHECK-IR-NOT: stream.resource.dealloca

// RUN: (iree-compile %s \
// RUN:    --iree-hal-target-device=local \
// RUN:    --iree-hal-local-target-device-backends=vmvx | \
// RUN:  iree-run-module \
// RUN:    --device=local-task \
// RUN:    --module=- \
// RUN:    --function=in_place_computation \
// RUN:    --input="64xf32=1.0" \
// RUN:    --input="64xf32=0" \
// RUN:    --input="&512xi8=0") | \
// RUN:  FileCheck %s --check-prefix=CHECK-EXEC

// Ensure we produce the right value (the returned buffer view is aliasing the
// %output storage provided, but there's no way to test that here).
// CHECK-EXEC-LABEL: EXEC @in_place_computation
// CHECK-EXEC: 64xf32=5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5

// Test for external transients feature: verifies that transient allocations
// can be emplaced into externally-provided storage.
//
// The test performs three sequential operations wrapped in dispatch regions
// to prevent fusion:
// 1. temp1 = input + 1.0  (uses input + transient storage)
// 2. temp2 = temp1 * 2.0  (uses transient + transient storage)
// 3. output = temp2 + input (uses transient + output buffer)
//
// With input=[1.0, 1.0, ...], the expected output is [5.0, 5.0, ...].

util.func public @in_place_computation(
  %input: tensor<64xf32>,
  %output: tensor<64xf32> {iree.abi.output = 0 : index},
  %transient_storage: !hal.buffer {iree.abi.transients}
) -> tensor<64xf32> {
  // Dispatch 1: temp1 = input + 1.0
  %temp1 = flow.dispatch.region -> (tensor<64xf32>) {
    %one = arith.constant dense<1.0> : tensor<64xf32>
    %result = arith.addf %input, %one : tensor<64xf32>
    flow.return %result : tensor<64xf32>
  }

  // Dispatch 2: temp2 = temp1 * 2.0
  %temp2 = flow.dispatch.region -> (tensor<64xf32>) {
    %two = arith.constant dense<2.0> : tensor<64xf32>
    %result = arith.mulf %temp1, %two : tensor<64xf32>
    flow.return %result : tensor<64xf32>
  }

  // Dispatch 3: output = temp2 + input
  %result = flow.dispatch.region -> (tensor<64xf32>) {
    %final = arith.addf %temp2, %input : tensor<64xf32>
    flow.return %final : tensor<64xf32>
  }

  util.return %result : tensor<64xf32>
}
