// Tests that multiple devices are supported through iree-run-module by
// providing two local thread pools. This is not optimal and not an intended
// route for multi-device CPU workloads but requires no additional hardware
// resources for the test and still verifies the compiler/runtime tooling
// rendezvous of devices as specified on the command line.

// RUN: (iree-compile %s \
// RUN:      --iree-execution-model=async-external \
// RUN:      --iree-hal-target-device=device_a=local[0] \
// RUN:      --iree-hal-target-device=device_b=local[1] \
// RUN:      --iree-hal-local-target-device-backends=vmvx | \
// RUN:  iree-run-module \
// RUN:      --module=- \
// RUN:      --function=mutli_device_mul \
// RUN:      --input=4xf32=10,11,12,13 \
// RUN:      --device=local-task \
// RUN:      --device=local-task \
// RUN:      --task_topology_group_count=1) | \
// RUN: FileCheck %s

// CHECK: EXEC @mutli_device_mul
// CHECK-NEXT: result[0]: hal.buffer_view
// CHECK-NEXT: 4xf32=0 55 144 273
func.func public @mutli_device_mul(
  // Input argument is resident on device_a (tooling default to first device).
  %input_a: tensor<4xf32> {iree.abi.affinity = #hal.device.promise<@device_a>}
) -> (
  // Output result is expected to be on device_a (though not required).
  tensor<4xf32> {iree.abi.affinity = #hal.device.promise<@device_a>}
) {
  // Compute on device_a (input is there).
  %constant_a = arith.constant dense<[0.0, 1.0, 2.0, 3.0]> : tensor<4xf32>
  %transient_a = arith.mulf %input_a, %constant_a : tensor<4xf32>
  // Transfer the result from device_a -> device_b.
  %transient_b = flow.tensor.transfer %transient_a : tensor<4xf32> to #hal.device.promise<@device_b>
  // Compute on device_b.
  %constant_b = arith.constant dense<[4.0, 5.0, 6.0, 7.0]> : tensor<4xf32>
  %result_b = arith.mulf %transient_b, %constant_b : tensor<4xf32>
  // Transfer the result from device_b -> device_a.
  %result_a = flow.tensor.transfer %result_b : tensor<4xf32> to #hal.device.promise<@device_a>
  // Return the result on device_a (as required by ABI attr).
  func.return %result_a : tensor<4xf32>
}
