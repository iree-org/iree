// Tests that iree-check-module profiles every device (device_a + device_b).

// RUN: iree-compile %s \
// RUN:     --iree-hal-target-device=device_a=local[0] \
// RUN:     --iree-hal-target-device=device_b=local[1] \
// RUN:     --iree-hal-local-target-device-backends=vmvx | \
// RUN: iree-check-module \
// RUN:     --module=- \
// RUN:     --device=local-task \
// RUN:     --device=local-task \
// RUN:     --task_topology_group_count=1 \
// RUN:     --print_device_statistics 2>&1 | \
// RUN: FileCheck %s

// CHECK: IREE HAL device statistics (device[0] local-task):
// CHECK-NOT: dispatch_1_
// CHECK: host_execute _multi_device_mul_dispatch_0_
// CHECK-NOT: dispatch_1_
// CHECK: IREE HAL device statistics (device[1] local-task):
// CHECK-NOT: dispatch_0_
// CHECK: host_execute _multi_device_mul_dispatch_1_
// CHECK-NOT: dispatch_0_

func.func @multi_device_mul() {
  // Compute on device_a.
  %input_a = util.unfoldable_constant dense<[10.0, 11.0, 12.0, 13.0]> : tensor<4xf32>
  %constant_a = arith.constant dense<[0.0, 1.0, 2.0, 3.0]> : tensor<4xf32>
  %transient_a = arith.mulf %input_a, %constant_a : tensor<4xf32>
  // Transfer the result from device_a -> device_b.
  %transient_b = flow.tensor.transfer %transient_a : tensor<4xf32> to #hal.device.promise<@device_b>
  // Compute on device_b.
  %constant_b = arith.constant dense<[4.0, 5.0, 6.0, 7.0]> : tensor<4xf32>
  %result_b = arith.mulf %transient_b, %constant_b : tensor<4xf32>
  // Transfer the result from device_b -> device_a.
  %result_a = flow.tensor.transfer %result_b : tensor<4xf32> to #hal.device.promise<@device_a>
  check.expect_eq_const(%result_a, dense<[0.0, 55.0, 144.0, 273.0]> : tensor<4xf32>) : tensor<4xf32>
  return
}
