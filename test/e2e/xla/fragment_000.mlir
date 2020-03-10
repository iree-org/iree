// RUN: iree-run-mlir -iree-hal-target-backends=vmla %s -input-value="f32=0" -input-value="5x1xf32=[1][-2][-3][4][-5]" -input-value="f32=1" -input-value="5x5xf32=[3.46499 -7.64389 -5.72249 5.98053 17.6892][2.9707 -6.20734 -4.25962 4.76055 13.8784][2.47641 -4.77079 -2.79675 3.54056 10.0675][1.98212 -3.33424 -1.33388 2.32058 6.25666][1.48783 -1.8977 0.12899 1.1006 2.4458]" -input-value="5xf32=0 0 0 0 0" | IreeFileCheck %s --implicit-check-not="[" --implicit-check-not="]"
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=vulkan-spirv %s -input-value="f32=0" -input-value="5x1xf32=[1][-2][-3][4][-5]" -input-value="f32=1" -input-value="5x5xf32=[3.46499 -7.64389 -5.72249 5.98053 17.6892][2.9707 -6.20734 -4.25962 4.76055 13.8784][2.47641 -4.77079 -2.79675 3.54056 10.0675][1.98212 -3.33424 -1.33388 2.32058 6.25666][1.48783 -1.8977 0.12899 1.1006 2.4458]" -input-value="5xf32=0 0 0 0 0" | IreeFileCheck %s --implicit-check-not="[" --implicit-check-not="]")

// CHECK-LABEL: EXEC @main_entry_dispatch_3
func @main_entry_dispatch_3(
    %0: tensor<f32>,
    %1: tensor<5x1xf32>,
    %2: tensor<f32>,
    %3: tensor<5x5xf32>,
    %4: tensor<5xf32>) -> tensor<5x5xf32> {
  %5 = "xla_hlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>, name = "broadcast.44"} : (tensor<5x1xf32>) -> tensor<5x1x5xf32>
  %6 = "xla_hlo.broadcast_in_dim"(%2) {name = "broadcast.9"} : (tensor<f32>) -> tensor<5x1x5xf32>
  %7 = mulf %5, %6 : tensor<5x1x5xf32>
  %8 = "xla_hlo.broadcast_in_dim"(%0) {name = "broadcast.47"} : (tensor<f32>) -> tensor<5x1x5xf32>
  %9 = cmpf "ogt", %7, %8 : tensor<5x1x5xf32>
  %10 = "xla_hlo.broadcast_in_dim"(%0) {name = "broadcast.11"} : (tensor<f32>) -> tensor<5x1x5xf32>
  %11 = "xla_hlo.broadcast_in_dim"(%0) {name = "broadcast.67"} : (tensor<f32>) -> tensor<5x5xf32>
  %12 = "xla_hlo.broadcast_in_dim"(%4) {broadcast_dimensions = dense<1> : tensor<1xi64>, name = "broadcast.64"} : (tensor<5xf32>) -> tensor<5x5xf32>
  %13 = addf %3, %12 : tensor<5x5xf32>
  %14 = xla_hlo.max %11, %13 {name = "maximum.68"} : tensor<5x5xf32>
  %15 = "xla_hlo.reshape"(%14) {name = "reshape.70"} : (tensor<5x5xf32>) -> tensor<5x1x5xf32>
  %16 = "xla_hlo.select"(%9, %10, %15) {name = "select.71"} : (tensor<5x1x5xi1>, tensor<5x1x5xf32>, tensor<5x1x5xf32>) -> tensor<5x1x5xf32>
  %17 = "xla_hlo.copy"(%16) {name = "copy.4"} : (tensor<5x1x5xf32>) -> tensor<5x1x5xf32>
  %18 = "xla_hlo.reshape"(%17) {name = "reshape.72"} : (tensor<5x1x5xf32>) -> tensor<5x5xf32>
  return %18 : tensor<5x5xf32>
}

// On separate lines to avoid "[[" which IreeFileCheck interprets as substitutions
// CHECK: 5x5xf32=
// CHECK-SAME: [0 0 0 0 0]
// CHECK-SAME: [2.97{{[0-9]+}} 0 0 4.76{{[0-9]+}} 13.87{{[0-9]+}}]
// CHECK-SAME: [2.47{{[0-9]+}} 0 0 3.54{{[0-9]+}} 10.06{{[0-9]+}}]
// CHECK-SAME: [0 0 0 0 0]
// CHECK-SAME: [1.48{{[0-9]+}} 0 0.12{{[0-9]+}} 1.10{{[0-9]+}} 2.44{{[0-9]+}}]
