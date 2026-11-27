// Convolutions whose batch, output spatial or output feature extent is zero
// at run time return empty results.

// RUN: iree-compile %s --iree-input-type=stablehlo --iree-hal-target-device=local \
// RUN:   --iree-hal-local-target-device-backends=vmvx -o %t.vmvx.vmfb
// RUN: iree-run-module --device=local-task --module=%t.vmvx.vmfb --function=zero_batch \
// RUN:   --input=0x9x5x4xf32= --input=3x2x4x6xf32=1 --expected_output=0x7x4x6xi32= | FileCheck %s
// RUN: iree-run-module --device=local-task --module=%t.vmvx.vmfb --function=zero_spatial \
// RUN:   --input=2x2x5x4xf32=1 --input=3x2x4x6xf32=1 --expected_output=2x0x4x6xi32= | FileCheck %s
// RUN: iree-run-module --device=local-task --module=%t.vmvx.vmfb --function=zero_feature \
// RUN:   --input=2x9x5x4xf32=1 --input=3x2x4x0xf32= --expected_output=2x7x4x0xi32= | FileCheck %s
// RUN: iree-compile %s --iree-input-type=stablehlo --iree-hal-target-device=local \
// RUN:   --iree-hal-local-target-device-backends=llvm-cpu --iree-llvmcpu-target-cpu=generic -o %t.cpu.vmfb
// RUN: iree-run-module --device=local-task --module=%t.cpu.vmfb --function=zero_batch \
// RUN:   --input=0x9x5x4xf32= --input=3x2x4x6xf32=1 --expected_output=0x7x4x6xi32= | FileCheck %s

// CHECK: [SUCCESS]

func.func @zero_batch(%x: tensor<?x9x5x4xf32>, %w: tensor<3x2x4x6xf32>) -> tensor<?x7x4x6xi32> {
  %r = stablehlo.convolution(%x, %w) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<?x9x5x4xf32>, tensor<3x2x4x6xf32>) -> tensor<?x7x4x6xf32>
  %bits = stablehlo.bitcast_convert %r : (tensor<?x7x4x6xf32>) -> tensor<?x7x4x6xi32>
  return %bits : tensor<?x7x4x6xi32>
}

func.func @zero_spatial(%x: tensor<2x?x5x4xf32>, %w: tensor<3x2x4x6xf32>) -> tensor<2x?x4x6xi32> {
  %r = stablehlo.convolution(%x, %w) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<2x?x5x4xf32>, tensor<3x2x4x6xf32>) -> tensor<2x?x4x6xf32>
  %bits = stablehlo.bitcast_convert %r : (tensor<2x?x4x6xf32>) -> tensor<2x?x4x6xi32>
  return %bits : tensor<2x?x4x6xi32>
}

func.func @zero_feature(%x: tensor<2x9x5x4xf32>, %w: tensor<3x2x4x?xf32>) -> tensor<2x7x4x?xi32> {
  %r = stablehlo.convolution(%x, %w) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<2x9x5x4xf32>, tensor<3x2x4x?xf32>) -> tensor<2x7x4x?xf32>
  %bits = stablehlo.bitcast_convert %r : (tensor<2x7x4x?xf32>) -> tensor<2x7x4x?xi32>
  return %bits : tensor<2x7x4x?xi32>
}
