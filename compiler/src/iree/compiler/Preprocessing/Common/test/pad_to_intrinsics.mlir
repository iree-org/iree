
// RUN: iree-opt %s --pass-pipeline="builtin.module(func.func(iree-preprocessing-pad-to-intrinsics))" \
// RUN:   | FileCheck %s

#rocm_executable_target0 = #hal.executable.target<"rocm", "rocm-hsaco-fb",
                             {mma_intrinsics = [#iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>,
                                                #iree_gpu.mma_layout<MFMA_F16_32x32x8_F32>],
                              target_arch = "gfx942", ukernels = "none"}>

// CHECK-LABEL: func.func @main0(
// CHECK-SAME:    %[[ARG0:.+]]: tensor<2x130x130x4xf16>,
// CHECK-SAME:    %[[ARG1:.+]]: tensor<3x3x4x320xf16>,
// CHECK-SAME:    %[[ARG2:.+]]: tensor<2x128x128x320xf32>)
func.func @main0(%arg0: tensor<2x130x130x4xf16>, %arg1: tensor<3x3x4x320xf16>, %arg2: tensor<2x128x128x320xf32>)
    -> tensor<2x128x128x320xf32>
    attributes {hal.device.targets = [#hal.device.target<"rocm", [#rocm_executable_target0]>]} {
  %conv0 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
             ins(%arg0, %arg1 : tensor<2x130x130x4xf16>, tensor<3x3x4x320xf16>)
             outs(%arg2 : tensor<2x128x128x320xf32>) -> tensor<2x128x128x320xf32>
  return %conv0 : tensor<2x128x128x320xf32>
}

// CHECK:      %[[CST0:.+]] = arith.constant 0.0{{.*}} : f16
// CHECK:      %[[PAD0:.+]] = tensor.pad %[[ARG0]] low[0, 0, 0, 0] high[0, 0, 0, 12]
// CHECK:        tensor.yield %[[CST0]] : f16
// CHECK-NEXT:   tensor<2x130x130x4xf16> to tensor<2x130x130x16xf16>
// CHECK:      %[[PAD1:.+]] = tensor.pad %[[ARG1]] low[0, 0, 0, 0] high[0, 0, 12, 0]
// CHECK:        tensor<3x3x4x320xf16> to tensor<3x3x16x320xf16>
// CHECK:      %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
// CHECK-SAME:   ins(%[[PAD0]], %[[PAD1]] : tensor<2x130x130x16xf16>, tensor<3x3x16x320xf16>)
// CHECK-SAME:   outs(%[[ARG2]] : tensor<2x128x128x320xf32>) -> tensor<2x128x128x320xf32>
// CHECK:      return %[[CONV]] : tensor<2x128x128x320xf32>


// CHECK-LABEL: func.func @main1(
// CHECK-SAME:    %[[ARG0:.+]]: tensor<2x130x130x320xf16>,
// CHECK-SAME:    %[[ARG1:.+]]: tensor<3x3x320x4xf16>,
// CHECK-SAME:    %[[ARG2:.+]]: tensor<2x128x128x4xf32>)
func.func @main1(%arg0: tensor<2x130x130x320xf16>, %arg1: tensor<3x3x320x4xf16>, %arg2: tensor<2x128x128x4xf32>)
    -> tensor<2x128x128x4xf32>
    attributes {hal.device.targets = [#hal.device.target<"rocm", [#rocm_executable_target0]>]} {
  %conv0 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
             ins(%arg0, %arg1 : tensor<2x130x130x320xf16>, tensor<3x3x320x4xf16>)
             outs(%arg2 : tensor<2x128x128x4xf32>) -> tensor<2x128x128x4xf32>
  return %conv0 : tensor<2x128x128x4xf32>
}

// CHECK:      %[[CST0:.+]] = arith.constant 0.0{{.*}} : f16
// CHECK:      %[[PAD1:.+]] = tensor.pad %[[ARG1]] low[0, 0, 0, 0] high[0, 0, 0, 12]
// CHECK:        tensor<3x3x320x4xf16> to tensor<3x3x320x16xf16>
// CHECK:      %[[PAD2:.+]] = tensor.pad %[[ARG2]] low[0, 0, 0, 0] high[0, 0, 0, 12]
// CHECK:        tensor<2x128x128x4xf32> to tensor<2x128x128x16xf32>
// CHECK:      %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
// CHECK-SAME:   ins(%[[ARG0]], %[[PAD1]] : tensor<2x130x130x320xf16>, tensor<3x3x320x16xf16>)
// CHECK-SAME:   outs(%[[PAD2]] : tensor<2x128x128x16xf32>) -> tensor<2x128x128x16xf32>
// CHECK:      %[[RET:.+]] = tensor.extract_slice %[[CONV]][0, 0, 0, 0] [2, 128, 128, 4] [1, 1, 1, 1]
// CHECK:      return %[[RET]] : tensor<2x128x128x4xf32>

#rocm_executable_target1 = #hal.executable.target<"rocm", "rocm-hsaco-fb",
                             {mma_intrinsics = [#iree_gpu.mma_layout<MFMA_F16_32x32x8_F32>],
                              target_arch = "gfx942", ukernels = "none"}>

// CHECK-LABEL: func.func @main2(
// CHECK-SAME:    %[[ARG0:.+]]: tensor<2x130x130x4xf16>,
// CHECK-SAME:    %[[ARG1:.+]]: tensor<3x3x4x320xf16>,
// CHECK-SAME:    %[[ARG2:.+]]: tensor<2x128x128x320xf32>)
func.func @main2(%arg0: tensor<2x130x130x4xf16>, %arg1: tensor<3x3x4x320xf16>, %arg2: tensor<2x128x128x320xf32>)
    -> tensor<2x128x128x320xf32>
    attributes {hal.device.targets = [#hal.device.target<"rocm", [#rocm_executable_target1]>]} {
  %conv0 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
             ins(%arg0, %arg1 : tensor<2x130x130x4xf16>, tensor<3x3x4x320xf16>)
             outs(%arg2 : tensor<2x128x128x320xf32>) -> tensor<2x128x128x320xf32>
  return %conv0 : tensor<2x128x128x320xf32>
}

// CHECK:      %[[CST0:.+]] = arith.constant 0.0{{.*}} : f16
// CHECK:      %[[PAD0:.+]] = tensor.pad %[[ARG0]] low[0, 0, 0, 0] high[0, 0, 0, 4]
// CHECK:        tensor<2x130x130x4xf16> to tensor<2x130x130x8xf16>
// CHECK:      %[[PAD1:.+]] = tensor.pad %[[ARG1]] low[0, 0, 0, 0] high[0, 0, 4, 0]
// CHECK:        tensor<3x3x4x320xf16> to tensor<3x3x8x320xf16>
// CHECK:      %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
// CHECK-SAME:   ins(%[[PAD0]], %[[PAD1]] : tensor<2x130x130x8xf16>, tensor<3x3x8x320xf16>)
// CHECK-SAME:   outs(%[[ARG2]] : tensor<2x128x128x320xf32>) -> tensor<2x128x128x320xf32>
// CHECK:      return %[[CONV]] : tensor<2x128x128x320xf32>
