// RUN: iree-opt --split-input-file %s --iree-gpu-test-target=gfx1100 --pass-pipeline="builtin.module(iree-preprocessing-pad-to-intrinsics,func.func(canonicalize))" | FileCheck %s
// RUN: iree-opt --split-input-file %s --iree-gpu-test-target=gfx1100 --pass-pipeline="builtin.module(iree-preprocessing-pad-to-intrinsics{pad-target-type=conv},func.func(canonicalize))" | FileCheck %s -check-prefix=CONVOLUTION
// RUN: iree-opt --split-input-file %s --iree-gpu-test-target=gfx1100 --pass-pipeline="builtin.module(iree-preprocessing-pad-to-intrinsics{pad-target-type=contraction},func.func(canonicalize))" | FileCheck %s -check-prefix=CONTRACT


// CHECK-LABEL: func.func @main0(
// CHECK-SAME:    %[[ARG0:.+]]: tensor<2x130x130x4xf16>,
// CHECK-SAME:    %[[ARG1:.+]]: tensor<3x3x4x320xf16>,
// CHECK-SAME:    %[[ARG2:.+]]: tensor<2x128x128x320xf32>)
func.func @main0(%arg0: tensor<2x130x130x4xf16>, %arg1: tensor<3x3x4x320xf16>, %arg2: tensor<2x128x128x320xf32>)
    -> tensor<2x128x128x320xf32> {
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

// CONVOLUTION:      tensor.pad {{.*}} low[0, 0, 0, 0] high[0, 0, 0, 12]
// CONVOLUTION:      tensor.pad {{.*}} low[0, 0, 0, 0] high[0, 0, 12, 0]

// CONTRACT-NOT:     tensor.pad {{.*}} low[0, 0, 0, 0] high[0, 0, 0, 12]
// CONTRACT-NOT:     tensor.pad {{.*}} low[0, 0, 0, 0] high[0, 0, 12, 0]

// -----

// CHECK-LABEL: func.func @conv_nchw_fchw(
// CHECK-SAME:    %[[ARG0:.+]]: tensor<2x4x130x130xf16>,
// CHECK-SAME:    %[[ARG1:.+]]: tensor<320x4x3x3xf16>,
// CHECK-SAME:    %[[ARG2:.+]]: tensor<2x320x128x128xf32>)
func.func @conv_nchw_fchw(%arg0: tensor<2x4x130x130xf16>, %arg1: tensor<320x4x3x3xf16>, %arg2: tensor<2x320x128x128xf32>)
    -> tensor<2x320x128x128xf32> {
  %conv0 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
             ins(%arg0, %arg1 : tensor<2x4x130x130xf16>, tensor<320x4x3x3xf16>)
             outs(%arg2 : tensor<2x320x128x128xf32>) -> tensor<2x320x128x128xf32>
  return %conv0 : tensor<2x320x128x128xf32>
}

// CHECK:      %[[CST0:.+]] = arith.constant 0.0{{.*}} : f16
// CHECK:      %[[PAD0:.+]] = tensor.pad %[[ARG0]] low[0, 0, 0, 0] high[0, 12, 0, 0]
// CHECK:        tensor.yield %[[CST0]] : f16
// CHECK-NEXT:   tensor<2x4x130x130xf16> to tensor<2x16x130x130xf16>
// CHECK:      %[[PAD1:.+]] = tensor.pad %[[ARG1]] low[0, 0, 0, 0] high[0, 12, 0, 0]
// CHECK:        tensor<320x4x3x3xf16> to tensor<320x16x3x3xf16>
// CHECK:      %[[CONV:.+]] = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
// CHECK-SAME:   ins(%[[PAD0]], %[[PAD1]] : tensor<2x16x130x130xf16>, tensor<320x16x3x3xf16>)
// CHECK-SAME:   outs(%[[ARG2]] : tensor<2x320x128x128xf32>)
// CHECK:      return %[[CONV]] : tensor<2x320x128x128xf32>

// CONVOLUTION:      tensor.pad {{.*}} low[0, 0, 0, 0] high[0, 12, 0, 0]
// CONVOLUTION:      tensor.pad {{.*}} low[0, 0, 0, 0] high[0, 12, 0, 0]

// CONTRACT-NOT:     tensor.pad {{.*}} low[0, 0, 0, 0] high[0, 12, 0, 0]
// CONTRACT-NOT:     tensor.pad {{.*}} low[0, 0, 0, 0] high[0, 12, 0, 0]

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @conv_generic_nhwc_fhwc(
// CHECK-SAME:    %[[ARG0:.+]]: tensor<2x130x130x4xf16>,
// CHECK-SAME:    %[[ARG1:.+]]: tensor<320x3x3x4xf16>,
// CHECK-SAME:    %[[ARG2:.+]]: tensor<2x128x128x320xf32>)
func.func @conv_generic_nhwc_fhwc(%arg0: tensor<2x130x130x4xf16>, %arg1: tensor<320x3x3x4xf16>, %arg2: tensor<2x128x128x320xf32>) -> tensor<2x128x128x320xf32> {
  %1 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<2x130x130x4xf16>, tensor<320x3x3x4xf16>) outs(%arg2 : tensor<2x128x128x320xf32>) {
  ^bb0(%in: f16, %in_0: f16, %out: f32):
    %2 = arith.extf %in : f16 to f32
    %3 = arith.extf %in_0 : f16 to f32
    %4 = arith.mulf %2, %3 : f32
    %5 = arith.addf %out, %4 : f32
    linalg.yield %5 : f32
  } -> tensor<2x128x128x320xf32>
  return %1 : tensor<2x128x128x320xf32>
}

// CHECK:      %[[CST0:.+]] = arith.constant 0.0{{.*}} : f16
// CHECK:      %[[PAD0:.+]] = tensor.pad %[[ARG0]] low[0, 0, 0, 0] high[0, 0, 0, 12]
// CHECK:        tensor.yield %[[CST0]] : f16
// CHECK-NEXT:   tensor<2x130x130x4xf16> to tensor<2x130x130x16xf16>
// CHECK:      %[[PAD1:.+]] = tensor.pad %[[ARG1]] low[0, 0, 0, 0] high[0, 0, 0, 12]
// CHECK:        tensor<320x3x3x4xf16> to tensor<320x3x3x16xf16>
// CHECK:      %[[CONV:.+]] = linalg.generic
// CHECK-SAME:   indexing_maps = [#map, #map1, #map2],
// CHECK-SAME:   ins(%[[PAD0]], %[[PAD1]] : tensor<2x130x130x16xf16>, tensor<320x3x3x16xf16>)
// CHECK-SAME:   outs(%[[ARG2]] : tensor<2x128x128x320xf32>)
// CHECK:      return %[[CONV]] : tensor<2x128x128x320xf32>

// CONVOLUTION:      tensor.pad {{.*}} low[0, 0, 0, 0] high[0, 0, 0, 12]
// CONVOLUTION:      tensor.pad {{.*}} low[0, 0, 0, 0] high[0, 0, 0, 12]

// CONTRACT-NOT:     tensor.pad {{.*}} low[0, 0, 0, 0] high[0, 0, 0, 12]
// CONTRACT-NOT:     tensor.pad {{.*}} low[0, 0, 0, 0] high[0, 0, 0, 12]

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 + d3, d1 + d4, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d3, d4, d5)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>

// CHECK-LABEL: func.func @conv2d_no_batch(
// CHECK-SAME:    %[[ARG0:.+]]: tensor<1026x1026x128xf16>
// CHECK-SAME:    %[[ARG1:.+]]: tensor<3x3x3x128xf16>
// CHECK-SAME:    %[[ARG2:.+]]: tensor<1024x1024x3xf32>)
func.func @conv2d_no_batch(%arg0: tensor<1026x1026x128xf16>, %arg1: tensor<3x3x3x128xf16>, %arg2:tensor<1024x1024x3xf32>) -> tensor<1024x1024x3xf32> {
  %conv = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<1026x1026x128xf16>, tensor<3x3x3x128xf16>) outs(%arg2 : tensor<1024x1024x3xf32>) {
  ^bb0(%in: f16, %in_0: f16, %out: f32):
    %0 = arith.extf %in : f16 to f32
    %1 = arith.extf %in_0 : f16 to f32
    %2 = arith.mulf %0, %1 : f32
    %3 = arith.addf %out, %2 : f32
    linalg.yield %3 : f32
  } -> tensor<1024x1024x3xf32>
  return %conv : tensor<1024x1024x3xf32>
}

// CHECK:      %[[PAD0:.+]] = tensor.pad %[[ARG1]] low[0, 0, 0, 0] high[13, 0, 0, 0] {
// CHECK:      tensor<3x3x3x128xf16> to tensor<16x3x3x128xf16>
// CHECK:      %[[PAD1:.+]] = tensor.pad %[[ARG2]] low[0, 0, 0] high[0, 0, 13] {
// CHECK:      tensor<1024x1024x3xf32> to tensor<1024x1024x16xf32>
// CHECK:      %[[CONV:.+]] = linalg.generic {indexing_maps = [#map, #map1, #map2]
// CHECK-SAME: ins(%arg0, %[[PAD0]] : tensor<1026x1026x128xf16>, tensor<16x3x3x128xf16>)
// CHECK-SAME: outs(%[[PAD1]] : tensor<1024x1024x16xf32>) {
// CHECK:      %[[RES:.+]] = tensor.extract_slice %[[CONV]][0, 0, 0] [1024, 1024, 3] [1, 1, 1]
// CHECK:      return %[[RES]] : tensor<1024x1024x3xf32>

// CONVOLUTION:      tensor.pad {{.*}} low[0, 0, 0, 0] high[13, 0, 0, 0]
// CONVOLUTION:      tensor.pad {{.*}} low[0, 0, 0] high[0, 0, 13]

// CONTRACT-NOT:     tensor.pad {{.*}} low[0, 0, 0, 0] high[13, 0, 0, 0]
// CONTRACT-NOT:     tensor.pad {{.*}} low[0, 0, 0] high[0, 0, 13]

// -----

// CHECK-LABEL: func.func @main1(
// CHECK-SAME:    %[[ARG0:.+]]: tensor<2x130x130x320xf16>,
// CHECK-SAME:    %[[ARG1:.+]]: tensor<3x3x320x4xf16>,
// CHECK-SAME:    %[[ARG2:.+]]: tensor<2x128x128x4xf32>)
func.func @main1(%arg0: tensor<2x130x130x320xf16>, %arg1: tensor<3x3x320x4xf16>, %arg2: tensor<2x128x128x4xf32>)
    -> tensor<2x128x128x4xf32> {
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

// CONVOLUTION:      tensor.pad {{.*}} low[0, 0, 0, 0] high[0, 0, 0, 12]

// CONTRACT-NOT:     tensor.pad {{.*}} low[0, 0, 0, 0] high[0, 0, 0, 12]

// -----

// Use an explicit target here given we are swapping the preferred order of MFMA intrinsics.
#target = #iree_gpu.target<arch = "gfx942", features = "",
  wgp = <compute = fp64|fp32|fp16|int64|int32|int16|int8, storage = b64|b32|b16|b8,
  subgroup = shuffle|arithmetic, dot = dp4xi8toi32,
  mma = [<MFMA_F32_32x32x8_F16>],
  subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024],
  max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536,
  max_workgroup_counts = [2147483647, 2147483647, 2147483647]>>
#rocm_executable_target = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.target_info = #target, ukernels = "none"}>

// CHECK-LABEL: func.func @main2(
// CHECK-SAME:    %[[ARG0:.+]]: tensor<2x130x130x4xf16>,
// CHECK-SAME:    %[[ARG1:.+]]: tensor<3x3x4x320xf16>,
// CHECK-SAME:    %[[ARG2:.+]]: tensor<2x128x128x320xf32>)
func.func @main2(%arg0: tensor<2x130x130x4xf16>, %arg1: tensor<3x3x4x320xf16>, %arg2: tensor<2x128x128x320xf32>)
    -> tensor<2x128x128x320xf32>
    attributes {hal.executable.target = #rocm_executable_target} {
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

// -----

// We want to skip padding skinny matmul cases, since warpReduction is more performant for it.

#rocm_executable_target = #hal.executable.target<"rocm", "rocm-hsaco-fb">

//       CHECK: func.func @skip_skinny_m_matmul(
//  CHECK-SAME:    %[[ARG0:.+]]: tensor<2x20xf16>,
//  CHECK-SAME:    %[[ARG1:.+]]: tensor<20x30xf16>,
//  CHECK-SAME:    %[[ARG2:.+]]: tensor<2x30xf16>)
func.func @skip_skinny_m_matmul(%arg0 : tensor<2x20xf16>, %arg1 : tensor<20x30xf16>, %arg2 : tensor<2x30xf16>) -> tensor<2x30xf16>
    attributes {hal.device.targets = [#hal.device.target<"rocm", [#rocm_executable_target]>]} {
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<2x20xf16>, tensor<20x30xf16>)
        outs(%arg2 : tensor<2x30xf16>) -> tensor<2x30xf16>
    return %0 : tensor<2x30xf16>
}

// CHECK-NOT:  tensor.pad

// -----

// We want to skip padding skinny matmul cases, since warpReduction is more performant for it.

#rocm_executable_target = #hal.executable.target<"rocm", "rocm-hsaco-fb">

//       CHECK: func.func @skip_skinny_n_mmtb(
//  CHECK-SAME:    %[[ARG0:.+]]: tensor<10x20xf16>,
//  CHECK-SAME:    %[[ARG1:.+]]: tensor<4x20xf16>,
//  CHECK-SAME:    %[[ARG2:.+]]: tensor<10x4xf16>)
func.func @skip_skinny_n_mmtb(%arg0 : tensor<10x20xf16>, %arg1 : tensor<4x20xf16>, %arg2 : tensor<10x4xf16>) -> tensor<10x4xf16>
    attributes {hal.device.targets = [#hal.device.target<"rocm", [#rocm_executable_target]>]} {
    %0 = linalg.matmul
        indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d2)>,
          affine_map<(d0, d1, d2) -> (d1, d2)>,
          affine_map<(d0, d1, d2) -> (d0, d1)>
        ]
        ins(%arg0, %arg1 : tensor<10x20xf16>, tensor<4x20xf16>)
        outs(%arg2 : tensor<10x4xf16>) -> tensor<10x4xf16>
    return %0 : tensor<10x4xf16>
}

// CHECK-NOT:  tensor.pad
