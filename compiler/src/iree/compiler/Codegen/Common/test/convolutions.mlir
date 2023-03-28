// RUN: iree-opt %s --iree-transform-dialect-interpreter='transform-file-name=%p/convolution_implicit_gemm_nchw_spec.mlir' --split-input-file | FileCheck %s --check-prefix=CHECK-NCHW
// RUN: iree-opt %s --iree-transform-dialect-interpreter='transform-file-name=%p/convolution_implicit_gemm_nhwc_spec.mlir' --split-input-file | FileCheck %s --check-prefix=CHECK-NHWC
// RUN: iree-opt %s --iree-transform-dialect-interpreter='transform-file-name=%p/convolution_match_spec.mlir' --split-input-file --verify-diagnostics

!input_tensor_t = tensor<2x16x130x130xf32>
!weight_tensor_t = tensor<32x16x3x3xf32>
!output_tensor_t = tensor<2x32x128x128xf32>
func.func @conv_2d_nchw_fchw_trailing_eltwise(%in: !input_tensor_t, %wei: !weight_tensor_t,
                             %out: !output_tensor_t) -> !output_tensor_t {
  // expected-remark @below {{convolution}}
  %0 = linalg.conv_2d_nchw_fchw
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
     ins(%in, %wei: !input_tensor_t, !weight_tensor_t)
    outs(%out: !output_tensor_t) -> !output_tensor_t

  %1 = tensor.empty() : !output_tensor_t
  // expected-remark @below {{trailing}}
  %2 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%1 : !output_tensor_t) outs(%0 : !output_tensor_t) {
    ^bb0(%arg3: f32, %arg4: f32):
      %3 = math.sqrt %arg3 : f32
      linalg.yield %3 : f32
    } -> !output_tensor_t
  return %2 : !output_tensor_t
}

// CHECK-NCHW-LABEL: func.func @conv_2d_nchw_fchw_trailing_eltwise
// CHECK-NCHW-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-NCHW-DAG:     %[[C16:.+]] = arith.constant 16 : index
// CHECK-NCHW-DAG:     %[[C144:.+]] = arith.constant 144 : index

// CHECK-NCHW:         scf.forall ({{.*}}) in (2, 1, 512) shared_outs({{.*}}) -> (tensor<2x32x16384xf32>) {
// CHECK-NCHW:           scf.for %{{.*}} = %[[C0]] to %[[C144]] step %[[C16]] {{.*}} -> (tensor<1x32x32xf32>) {

/// Img2col packing
// CHECK-NCHW:             scf.forall ({{.*}}) in (32) shared_outs({{.*}}) -> (tensor<1x16x32xf32>) {
// CHECK-NCHW:               %[[COLVEC:.+]] = vector.gather {{.*}} : tensor<2x16x130x130xf32>, vector<16xindex>, vector<16xi1>, vector<16xf32> into vector<16xf32>
// CHECK-NCHW:               vector.transfer_write %[[COLVEC]], {{.*}} : vector<16xf32>, tensor<16xf32>
// CHECK-NCHW:             } {mapping = [#gpu.thread<x>]}

// Promoted filter
// CHECK-NCHW:             bufferization.alloc_tensor() {{.*}} : tensor<32x16xf32>

// GEMM from img2col
// CHECK-NCHW:             scf.forall ({{.*}}) in (1) shared_outs({{.*}}) -> (tensor<1x32x32xf32>) {
// CHECK-NCHW:               vector.transfer_read {{.*}} : tensor<32x16xf32>, vector<32x16xf32>
// CHECK-NCHW:               vector.transfer_read {{.*}} : tensor<1x16x32xf32>, vector<16x32xf32>
// CHECK-NCHW:               vector.transfer_read {{.*}} : tensor<1x32x32xf32>, vector<32x32xf32>
// CHECK-NCHW:               vector.contract {{.*}} : vector<32x16xf32>, vector<16x32xf32> into vector<32x32xf32>
// CHECK-NCHW:               vector.transfer_write {{.*}} : vector<32x32xf32>, tensor<1x32x32xf32>
// CHECK-NCHW:             } {mapping = [#gpu.warp<x>]}
// CHECK-NCHW:           }

// Trailing elementwise
// CHECK-NCHW:           scf.forall ({{.*}}) in (32) shared_outs({{.*}}) -> (tensor<1x32x32xf32>) {
// CHECK-NCHW:             math.sqrt %{{.*}} : vector<32xf32>
// CHECK-NCHW:           } {mapping = [#gpu.thread<x>]}

// CHECK-NCHW:         } {mapping = [#gpu.block<x>, #gpu.block<y>, #gpu.block<z>]}

// -----

!input_tensor_t = tensor<2x16x130x130xf32>
!weight_tensor_t = tensor<32x16x3x3xf32>
!output_tensor_t = tensor<2x32x128x128xf32>
func.func @conv_2d_nchw_fchw_fill(%in: !input_tensor_t, %wei: !weight_tensor_t) -> !output_tensor_t {

  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : !output_tensor_t
  // expected-remark @below {{fill}}
  %1 = linalg.fill ins(%cst : f32) outs(%0 : !output_tensor_t) -> !output_tensor_t

  // expected-remark @below {{convolution}}
  %2 = linalg.conv_2d_nchw_fchw
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
     ins(%in, %wei: !input_tensor_t, !weight_tensor_t)
    outs(%1: !output_tensor_t) -> !output_tensor_t
  return %2 : !output_tensor_t
}

// CHECK-NCHW-LABEL: func.func @conv_2d_nchw_fchw_fill

// CHECK-NCHW-DAG:     %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<1x32x1xf32>
// CHECK-NCHW-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-NCHW-DAG:     %[[C16:.+]] = arith.constant 16 : index
// CHECK-NCHW-DAG:     %[[C144:.+]] = arith.constant 144 : index

// CHECK-NCHW:         scf.forall ({{.*}}) in (2, 1, 512) shared_outs({{.*}}) -> (tensor<2x32x16384xf32>) {

/// Fill
// CHECK-NCHW:           scf.forall ({{.*}}) in (32) shared_outs({{.*}}) -> (tensor<1x32x32xf32>) {
// CHECK-NCHW:             vector.transfer_write %[[CST]], {{.*}} : vector<1x32x1xf32>, tensor<1x32x1xf32>
// CHECK-NCHW:           } {mapping = [#gpu.thread<x>]}

// CHECK-NCHW:           scf.for %{{.*}} = %[[C0]] to %[[C144]] step %[[C16]] {{.*}} -> (tensor<1x32x32xf32>) {

/// Img2col packing
// CHECK-NCHW:             scf.forall ({{.*}}) in (32) shared_outs({{.*}}) -> (tensor<1x16x32xf32>) {
// CHECK-NCHW:               %[[COLVEC:.+]] = vector.gather {{.*}} : tensor<2x16x130x130xf32>, vector<16xindex>, vector<16xi1>, vector<16xf32> into vector<16xf32>
// CHECK-NCHW:               vector.transfer_write %[[COLVEC]], {{.*}} : vector<16xf32>, tensor<16xf32>
// CHECK-NCHW:             } {mapping = [#gpu.thread<x>]}

/// Promoted filter
// CHECK-NCHW:             bufferization.alloc_tensor() {{.*}} : tensor<32x16xf32>

/// GEMM from img2col
// CHECK-NCHW:             scf.forall ({{.*}}) in (1) shared_outs({{.*}}) -> (tensor<1x32x32xf32>) {
// CHECK-NCHW:               vector.transfer_read {{.*}} : tensor<32x16xf32>, vector<32x16xf32>
// CHECK-NCHW:               vector.transfer_read {{.*}} : tensor<1x16x32xf32>, vector<16x32xf32>
// CHECK-NCHW:               vector.transfer_read {{.*}} : tensor<1x32x32xf32>, vector<32x32xf32>
// CHECK-NCHW:               vector.contract {{.*}} : vector<32x16xf32>, vector<16x32xf32> into vector<32x32xf32>
// CHECK-NCHW:               vector.transfer_write {{.*}} : vector<32x32xf32>, tensor<1x32x32xf32>
// CHECK-NCHW:             } {mapping = [#gpu.warp<x>]}
// CHECK-NCHW:           }

// CHECK-NCHW:         } {mapping = [#gpu.block<x>, #gpu.block<y>, #gpu.block<z>]}

// -----

!input_tensor_t = tensor<2x130x130x16xf32>
!weight_tensor_t = tensor<3x3x16x32xf32>
!output_tensor_t = tensor<2x128x128x32xf32>
func.func @conv_2d_nhwc_hwcf(%in: !input_tensor_t, %wei: !weight_tensor_t) -> !output_tensor_t {

  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : !output_tensor_t
  // expected-remark @below {{fill}}
  %1 = linalg.fill ins(%cst : f32) outs(%0 : !output_tensor_t) ->   !output_tensor_t

  // expected-remark @below {{convolution}}
  %2 = linalg.conv_2d_nhwc_hwcf
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
     ins(%in, %wei: !input_tensor_t, !weight_tensor_t)
    outs(%1: !output_tensor_t) -> !output_tensor_t

  %3 = tensor.empty() : !output_tensor_t
  // expected-remark @below {{trailing}}
  %4 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%2 : !output_tensor_t) outs(%3 : !output_tensor_t) {
    ^bb0(%arg3: f32, %arg4: f32):
      %5 = math.sqrt %arg3 : f32
      linalg.yield %5 : f32
    } -> !output_tensor_t
  return %4 : !output_tensor_t
}

// CHECK-NHWC-LABEL: func.func @conv_2d_nhwc_hwcf

// CHECK-NHWC-DAG:     %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<1x1x32xf32>
// CHECK-NHWC-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-NHWC-DAG:     %[[C16:.+]] = arith.constant 16 : index
// CHECK-NHWC-DAG:     %[[C144:.+]] = arith.constant 144 : index

// CHECK-NHWC:         scf.forall ({{.*}}) in (2, 512, 1) shared_outs({{.*}}) -> (tensor<2x16384x32xf32>) {

/// Fill
// CHECK-NHWC:           scf.forall ({{.*}}) in (32) shared_outs({{.*}}) -> (tensor<1x32x32xf32>) {
// CHECK-NHWC:             vector.transfer_write %[[CST]], {{.*}} : vector<1x1x32xf32>, tensor<1x1x32xf32>
// CHECK-NHWC:           } {mapping = [#gpu.thread<x>]}

// CHECK-NHWC:           scf.for %{{.*}} = %[[C0]] to %[[C144]] step %[[C16]] {{.*}} -> (tensor<1x32x32xf32>) {

/// Img2col packing
// CHECK-NHWC:             scf.forall ({{.*}}) in (32) shared_outs({{.*}}) -> (tensor<1x32x16xf32>) {
// CHECK-NHWC:               %[[COLVEC:.+]] = vector.gather {{.*}} : tensor<2x130x130x16xf32>, vector<16xindex>, vector<16xi1>, vector<16xf32> into vector<16xf32>
// CHECK-NHWC:               vector.transfer_write %[[COLVEC]], {{.*}} : vector<16xf32>, tensor<1x1x16xf32>
// CHECK-NHWC:             } {mapping = [#gpu.thread<x>]}

/// Promoted filter
// CHECK-NHWC:             bufferization.alloc_tensor() {{.*}} : tensor<16x32xf32>

/// GEMM from img2col
// CHECK-NHWC:             scf.forall ({{.*}}) in (1) shared_outs({{.*}}) -> (tensor<1x32x32xf32>) {
// CHECK-NHWC:               vector.transfer_read {{.*}} : tensor<1x32x16xf32>, vector<32x16xf32>
// CHECK-NHWC:               vector.transfer_read {{.*}} : tensor<16x32xf32>, vector<16x32xf32>
// CHECK-NHWC:               vector.transfer_read {{.*}} : tensor<1x32x32xf32>, vector<32x32xf32>
// CHECK-NHWC:               vector.contract {{.*}} : vector<32x16xf32>, vector<16x32xf32> into vector<32x32xf32>
// CHECK-NHWC:               vector.transfer_write {{.*}} : vector<32x32xf32>, tensor<1x32x32xf32>
// CHECK-NHWC:             } {mapping = [#gpu.warp<x>]}
// CHECK-NHWC:           }

// Trailing elementwise
// CHECK-NHWC:           scf.forall ({{.*}}) in (32) shared_outs({{.*}}) -> (tensor<1x32x32xf32>) {
// CHECK-NHWC:             math.sqrt %{{.*}} : vector<32xf32>
// CHECK-NHWC:           } {mapping = [#gpu.thread<x>]}

// CHECK-NHWC:         } {mapping = [#gpu.block<x>, #gpu.block<y>, #gpu.block<z>]}
