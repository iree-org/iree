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
