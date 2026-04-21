// Padded conv2d regression test for the gfx950 im2col + DMA
// (gather_to_lds) path. Covers the OOB-sentinel emission introduced for
// padded-im2col async-copy: corner positions must read pad_value = 0
// (proving fat_raw_buffer OOB → 0 worked), interior positions must get
// the full 3*3*C contribution.
//
// Input and filter are all-ones; the 3x3 same-padding conv then produces:
//   interior (1 <= i <= H-2, 1 <= j <= W-2): sum = 9 * C = 9 * 512 = 4608
//   corner   (0,0 / 0,W-1 / H-1,0 / H-1,W-1): sum = 4 * C = 4 * 512 = 2048
//
// Interior ≠ corner ≠ 0 iff the sentinel worked correctly. If the sentinel
// were broken, corner values would exceed 2048; if OOB-to-zero were broken,
// interior values would change.

!input_type = tensor<1x8x8x512xf16>
!padded_type = tensor<1x10x10x512xf16>
!filter_type = tensor<3x3x512x512xf16>
!output_type = tensor<1x8x8x512xf32>
!interior_type = tensor<1x6x6x512xf32>
!corner_type = tensor<1x1x1x512xf32>

func.func @im2col_dma_conv_padded() {
  %input = util.unfoldable_constant dense<1.0> : !input_type
  %filter = util.unfoldable_constant dense<1.0> : !filter_type
  %cst_f16 = arith.constant 0.000000e+00 : f16
  %cst_f32 = arith.constant 0.000000e+00 : f32
  %padded = tensor.pad %input low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):
    tensor.yield %cst_f16 : f16
  } : !input_type to !padded_type
  %empty = tensor.empty() : !output_type
  %fill = linalg.fill ins(%cst_f32 : f32) outs(%empty : !output_type) -> !output_type
  %result = linalg.conv_2d_nhwc_hwcf {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%padded, %filter : !padded_type, !filter_type)
    outs(%fill : !output_type) -> !output_type

  // Interior tile: all 9 window cells are in-bounds. Expected = 9 * 512.
  %interior = tensor.extract_slice %result[0, 1, 1, 0] [1, 6, 6, 512] [1, 1, 1, 1]
      : !output_type to !interior_type
  check.expect_almost_eq_const(
    %interior, dense<4608.0> : !interior_type) : !interior_type

  // Top-left corner: 4 window cells in-bounds, 5 OOB (must read 0).
  // Expected = 4 * 512 = 2048. Any sentinel bug shifts this away from 2048.
  %corner = tensor.extract_slice %result[0, 0, 0, 0] [1, 1, 1, 512] [1, 1, 1, 1]
      : !output_type to !corner_type
  check.expect_almost_eq_const(
    %corner, dense<2048.0> : !corner_type) : !corner_type

  return
}
