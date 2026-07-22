// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-materialize-device-encoding))" --split-input-file %s | FileCheck %s

// Map invariant materialization (all input formats must produce the same canonical generics):
//   input:   (d0..d8) -> (d0, d4, d2 + d5, d3 + d6, d8)
//   filter:  (d0..d8) -> (d1, d4, d5, d6, d8, d7)
//   output:  (d0..d8) -> (d0, d1, d2, d3, d7)

// conv_2d_nhwc_hwcf materialization.

// Pack input from [N, H, W, C] -> [N, IC/c0, H, W, c0].

#nhwc_hwcf_map_in  = affine_map<(n, oh, ow, oc, fh, fw, ic) -> (n, oh + fh, ow + fw, ic)>
#nhwc_hwcf_map_f   = affine_map<(n, oh, ow, oc, fh, fw, ic) -> (fh, fw, ic, oc)>
#nhwc_hwcf_map_out = affine_map<(n, oh, ow, oc, fh, fw, ic) -> (n, oh, ow, oc)>

// CHECK: #[[$M_IN:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d4, d2 + d5, d3 + d6, d8)>
// CHECK: #[[$M_FLT:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d4, d5, d6, d8, d7)>
// CHECK: #[[$M_OUT:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d7)>

#encoding_nhwc_hwcf_input = #iree_encoding.encoding<operand_index = 0, op_type = conv,
  element_types = [f32, f32, f32],
  user_indexing_maps = [#nhwc_hwcf_map_in, #nhwc_hwcf_map_f, #nhwc_hwcf_map_out],
  iteration_sizes = [1, 14, 14, 8, 3, 3, 4]>

func.func @conv_input_pack(%arg0: tensor<1x16x16x4xf32>)
    -> tensor<1x16x16x4xf32, #encoding_nhwc_hwcf_input>
    attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz",
      {target_triple="aarch64-xyz-xyz", cpu_features="+neon",
       iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %0 = iree_encoding.set_encoding %arg0
       : tensor<1x16x16x4xf32> -> tensor<1x16x16x4xf32, #encoding_nhwc_hwcf_input>
  return %0 : tensor<1x16x16x4xf32, #encoding_nhwc_hwcf_input>
}
// CHECK-LABEL: func.func @conv_input_pack
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]: tensor<1x16x16x4xf32>
// CHECK:         %[[PACK:.+]] = linalg.pack %[[ARG0]]
// CHECK-SAME:      outer_dims_perm = [0, 3, 1, 2]
// CHECK-SAME:      inner_dims_pos = [3]
// CHECK-SAME:      inner_tiles = [8]
// CHECK-SAME:      : tensor<1x16x16x4xf32> -> tensor<1x1x16x16x8xf32>
// CHECK:         return %[[PACK]]


// Pack filter from [KH, KW, C, F] -> [OC/k0, IC/c0, KH, KW, c0, k0].

#encoding_nhwc_hwcf_filter = #iree_encoding.encoding<operand_index = 1, op_type = conv,
  element_types = [f32, f32, f32],
  user_indexing_maps = [#nhwc_hwcf_map_in, #nhwc_hwcf_map_f, #nhwc_hwcf_map_out],
  iteration_sizes = [1, 14, 14, 8, 3, 3, 4]>

func.func @conv_filter_pack(%arg0: tensor<3x3x4x8xf32>)
    -> tensor<3x3x4x8xf32, #encoding_nhwc_hwcf_filter>
    attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz",
      {target_triple="aarch64-xyz-xyz", cpu_features="+neon",
       iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %0 = iree_encoding.set_encoding %arg0
       : tensor<3x3x4x8xf32> -> tensor<3x3x4x8xf32, #encoding_nhwc_hwcf_filter>
  return %0 : tensor<3x3x4x8xf32, #encoding_nhwc_hwcf_filter>
}
// CHECK-LABEL: func.func @conv_filter_pack
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]: tensor<3x3x4x8xf32>
// CHECK:         %[[PACK:.+]] = linalg.pack %[[ARG0]]
// CHECK-SAME:      outer_dims_perm = [3, 2, 0, 1]
// CHECK-SAME:      inner_dims_pos = [2, 3]
// CHECK-SAME:      inner_tiles = [8, 8]
// CHECK-SAME:      : tensor<3x3x4x8xf32> -> tensor<1x1x3x3x8x8xf32>
// CHECK:         return %[[PACK]]


// Unpack output from [N, OC/k0, OH, OW, k0] -> [N, OH, OW, OC].

#encoding_nhwc_hwcf_output = #iree_encoding.encoding<operand_index = 2, op_type = conv,
  element_types = [f32, f32, f32],
  user_indexing_maps = [#nhwc_hwcf_map_in, #nhwc_hwcf_map_f, #nhwc_hwcf_map_out],
  iteration_sizes = [1, 14, 14, 8, 3, 3, 4]>

func.func @conv_output_unset(%arg0: tensor<1x14x14x8xf32, #encoding_nhwc_hwcf_output>)
    -> tensor<1x14x14x8xf32>
    attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz",
      {target_triple="aarch64-xyz-xyz", cpu_features="+neon",
       iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %0 = iree_encoding.unset_encoding %arg0
       : tensor<1x14x14x8xf32, #encoding_nhwc_hwcf_output> -> tensor<1x14x14x8xf32>
  return %0 : tensor<1x14x14x8xf32>
}
// CHECK-LABEL: func.func @conv_output_unset
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]: tensor<1x1x14x14x8xf32>
// CHECK:         %[[UNPACK:.+]] = linalg.unpack %[[ARG0]]
// CHECK-SAME:      outer_dims_perm = [0, 3, 1, 2]
// CHECK-SAME:      inner_dims_pos = [3]
// CHECK-SAME:      inner_tiles = [8]
// CHECK-SAME:      : tensor<1x1x14x14x8xf32> -> tensor<1x14x14x8xf32>
// CHECK:         return %[[UNPACK]]


// Conv materialization.

func.func @conv2d_nhwc_hwcf_materialize(
    %input  : tensor<1x16x16x4xf32, #encoding_nhwc_hwcf_input>,
    %filter : tensor<3x3x4x8xf32, #encoding_nhwc_hwcf_filter>,
    %output : tensor<1x14x14x8xf32, #encoding_nhwc_hwcf_output>)
    -> tensor<1x14x14x8xf32, #encoding_nhwc_hwcf_output>
    attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz",
      {target_triple="aarch64-xyz-xyz", cpu_features="+neon",
       iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %0 = linalg.conv_2d_nhwc_hwcf
         {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
         ins(%input, %filter
           : tensor<1x16x16x4xf32, #encoding_nhwc_hwcf_input>,
             tensor<3x3x4x8xf32, #encoding_nhwc_hwcf_filter>)
         outs(%output : tensor<1x14x14x8xf32, #encoding_nhwc_hwcf_output>)
         -> tensor<1x14x14x8xf32, #encoding_nhwc_hwcf_output>
  return %0 : tensor<1x14x14x8xf32, #encoding_nhwc_hwcf_output>
}
// CHECK-LABEL: func.func @conv2d_nhwc_hwcf_materialize
// CHECK-SAME:    %[[INPUT:.+]]: tensor<1x1x16x16x8xf32>
// CHECK-SAME:    %[[FILTER:.+]]: tensor<1x1x3x3x8x8xf32>
// CHECK-SAME:    %[[OUTPUT:.+]]: tensor<1x1x14x14x8xf32>
//
// 9D data-tiled conv generic with three operands (input, filter, output):
// CHECK:         %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:      indexing_maps = [#[[$M_IN]], #[[$M_FLT]], #[[$M_OUT]]]
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel", "parallel",
// CHECK-SAME:                        "reduction", "reduction", "reduction",
// CHECK-SAME:                        "parallel", "reduction"]
// CHECK-SAME:      ins(%[[INPUT]], %[[FILTER]]
// CHECK-SAME:      outs(%[[OUTPUT]]
// CHECK:           %[[MUL:.+]] = arith.mulf
// CHECK:           %[[ADD:.+]] = arith.addf %[[MUL]]
// CHECK:           linalg.yield %[[ADD]]
// CHECK:         return %[[RESULT]]


// Full conv materialization.

func.func @conv2d_nhwc_hwcf_materialize_full(
    %input  : tensor<1x16x16x4xf32>,
    %filter : tensor<3x3x4x8xf32>,
    %output : tensor<1x14x14x8xf32>)
    -> tensor<1x14x14x8xf32>
    attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz",
      {target_triple="aarch64-xyz-xyz", cpu_features="+neon",
       iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %enc_in  = iree_encoding.set_encoding %input
      : tensor<1x16x16x4xf32> -> tensor<1x16x16x4xf32, #encoding_nhwc_hwcf_input>
  %enc_f   = iree_encoding.set_encoding %filter
      : tensor<3x3x4x8xf32> -> tensor<3x3x4x8xf32, #encoding_nhwc_hwcf_filter>
  %enc_out = iree_encoding.set_encoding %output
      : tensor<1x14x14x8xf32> -> tensor<1x14x14x8xf32, #encoding_nhwc_hwcf_output>
  %0 = linalg.conv_2d_nhwc_hwcf
         {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
         ins(%enc_in, %enc_f
           : tensor<1x16x16x4xf32, #encoding_nhwc_hwcf_input>,
             tensor<3x3x4x8xf32, #encoding_nhwc_hwcf_filter>)
         outs(%enc_out : tensor<1x14x14x8xf32, #encoding_nhwc_hwcf_output>)
         -> tensor<1x14x14x8xf32, #encoding_nhwc_hwcf_output>
  %1 = iree_encoding.unset_encoding %0
      : tensor<1x14x14x8xf32, #encoding_nhwc_hwcf_output> -> tensor<1x14x14x8xf32>
  return %1 : tensor<1x14x14x8xf32>
}
// CHECK-LABEL: func.func @conv2d_nhwc_hwcf_materialize_full
// CHECK-SAME:    %[[IN:[a-zA-Z0-9]+]]: tensor<1x16x16x4xf32>
// CHECK-SAME:    %[[F:[a-zA-Z0-9]+]]: tensor<3x3x4x8xf32>
// CHECK-SAME:    %[[OUT:[a-zA-Z0-9]+]]: tensor<1x14x14x8xf32>
//
// CHECK:         %[[PACK_IN:.+]] = linalg.pack %[[IN]]
// CHECK-SAME:      outer_dims_perm = [0, 3, 1, 2]
// CHECK-SAME:      inner_dims_pos = [3]
// CHECK-SAME:      inner_tiles = [8]
// CHECK-SAME:      : tensor<1x16x16x4xf32> -> tensor<1x1x16x16x8xf32>
//
// CHECK:         %[[PACK_F:.+]] = linalg.pack %[[F]]
// CHECK-SAME:      outer_dims_perm = [3, 2, 0, 1]
// CHECK-SAME:      inner_dims_pos = [2, 3]
// CHECK-SAME:      inner_tiles = [8, 8]
// CHECK-SAME:      : tensor<3x3x4x8xf32> -> tensor<1x1x3x3x8x8xf32>
//
// CHECK:         %[[PACK_OUT:.+]] = linalg.pack %[[OUT]]
// CHECK-SAME:      outer_dims_perm = [0, 3, 1, 2]
// CHECK-SAME:      inner_dims_pos = [3]
// CHECK-SAME:      inner_tiles = [8]
// CHECK-SAME:      : tensor<1x14x14x8xf32> -> tensor<1x1x14x14x8xf32>
//
// CHECK:         %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:      ins(%[[PACK_IN]], %[[PACK_F]]
// CHECK-SAME:      outs(%[[PACK_OUT]]
//
// CHECK:         %[[UNPACK:.+]] = linalg.unpack %[[RESULT]]
// CHECK-SAME:      outer_dims_perm = [0, 3, 1, 2]
// CHECK-SAME:      inner_dims_pos = [3]
// CHECK-SAME:      inner_tiles = [8]
// CHECK-SAME:      : tensor<1x1x14x14x8xf32> -> tensor<1x14x14x8xf32>


// NHWC-HWCF conv expressed as a linalg.generic carrying conv encodings.
// It must lower to the same pack / 9D generic / unpack as the named conv2d_nhwc_hwcf_materialize_full above.

func.func @conv2d_nhwc_hwcf_materialize_generic(
    %input  : tensor<1x16x16x4xf32>,
    %filter : tensor<3x3x4x8xf32>,
    %output : tensor<1x14x14x8xf32>)
    -> tensor<1x14x14x8xf32>
    attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz",
      {target_triple="aarch64-xyz-xyz", cpu_features="+neon",
       iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %enc_in  = iree_encoding.set_encoding %input
      : tensor<1x16x16x4xf32> -> tensor<1x16x16x4xf32, #encoding_nhwc_hwcf_input>
  %enc_f   = iree_encoding.set_encoding %filter
      : tensor<3x3x4x8xf32> -> tensor<3x3x4x8xf32, #encoding_nhwc_hwcf_filter>
  %enc_out = iree_encoding.set_encoding %output
      : tensor<1x14x14x8xf32> -> tensor<1x14x14x8xf32, #encoding_nhwc_hwcf_output>
  %0 = linalg.generic
         {indexing_maps = [#nhwc_hwcf_map_in, #nhwc_hwcf_map_f, #nhwc_hwcf_map_out],
          iterator_types = ["parallel", "parallel", "parallel", "parallel",
                            "reduction", "reduction", "reduction"]}
         ins(%enc_in, %enc_f
           : tensor<1x16x16x4xf32, #encoding_nhwc_hwcf_input>,
             tensor<3x3x4x8xf32, #encoding_nhwc_hwcf_filter>)
         outs(%enc_out : tensor<1x14x14x8xf32, #encoding_nhwc_hwcf_output>) {
  ^bb0(%in: f32, %f: f32, %acc: f32):
    %m = arith.mulf %in, %f : f32
    %a = arith.addf %acc, %m : f32
    linalg.yield %a : f32
  } -> tensor<1x14x14x8xf32, #encoding_nhwc_hwcf_output>
  %1 = iree_encoding.unset_encoding %0
      : tensor<1x14x14x8xf32, #encoding_nhwc_hwcf_output> -> tensor<1x14x14x8xf32>
  return %1 : tensor<1x14x14x8xf32>
}
// CHECK-LABEL: func.func @conv2d_nhwc_hwcf_materialize_generic
// CHECK-SAME:    %[[IN:[a-zA-Z0-9]+]]: tensor<1x16x16x4xf32>
// CHECK-SAME:    %[[F:[a-zA-Z0-9]+]]: tensor<3x3x4x8xf32>
// CHECK-SAME:    %[[OUT:[a-zA-Z0-9]+]]: tensor<1x14x14x8xf32>
//
// CHECK:         %[[PACK_IN:.+]] = linalg.pack %[[IN]]
// CHECK-SAME:      outer_dims_perm = [0, 3, 1, 2]
// CHECK-SAME:      inner_dims_pos = [3]
// CHECK-SAME:      inner_tiles = [8]
// CHECK-SAME:      : tensor<1x16x16x4xf32> -> tensor<1x1x16x16x8xf32>
//
// CHECK:         %[[PACK_F:.+]] = linalg.pack %[[F]]
// CHECK-SAME:      outer_dims_perm = [3, 2, 0, 1]
// CHECK-SAME:      inner_dims_pos = [2, 3]
// CHECK-SAME:      inner_tiles = [8, 8]
// CHECK-SAME:      : tensor<3x3x4x8xf32> -> tensor<1x1x3x3x8x8xf32>
//
// CHECK:         %[[PACK_OUT:.+]] = linalg.pack %[[OUT]]
// CHECK-SAME:      outer_dims_perm = [0, 3, 1, 2]
// CHECK-SAME:      inner_dims_pos = [3]
// CHECK-SAME:      inner_tiles = [8]
// CHECK-SAME:      : tensor<1x14x14x8xf32> -> tensor<1x1x14x14x8xf32>
//
// CHECK:         %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:      ins(%[[PACK_IN]], %[[PACK_F]]
// CHECK-SAME:      outs(%[[PACK_OUT]]
//
// CHECK:         %[[UNPACK:.+]] = linalg.unpack %[[RESULT]]
// CHECK-SAME:      outer_dims_perm = [0, 3, 1, 2]
// CHECK-SAME:      inner_dims_pos = [3]
// CHECK-SAME:      inner_tiles = [8]
// CHECK-SAME:      : tensor<1x1x14x14x8xf32> -> tensor<1x14x14x8xf32>


// -----

// conv_2d_nchw_fchw materialization.

// Pack input from [N, C, H, W] -> [N, IC/c0, H, W, c0].

#nchw_fchw_map_in  = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d2 + d5, d3 + d6)>
#nchw_fchw_map_f   = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>
#nchw_fchw_map_out = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>

// CHECK: #[[$M_IN:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d4, d2 + d5, d3 + d6, d8)>
// CHECK: #[[$M_FLT:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d4, d5, d6, d8, d7)>
// CHECK: #[[$M_OUT:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d7)>

#encoding_nhwc_fchw_input = #iree_encoding.encoding<operand_index = 0, op_type = conv,
  element_types = [f32, f32, f32],
  user_indexing_maps = [#nchw_fchw_map_in, #nchw_fchw_map_f, #nchw_fchw_map_out],
  iteration_sizes = [1, 8, 14, 14, 4, 3, 3]>

func.func @conv_nchw_input_pack(%arg0: tensor<1x4x16x16xf32>)
    -> tensor<1x4x16x16xf32, #encoding_nhwc_fchw_input>
    attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz",
      {target_triple="aarch64-xyz-xyz", cpu_features="+neon",
       iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %0 = iree_encoding.set_encoding %arg0
       : tensor<1x4x16x16xf32> -> tensor<1x4x16x16xf32, #encoding_nhwc_fchw_input>
  return %0 : tensor<1x4x16x16xf32, #encoding_nhwc_fchw_input>
}
// CHECK-LABEL: func.func @conv_nchw_input_pack
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]: tensor<1x4x16x16xf32>
// CHECK:         %[[PACK:.+]] = linalg.pack %[[ARG0]]
// CHECK-SAME:      outer_dims_perm = [0, 1, 2, 3]
// CHECK-SAME:      inner_dims_pos = [1]
// CHECK-SAME:      inner_tiles = [8]
// CHECK-SAME:      : tensor<1x4x16x16xf32> -> tensor<1x1x16x16x8xf32>
// CHECK:         return %[[PACK]]


// Pack filter from [F, C, KH, KW] -> [OC/k0, IC/c0, KH, KW, c0, k0].

#encoding_nhwc_fchw_filter = #iree_encoding.encoding<operand_index = 1, op_type = conv,
  element_types = [f32, f32, f32],
  user_indexing_maps = [#nchw_fchw_map_in, #nchw_fchw_map_f, #nchw_fchw_map_out],
  iteration_sizes = [1, 8, 14, 14, 4, 3, 3]>

func.func @conv_fchw_filter_pack(%arg0: tensor<8x4x3x3xf32>)
    -> tensor<8x4x3x3xf32, #encoding_nhwc_fchw_filter>
    attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz",
      {target_triple="aarch64-xyz-xyz", cpu_features="+neon",
       iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %0 = iree_encoding.set_encoding %arg0
       : tensor<8x4x3x3xf32> -> tensor<8x4x3x3xf32, #encoding_nhwc_fchw_filter>
  return %0 : tensor<8x4x3x3xf32, #encoding_nhwc_fchw_filter>
}
// CHECK-LABEL: func.func @conv_fchw_filter_pack
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]: tensor<8x4x3x3xf32>
// CHECK:         %[[PACK:.+]] = linalg.pack %[[ARG0]]
// CHECK-SAME:      outer_dims_perm = [0, 1, 2, 3]
// CHECK-SAME:      inner_dims_pos = [1, 0]
// CHECK-SAME:      inner_tiles = [8, 8]
// CHECK-SAME:      : tensor<8x4x3x3xf32> -> tensor<1x1x3x3x8x8xf32>
// CHECK:         return %[[PACK]]


// Unpack output from [N, OC/k0, OH, OW, k0] -> [N, OC, OH, OW].

#encoding_nhwc_fchw_output = #iree_encoding.encoding<operand_index = 2, op_type = conv,
  element_types = [f32, f32, f32],
  user_indexing_maps = [#nchw_fchw_map_in, #nchw_fchw_map_f, #nchw_fchw_map_out],
  iteration_sizes = [1, 8, 14, 14, 4, 3, 3]>

func.func @conv_nchw_output_unset(%arg0: tensor<1x8x14x14xf32, #encoding_nhwc_fchw_output>)
    -> tensor<1x8x14x14xf32>
    attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz",
      {target_triple="aarch64-xyz-xyz", cpu_features="+neon",
       iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %0 = iree_encoding.unset_encoding %arg0
       : tensor<1x8x14x14xf32, #encoding_nhwc_fchw_output> -> tensor<1x8x14x14xf32>
  return %0 : tensor<1x8x14x14xf32>
}
// CHECK-LABEL: func.func @conv_nchw_output_unset
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]: tensor<1x1x14x14x8xf32>
// CHECK:         %[[UNPACK:.+]] = linalg.unpack %[[ARG0]]
// CHECK-SAME:      outer_dims_perm = [0, 1, 2, 3]
// CHECK-SAME:      inner_dims_pos = [1]
// CHECK-SAME:      inner_tiles = [8]
// CHECK-SAME:      : tensor<1x1x14x14x8xf32> -> tensor<1x8x14x14xf32>
// CHECK:         return %[[UNPACK]]


// Conv materialization.

func.func @conv2d_nchw_fchw_materialize(
    %input  : tensor<1x4x16x16xf32, #encoding_nhwc_fchw_input>,
    %filter : tensor<8x4x3x3xf32, #encoding_nhwc_fchw_filter>,
    %output : tensor<1x8x14x14xf32, #encoding_nhwc_fchw_output>)
    -> tensor<1x8x14x14xf32, #encoding_nhwc_fchw_output>
    attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz",
      {target_triple="aarch64-xyz-xyz", cpu_features="+neon",
       iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %0 = linalg.conv_2d_nchw_fchw
         {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
         ins(%input, %filter
           : tensor<1x4x16x16xf32, #encoding_nhwc_fchw_input>,
             tensor<8x4x3x3xf32, #encoding_nhwc_fchw_filter>)
         outs(%output : tensor<1x8x14x14xf32, #encoding_nhwc_fchw_output>)
         -> tensor<1x8x14x14xf32, #encoding_nhwc_fchw_output>
  return %0 : tensor<1x8x14x14xf32, #encoding_nhwc_fchw_output>
}
// CHECK-LABEL: func.func @conv2d_nchw_fchw_materialize
// CHECK-SAME:    %[[INPUT:.+]]: tensor<1x1x16x16x8xf32>
// CHECK-SAME:    %[[FILTER:.+]]: tensor<1x1x3x3x8x8xf32>
// CHECK-SAME:    %[[OUTPUT:.+]]: tensor<1x1x14x14x8xf32>
//
// CHECK:         %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:      indexing_maps = [#[[$M_IN]], #[[$M_FLT]], #[[$M_OUT]]]
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel", "parallel",
// CHECK-SAME:                        "reduction", "reduction", "reduction",
// CHECK-SAME:                        "parallel", "reduction"]
// CHECK-SAME:      ins(%[[INPUT]], %[[FILTER]]
// CHECK-SAME:      outs(%[[OUTPUT]]
// CHECK:           %[[MUL:.+]] = arith.mulf
// CHECK:           %[[ADD:.+]] = arith.addf %[[MUL]]
// CHECK:           linalg.yield %[[ADD]]
// CHECK:         return %[[RESULT]]


// Full conv materialization.

func.func @conv2d_nchw_fchw_materialize_full(
    %input  : tensor<1x4x16x16xf32>,
    %filter : tensor<8x4x3x3xf32>,
    %output : tensor<1x8x14x14xf32>)
    -> tensor<1x8x14x14xf32>
    attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz",
      {target_triple="aarch64-xyz-xyz", cpu_features="+neon",
       iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %enc_in  = iree_encoding.set_encoding %input
      : tensor<1x4x16x16xf32> -> tensor<1x4x16x16xf32, #encoding_nhwc_fchw_input>
  %enc_f   = iree_encoding.set_encoding %filter
      : tensor<8x4x3x3xf32> -> tensor<8x4x3x3xf32, #encoding_nhwc_fchw_filter>
  %enc_out = iree_encoding.set_encoding %output
      : tensor<1x8x14x14xf32> -> tensor<1x8x14x14xf32, #encoding_nhwc_fchw_output>
  %0 = linalg.conv_2d_nchw_fchw
         {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
         ins(%enc_in, %enc_f
           : tensor<1x4x16x16xf32, #encoding_nhwc_fchw_input>,
             tensor<8x4x3x3xf32, #encoding_nhwc_fchw_filter>)
         outs(%enc_out : tensor<1x8x14x14xf32, #encoding_nhwc_fchw_output>)
         -> tensor<1x8x14x14xf32, #encoding_nhwc_fchw_output>
  %1 = iree_encoding.unset_encoding %0
      : tensor<1x8x14x14xf32, #encoding_nhwc_fchw_output> -> tensor<1x8x14x14xf32>
  return %1 : tensor<1x8x14x14xf32>
}
// CHECK-LABEL: func.func @conv2d_nchw_fchw_materialize_full
// CHECK-SAME:    %[[IN:[a-zA-Z0-9]+]]: tensor<1x4x16x16xf32>
// CHECK-SAME:    %[[F:[a-zA-Z0-9]+]]: tensor<8x4x3x3xf32>
// CHECK-SAME:    %[[OUT:[a-zA-Z0-9]+]]: tensor<1x8x14x14xf32>
//
// CHECK:         %[[PACK_IN:.+]] = linalg.pack %[[IN]]
// CHECK-SAME:      outer_dims_perm = [0, 1, 2, 3]
// CHECK-SAME:      inner_dims_pos = [1]
// CHECK-SAME:      inner_tiles = [8]
// CHECK-SAME:      : tensor<1x4x16x16xf32> -> tensor<1x1x16x16x8xf32>
//
// CHECK:         %[[PACK_F:.+]] = linalg.pack %[[F]]
// CHECK-SAME:      outer_dims_perm = [0, 1, 2, 3]
// CHECK-SAME:      inner_dims_pos = [1, 0]
// CHECK-SAME:      inner_tiles = [8, 8]
// CHECK-SAME:      : tensor<8x4x3x3xf32> -> tensor<1x1x3x3x8x8xf32>
//
// CHECK:         %[[PACK_OUT:.+]] = linalg.pack %[[OUT]]
// CHECK-SAME:      outer_dims_perm = [0, 1, 2, 3]
// CHECK-SAME:      inner_dims_pos = [1]
// CHECK-SAME:      inner_tiles = [8]
// CHECK-SAME:      : tensor<1x8x14x14xf32> -> tensor<1x1x14x14x8xf32>
//
// CHECK:         %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:      ins(%[[PACK_IN]], %[[PACK_F]]
// CHECK-SAME:      outs(%[[PACK_OUT]]
//
// CHECK:         %[[UNPACK:.+]] = linalg.unpack %[[RESULT]]
// CHECK-SAME:      outer_dims_perm = [0, 1, 2, 3]
// CHECK-SAME:      inner_dims_pos = [1]
// CHECK-SAME:      inner_tiles = [8]
// CHECK-SAME:      : tensor<1x1x14x14x8xf32> -> tensor<1x8x14x14xf32>

// -----

// conv_2d_nhwc_fhwc materialization.

// Pack filter from [F, KH, KW, C] -> [OC/k0, IC/c0, KH, KW, c0, k0].

#nhwc_fhwc_map_in  = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
#nhwc_fhwc_map_f   = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>
#nhwc_fhwc_map_out = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>

#encoding_nhwc_fhwc_filter = #iree_encoding.encoding<operand_index = 1, op_type = conv,
  element_types = [f32, f32, f32],
  user_indexing_maps = [#nhwc_fhwc_map_in, #nhwc_fhwc_map_f, #nhwc_fhwc_map_out],
  iteration_sizes = [1, 14, 14, 8, 3, 3, 4]>

func.func @conv_fhwc_filter_pack(%arg0: tensor<8x3x3x4xf32>)
    -> tensor<8x3x3x4xf32, #encoding_nhwc_fhwc_filter>
    attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz",
      {target_triple="aarch64-xyz-xyz", cpu_features="+neon",
       iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %0 = iree_encoding.set_encoding %arg0
       : tensor<8x3x3x4xf32> -> tensor<8x3x3x4xf32, #encoding_nhwc_fhwc_filter>
  return %0 : tensor<8x3x3x4xf32, #encoding_nhwc_fhwc_filter>
}
// CHECK-LABEL: func.func @conv_fhwc_filter_pack
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]: tensor<8x3x3x4xf32>
// CHECK:         %[[PACK:.+]] = linalg.pack %[[ARG0]]
// CHECK-SAME:      outer_dims_perm = [0, 3, 1, 2]
// CHECK-SAME:      inner_dims_pos = [3, 0]
// CHECK-SAME:      inner_tiles = [8, 8]
// CHECK-SAME:      : tensor<8x3x3x4xf32> -> tensor<1x1x3x3x8x8xf32>
// CHECK:         return %[[PACK]]

// -----

// Unit batch-dim folded conv: N=1 stripped upstream, so operands pack to rank-4
// and the lowering restores the batch via expand/collapse around the 9-D generic.

#bl_map_in  = affine_map<(oh, ow, oc, fh, fw, ic) -> (oh + fh, ow + fw, ic)>
#bl_map_f   = affine_map<(oh, ow, oc, fh, fw, ic) -> (fh, fw, ic, oc)>
#bl_map_out = affine_map<(oh, ow, oc, fh, fw, ic) -> (oh, ow, oc)>

#encoding_bl_input = #iree_encoding.encoding<operand_index = 0, op_type = conv,
  element_types = [f32, f32, f32],
  user_indexing_maps = [#bl_map_in, #bl_map_f, #bl_map_out],
  iteration_sizes = [14, 14, 8, 3, 3, 4]>
#encoding_bl_filter = #iree_encoding.encoding<operand_index = 1, op_type = conv,
  element_types = [f32, f32, f32],
  user_indexing_maps = [#bl_map_in, #bl_map_f, #bl_map_out],
  iteration_sizes = [14, 14, 8, 3, 3, 4]>
#encoding_bl_output = #iree_encoding.encoding<operand_index = 2, op_type = conv,
  element_types = [f32, f32, f32],
  user_indexing_maps = [#bl_map_in, #bl_map_f, #bl_map_out],
  iteration_sizes = [14, 14, 8, 3, 3, 4]>

func.func @conv2d_batchless_materialize(
    %input  : tensor<16x16x4xf32, #encoding_bl_input>,
    %filter : tensor<3x3x4x8xf32, #encoding_bl_filter>,
    %output : tensor<14x14x8xf32, #encoding_bl_output>)
    -> tensor<14x14x8xf32, #encoding_bl_output>
    attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz",
      {target_triple="aarch64-xyz-xyz", cpu_features="+neon",
       iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %0 = linalg.generic
         {indexing_maps = [#bl_map_in, #bl_map_f, #bl_map_out],
          iterator_types = ["parallel", "parallel", "parallel",
                            "reduction", "reduction", "reduction"]}
         ins(%input, %filter
           : tensor<16x16x4xf32, #encoding_bl_input>,
             tensor<3x3x4x8xf32, #encoding_bl_filter>)
         outs(%output : tensor<14x14x8xf32, #encoding_bl_output>) {
  ^bb0(%in: f32, %f: f32, %acc: f32):
    %m = arith.mulf %in, %f : f32
    %a = arith.addf %acc, %m : f32
    linalg.yield %a : f32
  } -> tensor<14x14x8xf32, #encoding_bl_output>
  return %0 : tensor<14x14x8xf32, #encoding_bl_output>
}
// CHECK-LABEL: func.func @conv2d_batchless_materialize
// CHECK-SAME:    %[[INPUT:.+]]: tensor<1x16x16x8xf32>
// CHECK-SAME:    %[[FILTER:.+]]: tensor<1x1x3x3x8x8xf32>
// CHECK-SAME:    %[[OUTPUT:.+]]: tensor<1x14x14x8xf32>
// CHECK:         %[[EXP_IN:.+]] = tensor.expand_shape %[[INPUT]] {{\[}}[0, 1], [2], [3], [4]{{\]}}
// CHECK-SAME:      : tensor<1x16x16x8xf32> into tensor<1x1x16x16x8xf32>
// CHECK:         %[[EXP_OUT:.+]] = tensor.expand_shape %[[OUTPUT]] {{\[}}[0, 1], [2], [3], [4]{{\]}}
// CHECK-SAME:      : tensor<1x14x14x8xf32> into tensor<1x1x14x14x8xf32>
// CHECK:         %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:      indexing_maps = [#[[$M_IN]], #[[$M_FLT]], #[[$M_OUT]]]
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel", "parallel",
// CHECK-SAME:                        "reduction", "reduction", "reduction",
// CHECK-SAME:                        "parallel", "reduction"]
// CHECK-SAME:      ins(%[[EXP_IN]], %[[FILTER]]
// CHECK-SAME:      outs(%[[EXP_OUT]]
// CHECK:           %[[MUL:.+]] = arith.mulf
// CHECK:           %[[ADD:.+]] = arith.addf %[[MUL]]
// CHECK:           linalg.yield %[[ADD]]
// CHECK:         %[[COLLAPSE:.+]] = tensor.collapse_shape %[[RESULT]] {{\[}}[0, 1], [2], [3], [4]{{\]}}
// CHECK-SAME:      : tensor<1x1x14x14x8xf32> into tensor<1x14x14x8xf32>
// CHECK:         return %[[COLLAPSE]]
