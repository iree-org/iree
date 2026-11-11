// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-global-opt-rebase-quantized-operands)" %s | FileCheck %s

// Targets an arm64 CPU with dotprod. Its data-tiling tables have tiles for
// signed i8 x i8 and i8 x i4 contractions only, so an unsigned operand gets
// no tiled layout until it is rebased.

#arm64 = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+dotprod", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, target_triple = "aarch64-unknown-unknown-eabi-elf"}>
#maps_q = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
           affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>]
#maps_sym = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
             affine_map<(d0, d1) -> (d0, d1)>]
#matmul_nt = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>,
              affine_map<(m, n, k) -> (m, n)>]

// A u8 activation times a constant s8 weight. The activation moves to s8: its
// zero point and bounds shift by -128 at the quantize and at the dequantize.
module attributes {stream.affinity.default = #hal.device.affinity<@device>} {
  util.global private @device = #hal.device.target<"local", [#arm64]> : !hal.device
  util.func public @u8_activation(%x: tensor<16x64xf32>, %sx: f32, %zx: i8, %sw: f32) -> tensor<16x32xf32> {
    %w = arith.constant dense<3> : tensor<32x64xi8>
    %qi = tensor.empty() : tensor<16x64xi8>
    %xq = iree_linalg_ext.quantize_affine <indexing_maps = #maps_q, quant_min = 0, quant_max = 255, storage_unsigned, zp_unsigned>
        ins(%x, %sx, %zx : tensor<16x64xf32>, f32, i8) outs(%qi : tensor<16x64xi8>) -> tensor<16x64xi8>
    %xi = tensor.empty() : tensor<16x64xf32>
    %xd = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_q, input_unsigned, zp_unsigned>
        ins(%xq, %sx, %zx : tensor<16x64xi8>, f32, i8) outs(%xi : tensor<16x64xf32>) -> tensor<16x64xf32>
    %wi = tensor.empty() : tensor<32x64xf32>
    %wd = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_sym>
        ins(%w, %sw : tensor<32x64xi8>, f32) outs(%wi : tensor<32x64xf32>) -> tensor<32x64xf32>
    %zero = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<16x32xf32>
    %f = linalg.fill ins(%zero : f32) outs(%e : tensor<16x32xf32>) -> tensor<16x32xf32>
    %mm = linalg.matmul indexing_maps = #matmul_nt
        ins(%xd, %wd : tensor<16x64xf32>, tensor<32x64xf32>) outs(%f : tensor<16x32xf32>) -> tensor<16x32xf32>
    util.return %mm : tensor<16x32xf32>
  }
}

// CHECK-LABEL: util.func public @u8_activation(
//  CHECK-SAME:   %{{.+}}: tensor<16x64xf32>, %{{.+}}: f32, %[[ZX:[a-zA-Z0-9_]+]]: i8
//       CHECK:   %[[SHIFT:.+]] = arith.constant -128 : i8
//       CHECK:   %[[ZP:.+]] = arith.addi %[[ZX]], %[[SHIFT]] : i8
//       CHECK:   %[[Q:.+]] = iree_linalg_ext.quantize_affine
//  CHECK-SAME:     quant_min = -128, quant_max = 127>
//  CHECK-SAME:     ins(%{{.+}}, %{{.+}}, %[[ZP]] :
//       CHECK:   iree_linalg_ext.dequantize_affine
//   CHECK-NOT:     unsigned
//  CHECK-SAME:     ins(%[[Q]], %{{.+}}, %[[ZP]] :

// -----

#arm64 = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+dotprod", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, target_triple = "aarch64-unknown-unknown-eabi-elf"}>
#maps_q = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
           affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>]
#maps_sym = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
             affine_map<(d0, d1) -> (d0, d1)>]
#matmul_nt = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>,
              affine_map<(m, n, k) -> (m, n)>]

// A u8 zero point of 128 moves to 0, and the zero point is dropped.
module attributes {stream.affinity.default = #hal.device.affinity<@device>} {
  util.global private @device = #hal.device.target<"local", [#arm64]> : !hal.device
  util.func public @zero_point_becomes_zero(%x: tensor<16x64xf32>, %sx: f32, %sw: f32) -> tensor<16x32xf32> {
    %zx = arith.constant 128 : i8
    %w = arith.constant dense<3> : tensor<32x64xi8>
    %qi = tensor.empty() : tensor<16x64xi8>
    %xq = iree_linalg_ext.quantize_affine <indexing_maps = #maps_q, quant_min = 0, quant_max = 255, storage_unsigned, zp_unsigned>
        ins(%x, %sx, %zx : tensor<16x64xf32>, f32, i8) outs(%qi : tensor<16x64xi8>) -> tensor<16x64xi8>
    %xi = tensor.empty() : tensor<16x64xf32>
    %xd = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_q, input_unsigned, zp_unsigned>
        ins(%xq, %sx, %zx : tensor<16x64xi8>, f32, i8) outs(%xi : tensor<16x64xf32>) -> tensor<16x64xf32>
    %wi = tensor.empty() : tensor<32x64xf32>
    %wd = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_sym>
        ins(%w, %sw : tensor<32x64xi8>, f32) outs(%wi : tensor<32x64xf32>) -> tensor<32x64xf32>
    %zero = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<16x32xf32>
    %f = linalg.fill ins(%zero : f32) outs(%e : tensor<16x32xf32>) -> tensor<16x32xf32>
    %mm = linalg.matmul indexing_maps = #matmul_nt
        ins(%xd, %wd : tensor<16x64xf32>, tensor<32x64xf32>) outs(%f : tensor<16x32xf32>) -> tensor<16x32xf32>
    util.return %mm : tensor<16x32xf32>
  }
}

// CHECK-LABEL: util.func public @zero_point_becomes_zero(
//       CHECK:   iree_linalg_ext.quantize_affine
//  CHECK-SAME:     ins(%{{.+}}, %{{.+}} : tensor<16x64xf32>, f32)
//       CHECK:   iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     ins(%{{.+}}, %{{.+}} : tensor<16x64xi8>, f32)

// -----

#arm64 = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+dotprod", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, target_triple = "aarch64-unknown-unknown-eabi-elf"}>
#maps_sym = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
             affine_map<(d0, d1) -> (d0, d1)>]
#matmul_nt = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>,
              affine_map<(m, n, k) -> (m, n)>]

// A symmetric u8 activation times a constant weight gains a zero point of
// -128.
module attributes {stream.affinity.default = #hal.device.affinity<@device>} {
  util.global private @device = #hal.device.target<"local", [#arm64]> : !hal.device
  util.func public @symmetric_activation_gains_zero_point(%x: tensor<16x64xf32>, %sx: f32, %sw: f32) -> tensor<16x32xf32> {
    %w = arith.constant dense<3> : tensor<32x64xi8>
    %qi = tensor.empty() : tensor<16x64xi8>
    %xq = iree_linalg_ext.quantize_affine <indexing_maps = #maps_sym, quant_min = 0, quant_max = 255, storage_unsigned>
        ins(%x, %sx : tensor<16x64xf32>, f32) outs(%qi : tensor<16x64xi8>) -> tensor<16x64xi8>
    %xi = tensor.empty() : tensor<16x64xf32>
    %xd = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_sym, input_unsigned>
        ins(%xq, %sx : tensor<16x64xi8>, f32) outs(%xi : tensor<16x64xf32>) -> tensor<16x64xf32>
    %wi = tensor.empty() : tensor<32x64xf32>
    %wd = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_sym>
        ins(%w, %sw : tensor<32x64xi8>, f32) outs(%wi : tensor<32x64xf32>) -> tensor<32x64xf32>
    %zero = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<16x32xf32>
    %f = linalg.fill ins(%zero : f32) outs(%e : tensor<16x32xf32>) -> tensor<16x32xf32>
    %mm = linalg.matmul indexing_maps = #matmul_nt
        ins(%xd, %wd : tensor<16x64xf32>, tensor<32x64xf32>) outs(%f : tensor<16x32xf32>) -> tensor<16x32xf32>
    util.return %mm : tensor<16x32xf32>
  }
}

// CHECK-LABEL: util.func public @symmetric_activation_gains_zero_point(
//       CHECK:   %[[ZP:.+]] = arith.constant -128 : i8
//       CHECK:   iree_linalg_ext.quantize_affine
//  CHECK-SAME:     ins(%{{.+}}, %{{.+}}, %[[ZP]] : tensor<16x64xf32>, f32, i8)
//       CHECK:   iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     ins(%{{.+}}, %{{.+}}, %{{.+}} : tensor<16x64xi8>, f32, i8)

// -----

#arm64 = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+dotprod", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, target_triple = "aarch64-unknown-unknown-eabi-elf"}>

// Blockwise W4A8 as torchao exports it: a u4 weight in u8 storage and its block
// zero points, both in globals, times a symmetric s8 activation. The weight
// moves by -8 to [-8, 7] with a shift of the loaded global, and its zero points
// become a constant.
module attributes {stream.affinity.default = #hal.device.affinity<@device>} {
  util.global private @device = #hal.device.target<"local", [#arm64]> : !hal.device
  util.global private @weight = dense<9> : tensor<8x2x32xi8>
  util.global private @zw = dense<[[8, 7], [8, 9], [8, 8], [1, 15], [8, 8], [8, 8], [8, 8], [8, 8]]> : tensor<8x2xi8>
  util.func public @u4_weight_blockwise(%x: tensor<16x2x32xf32>, %sx: tensor<16x2xf32>, %sw: tensor<8x2xf32>) -> tensor<16x8xf32> {
    %w = util.global.load @weight : tensor<8x2x32xi8>
    %zw = util.global.load @zw : tensor<8x2xi8>
    %qi = tensor.empty() : tensor<16x2x32xi8>
    %xq = iree_linalg_ext.quantize_affine
        <indexing_maps = [affine_map<(m, g, l) -> (m, g, l)>, affine_map<(m, g, l) -> (m, g)>,
                          affine_map<(m, g, l) -> (m, g, l)>],
         quant_min = -128, quant_max = 127>
        ins(%x, %sx : tensor<16x2x32xf32>, tensor<16x2xf32>) outs(%qi : tensor<16x2x32xi8>) -> tensor<16x2x32xi8>
    %xi = tensor.empty() : tensor<16x2x32xf32>
    %xd = iree_linalg_ext.dequantize_affine
        <indexing_maps = [affine_map<(m, g, l) -> (m, g, l)>, affine_map<(m, g, l) -> (m, g)>,
                          affine_map<(m, g, l) -> (m, g, l)>]>
        ins(%xq, %sx : tensor<16x2x32xi8>, tensor<16x2xf32>) outs(%xi : tensor<16x2x32xf32>) -> tensor<16x2x32xf32>
    %wi = tensor.empty() : tensor<8x2x32xf32>
    %wd = iree_linalg_ext.dequantize_affine
        <indexing_maps = [affine_map<(n, g, l) -> (n, g, l)>, affine_map<(n, g, l) -> (n, g)>,
                          affine_map<(n, g, l) -> (n, g)>, affine_map<(n, g, l) -> (n, g, l)>],
         quant_min = 0, quant_max = 15, input_unsigned, zp_unsigned>
        ins(%w, %sw, %zw : tensor<8x2x32xi8>, tensor<8x2xf32>, tensor<8x2xi8>) outs(%wi : tensor<8x2x32xf32>) -> tensor<8x2x32xf32>
    %zero = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<16x8xf32>
    %f = linalg.fill ins(%zero : f32) outs(%e : tensor<16x8xf32>) -> tensor<16x8xf32>
    %mm = linalg.contract
        indexing_maps = [affine_map<(m, n, g, l) -> (m, g, l)>, affine_map<(m, n, g, l) -> (n, g, l)>,
                         affine_map<(m, n, g, l) -> (m, n)>]
        ins(%xd, %wd : tensor<16x2x32xf32>, tensor<8x2x32xf32>) outs(%f : tensor<16x8xf32>) -> tensor<16x8xf32>
    util.return %mm : tensor<16x8xf32>
  }
}

// CHECK-LABEL: util.func public @u4_weight_blockwise(
//       CHECK:   %[[W:.+]] = util.global.load @weight
//       CHECK:   %[[SHIFT:.+]] = arith.constant -8 : i8
//       CHECK:   %[[WS:.+]] = linalg.generic
//  CHECK-SAME:     ins(%[[W]] : tensor<8x2x32xi8>)
//       CHECK:     arith.addi %{{.+}}, %[[SHIFT]] : i8
//       CHECK:   %[[ZW:.+]] = arith.constant dense<{{\[}}[0, -1], [0, 1], [0, 0], [-7, 7], [0, 0], [0, 0], [0, 0], [0, 0]]> : tensor<8x2xi8>
//       CHECK:   iree_linalg_ext.dequantize_affine <indexing_maps = [#{{.+}}, #{{.+}}, #{{.+}}, #{{.+}}], quant_min = -8, quant_max = 7>
//  CHECK-SAME:     ins(%[[WS]], %{{.+}}, %[[ZW]] :

// -----

#arm64 = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+dotprod", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, target_triple = "aarch64-unknown-unknown-eabi-elf"}>
#maps_q = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
           affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>]
#maps_sym = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
             affine_map<(d0, d1) -> (d0, d1)>]
#matmul_nt = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>,
              affine_map<(m, n, k) -> (m, n)>]

// A function the initializer calls stores over the initial value of the zero
// point. The weight and the loaded zero point move by -128 on the whole 8-bit
// grid.
module attributes {stream.affinity.default = #hal.device.affinity<@device>} {
  util.global private @device = #hal.device.target<"local", [#arm64]> : !hal.device
  util.global private @weight = dense<9> : tensor<32x64xi8>
  util.global private @zw = 8 : i8
  util.func private @store_zw() {
    %zw = arith.constant 3 : i8
    util.global.store %zw, @zw : i8
    util.return
  }
  util.initializer {
    util.call @store_zw() : () -> ()
    util.return
  }
  util.func public @zero_point_stored_by_initializer(%x: tensor<16x64xf32>, %sx: f32, %sw: f32) -> tensor<16x32xf32> {
    %w = util.global.load @weight : tensor<32x64xi8>
    %zw = util.global.load @zw : i8
    %qi = tensor.empty() : tensor<16x64xi8>
    %xq = iree_linalg_ext.quantize_affine <indexing_maps = #maps_sym, quant_min = -128, quant_max = 127>
        ins(%x, %sx : tensor<16x64xf32>, f32) outs(%qi : tensor<16x64xi8>) -> tensor<16x64xi8>
    %xi = tensor.empty() : tensor<16x64xf32>
    %xd = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_sym>
        ins(%xq, %sx : tensor<16x64xi8>, f32) outs(%xi : tensor<16x64xf32>) -> tensor<16x64xf32>
    %wi = tensor.empty() : tensor<32x64xf32>
    %wd = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_q, quant_min = 0, quant_max = 15, input_unsigned, zp_unsigned>
        ins(%w, %sw, %zw : tensor<32x64xi8>, f32, i8) outs(%wi : tensor<32x64xf32>) -> tensor<32x64xf32>
    %zero = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<16x32xf32>
    %f = linalg.fill ins(%zero : f32) outs(%e : tensor<16x32xf32>) -> tensor<16x32xf32>
    %mm = linalg.matmul indexing_maps = #matmul_nt
        ins(%xd, %wd : tensor<16x64xf32>, tensor<32x64xf32>) outs(%f : tensor<16x32xf32>) -> tensor<16x32xf32>
    util.return %mm : tensor<16x32xf32>
  }
}

// CHECK-LABEL: util.func public @zero_point_stored_by_initializer(
//       CHECK:   %[[ZW:.+]] = util.global.load @zw : i8
//       CHECK:   %[[SHIFT:.+]] = arith.constant -128 : i8
//       CHECK:   %[[ZWS:.+]] = arith.addi %[[ZW]], %[[SHIFT]] : i8
//       CHECK:   iree_linalg_ext.dequantize_affine <indexing_maps = [#{{.+}}, #{{.+}}, #{{.+}}, #{{.+}}], quant_min = -128, quant_max = -113>
//  CHECK-SAME:     ins(%{{.+}}, %{{.+}}, %[[ZWS]] :

// -----

#arm64 = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+dotprod", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, target_triple = "aarch64-unknown-unknown-eabi-elf"}>
#maps_q = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
           affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>]
#maps_sym = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
             affine_map<(d0, d1) -> (d0, d1)>]
#matmul_nt = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>,
              affine_map<(m, n, k) -> (m, n)>]

// One constant weight read in two shapes. It is shifted once, before the
// reshapes, so both readers share one shifted copy.
module attributes {stream.affinity.default = #hal.device.affinity<@device>} {
  util.global private @device = #hal.device.target<"local", [#arm64]> : !hal.device
  util.global private @weight = dense<200> : tensor<32x64xi8>
  util.func public @weight_read_in_two_shapes(%x: tensor<16x64xf32>, %y: tensor<16x32xf32>, %s: f32) -> (tensor<16x32xf32>, tensor<16x64xf32>) {
    %w = util.global.load @weight : tensor<32x64xi8>
    %w3 = tensor.expand_shape %w [[0], [1, 2]] output_shape [32, 2, 32] : tensor<32x64xi8> into tensor<32x2x32xi8>
    %wt = tensor.collapse_shape %w3 [[0, 1], [2]] : tensor<32x2x32xi8> into tensor<64x32xi8>
    %zw = arith.constant 128 : i8
    %zero = arith.constant 0.0 : f32
    %xqi = tensor.empty() : tensor<16x64xi8>
    %xq = iree_linalg_ext.quantize_affine <indexing_maps = #maps_sym, quant_min = -128, quant_max = 127>
        ins(%x, %s : tensor<16x64xf32>, f32) outs(%xqi : tensor<16x64xi8>) -> tensor<16x64xi8>
    %xi = tensor.empty() : tensor<16x64xf32>
    %xd = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_sym>
        ins(%xq, %s : tensor<16x64xi8>, f32) outs(%xi : tensor<16x64xf32>) -> tensor<16x64xf32>
    %wi = tensor.empty() : tensor<32x64xf32>
    %wd = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_q, input_unsigned, zp_unsigned>
        ins(%w, %s, %zw : tensor<32x64xi8>, f32, i8) outs(%wi : tensor<32x64xf32>) -> tensor<32x64xf32>
    %e = tensor.empty() : tensor<16x32xf32>
    %f = linalg.fill ins(%zero : f32) outs(%e : tensor<16x32xf32>) -> tensor<16x32xf32>
    %mm = linalg.matmul indexing_maps = #matmul_nt
        ins(%xd, %wd : tensor<16x64xf32>, tensor<32x64xf32>) outs(%f : tensor<16x32xf32>) -> tensor<16x32xf32>
    %yqi = tensor.empty() : tensor<16x32xi8>
    %yq = iree_linalg_ext.quantize_affine <indexing_maps = #maps_sym, quant_min = -128, quant_max = 127>
        ins(%y, %s : tensor<16x32xf32>, f32) outs(%yqi : tensor<16x32xi8>) -> tensor<16x32xi8>
    %yi = tensor.empty() : tensor<16x32xf32>
    %yd = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_sym>
        ins(%yq, %s : tensor<16x32xi8>, f32) outs(%yi : tensor<16x32xf32>) -> tensor<16x32xf32>
    %wti = tensor.empty() : tensor<64x32xf32>
    %wtd = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_q, input_unsigned, zp_unsigned>
        ins(%wt, %s, %zw : tensor<64x32xi8>, f32, i8) outs(%wti : tensor<64x32xf32>) -> tensor<64x32xf32>
    %e2 = tensor.empty() : tensor<16x64xf32>
    %f2 = linalg.fill ins(%zero : f32) outs(%e2 : tensor<16x64xf32>) -> tensor<16x64xf32>
    %mm2 = linalg.matmul indexing_maps = #matmul_nt
        ins(%yd, %wtd : tensor<16x32xf32>, tensor<64x32xf32>) outs(%f2 : tensor<16x64xf32>) -> tensor<16x64xf32>
    util.return %mm, %mm2 : tensor<16x32xf32>, tensor<16x64xf32>
  }
}

// CHECK-LABEL: util.func public @weight_read_in_two_shapes(
//       CHECK:   %[[W:.+]] = util.global.load @weight
//       CHECK:   %[[WS:.+]] = linalg.generic
//  CHECK-SAME:     ins(%[[W]] : tensor<32x64xi8>)
//       CHECK:   %[[W3:.+]] = tensor.expand_shape %[[WS]]
//       CHECK:   %[[WT:.+]] = tensor.collapse_shape %[[W3]]
//       CHECK:   iree_linalg_ext.dequantize_affine {{.+}} ins(%[[WS]],
//       CHECK:   iree_linalg_ext.dequantize_affine {{.+}} ins(%[[WT]],

// -----

#arm64 = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+dotprod", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, target_triple = "aarch64-unknown-unknown-eabi-elf"}>
#maps_q = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
           affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>]
#maps_sym = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
             affine_map<(d0, d1) -> (d0, d1)>]
#matmul_nt = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>,
              affine_map<(m, n, k) -> (m, n)>]

// A u8 weight of dynamic shape, computed from a constant. The shift reads its
// sizes from the unshifted weight, and the dequantize reads the shift.
module attributes {stream.affinity.default = #hal.device.affinity<@device>} {
  util.global private @device = #hal.device.target<"local", [#arm64]> : !hal.device
  util.func public @dynamic_weight(%x: tensor<16x64xf32>, %s: f32, %n: index) -> tensor<16x?xf32> {
    %cst = arith.constant dense<200> : tensor<32x64xi8>
    %c32 = arith.constant 32 : index
    %wi8 = tensor.empty(%c32) : tensor<?x64xi8>
    %w = linalg.copy ins(%cst : tensor<32x64xi8>) outs(%wi8 : tensor<?x64xi8>) -> tensor<?x64xi8>
    %zw = arith.constant 128 : i8
    %qi = tensor.empty() : tensor<16x64xi8>
    %xq = iree_linalg_ext.quantize_affine <indexing_maps = #maps_sym, quant_min = -128, quant_max = 127>
        ins(%x, %s : tensor<16x64xf32>, f32) outs(%qi : tensor<16x64xi8>) -> tensor<16x64xi8>
    %xi = tensor.empty() : tensor<16x64xf32>
    %xd = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_sym>
        ins(%xq, %s : tensor<16x64xi8>, f32) outs(%xi : tensor<16x64xf32>) -> tensor<16x64xf32>
    %wi = tensor.empty(%n) : tensor<?x64xf32>
    %wd = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_q, input_unsigned, zp_unsigned>
        ins(%w, %s, %zw : tensor<?x64xi8>, f32, i8) outs(%wi : tensor<?x64xf32>) -> tensor<?x64xf32>
    %zero = arith.constant 0.0 : f32
    %e = tensor.empty(%n) : tensor<16x?xf32>
    %f = linalg.fill ins(%zero : f32) outs(%e : tensor<16x?xf32>) -> tensor<16x?xf32>
    %mm = linalg.matmul indexing_maps = #matmul_nt
        ins(%xd, %wd : tensor<16x64xf32>, tensor<?x64xf32>) outs(%f : tensor<16x?xf32>) -> tensor<16x?xf32>
    util.return %mm : tensor<16x?xf32>
  }
}

// CHECK-LABEL: util.func public @dynamic_weight(
//       CHECK:   %[[W:.+]] = linalg.copy
//       CHECK:   tensor.dim %[[W]]
//       CHECK:   %[[WS:.+]] = linalg.generic
//  CHECK-SAME:     ins(%[[W]] :
//       CHECK:   iree_linalg_ext.dequantize_affine {{.+}} ins(%[[WS]],

// -----

#arm64 = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+dotprod", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, target_triple = "aarch64-unknown-unknown-eabi-elf"}>
#maps_q = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
           affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>]
#matmul_nt = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>,
              affine_map<(m, n, k) -> (m, n)>]

// A u8 activation and a constant u8 weight both move to s8.
module attributes {stream.affinity.default = #hal.device.affinity<@device>} {
  util.global private @device = #hal.device.target<"local", [#arm64]> : !hal.device
  util.func public @both_operands(%x: tensor<16x64xf32>, %sx: f32, %zx: i8, %sw: f32) -> tensor<16x32xf32> {
    %w = arith.constant dense<200> : tensor<32x64xi8>
    %zw = arith.constant 100 : i8
    %qi = tensor.empty() : tensor<16x64xi8>
    %xq = iree_linalg_ext.quantize_affine <indexing_maps = #maps_q, quant_min = 0, quant_max = 255, storage_unsigned, zp_unsigned>
        ins(%x, %sx, %zx : tensor<16x64xf32>, f32, i8) outs(%qi : tensor<16x64xi8>) -> tensor<16x64xi8>
    %xi = tensor.empty() : tensor<16x64xf32>
    %xd = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_q, input_unsigned, zp_unsigned>
        ins(%xq, %sx, %zx : tensor<16x64xi8>, f32, i8) outs(%xi : tensor<16x64xf32>) -> tensor<16x64xf32>
    %wi = tensor.empty() : tensor<32x64xf32>
    %wd = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_q, input_unsigned, zp_unsigned>
        ins(%w, %sw, %zw : tensor<32x64xi8>, f32, i8) outs(%wi : tensor<32x64xf32>) -> tensor<32x64xf32>
    %zero = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<16x32xf32>
    %f = linalg.fill ins(%zero : f32) outs(%e : tensor<16x32xf32>) -> tensor<16x32xf32>
    %mm = linalg.matmul indexing_maps = #matmul_nt
        ins(%xd, %wd : tensor<16x64xf32>, tensor<32x64xf32>) outs(%f : tensor<16x32xf32>) -> tensor<16x32xf32>
    util.return %mm : tensor<16x32xf32>
  }
}

// CHECK-LABEL: util.func public @both_operands(
//       CHECK:   %[[WS:.+]] = linalg.generic
//       CHECK:   iree_linalg_ext.quantize_affine
//  CHECK-SAME:     quant_min = -128, quant_max = 127>
//       CHECK:   %[[ZW:.+]] = arith.constant -28 : i8
//       CHECK:   iree_linalg_ext.dequantize_affine <indexing_maps = [#{{.+}}, #{{.+}}, #{{.+}}, #{{.+}}]> ins(%[[WS]], %{{.+}}, %[[ZW]] : tensor<32x64xi8>, f32, i8)

// -----

#arm64 = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+dotprod", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, target_triple = "aarch64-unknown-unknown-eabi-elf"}>
#maps_q = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
           affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>]
#maps_sym = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
             affine_map<(d0, d1) -> (d0, d1)>]
#matmul_nt = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>,
              affine_map<(m, n, k) -> (m, n)>]

// A reader may state wider bounds than its quantize. The grid holds both, so
// both shift by -128.
module attributes {stream.affinity.default = #hal.device.affinity<@device>} {
  util.global private @device = #hal.device.target<"local", [#arm64]> : !hal.device
  util.func public @reader_bounds_wider_than_quantize(%x: tensor<16x64xf32>, %sx: f32, %zx: i8, %sw: f32) -> tensor<16x32xf32> {
    %w = arith.constant dense<3> : tensor<32x64xi8>
    %qi = tensor.empty() : tensor<16x64xi8>
    %xq = iree_linalg_ext.quantize_affine <indexing_maps = #maps_q, quant_min = 0, quant_max = 15, storage_unsigned, zp_unsigned>
        ins(%x, %sx, %zx : tensor<16x64xf32>, f32, i8) outs(%qi : tensor<16x64xi8>) -> tensor<16x64xi8>
    %xi = tensor.empty() : tensor<16x64xf32>
    %xd = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_q, quant_min = 0, quant_max = 255, input_unsigned, zp_unsigned>
        ins(%xq, %sx, %zx : tensor<16x64xi8>, f32, i8) outs(%xi : tensor<16x64xf32>) -> tensor<16x64xf32>
    %wi = tensor.empty() : tensor<32x64xf32>
    %wd = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_sym>
        ins(%w, %sw : tensor<32x64xi8>, f32) outs(%wi : tensor<32x64xf32>) -> tensor<32x64xf32>
    %zero = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<16x32xf32>
    %f = linalg.fill ins(%zero : f32) outs(%e : tensor<16x32xf32>) -> tensor<16x32xf32>
    %mm = linalg.matmul indexing_maps = #matmul_nt
        ins(%xd, %wd : tensor<16x64xf32>, tensor<32x64xf32>) outs(%f : tensor<16x32xf32>) -> tensor<16x32xf32>
    util.return %mm : tensor<16x32xf32>
  }
}

// CHECK-LABEL: util.func public @reader_bounds_wider_than_quantize(
//       CHECK:   iree_linalg_ext.quantize_affine
//  CHECK-SAME:     quant_min = -128, quant_max = -113>
//       CHECK:   iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     quant_min = -128, quant_max = 127>

// -----

#arm64 = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+dotprod", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, target_triple = "aarch64-unknown-unknown-eabi-elf"}>
#maps_q = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
           affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>]
#maps_sym = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
             affine_map<(d0, d1) -> (d0, d1)>]
#matmul_nt = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>,
              affine_map<(m, n, k) -> (m, n)>]

// A quantize may state wider bounds than its reader. The grid holds both, so
// both shift by -128.
module attributes {stream.affinity.default = #hal.device.affinity<@device>} {
  util.global private @device = #hal.device.target<"local", [#arm64]> : !hal.device
  util.func public @quantize_bounds_wider_than_reader(%x: tensor<16x64xf32>, %sx: f32, %sw: f32) -> tensor<16x32xf32> {
    %w = arith.constant dense<3> : tensor<32x64xi8>
    %zx = arith.constant 8 : i8
    %qi = tensor.empty() : tensor<16x64xi8>
    %xq = iree_linalg_ext.quantize_affine <indexing_maps = #maps_q, quant_min = 0, quant_max = 255, storage_unsigned, zp_unsigned>
        ins(%x, %sx, %zx : tensor<16x64xf32>, f32, i8) outs(%qi : tensor<16x64xi8>) -> tensor<16x64xi8>
    %xi = tensor.empty() : tensor<16x64xf32>
    %xd = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_q, quant_min = 0, quant_max = 15, input_unsigned, zp_unsigned>
        ins(%xq, %sx, %zx : tensor<16x64xi8>, f32, i8) outs(%xi : tensor<16x64xf32>) -> tensor<16x64xf32>
    %wi = tensor.empty() : tensor<32x64xf32>
    %wd = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_sym>
        ins(%w, %sw : tensor<32x64xi8>, f32) outs(%wi : tensor<32x64xf32>) -> tensor<32x64xf32>
    %zero = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<16x32xf32>
    %f = linalg.fill ins(%zero : f32) outs(%e : tensor<16x32xf32>) -> tensor<16x32xf32>
    %mm = linalg.matmul indexing_maps = #matmul_nt
        ins(%xd, %wd : tensor<16x64xf32>, tensor<32x64xf32>) outs(%f : tensor<16x32xf32>) -> tensor<16x32xf32>
    util.return %mm : tensor<16x32xf32>
  }
}

// CHECK-LABEL: util.func public @quantize_bounds_wider_than_reader(
//       CHECK:   iree_linalg_ext.quantize_affine
//  CHECK-SAME:     quant_min = -128, quant_max = 127>
//       CHECK:   iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     quant_min = -128, quant_max = -113>

// -----

#arm64 = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+dotprod", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, target_triple = "aarch64-unknown-unknown-eabi-elf"}>
#maps_q = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
           affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>]
#maps_sym = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
             affine_map<(d0, d1) -> (d0, d1)>]
#matmul_nt = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>,
              affine_map<(m, n, k) -> (m, n)>]

// The quantized activation is also read as signed, so it keeps its signedness.
module attributes {stream.affinity.default = #hal.device.affinity<@device>} {
  util.global private @device = #hal.device.target<"local", [#arm64]> : !hal.device
  util.func public @reader_of_other_signedness(%x: tensor<16x64xf32>, %sx: f32, %zx: i8, %sw: f32) -> (tensor<16x32xf32>, tensor<16x64xf32>) {
    %w = arith.constant dense<3> : tensor<32x64xi8>
    %qi = tensor.empty() : tensor<16x64xi8>
    %xq = iree_linalg_ext.quantize_affine <indexing_maps = #maps_q, quant_min = 0, quant_max = 255, storage_unsigned, zp_unsigned>
        ins(%x, %sx, %zx : tensor<16x64xf32>, f32, i8) outs(%qi : tensor<16x64xi8>) -> tensor<16x64xi8>
    %xi = tensor.empty() : tensor<16x64xf32>
    %xd = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_q, input_unsigned, zp_unsigned>
        ins(%xq, %sx, %zx : tensor<16x64xi8>, f32, i8) outs(%xi : tensor<16x64xf32>) -> tensor<16x64xf32>
    %xs = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_q, quant_min = 0, quant_max = 127>
        ins(%xq, %sx, %zx : tensor<16x64xi8>, f32, i8) outs(%xi : tensor<16x64xf32>) -> tensor<16x64xf32>
    %wi = tensor.empty() : tensor<32x64xf32>
    %wd = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_sym>
        ins(%w, %sw : tensor<32x64xi8>, f32) outs(%wi : tensor<32x64xf32>) -> tensor<32x64xf32>
    %zero = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<16x32xf32>
    %f = linalg.fill ins(%zero : f32) outs(%e : tensor<16x32xf32>) -> tensor<16x32xf32>
    %mm = linalg.matmul indexing_maps = #matmul_nt
        ins(%xd, %wd : tensor<16x64xf32>, tensor<32x64xf32>) outs(%f : tensor<16x32xf32>) -> tensor<16x32xf32>
    util.return %mm, %xs : tensor<16x32xf32>, tensor<16x64xf32>
  }
}

// CHECK-LABEL: util.func public @reader_of_other_signedness(
//       CHECK:   iree_linalg_ext.quantize_affine
//  CHECK-SAME:     quant_min = 0, quant_max = 255, storage_unsigned, zp_unsigned>

// -----

#arm64 = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+dotprod", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, target_triple = "aarch64-unknown-unknown-eabi-elf"}>
#maps_q = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
           affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>]
#maps_sym = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
             affine_map<(d0, d1) -> (d0, d1)>]
#matmul_nt = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>,
              affine_map<(m, n, k) -> (m, n)>]

// The quantized activation is also the zero point of another dequantize, so it
// keeps its signedness.
module attributes {stream.affinity.default = #hal.device.affinity<@device>} {
  util.global private @device = #hal.device.target<"local", [#arm64]> : !hal.device
  util.func public @zero_point_reader(%x: tensor<16x64xf32>, %sx: f32, %zx: i8, %sw: f32, %y: tensor<16x64xi8>) -> (tensor<16x32xf32>, tensor<16x64xf32>) {
    %w = arith.constant dense<3> : tensor<32x64xi8>
    %qi = tensor.empty() : tensor<16x64xi8>
    %xq = iree_linalg_ext.quantize_affine <indexing_maps = #maps_q, quant_min = 0, quant_max = 255, storage_unsigned, zp_unsigned>
        ins(%x, %sx, %zx : tensor<16x64xf32>, f32, i8) outs(%qi : tensor<16x64xi8>) -> tensor<16x64xi8>
    %xi = tensor.empty() : tensor<16x64xf32>
    %xd = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_q, input_unsigned, zp_unsigned>
        ins(%xq, %sx, %zx : tensor<16x64xi8>, f32, i8) outs(%xi : tensor<16x64xf32>) -> tensor<16x64xf32>
    %yd = iree_linalg_ext.dequantize_affine
        <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
                          affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
         input_unsigned, zp_unsigned>
        ins(%y, %sx, %xq : tensor<16x64xi8>, f32, tensor<16x64xi8>) outs(%xi : tensor<16x64xf32>) -> tensor<16x64xf32>
    %wi = tensor.empty() : tensor<32x64xf32>
    %wd = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_sym>
        ins(%w, %sw : tensor<32x64xi8>, f32) outs(%wi : tensor<32x64xf32>) -> tensor<32x64xf32>
    %zero = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<16x32xf32>
    %f = linalg.fill ins(%zero : f32) outs(%e : tensor<16x32xf32>) -> tensor<16x32xf32>
    %mm = linalg.matmul indexing_maps = #matmul_nt
        ins(%xd, %wd : tensor<16x64xf32>, tensor<32x64xf32>) outs(%f : tensor<16x32xf32>) -> tensor<16x32xf32>
    util.return %mm, %yd : tensor<16x32xf32>, tensor<16x64xf32>
  }
}

// CHECK-LABEL: util.func public @zero_point_reader(
//       CHECK:   iree_linalg_ext.quantize_affine
//  CHECK-SAME:     quant_min = 0, quant_max = 255, storage_unsigned, zp_unsigned>

// -----

#arm64 = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+dotprod", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, target_triple = "aarch64-unknown-unknown-eabi-elf"}>
#maps_q = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
           affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>]
#maps_sym = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
             affine_map<(d0, d1) -> (d0, d1)>]
#matmul_nt = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>,
              affine_map<(m, n, k) -> (m, n)>]

// The values of a u8 weight fit 4 bits, but its zero point of 200 does not, so
// the weight moves on the 8-bit grid, by -128.
module attributes {stream.affinity.default = #hal.device.affinity<@device>} {
  util.global private @device = #hal.device.target<"local", [#arm64]> : !hal.device
  util.func public @zero_point_outside_value_range(%x: tensor<16x64xf32>, %sx: f32, %sw: f32) -> tensor<16x32xf32> {
    %w = arith.constant dense<3> : tensor<32x64xi8>
    %zw = arith.constant 200 : i8
    %qi = tensor.empty() : tensor<16x64xi8>
    %xq = iree_linalg_ext.quantize_affine <indexing_maps = #maps_sym, quant_min = -128, quant_max = 127>
        ins(%x, %sx : tensor<16x64xf32>, f32) outs(%qi : tensor<16x64xi8>) -> tensor<16x64xi8>
    %xi = tensor.empty() : tensor<16x64xf32>
    %xd = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_sym>
        ins(%xq, %sx : tensor<16x64xi8>, f32) outs(%xi : tensor<16x64xf32>) -> tensor<16x64xf32>
    %wi = tensor.empty() : tensor<32x64xf32>
    %wd = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_q, quant_min = 0, quant_max = 15, input_unsigned, zp_unsigned>
        ins(%w, %sw, %zw : tensor<32x64xi8>, f32, i8) outs(%wi : tensor<32x64xf32>) -> tensor<32x64xf32>
    %zero = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<16x32xf32>
    %f = linalg.fill ins(%zero : f32) outs(%e : tensor<16x32xf32>) -> tensor<16x32xf32>
    %mm = linalg.matmul indexing_maps = #matmul_nt
        ins(%xd, %wd : tensor<16x64xf32>, tensor<32x64xf32>) outs(%f : tensor<16x32xf32>) -> tensor<16x32xf32>
    util.return %mm : tensor<16x32xf32>
  }
}

// CHECK-LABEL: util.func public @zero_point_outside_value_range(
//       CHECK:   iree_linalg_ext.dequantize_affine <indexing_maps = [#{{.+}}, #{{.+}}, #{{.+}}, #{{.+}}], quant_min = -128, quant_max = -113>

// -----

#arm64 = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+dotprod", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, target_triple = "aarch64-unknown-unknown-eabi-elf"}>
#maps_q = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
           affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>]
#maps_sym = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
             affine_map<(d0, d1) -> (d0, d1)>]
#matmul_nt = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>,
              affine_map<(m, n, k) -> (m, n)>]

// A u4 weight that a parameter holds keeps its signedness.
module attributes {stream.affinity.default = #hal.device.affinity<@device>} {
  util.global private @device = #hal.device.target<"local", [#arm64]> : !hal.device
  util.global private @weight = #stream.parameter.named<"model"::"weight"> : tensor<32x64xi8>
  util.func public @parameter_weight(%x: tensor<16x64xf32>, %sx: f32, %sw: f32) -> tensor<16x32xf32> {
    %w = util.global.load @weight : tensor<32x64xi8>
    %zw = arith.constant 8 : i8
    %qi = tensor.empty() : tensor<16x64xi8>
    %xq = iree_linalg_ext.quantize_affine <indexing_maps = #maps_sym, quant_min = -128, quant_max = 127>
        ins(%x, %sx : tensor<16x64xf32>, f32) outs(%qi : tensor<16x64xi8>) -> tensor<16x64xi8>
    %xi = tensor.empty() : tensor<16x64xf32>
    %xd = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_sym>
        ins(%xq, %sx : tensor<16x64xi8>, f32) outs(%xi : tensor<16x64xf32>) -> tensor<16x64xf32>
    %wi = tensor.empty() : tensor<32x64xf32>
    %wd = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_q, quant_min = 0, quant_max = 15, input_unsigned, zp_unsigned>
        ins(%w, %sw, %zw : tensor<32x64xi8>, f32, i8) outs(%wi : tensor<32x64xf32>) -> tensor<32x64xf32>
    %zero = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<16x32xf32>
    %f = linalg.fill ins(%zero : f32) outs(%e : tensor<16x32xf32>) -> tensor<16x32xf32>
    %mm = linalg.matmul indexing_maps = #matmul_nt
        ins(%xd, %wd : tensor<16x64xf32>, tensor<32x64xf32>) outs(%f : tensor<16x32xf32>) -> tensor<16x32xf32>
    util.return %mm : tensor<16x32xf32>
  }
}

// CHECK-LABEL: util.func public @parameter_weight(
//       CHECK:   iree_linalg_ext.dequantize_affine <indexing_maps = [#{{.+}}, #{{.+}}, #{{.+}}, #{{.+}}], quant_min = 0, quant_max = 15, input_unsigned, zp_unsigned>

// -----

#arm64 = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+dotprod", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, target_triple = "aarch64-unknown-unknown-eabi-elf"}>
#maps_q = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
           affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>]
#maps_sym = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
             affine_map<(d0, d1) -> (d0, d1)>]
#matmul_nt = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>,
              affine_map<(m, n, k) -> (m, n)>]

// A u4 weight in a public global keeps its signedness.
module attributes {stream.affinity.default = #hal.device.affinity<@device>} {
  util.global private @device = #hal.device.target<"local", [#arm64]> : !hal.device
  util.global public @weight = dense<9> : tensor<32x64xi8>
  util.func public @public_weight(%x: tensor<16x64xf32>, %sx: f32, %sw: f32) -> tensor<16x32xf32> {
    %w = util.global.load @weight : tensor<32x64xi8>
    %zw = arith.constant 8 : i8
    %qi = tensor.empty() : tensor<16x64xi8>
    %xq = iree_linalg_ext.quantize_affine <indexing_maps = #maps_sym, quant_min = -128, quant_max = 127>
        ins(%x, %sx : tensor<16x64xf32>, f32) outs(%qi : tensor<16x64xi8>) -> tensor<16x64xi8>
    %xi = tensor.empty() : tensor<16x64xf32>
    %xd = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_sym>
        ins(%xq, %sx : tensor<16x64xi8>, f32) outs(%xi : tensor<16x64xf32>) -> tensor<16x64xf32>
    %wi = tensor.empty() : tensor<32x64xf32>
    %wd = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_q, quant_min = 0, quant_max = 15, input_unsigned, zp_unsigned>
        ins(%w, %sw, %zw : tensor<32x64xi8>, f32, i8) outs(%wi : tensor<32x64xf32>) -> tensor<32x64xf32>
    %zero = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<16x32xf32>
    %f = linalg.fill ins(%zero : f32) outs(%e : tensor<16x32xf32>) -> tensor<16x32xf32>
    %mm = linalg.matmul indexing_maps = #matmul_nt
        ins(%xd, %wd : tensor<16x64xf32>, tensor<32x64xf32>) outs(%f : tensor<16x32xf32>) -> tensor<16x32xf32>
    util.return %mm : tensor<16x32xf32>
  }
}

// CHECK-LABEL: util.func public @public_weight(
//       CHECK:   iree_linalg_ext.dequantize_affine <indexing_maps = [#{{.+}}, #{{.+}}, #{{.+}}, #{{.+}}], quant_min = 0, quant_max = 15, input_unsigned, zp_unsigned>

// -----

#arm64 = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+dotprod", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, target_triple = "aarch64-unknown-unknown-eabi-elf"}>
#maps_q = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
           affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>]
#maps_sym = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
             affine_map<(d0, d1) -> (d0, d1)>]
#matmul_nt = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>,
              affine_map<(m, n, k) -> (m, n)>]

// A u4 weight stored as an i4 constant keeps its signedness.
module attributes {stream.affinity.default = #hal.device.affinity<@device>} {
  util.global private @device = #hal.device.target<"local", [#arm64]> : !hal.device
  util.func public @sub_byte_constant_weight(%x: tensor<16x64xf32>, %sx: f32, %sw: f32) -> tensor<16x32xf32> {
    %w = arith.constant dense<9> : tensor<32x64xi4>
    %zw = arith.constant 8 : i4
    %qi = tensor.empty() : tensor<16x64xi8>
    %xq = iree_linalg_ext.quantize_affine <indexing_maps = #maps_sym, quant_min = -128, quant_max = 127>
        ins(%x, %sx : tensor<16x64xf32>, f32) outs(%qi : tensor<16x64xi8>) -> tensor<16x64xi8>
    %xi = tensor.empty() : tensor<16x64xf32>
    %xd = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_sym>
        ins(%xq, %sx : tensor<16x64xi8>, f32) outs(%xi : tensor<16x64xf32>) -> tensor<16x64xf32>
    %wi = tensor.empty() : tensor<32x64xf32>
    %wd = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_q, input_unsigned, zp_unsigned>
        ins(%w, %sw, %zw : tensor<32x64xi4>, f32, i4) outs(%wi : tensor<32x64xf32>) -> tensor<32x64xf32>
    %zero = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<16x32xf32>
    %f = linalg.fill ins(%zero : f32) outs(%e : tensor<16x32xf32>) -> tensor<16x32xf32>
    %mm = linalg.matmul indexing_maps = #matmul_nt
        ins(%xd, %wd : tensor<16x64xf32>, tensor<32x64xf32>) outs(%f : tensor<16x32xf32>) -> tensor<16x32xf32>
    util.return %mm : tensor<16x32xf32>
  }
}

// CHECK-LABEL: util.func public @sub_byte_constant_weight(
//       CHECK:   iree_linalg_ext.dequantize_affine <indexing_maps = [#{{.+}}, #{{.+}}, #{{.+}}, #{{.+}}], input_unsigned, zp_unsigned>

// -----

#arm64 = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+dotprod", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, target_triple = "aarch64-unknown-unknown-eabi-elf"}>
#maps_q = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
           affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>]
#maps_sym = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
             affine_map<(d0, d1) -> (d0, d1)>]
#matmul_nt = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>,
              affine_map<(m, n, k) -> (m, n)>]

// An i4 zero point cannot hold the zero point of an 8-bit grid, so the
// activation keeps its signedness.
module attributes {stream.affinity.default = #hal.device.affinity<@device>} {
  util.global private @device = #hal.device.target<"local", [#arm64]> : !hal.device
  util.func public @narrow_zero_point_type(%x: tensor<16x64xf32>, %sx: f32, %zx: i4, %sw: f32) -> tensor<16x32xf32> {
    %w = arith.constant dense<3> : tensor<32x64xi8>
    %qi = tensor.empty() : tensor<16x64xi8>
    %xq = iree_linalg_ext.quantize_affine <indexing_maps = #maps_q, quant_min = 0, quant_max = 255, storage_unsigned, zp_unsigned>
        ins(%x, %sx, %zx : tensor<16x64xf32>, f32, i4) outs(%qi : tensor<16x64xi8>) -> tensor<16x64xi8>
    %xi = tensor.empty() : tensor<16x64xf32>
    %xd = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_q, input_unsigned, zp_unsigned>
        ins(%xq, %sx, %zx : tensor<16x64xi8>, f32, i4) outs(%xi : tensor<16x64xf32>) -> tensor<16x64xf32>
    %wi = tensor.empty() : tensor<32x64xf32>
    %wd = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_sym>
        ins(%w, %sw : tensor<32x64xi8>, f32) outs(%wi : tensor<32x64xf32>) -> tensor<32x64xf32>
    %zero = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<16x32xf32>
    %f = linalg.fill ins(%zero : f32) outs(%e : tensor<16x32xf32>) -> tensor<16x32xf32>
    %mm = linalg.matmul indexing_maps = #matmul_nt
        ins(%xd, %wd : tensor<16x64xf32>, tensor<32x64xf32>) outs(%f : tensor<16x32xf32>) -> tensor<16x32xf32>
    util.return %mm : tensor<16x32xf32>
  }
}

// CHECK-LABEL: util.func public @narrow_zero_point_type(
//       CHECK:   iree_linalg_ext.quantize_affine
//  CHECK-SAME:     quant_min = 0, quant_max = 255, storage_unsigned, zp_unsigned>

// -----

#arm64 = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+dotprod", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, target_triple = "aarch64-unknown-unknown-eabi-elf"}>
#maps_q = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
           affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>]
#maps_sym = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
             affine_map<(d0, d1) -> (d0, d1)>]
#matmul_nt = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>,
              affine_map<(m, n, k) -> (m, n)>]

// A quantize whose f8E4M3FN scale does not hold every integer of an 8-bit grid
// keeps its signedness.
module attributes {stream.affinity.default = #hal.device.affinity<@device>} {
  util.global private @device = #hal.device.target<"local", [#arm64]> : !hal.device
  util.func public @quantize_scale_type_too_narrow(%x: tensor<16x64xf32>, %sx: f8E4M3FN, %zx: i8, %sw: f32) -> tensor<16x32xf32> {
    %w = arith.constant dense<3> : tensor<32x64xi8>
    %qi = tensor.empty() : tensor<16x64xi8>
    %xq = iree_linalg_ext.quantize_affine <indexing_maps = #maps_q, quant_min = 0, quant_max = 255, storage_unsigned, zp_unsigned>
        ins(%x, %sx, %zx : tensor<16x64xf32>, f8E4M3FN, i8) outs(%qi : tensor<16x64xi8>) -> tensor<16x64xi8>
    %xi = tensor.empty() : tensor<16x64xf32>
    %xd = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_q, input_unsigned, zp_unsigned>
        ins(%xq, %sx, %zx : tensor<16x64xi8>, f8E4M3FN, i8) outs(%xi : tensor<16x64xf32>) -> tensor<16x64xf32>
    %wi = tensor.empty() : tensor<32x64xf32>
    %wd = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_sym>
        ins(%w, %sw : tensor<32x64xi8>, f32) outs(%wi : tensor<32x64xf32>) -> tensor<32x64xf32>
    %zero = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<16x32xf32>
    %f = linalg.fill ins(%zero : f32) outs(%e : tensor<16x32xf32>) -> tensor<16x32xf32>
    %mm = linalg.matmul indexing_maps = #matmul_nt
        ins(%xd, %wd : tensor<16x64xf32>, tensor<32x64xf32>) outs(%f : tensor<16x32xf32>) -> tensor<16x32xf32>
    util.return %mm : tensor<16x32xf32>
  }
}

// CHECK-LABEL: util.func public @quantize_scale_type_too_narrow(
//       CHECK:   iree_linalg_ext.quantize_affine
//  CHECK-SAME:     quant_min = 0, quant_max = 255, storage_unsigned, zp_unsigned>

// -----

#arm64 = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+dotprod", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, target_triple = "aarch64-unknown-unknown-eabi-elf"}>
#maps_sym = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
             affine_map<(d0, d1) -> (d0, d1)>]
#matmul_nt = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>,
              affine_map<(m, n, k) -> (m, n)>]

// A symmetric u8 weight times a run-time activation keeps its signedness.
module attributes {stream.affinity.default = #hal.device.affinity<@device>} {
  util.global private @device = #hal.device.target<"local", [#arm64]> : !hal.device
  util.func public @symmetric_weight_with_runtime_activation(%x: tensor<16x64xf32>, %sx: f32, %sw: f32) -> tensor<16x32xf32> {
    %w = arith.constant dense<200> : tensor<32x64xi8>
    %qi = tensor.empty() : tensor<16x64xi8>
    %xq = iree_linalg_ext.quantize_affine <indexing_maps = #maps_sym, quant_min = -128, quant_max = 127>
        ins(%x, %sx : tensor<16x64xf32>, f32) outs(%qi : tensor<16x64xi8>) -> tensor<16x64xi8>
    %xi = tensor.empty() : tensor<16x64xf32>
    %xd = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_sym>
        ins(%xq, %sx : tensor<16x64xi8>, f32) outs(%xi : tensor<16x64xf32>) -> tensor<16x64xf32>
    %wi = tensor.empty() : tensor<32x64xf32>
    %wd = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_sym, input_unsigned>
        ins(%w, %sw : tensor<32x64xi8>, f32) outs(%wi : tensor<32x64xf32>) -> tensor<32x64xf32>
    %zero = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<16x32xf32>
    %f = linalg.fill ins(%zero : f32) outs(%e : tensor<16x32xf32>) -> tensor<16x32xf32>
    %mm = linalg.matmul indexing_maps = #matmul_nt
        ins(%xd, %wd : tensor<16x64xf32>, tensor<32x64xf32>) outs(%f : tensor<16x32xf32>) -> tensor<16x32xf32>
    util.return %mm : tensor<16x32xf32>
  }
}

// CHECK-LABEL: util.func public @symmetric_weight_with_runtime_activation(
//       CHECK:   iree_linalg_ext.dequantize_affine <indexing_maps = [#{{.+}}, #{{.+}}, #{{.+}}], input_unsigned>

// -----

#arm64 = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+dotprod", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, target_triple = "aarch64-unknown-unknown-eabi-elf"}>
#maps_q = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
           affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>]
#maps_sym = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
             affine_map<(d0, d1) -> (d0, d1)>]
#matmul_nt = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>,
              affine_map<(m, n, k) -> (m, n)>]

// The quantized activation is also returned, so it keeps its signedness.
module attributes {stream.affinity.default = #hal.device.affinity<@device>} {
  util.global private @device = #hal.device.target<"local", [#arm64]> : !hal.device
  util.func public @other_reader(%x: tensor<16x64xf32>, %sx: f32, %zx: i8, %sw: f32) -> (tensor<16x32xf32>, tensor<16x64xi8>) {
    %w = arith.constant dense<3> : tensor<32x64xi8>
    %qi = tensor.empty() : tensor<16x64xi8>
    %xq = iree_linalg_ext.quantize_affine <indexing_maps = #maps_q, quant_min = 0, quant_max = 255, storage_unsigned, zp_unsigned>
        ins(%x, %sx, %zx : tensor<16x64xf32>, f32, i8) outs(%qi : tensor<16x64xi8>) -> tensor<16x64xi8>
    %xi = tensor.empty() : tensor<16x64xf32>
    %xd = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_q, input_unsigned, zp_unsigned>
        ins(%xq, %sx, %zx : tensor<16x64xi8>, f32, i8) outs(%xi : tensor<16x64xf32>) -> tensor<16x64xf32>
    %wi = tensor.empty() : tensor<32x64xf32>
    %wd = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_sym>
        ins(%w, %sw : tensor<32x64xi8>, f32) outs(%wi : tensor<32x64xf32>) -> tensor<32x64xf32>
    %zero = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<16x32xf32>
    %f = linalg.fill ins(%zero : f32) outs(%e : tensor<16x32xf32>) -> tensor<16x32xf32>
    %mm = linalg.matmul indexing_maps = #matmul_nt
        ins(%xd, %wd : tensor<16x64xf32>, tensor<32x64xf32>) outs(%f : tensor<16x32xf32>) -> tensor<16x32xf32>
    util.return %mm, %xq : tensor<16x32xf32>, tensor<16x64xi8>
  }
}

// CHECK-LABEL: util.func public @other_reader(
//       CHECK:   iree_linalg_ext.quantize_affine
//  CHECK-SAME:     quant_min = 0, quant_max = 255, storage_unsigned, zp_unsigned>

// -----

#arm64 = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+dotprod", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, target_triple = "aarch64-unknown-unknown-eabi-elf"}>
#maps_q = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
           affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>]
#maps_sym = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
             affine_map<(d0, d1) -> (d0, d1)>]
#matmul_nt = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>,
              affine_map<(m, n, k) -> (m, n)>]

// One quantized activation feeds two contractions. It moves once, with both
// dequantizes.
module attributes {stream.affinity.default = #hal.device.affinity<@device>} {
  util.global private @device = #hal.device.target<"local", [#arm64]> : !hal.device
  util.func public @shared_activation(%x: tensor<16x64xf32>, %sx: f32, %zx: i8, %sw: f32) -> (tensor<16x32xf32>, tensor<16x32xf32>) {
    %w0 = arith.constant dense<3> : tensor<32x64xi8>
    %w1 = arith.constant dense<5> : tensor<32x64xi8>
    %qi = tensor.empty() : tensor<16x64xi8>
    %xq = iree_linalg_ext.quantize_affine <indexing_maps = #maps_q, quant_min = 0, quant_max = 255, storage_unsigned, zp_unsigned>
        ins(%x, %sx, %zx : tensor<16x64xf32>, f32, i8) outs(%qi : tensor<16x64xi8>) -> tensor<16x64xi8>
    %xi = tensor.empty() : tensor<16x64xf32>
    %xd0 = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_q, input_unsigned, zp_unsigned>
        ins(%xq, %sx, %zx : tensor<16x64xi8>, f32, i8) outs(%xi : tensor<16x64xf32>) -> tensor<16x64xf32>
    %xd1 = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_q, input_unsigned, zp_unsigned>
        ins(%xq, %sx, %zx : tensor<16x64xi8>, f32, i8) outs(%xi : tensor<16x64xf32>) -> tensor<16x64xf32>
    %wi = tensor.empty() : tensor<32x64xf32>
    %wd0 = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_sym>
        ins(%w0, %sw : tensor<32x64xi8>, f32) outs(%wi : tensor<32x64xf32>) -> tensor<32x64xf32>
    %wd1 = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_sym>
        ins(%w1, %sw : tensor<32x64xi8>, f32) outs(%wi : tensor<32x64xf32>) -> tensor<32x64xf32>
    %zero = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<16x32xf32>
    %f = linalg.fill ins(%zero : f32) outs(%e : tensor<16x32xf32>) -> tensor<16x32xf32>
    %mm0 = linalg.matmul indexing_maps = #matmul_nt
        ins(%xd0, %wd0 : tensor<16x64xf32>, tensor<32x64xf32>) outs(%f : tensor<16x32xf32>) -> tensor<16x32xf32>
    %mm1 = linalg.matmul indexing_maps = #matmul_nt
        ins(%xd1, %wd1 : tensor<16x64xf32>, tensor<32x64xf32>) outs(%f : tensor<16x32xf32>) -> tensor<16x32xf32>
    util.return %mm0, %mm1 : tensor<16x32xf32>, tensor<16x32xf32>
  }
}

// CHECK-LABEL: util.func public @shared_activation(
//       CHECK:   %[[ZP:.+]] = arith.addi
//       CHECK:   %[[Q:.+]] = iree_linalg_ext.quantize_affine
//       CHECK:   iree_linalg_ext.dequantize_affine <indexing_maps = [#{{.+}}, #{{.+}}, #{{.+}}, #{{.+}}]> ins(%[[Q]], %{{.+}}, %[[ZP]] :
//       CHECK:   iree_linalg_ext.dequantize_affine <indexing_maps = [#{{.+}}, #{{.+}}, #{{.+}}, #{{.+}}]> ins(%[[Q]], %{{.+}}, %[[ZP]] :

// -----

#arm64 = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+dotprod", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, target_triple = "aarch64-unknown-unknown-eabi-elf"}>
#maps_q = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
           affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>]
#maps_sym = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
             affine_map<(d0, d1) -> (d0, d1)>]
#matmul_nt = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>,
              affine_map<(m, n, k) -> (m, n)>]

// One activation feeds two contractions, and the second, whose u8 weight is a
// run-time value, can tile no rebase. The activation keeps its signedness.
module attributes {stream.affinity.default = #hal.device.affinity<@device>} {
  util.global private @device = #hal.device.target<"local", [#arm64]> : !hal.device
  util.func public @contraction_disagrees(%x: tensor<16x64xf32>, %sx: f32, %zx: i8, %sw: f32, %w1: tensor<32x64xi8>) -> (tensor<16x32xf32>, tensor<16x32xf32>) {
    %w0 = arith.constant dense<3> : tensor<32x64xi8>
    %qi = tensor.empty() : tensor<16x64xi8>
    %xq = iree_linalg_ext.quantize_affine <indexing_maps = #maps_q, quant_min = 0, quant_max = 255, storage_unsigned, zp_unsigned>
        ins(%x, %sx, %zx : tensor<16x64xf32>, f32, i8) outs(%qi : tensor<16x64xi8>) -> tensor<16x64xi8>
    %xi = tensor.empty() : tensor<16x64xf32>
    %xd = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_q, input_unsigned, zp_unsigned>
        ins(%xq, %sx, %zx : tensor<16x64xi8>, f32, i8) outs(%xi : tensor<16x64xf32>) -> tensor<16x64xf32>
    %wi = tensor.empty() : tensor<32x64xf32>
    %wd0 = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_sym>
        ins(%w0, %sw : tensor<32x64xi8>, f32) outs(%wi : tensor<32x64xf32>) -> tensor<32x64xf32>
    %wd1 = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_sym, input_unsigned>
        ins(%w1, %sw : tensor<32x64xi8>, f32) outs(%wi : tensor<32x64xf32>) -> tensor<32x64xf32>
    %zero = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<16x32xf32>
    %f = linalg.fill ins(%zero : f32) outs(%e : tensor<16x32xf32>) -> tensor<16x32xf32>
    %mm0 = linalg.matmul indexing_maps = #matmul_nt
        ins(%xd, %wd0 : tensor<16x64xf32>, tensor<32x64xf32>) outs(%f : tensor<16x32xf32>) -> tensor<16x32xf32>
    %mm1 = linalg.matmul indexing_maps = #matmul_nt
        ins(%xd, %wd1 : tensor<16x64xf32>, tensor<32x64xf32>) outs(%f : tensor<16x32xf32>) -> tensor<16x32xf32>
    util.return %mm0, %mm1 : tensor<16x32xf32>, tensor<16x32xf32>
  }
}

// CHECK-LABEL: util.func public @contraction_disagrees(
//       CHECK:   iree_linalg_ext.quantize_affine
//  CHECK-SAME:     quant_min = 0, quant_max = 255, storage_unsigned, zp_unsigned>

// -----

#arm64 = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+dotprod", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, target_triple = "aarch64-unknown-unknown-eabi-elf"}>
#maps_sym = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
             affine_map<(d0, d1) -> (d0, d1)>]
#maps_sym3 = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> ()>,
              affine_map<(d0, d1, d2) -> (d0, d1, d2)>]
#matmul_nt = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>,
              affine_map<(m, n, k) -> (m, n)>]

// A convolution, which is never rebased, reads the same activation as a
// matmul, so the activation keeps its signedness.
module attributes {stream.affinity.default = #hal.device.affinity<@device>} {
  util.global private @device = #hal.device.target<"local", [#arm64]> : !hal.device
  util.func public @convolution_reader(%x: tensor<16x64xf32>, %sx: f32, %sw: f32, %filter: tensor<3x64x8xf32>) -> (tensor<16x32xf32>, tensor<1x14x8xf32>) {
    %w = arith.constant dense<3> : tensor<32x64xi8>
    %qi = tensor.empty() : tensor<16x64xi8>
    %xq = iree_linalg_ext.quantize_affine <indexing_maps = #maps_sym, quant_min = 0, quant_max = 255, storage_unsigned>
        ins(%x, %sx : tensor<16x64xf32>, f32) outs(%qi : tensor<16x64xi8>) -> tensor<16x64xi8>
    %xi = tensor.empty() : tensor<16x64xf32>
    %xd = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_sym, input_unsigned>
        ins(%xq, %sx : tensor<16x64xi8>, f32) outs(%xi : tensor<16x64xf32>) -> tensor<16x64xf32>
    %wi = tensor.empty() : tensor<32x64xf32>
    %wd = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_sym>
        ins(%w, %sw : tensor<32x64xi8>, f32) outs(%wi : tensor<32x64xf32>) -> tensor<32x64xf32>
    %zero = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<16x32xf32>
    %f = linalg.fill ins(%zero : f32) outs(%e : tensor<16x32xf32>) -> tensor<16x32xf32>
    %mm = linalg.matmul indexing_maps = #matmul_nt
        ins(%xd, %wd : tensor<16x64xf32>, tensor<32x64xf32>) outs(%f : tensor<16x32xf32>) -> tensor<16x32xf32>
    %xq3 = tensor.expand_shape %xq [[0, 1], [2]] output_shape [1, 16, 64] : tensor<16x64xi8> into tensor<1x16x64xi8>
    %xi3 = tensor.empty() : tensor<1x16x64xf32>
    %xd3 = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_sym3, input_unsigned>
        ins(%xq3, %sx : tensor<1x16x64xi8>, f32) outs(%xi3 : tensor<1x16x64xf32>) -> tensor<1x16x64xf32>
    %fqi = tensor.empty() : tensor<3x64x8xi8>
    %fq = iree_linalg_ext.quantize_affine <indexing_maps = #maps_sym3, quant_min = -127, quant_max = 127>
        ins(%filter, %sw : tensor<3x64x8xf32>, f32) outs(%fqi : tensor<3x64x8xi8>) -> tensor<3x64x8xi8>
    %cwi = tensor.empty() : tensor<3x64x8xf32>
    %cwd = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_sym3>
        ins(%fq, %sw : tensor<3x64x8xi8>, f32) outs(%cwi : tensor<3x64x8xf32>) -> tensor<3x64x8xf32>
    %ce = tensor.empty() : tensor<1x14x8xf32>
    %cf = linalg.fill ins(%zero : f32) outs(%ce : tensor<1x14x8xf32>) -> tensor<1x14x8xf32>
    %conv = linalg.conv_1d_nwc_wcf {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
        ins(%xd3, %cwd : tensor<1x16x64xf32>, tensor<3x64x8xf32>) outs(%cf : tensor<1x14x8xf32>) -> tensor<1x14x8xf32>
    util.return %mm, %conv : tensor<16x32xf32>, tensor<1x14x8xf32>
  }
}

// CHECK-LABEL: util.func public @convolution_reader(
//       CHECK:   iree_linalg_ext.quantize_affine
//  CHECK-SAME:     quant_min = 0, quant_max = 255, storage_unsigned>

// -----

#maps_q = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
           affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>]
#maps_sym = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>,
             affine_map<(d0, d1) -> (d0, d1)>]
#matmul_nt = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>,
              affine_map<(m, n, k) -> (m, n)>]

// Without a device there is no resolver to ask.
util.func public @no_device(%x: tensor<16x64xf32>, %sx: f32, %zx: i8, %sw: f32) -> tensor<16x32xf32> {
  %w = arith.constant dense<3> : tensor<32x64xi8>
  %qi = tensor.empty() : tensor<16x64xi8>
  %xq = iree_linalg_ext.quantize_affine <indexing_maps = #maps_q, quant_min = 0, quant_max = 255, storage_unsigned, zp_unsigned>
      ins(%x, %sx, %zx : tensor<16x64xf32>, f32, i8) outs(%qi : tensor<16x64xi8>) -> tensor<16x64xi8>
  %xi = tensor.empty() : tensor<16x64xf32>
  %xd = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_q, input_unsigned, zp_unsigned>
      ins(%xq, %sx, %zx : tensor<16x64xi8>, f32, i8) outs(%xi : tensor<16x64xf32>) -> tensor<16x64xf32>
  %wi = tensor.empty() : tensor<32x64xf32>
  %wd = iree_linalg_ext.dequantize_affine <indexing_maps = #maps_sym>
      ins(%w, %sw : tensor<32x64xi8>, f32) outs(%wi : tensor<32x64xf32>) -> tensor<32x64xf32>
  %zero = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<16x32xf32>
  %f = linalg.fill ins(%zero : f32) outs(%e : tensor<16x32xf32>) -> tensor<16x32xf32>
  %mm = linalg.matmul indexing_maps = #matmul_nt
      ins(%xd, %wd : tensor<16x64xf32>, tensor<32x64xf32>) outs(%f : tensor<16x32xf32>) -> tensor<16x32xf32>
  util.return %mm : tensor<16x32xf32>
}

// CHECK-LABEL: util.func public @no_device(
//       CHECK:   iree_linalg_ext.quantize_affine
//  CHECK-SAME:     storage_unsigned, zp_unsigned>
