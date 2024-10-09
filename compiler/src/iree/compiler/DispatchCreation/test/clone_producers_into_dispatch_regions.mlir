// RUN: iree-opt --split-input-file --iree-flow-enable-gather-fusion --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-clone-producers-into-dispatch-regions))" %s | FileCheck %s

util.func public @complex_element_type(%input: tensor<4xi32>, %table: tensor<8x2xcomplex<f32>>) -> tensor<4x2xcomplex<f32>> {
  %c4095 = arith.constant 4095 : i32
  %const = arith.constant dense<[
    [(0x7FC00000,0.000000e+00), (0x7FC00000,1.000000e+00)], [(0x7FC00000,2.000000e+00), (0x7FC00000,3.000000e+00)],
    [(0x7FC00000,4.000000e+00), (0x7FC00000,5.000000e+00)], [(0x7FC00000,6.000000e+00), (0x7FC00000,7.000000e+00)]
  ]> : tensor<4x2xcomplex<f32>>
  %empty = tensor.empty() : tensor<4x2xcomplex<f32>>
  %0 = flow.dispatch.region -> (tensor<4x2xcomplex<f32>>) {
    %generic = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%input, %const : tensor<4xi32>, tensor<4x2xcomplex<f32>>) outs(%empty : tensor<4x2xcomplex<f32>>) {
    ^bb0(%in0: i32, %in1: complex<f32>, %out: complex<f32>):
      %i1 = linalg.index 1 : index
      %i0 = arith.index_cast %in0 : i32 to index
      %extract = tensor.extract %table[%i0, %i1] : tensor<8x2xcomplex<f32>>
      %cmp = arith.cmpi sle, %in0, %c4095 : i32
      %select = arith.select %cmp, %extract, %in1 : complex<f32>
      linalg.yield %select : complex<f32>
    } -> tensor<4x2xcomplex<f32>>
    flow.return %generic : tensor<4x2xcomplex<f32>>
  }
  util.return %0 : tensor<4x2xcomplex<f32>>
}

// CHECK-LABEL: util.func public @complex_element_type
//       CHECK:   flow.dispatch.region
//       CHECK:     %[[EMPTY:.+]] = tensor.empty() : tensor<4x2xcomplex<f32>>
//       CHECK:     %[[CST:.+]] = arith.constant dense<{{.+}}> : tensor<4x2xcomplex<f32>>
//       CHECK:     linalg.generic
//  CHECK-SAME:       ins(%{{.+}}, %[[CST]] : tensor<4xi32>, tensor<4x2xcomplex<f32>>)
//  CHECK-SAME:       outs(%[[EMPTY]] : tensor<4x2xcomplex<f32>>)
//       CHECK:   flow.return

// -----

util.func public @complex_constant_clone(%input: tensor<4x2xcomplex<f32>>) -> tensor<4x2xcomplex<f32>> {
  %cst = complex.constant [1.000000e+00 : f32, 2.000000e+00 : f32] : complex<f32>
  %empty = tensor.empty() : tensor<4x2xcomplex<f32>>
  %0 = linalg.fill ins(%cst : complex<f32>) outs(%empty : tensor<4x2xcomplex<f32>>) -> tensor<4x2xcomplex<f32>>
  %1 = flow.dispatch.region -> (tensor<4x2xcomplex<f32>>) {
    %generic = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%input : tensor<4x2xcomplex<f32>>) outs(%0 : tensor<4x2xcomplex<f32>>) {
    ^bb0(%in: complex<f32>, %out: complex<f32>):
      %2 = complex.mul %in, %out : complex<f32>
      linalg.yield %2 : complex<f32>
    } -> tensor<4x2xcomplex<f32>>
    flow.return %generic : tensor<4x2xcomplex<f32>>
  }
  util.return %1 : tensor<4x2xcomplex<f32>>
}

// CHECK-LABEL: @complex_constant_clone
// CHECK: flow.dispatch.region
// CHECK: tensor.empty
// CHECK: complex.constant
// CHECK: linalg.fill
// CHECK: linalg.generic
// CHECK: complex.mul
// CHECK: linalg.yield
// CHECK: flow.return

// -----

util.func public @complex_create(%real : f32, %imag : f32, %input: tensor<4x2xcomplex<f32>>) -> tensor<4x2xcomplex<f32>> {
  %cst = complex.create %real, %imag : complex<f32>
  %empty = tensor.empty() : tensor<4x2xcomplex<f32>>
  %0 = linalg.fill ins(%cst : complex<f32>) outs(%empty : tensor<4x2xcomplex<f32>>) -> tensor<4x2xcomplex<f32>>
  %1 = flow.dispatch.region -> (tensor<4x2xcomplex<f32>>) {
    %generic = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%input : tensor<4x2xcomplex<f32>>) outs(%0 : tensor<4x2xcomplex<f32>>) {
    ^bb0(%in: complex<f32>, %out: complex<f32>):
      %2 = complex.mul %in, %out : complex<f32>
      linalg.yield %2 : complex<f32>
    } -> tensor<4x2xcomplex<f32>>
    flow.return %generic : tensor<4x2xcomplex<f32>>
  }
  util.return %1 : tensor<4x2xcomplex<f32>>
}

// CHECK-LABEL: @complex_create
// CHECK: flow.dispatch.region
// CHECK: tensor.empty
// CHECK: complex.create
// CHECK: linalg.fill
// CHECK: linalg.generic
// CHECK: complex.mul
// CHECK: linalg.yield
// CHECK: flow.return

// -----

util.func public @use_in_dispatch_count(%arg0: tensor<1xi32>, %arg1: tensor<1xi32>) -> tensor<i32> {
  %c1 = arith.constant 1 : index
  %c2_i32 = arith.constant 2 : i32
  %c0 = arith.constant 0 : index
  %2 = tensor.empty() : tensor<i32>
  %extracted = tensor.extract %arg0[%c1] : tensor<1xi32>
  %4 = flow.dispatch.region -> (tensor<i32>) {
    %6 = linalg.generic {indexing_maps = [affine_map<() -> ()>], iterator_types = []} outs(%2 : tensor<i32>) {
    ^bb0(%out: i32):
      %7 = arith.addi %extracted, %c2_i32 : i32
      linalg.yield %7 : i32
    } -> tensor<i32>
    flow.return %6 : tensor<i32>
  } count() -> (index, index, index) {
    flow.return %c1, %c1, %c1 : index, index, index
  }
  util.return %4 : tensor<i32>
}


// CHECK-LABEL: @use_in_dispatch_count
// CHECK: %[[C1:.+]] = arith.constant 1 : index
// CHECK: flow.dispatch.region
// CHECK: %[[C1_2:.+]] = arith.constant 1 : index
// CHECK: linalg.generic
// CHECK: count()
// CHECK: flow.return %[[C1]], %[[C1]], %[[C1]]

// -----

util.func public @clone_dequantization(%arg0: tensor<4096x32x128xi8>, %arg1: tensor<1x1x32x128xf32>, %arg2: tensor<4096x32xf32>, %arg3: tensor<4096x32xf32>) -> tensor<1x1x4096xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<1x1x4096xf32>
  %1 = tensor.empty() : tensor<4096x32x128xf32>
  %2 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x1x4096xf32>) -> tensor<1x1x4096xf32>
  %3 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2) -> (d0, d1)>,
                        affine_map<(d0, d1, d2) -> (d0, d1)>,
                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%arg0, %arg2, %arg3 : tensor<4096x32x128xi8>, tensor<4096x32xf32>, tensor<4096x32xf32>) outs(%1 : tensor<4096x32x128xf32>) {
  ^bb0(%in: i8, %in_0: f32, %in_1: f32, %out: f32):
    %5 = arith.extui %in : i8 to i32
    %6 = arith.uitofp %5 : i32 to f32
    %7 = arith.subf %6, %in_1 : f32
    %8 = arith.mulf %7, %in_0 : f32
    linalg.yield %8 : f32
  } -> tensor<4096x32x128xf32>
  %9 = flow.dispatch.region -> (tensor<1x1x4096xf32>) {
    %4 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>,
                          affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>,
                          affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>],
        iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]}
        ins(%arg1, %3 : tensor<1x1x32x128xf32>, tensor<4096x32x128xf32>) outs(%2 : tensor<1x1x4096xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %5 = arith.mulf %in, %in_0 : f32
      %6 = arith.addf %5, %out : f32
      linalg.yield %6 : f32
    } -> tensor<1x1x4096xf32>
    flow.return %4 : tensor<1x1x4096xf32>
  }
  util.return %9 : tensor<1x1x4096xf32>
}
//       CHECK: util.func public @clone_dequantization
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<4096x32x128xi8>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<1x1x32x128xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<4096x32xf32>
//  CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: tensor<4096x32xf32>
//       CHECK:   %[[DISP:.+]] = flow.dispatch.region -> (tensor<1x1x4096xf32>)
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0.000000e+00 : f32
//   CHECK-DAG:   %[[INIT1:.+]] = tensor.empty() : tensor<1x1x4096xf32>
//   CHECK-DAG:   %[[INIT0:.+]] = tensor.empty() : tensor<4096x32x128xf32>
//       CHECK:   %[[GEN0:.+]] = linalg.generic
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel"]
//  CHECK-SAME:       ins(%[[ARG0]], %[[ARG2]], %[[ARG3]] :
//  CHECK-SAME:       outs(%[[INIT0]] :
//       CHECK:   %[[FILL:.+]] = linalg.fill ins(%[[C0]]
//  CHECK-SAME:       outs(%[[INIT1]] :
//       CHECK:   %[[GEN1:.+]] = linalg.generic
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
//  CHECK-SAME:       ins(%[[ARG1]], %[[GEN0]] :
//  CHECK-SAME:       outs(%[[FILL]] :
//       CHECK:   flow.return %[[GEN1]] :
//       CHECK:   util.return %[[DISP]]

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
util.func public @clone_dequantization_like(%arg0: tensor<32x1x16x1x8xi16>, %arg1: tensor<32x344x16x32x8xi4>) -> tensor<32x1x344x1x32xi32> {
  %c0_i32 = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<32x1x16x1x8xi32>
  %1 = linalg.generic {indexing_maps = [#map, #map],
                        iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
                        ins(%arg0 : tensor<32x1x16x1x8xi16>) outs(%0 : tensor<32x1x16x1x8xi32>) {
  ^bb0(%in: i16, %out: i32):
    %7 = arith.extsi %in : i16 to i32
    linalg.yield %7 : i32
  } -> tensor<32x1x16x1x8xi32>
  %2 = tensor.empty() : tensor<32x344x16x32x8xi32>
  %3 = linalg.generic {indexing_maps = [#map, #map],
                        iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
                        ins(%arg1 : tensor<32x344x16x32x8xi4>) outs(%2 : tensor<32x344x16x32x8xi32>) {
  ^bb0(%in: i4, %out: i32):
    %7 = arith.extui %in : i4 to i32
    linalg.yield %7 : i32
  } -> tensor<32x344x16x32x8xi32>
  %4 = tensor.empty() : tensor<32x1x344x1x32xi32>
  %5 = linalg.fill ins(%c0_i32 : i32) outs(%4 : tensor<32x1x344x1x32xi32>) -> tensor<32x1x344x1x32xi32>
  %6 = flow.dispatch.region -> (tensor<32x1x344x1x32xi32>) {
    %7 = linalg.batch_mmt4d ins(%1, %3 : tensor<32x1x16x1x8xi32>, tensor<32x344x16x32x8xi32>) outs(%5 : tensor<32x1x344x1x32xi32>) -> tensor<32x1x344x1x32xi32>
    flow.return %7 : tensor<32x1x344x1x32xi32>
  }
  util.return %6 : tensor<32x1x344x1x32xi32>
}
//       CHECK: util.func public @clone_dequantization
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<32x1x16x1x8xi16>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<32x344x16x32x8xi4>
//       CHECK:   %[[DISP:.+]] = flow.dispatch.region -> (tensor<32x1x344x1x32xi32>)
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : i32
//   CHECK-DAG:   %[[INIT0:.+]] = tensor.empty() : tensor<32x1x16x1x8xi32>
//   CHECK-DAG:   %[[INIT1:.+]] = tensor.empty() : tensor<32x1x344x1x32xi32>
//   CHECK-DAG:   %[[INIT2:.+]] = tensor.empty() : tensor<32x344x16x32x8xi32>
//       CHECK:   %[[FILL:.+]] = linalg.fill ins(%[[C0]]
//  CHECK-SAME:       outs(%[[INIT1]] :
//       CHECK:   %[[GEN0:.+]] = linalg.generic
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
//  CHECK-SAME:       ins(%[[ARG0]] :
//  CHECK-SAME:       outs(%[[INIT0]] :
//       CHECK:   %[[GEN1:.+]] = linalg.generic
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
//  CHECK-SAME:       ins(%[[ARG1]] :
//  CHECK-SAME:       outs(%[[INIT2]] :
//       CHECK:   %[[MMT4D:.+]] = linalg.batch_mmt4d
//  CHECK-SAME:       ins(%[[GEN0]], %[[GEN1]] :
//  CHECK-SAME:       outs(%[[FILL]] :
//       CHECK:   flow.return %[[MMT4D]] :
//       CHECK:   util.return %[[DISP]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0)>
util.func public @dequant_like_extf_reduction(%arg0: tensor<11008x32x128xf16>) -> tensor<11008xf32> {
  %0 = tensor.empty() : tensor<11008x32x128xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<11008x32x128xf16>) outs(%0 : tensor<11008x32x128xf32>) {
  ^bb0(%in: f16, %out: f32):
    %4 = arith.extf %in : f16 to f32
    linalg.yield %4 : f32
  } -> tensor<11008x32x128xf32>
  %2 = tensor.empty() : tensor<11008xf32>
  %3 = flow.dispatch.region -> (tensor<11008xf32>) {
    %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction", "reduction"]} ins(%1 : tensor<11008x32x128xf32>) outs(%2 : tensor<11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      %5 = arith.addf %in, %out : f32
      linalg.yield %5 : f32
    } -> tensor<11008xf32>
    flow.return %4 : tensor<11008xf32>
  }
  util.return %3 : tensor<11008xf32>
}
//       CHECK: util.func public @dequant_like_extf_reduction
//       CHECK:   %[[DISP:.+]] = flow.dispatch.region -> (tensor<11008xf32>)
//       CHECK:   linalg.generic
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel"]
//       CHECK:   %[[GEN:.+]] = linalg.generic
//  CHECK-SAME:       iterator_types = ["parallel", "reduction", "reduction"]
//       CHECK:   flow.return %[[GEN]] :
//       CHECK:   util.return %[[DISP]]

// -----

#map1 = affine_map<(d0) -> (d0)>
util.func public @clone_elementwise_op_empty() -> tensor<1280xf32> {
  %0 = flow.tensor.constant #flow.parameter.named<"model"::"unet.time_embedding.linear_2.bias"> : tensor<1280xf16>
  %1 = tensor.empty() : tensor<1280xf32>
  %2 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%0 : tensor<1280xf16>) outs(%1 : tensor<1280xf32>) {
  ^bb0(%in: f16, %out: f32):
    %3 = arith.extf %in : f16 to f32
    linalg.yield %3 : f32
  } -> tensor<1280xf32>
  util.return %2 : tensor<1280xf32>
}
//      CHECK: util.func public @clone_elementwise_op_empty()
//      CHECK:   %[[RETURN:.+]] = flow.dispatch.region
//      CHECK:     %[[EMPTY:.+]] = tensor.empty()
//      CHECK:     %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:         outs(%[[EMPTY]] :
//      CHECK:     flow.return %[[GENERIC]]
//      CHECK:   util.return %[[RETURN]]

// -----

util.func public @clone_broadcast_dequant_op(
    %arg0 : tensor<10x20xi8>, %arg1 : tensor<2x10xi32>) -> tensor<2x10xi32> {
  %0 = tensor.empty() : tensor<2x10x20xi32>
  %1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%arg0 : tensor<10x20xi8>) outs(%0 : tensor<2x10x20xi32>) {
    ^bb0(%b0 : i8, %b1 : i32):
      %2 = arith.extsi %b0 : i8 to i32
      linalg.yield %2 : i32
  } -> tensor<2x10x20xi32>
  %2 = flow.dispatch.region -> (tensor<2x10xi32>) {
    %3 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                         affine_map<(d0, d1, d2) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%1 : tensor<2x10x20xi32>) outs(%arg1 : tensor<2x10xi32>) {
      ^bb0(%b0: i32, %b1 : i32) :
        %4 = arith.addi %b0, %b1 : i32
        linalg.yield %4 : i32
    } -> tensor<2x10xi32>
    flow.return %3 : tensor<2x10xi32>
  }
  util.return %2 : tensor<2x10xi32>
}
// CHECK-LABEL: func public @clone_broadcast_dequant_op(
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<10x20xi8>,
//  CHECK-SAME:     %[[ARG1:.+]]: tensor<2x10xi32>)
//       CHECK:   %[[RETURN:.+]] = flow.dispatch.region
//       CHECK:     %[[DEQUANT:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[ARG0]] :
//       CHECK:     %[[REDUCE:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[DEQUANT]] :
//       CHECK:     flow.return %[[REDUCE]]
//       CHECK:   return %[[RETURN]]

// -----

// Do no clone index cast operations when they are operands to the dispatch
util.func public @dont_clone_index_type_op(%arg0 : i64) -> tensor<?xf32> {
  %0 = arith.index_cast %arg0 : i64 to index
  %1 = flow.dispatch.region[] -> (tensor<?xf32>{%0}) {
    %2 = tensor.empty(%0) : tensor<?xf32>
    flow.return %2 : tensor<?xf32>
  }
  util.return %1 : tensor<?xf32>
}
// CHECK-LABEL: func public @dont_clone_index_type_op
//       CHECK:   arith.index_cast
//       CHECK:   flow.dispatch.region
//   CHECK-NOT:   arith.index_cast

// -----
// Do no clone index cast operations when they are in-direct operands to the dispatch
#map = affine_map<()[s0] -> (s0 * 12)>
util.func public @dont_clone_index_type_op_2(%arg0: i64) -> tensor<?xf32> {
  %0 = arith.index_cast %arg0 : i64 to index
  %1 = affine.apply #map()[%0]
  %2 = flow.dispatch.region -> (tensor<?xf32>{%1}) {
    %3 = tensor.empty(%1) : tensor<?xf32>
    flow.return %3 : tensor<?xf32>
  }
  util.return %2 : tensor<?xf32>
}
// CHECK-LABEL: func public @dont_clone_index_type_op_2
//       CHECK:   arith.index_cast
//       CHECK:   affine.apply
//       CHECK:   flow.dispatch.region
//   CHECK-NOT:   arith.index_cast
//   CHECK-NOT:   affine.apply

// -----

// Clone 'gather-like' operations
util.func public @clone_gather_like(%arg0: tensor<4x1x4xi64>, %arg1: tensor<16384x16x32x128xf16>, %arg2: f16, %arg3: i64, %arg4: tensor<4x4x16x32x128xf16>, %arg5: tensor<4x1x32x1x128xf16>) -> tensor<4x32x1x128xf16> {
  %0 = tensor.empty() : tensor<4x1x4x16x32x128xf16>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<4x1x4xi64>) outs(%0 : tensor<4x1x4x16x32x128xf16>) {
  ^bb0(%in: i64, %out: f16):
    %5 = arith.index_cast %in : i64 to index
    %6 = linalg.index 3 : index
    %7 = linalg.index 4 : index
    %8 = linalg.index 5 : index
    %extracted = tensor.extract %arg1[%5, %6, %7, %8] : tensor<16384x16x32x128xf16>
    linalg.yield %extracted : f16
  } -> tensor<4x1x4x16x32x128xf16>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<4x1x4xi64>) outs(%0 : tensor<4x1x4x16x32x128xf16>) {
  ^bb0(%in: i64, %out: f16):
    %5 = arith.addi %in, %arg3 : i64
    %6 = arith.index_cast %5 : i64 to index
    %7 = linalg.index 3 : index
    %8 = linalg.index 4 : index
    %9 = linalg.index 5 : index
    %extracted = tensor.extract %arg1[%6, %7, %8, %9] : tensor<16384x16x32x128xf16>
    linalg.yield %extracted : f16
  } -> tensor<4x1x4x16x32x128xf16>
  %3 = tensor.empty() : tensor<4x1x32x1x128xf16>
  %4 = flow.dispatch.region -> (tensor<4x1x32x1x128xf16>) {
    %5 = iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d5, d6, d2, d4)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d5, d6, d2, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> ()>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d7)>]} ins(%arg5, %1, %2, %arg2 : tensor<4x1x32x1x128xf16>, tensor<4x1x4x16x32x128xf16>, tensor<4x1x4x16x32x128xf16>, f16) outs(%3 : tensor<4x1x32x1x128xf16>) {
      ^bb0(%score: f16):
        iree_linalg_ext.yield %score: f16
    } -> tensor<4x1x32x1x128xf16>
    flow.return %5 : tensor<4x1x32x1x128xf16>
  }
  %collapsed = tensor.collapse_shape %4 [[0, 1], [2], [3], [4]] : tensor<4x1x32x1x128xf16> into tensor<4x32x1x128xf16>
  util.return %collapsed : tensor<4x32x1x128xf16>
}

// CHECK-LABEL:  util.func public @clone_gather_lik
//       CHECK:    %[[DISPATCH:.+]] = flow.dispatch.region
//       CHECK:      %[[GATHER0:.+]] = linalg.generic
//       CHECK:      %[[GATHER1:.+]] = linalg.generic
//       CHECK:      %[[ATTENTION:.+]] = iree_linalg_ext.attention
//       CHECK:        ins({{.*}}, %[[GATHER0]], %[[GATHER1]]
//       CHECK:      flow.return %[[ATTENTION]]

// -----

// Clone 'gather-like' operations
util.func public @clone_gather_like(%arg0: tensor<4x1x4xi64>, %arg1: tensor<16384x16x32x128xf16>, %arg2: f16, %arg3: i64, %arg4: tensor<4x4x16x32x128xf16>, %arg5: tensor<4x1x32x1x128xf16>) -> tensor<4x32x1x128xf16> {
  %0 = tensor.empty() : tensor<4x1x4x16x32x128xf16>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<4x1x4xi64>) outs(%0 : tensor<4x1x4x16x32x128xf16>) {
  ^bb0(%in: i64, %out: f16):
    %5 = arith.index_cast %in : i64 to index
    %6 = linalg.index 3 : index
    %7 = linalg.index 4 : index
    %8 = linalg.index 5 : index
    %extracted = tensor.extract %arg1[%5, %6, %7, %8] : tensor<16384x16x32x128xf16>
    linalg.yield %extracted : f16
  } -> tensor<4x1x4x16x32x128xf16>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<4x1x4xi64>) outs(%0 : tensor<4x1x4x16x32x128xf16>) {
  ^bb0(%in: i64, %out: f16):
    %5 = arith.addi %in, %arg3 : i64
    %6 = arith.index_cast %5 : i64 to index
    %7 = linalg.index 3 : index
    %8 = linalg.index 4 : index
    %9 = linalg.index 5 : index
    %extracted = tensor.extract %arg1[%6, %7, %8, %9] : tensor<16384x16x32x128xf16>
    linalg.yield %extracted : f16
  } -> tensor<4x1x4x16x32x128xf16>
  %3 = tensor.empty() : tensor<4x1x32x1x128xf16>
  %4 = flow.dispatch.region -> (tensor<4x1x32x1x128xf16>) {
    %5 = iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d5, d6, d2, d4)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d5, d6, d2, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> ()>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d7)>]} ins(%arg5, %1, %2, %arg2 : tensor<4x1x32x1x128xf16>, tensor<4x1x4x16x32x128xf16>, tensor<4x1x4x16x32x128xf16>, f16) outs(%3 : tensor<4x1x32x1x128xf16>) {
      ^bb0(%score: f16):
        iree_linalg_ext.yield %score: f16
    } -> tensor<4x1x32x1x128xf16>
    flow.return %5 : tensor<4x1x32x1x128xf16>
  }
  %collapsed = tensor.collapse_shape %4 [[0, 1], [2], [3], [4]] : tensor<4x1x32x1x128xf16> into tensor<4x32x1x128xf16>
  util.return %collapsed : tensor<4x32x1x128xf16>
}

// CHECK-LABEL:  util.func public @clone_gather_lik
//       CHECK:    %[[DISPATCH:.+]] = flow.dispatch.region
//       CHECK:      %[[GATHER0:.+]] = linalg.generic
//       CHECK:      %[[GATHER1:.+]] = linalg.generic
//       CHECK:      %[[ATTENTION:.+]] = iree_linalg_ext.attention
//       CHECK:        ins({{.*}}, %[[GATHER0]], %[[GATHER1]]
//       CHECK:      flow.return %[[ATTENTION]]

// -----

util.func public @clone_gather_like(%arg0: tensor<4x1x4xi64>, %arg1: tensor<16384x16x32x128xf16>, %arg2: f16, %arg3: i64, %arg4: tensor<4x4x16x32x128xf16>, %arg5: tensor<4x1x32x1x128xf16>, %arg6: tensor<4x1x32x128xf16>) -> tensor<4x32x1x128xf16> {
  %0 = tensor.empty() : tensor<4x1x4x16x32x128xf16>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<4x1x4xi64>) outs(%0 : tensor<4x1x4x16x32x128xf16>) {
  ^bb0(%in: i64, %out: f16):
    %5 = arith.index_cast %in : i64 to index
    %6 = linalg.index 3 : index
    %7 = linalg.index 4 : index
    %8 = linalg.index 5 : index
    %extracted = tensor.extract %arg1[%5, %6, %7, %8] : tensor<16384x16x32x128xf16>
    linalg.yield %extracted : f16
  } -> tensor<4x1x4x16x32x128xf16>

  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<4x1x4xi64>) outs(%0 : tensor<4x1x4x16x32x128xf16>) {
  ^bb0(%in: i64, %out: f16):
    %5 = arith.addi %in, %arg3 : i64
    %6 = arith.index_cast %5 : i64 to index
    %7 = linalg.index 3 : index
    %8 = linalg.index 4 : index
    %9 = linalg.index 5 : index
    %extracted = tensor.extract %arg1[%6, %7, %8, %9] : tensor<16384x16x32x128xf16>
    linalg.yield %extracted : f16
  } -> tensor<4x1x4x16x32x128xf16>

  %3 = tensor.empty() : tensor<4x1x32x1x128xf16>

  %4 = flow.dispatch.region -> (tensor<4x1x32x1x128xf16>) {
    %5 = iree_linalg_ext.attention {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d5, d6, d2, d4)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d5, d6, d2, d7)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> ()>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d4)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d7)>
      ]
    } ins(%arg5, %1, %2, %arg2, %arg6 : tensor<4x1x32x1x128xf16>, tensor<4x1x4x16x32x128xf16>, tensor<4x1x4x16x32x128xf16>, f16, tensor<4x1x32x128xf16>) outs(%3 : tensor<4x1x32x1x128xf16>) {
      ^bb0(%score: f16):
        iree_linalg_ext.yield %score: f16
    } -> tensor<4x1x32x1x128xf16>

    flow.return %5 : tensor<4x1x32x1x128xf16>
  }

  %collapsed = tensor.collapse_shape %4 [[0, 1], [2], [3], [4]] : tensor<4x1x32x1x128xf16> into tensor<4x32x1x128xf16>
  util.return %collapsed : tensor<4x32x1x128xf16>
}

// CHECK-LABEL:  util.func public @clone_gather_lik
//       CHECK:    %[[DISPATCH:.+]] = flow.dispatch.region
//       CHECK:      %[[GATHER0:.+]] = linalg.generic
//       CHECK:      %[[GATHER1:.+]] = linalg.generic
//       CHECK:      %[[ATTENTION:.+]] = iree_linalg_ext.attention
//       CHECK:        ins({{.*}}, %[[GATHER0]], %[[GATHER1]]
//       CHECK:      flow.return %[[ATTENTION]]
