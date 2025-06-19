// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-clone-producers-into-dispatch-regions{aggressive=true}))" %s | FileCheck %s

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

util.func public @attention_clone_mask(%arg0: tensor<?x?xf16>,
    %arg1: tensor<?x?xf16>, %arg2: tensor<?x?xf16>) -> tensor<?x?xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x?xf16>
  %dim_0 = tensor.dim %arg1, %c1 : tensor<?x?xf16>
  %dim_1 = tensor.dim %arg2, %c1 : tensor<?x?xf16>
  %false = arith.constant false
  %true = arith.constant true
  %cst = arith.constant 1.000000e+00 : f16
  %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xi1>
  %1 = tensor.empty(%dim, %dim_1) : tensor<?x?xf16>
  %2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      outs(%0 : tensor<?x?xi1>) {
  ^bb0(%out: i1):
    %4 = linalg.index 0 : index
    %5 = linalg.index 1 : index
    %6 = arith.cmpi sge, %4, %5 : index
    %7 = arith.select %6, %false, %true : i1
    linalg.yield %7 : i1
  } -> tensor<?x?xi1>
  %3 = flow.dispatch.region -> (tensor<?x?xf16>{%dim, %dim_1}) {
    %4 = iree_linalg_ext.attention {
        indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3)>,
                         affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
                         affine_map<(d0, d1, d2, d3) -> (d2, d1)>,
                         affine_map<(d0, d1, d2, d3) -> ()>,
                         affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
                         affine_map<(d0, d1, d2, d3) -> (d0, d1)>]}
        ins(%arg0, %arg1, %arg2, %cst, %2 : tensor<?x?xf16>, tensor<?x?xf16>,
            tensor<?x?xf16>, f16, tensor<?x?xi1>) outs(%1 : tensor<?x?xf16>) {
    ^bb0(%arg3: f32):
      iree_linalg_ext.yield %arg3 : f32
    } -> tensor<?x?xf16>
    flow.return %4 : tensor<?x?xf16>
  }
  util.return %3 : tensor<?x?xf16>
}
// CHECK-LABEL: func public @attention_clone_mask
//       CHECK:   %[[DISPATCH:.+]] = flow.dispatch.region
//       CHECK:     %[[MASK:.+]] = linalg.generic
//       CHECK:     %[[ATTENTION:.+]] = iree_linalg_ext.attention
//  CHECK-SAME:         ins({{.+}}, %[[MASK]] :
//       CHECK:     flow.return %[[ATTENTION]]
//       CHECK:   return %[[DISPATCH]]

// -----

util.func public @dont_clone_flow_ops(%arg0: tensor<?x?xf16>, %arg1: tensor<?x?xf16>, %arg2: tensor<?x?xf16>, %arg3: tensor<?x?xi1>) -> tensor<?x?xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x?xf16>
  %dim_0 = tensor.dim %arg1, %c1 : tensor<?x?xf16>
  %dim_1 = tensor.dim %arg2, %c1 : tensor<?x?xf16>
  %dim_2 = tensor.dim %arg3, %c0 : tensor<?x?xi1>
  %dim_3 = tensor.dim %arg3, %c1 : tensor<?x?xi1>
  %false = arith.constant false
  %true = arith.constant true
  %cst = arith.constant 1.000000e+00 : f16
  %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xi1>
  %1 = tensor.empty(%dim, %dim_1) : tensor<?x?xf16>
  %2 = flow.tensor.transfer %arg3 : tensor<?x?xi1>{%dim_2, %dim_3} to #hal.device.promise<@dev_a>
  %3 = flow.dispatch.region -> (tensor<?x?xf16>{%dim, %dim_1}) {
    %4 = iree_linalg_ext.attention {
        indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3)>,
                         affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
                         affine_map<(d0, d1, d2, d3) -> (d2, d1)>,
                         affine_map<(d0, d1, d2, d3) -> ()>,
                         affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
                         affine_map<(d0, d1, d2, d3) -> (d0, d1)>]}
        ins(%arg0, %arg1, %arg2, %cst, %2 : tensor<?x?xf16>, tensor<?x?xf16>,
            tensor<?x?xf16>, f16, tensor<?x?xi1>) outs(%1 : tensor<?x?xf16>) {
    ^bb0(%in: f32):
      iree_linalg_ext.yield %in : f32
    } -> tensor<?x?xf16>
    flow.return %4 : tensor<?x?xf16>
  }
  util.return %3 : tensor<?x?xf16>
}
// CHECK-LABEL: func public @dont_clone_flow_ops
//       CHECK:   %[[MASK:.+]] = flow.tensor.transfer
//       CHECK:   %[[DISPATCH:.+]] = flow.dispatch.region
//       CHECK:     %[[ATTENTION:.+]] = iree_linalg_ext.attention
//  CHECK-SAME:         ins({{.+}}, %[[MASK]] :
//       CHECK:     flow.return %[[ATTENTION]]
//       CHECK:   return %[[DISPATCH]]

// -----

util.func public @clone_scatter_indices(%arg0: tensor<1x?x32x8x128xf8E4M3FNUZ>, %arg1: tensor<1x?x32x8x128xbf16>, %arg2: tensor<1x?xi64>, %arg3: tensor<1x?xi64>, %arg4: tensor<1x?xi32>, %arg5: tensor<?x32x8x128xbf16>) -> (tensor<?x32x8x128xbf16>, tensor<1x?xi64>) {
  %c1 = arith.constant 1 : index
  %c10_i64 = arith.constant 10 : i64
  %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x?x32x8x128xf8E4M3FNUZ>) outs(%arg1 : tensor<1x?x32x8x128xbf16>) {
  ^bb0(%in: f8E4M3FNUZ, %out: bf16):
    %3 = arith.extf %in : f8E4M3FNUZ to bf16
    linalg.yield %3 : bf16
  } -> tensor<1x?x32x8x128xbf16>
  %1:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg2 : tensor<1x?xi64>) outs(%arg3, %arg4 : tensor<1x?xi64>, tensor<1x?xi32>) {
  ^bb0(%in: i64, %out: i64, %out_0: i32):
    %3 = arith.addi %in, %c10_i64 : i64
    %4 = arith.trunci %3 : i64 to i32
    linalg.yield %3, %4 : i64, i32
  } -> (tensor<1x?xi64>, tensor<1x?xi32>)
  %2 = flow.dispatch.region -> (tensor<?x32x8x128xbf16>{%c1}) {
    %3 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true) ins(%0, %1#1 : tensor<1x?x32x8x128xbf16>, tensor<1x?xi32>) outs(%arg5 : tensor<?x32x8x128xbf16>) {
    ^bb0(%arg6: bf16, %arg7: bf16):
      iree_linalg_ext.yield %arg6 : bf16
    } -> tensor<?x32x8x128xbf16>
    flow.return %3 : tensor<?x32x8x128xbf16>
  }
  util.return %2, %1#0 : tensor<?x32x8x128xbf16>, tensor<1x?xi64>
}

// CHECK-LABEL: func public @clone_scatter_indices
//       CHECK:   %[[DISPATCH0:.+]] = flow.dispatch.region
//       CHECK:     %[[GEN0:.+]]:2 = linalg.generic
//       CHECK:     flow.return %[[GEN0]]#0 : tensor<1x?xi64>
//       CHECK:   %[[DISPATCH1:.+]] = flow.dispatch.region
//       CHECK:     %[[GEN1:.+]] = linalg.generic
//       CHECK:     %[[GEN2:.+]]:2 = linalg.generic
//       CHECK:     %[[SCATTER:.+]] = iree_linalg_ext.scatter
//  CHECK-SAME:       ins(%[[GEN1]], %[[GEN2]]#1
//       CHECK:     flow.return %[[SCATTER]] : tensor<?x32x8x128xbf16>
//       CHECK:   util.return %[[DISPATCH1]], %[[DISPATCH0]]

// -----

// Still fuse Q, K, and V in the case of bit-extend ops.
util.func @attention_bitextend_fusion(%arg0: tensor<10x20x30x50xf8E4M3FNUZ>,
    %arg1: tensor<10x20x40x50xf8E4M3FNUZ>, %arg2: tensor<10x20x40x50xf8E4M3FNUZ>,
    %cst : bf16) -> tensor<10x20x30x40xbf16> {
  %query_empty = tensor.empty() : tensor<10x20x30x50xbf16>
  %query = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%arg0 : tensor<10x20x30x50xf8E4M3FNUZ>)
      outs(%query_empty : tensor<10x20x30x50xbf16>) {
      ^bb0(%b0: f8E4M3FNUZ, %out: bf16):
      %val = arith.extf %b0 : f8E4M3FNUZ to bf16
      linalg.yield %val : bf16
  } -> tensor<10x20x30x50xbf16>
  %key_empty = tensor.empty() : tensor<10x20x40x50xbf16>
  %key = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%arg1 : tensor<10x20x40x50xf8E4M3FNUZ>)
      outs(%key_empty : tensor<10x20x40x50xbf16>) {
      ^bb0(%b0: f8E4M3FNUZ, %out: bf16):
      %val = arith.extf %b0 : f8E4M3FNUZ to bf16
      linalg.yield %val : bf16
  } -> tensor<10x20x40x50xbf16>
  %value_empty = tensor.empty() : tensor<10x20x40x50xbf16>
  %value = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%arg2 : tensor<10x20x40x50xf8E4M3FNUZ>)
      outs(%value_empty : tensor<10x20x40x50xbf16>) {
      ^bb0(%b0: f8E4M3FNUZ, %out: bf16):
      %val = arith.extf %b0 : f8E4M3FNUZ to bf16
      linalg.yield %val : bf16
  } -> tensor<10x20x40x50xbf16>
  %empty = tensor.empty() : tensor<10x20x30x40xbf16>
  %dispatch = flow.dispatch.region -> (tensor<10x20x30x40xbf16>) {
    %attention = iree_linalg_ext.attention {
        indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d4)>,
                         affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d3, d4)>,
                         affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d3, d4)>,
                         affine_map<(d0, d1, d2, d3, d4, d5, d6) -> ()>,
                         affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>]}
        ins(%query, %key, %value, %cst
            : tensor<10x20x30x50xbf16>, tensor<10x20x40x50xbf16>, tensor<10x20x40x50xbf16>, bf16)
        outs(%empty : tensor<10x20x30x40xbf16>) {
      ^bb0(%arg6: f32):
        iree_linalg_ext.yield %arg6 : f32
    } -> tensor<10x20x30x40xbf16>
    flow.return %attention : tensor<10x20x30x40xbf16>
  }
  util.return %dispatch : tensor<10x20x30x40xbf16>
}
// CHECK-LABEL: func public @attention_bitextend_fusion
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<10x20x30x50xf8E4M3FNUZ>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<10x20x40x50xf8E4M3FNUZ>
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<10x20x40x50xf8E4M3FNUZ>
//       CHECK:   %[[DISPATCH:.+]] = flow.dispatch.region
//       CHECK:     %[[V:.+]] = linalg.generic
//       CHECK:     %[[Q:.+]] = linalg.generic
//       CHECK:     %[[K:.+]] = linalg.generic
//       CHECK:     %[[ATTENTION:.+]] = iree_linalg_ext.attention
//  CHECK-SAME:         ins(%[[Q]], %[[K]], %[[V]]
//       CHECK:     flow.return %[[ATTENTION]]
//       CHECK:   util.return %[[DISPATCH]]

// -----

#encoding = #iree_encoding.testing<>
util.func public @unset_encoding_elementwise_fusion(%arg0: tensor<?x?xf32, #encoding>, %arg1: tensor<?xf32>, %arg2: index, %arg3: index) -> tensor<?x?xf32> {
  %0 = iree_encoding.unset_encoding %arg0 : tensor<?x?xf32, #encoding> -> tensor<?x?xf32>{%arg2, %arg3}
  %1 = tensor.empty(%arg2, %arg3) : tensor<?x?xf32>
  %2 = flow.dispatch.region -> (tensor<?x?xf32>{%arg2, %arg3}) {
    %3 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%0, %arg1 : tensor<?x?xf32>, tensor<?xf32>)
      outs(%1 : tensor<?x?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %4 = arith.addf %in, %in_0 : f32
      linalg.yield %4 : f32
    } -> tensor<?x?xf32>
    flow.return %3 : tensor<?x?xf32>
  }
  util.return %2 : tensor<?x?xf32>
}
// CHECK-LABEL: util.func public @unset_encoding_elementwise_fusion(
//       CHECK: %[[DISPATCH:.+]] = flow.dispatch.region
//       CHECK:   %[[UNSET_ENCODING:.+]] = iree_encoding.unset_encoding
//  CHECK-NEXT:   %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:     ins(%[[UNSET_ENCODING]]
//       CHECK:   flow.return %[[GENERIC]]

// -----

// Check that hal ops are never cloned into dispatches
util.func @never_clone_hal_ops(%arg0 : !hal.buffer_view, %arg1 : !hal.fence,
    %arg2 : index, %arg3 : index, %arg4 : index, %update : tensor<?x?xf32>,
    %original : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = hal.tensor.import wait(%arg1) => %arg0 : !hal.buffer_view -> tensor<?xi32>{%arg2}
  %1 = flow.dispatch.region -> (tensor<?x?xf32>{%arg3, %arg4}) {
    %2 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true)
        ins(%update, %0 : tensor<?x?xf32>, tensor<?xi32>)
        outs(%original: tensor<?x?xf32>) {
      ^bb0(%b0: f32, %b1: f32):
        %1 = arith.addf %b0, %b1 : f32
        iree_linalg_ext.yield %1 : f32
    } -> tensor<?x?xf32>
    flow.return %2 : tensor<?x?xf32>
  }
  util.return %1 : tensor<?x?xf32>
}
// CHECK-LABEL: @never_clone_hal_ops
//       CHECK:   hal.tensor.import
//       CHECK:   %[[DISPATCH:.+]] = flow.dispatch.region
//   CHECK-NOT:   hal.tensor.import
//       CHECK:   return %[[DISPATCH]]

// -----

// Check that flow ops are never cloned into dispatches
util.func @never_clone_flow_ops(%arg0 : tensor<?xi32>,
    %arg2 : index, %arg3 : index, %arg4 : index, %update : tensor<?x?xf32>,
    %original : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = flow.tensor.transfer %arg0 : tensor<?xi32>{%arg2} to "target"
  %1 = flow.dispatch.region -> (tensor<?x?xf32>{%arg3, %arg4}) {
    %2 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true)
        ins(%update, %0 : tensor<?x?xf32>, tensor<?xi32>)
        outs(%original: tensor<?x?xf32>) {
      ^bb0(%b0: f32, %b1: f32):
        %1 = arith.addf %b0, %b1 : f32
        iree_linalg_ext.yield %1 : f32
    } -> tensor<?x?xf32>
    flow.return %2 : tensor<?x?xf32>
  }
  util.return %1 : tensor<?x?xf32>
}
// CHECK-LABEL: @never_clone_flow_ops
//       CHECK:   flow.tensor.transfer
//       CHECK:   %[[DISPATCH:.+]] = flow.dispatch.region
//   CHECK-NOT:   flow.tensor.transfer
//       CHECK:   return %[[DISPATCH]]

// -----

// Check that insert slices are never cloned
util.func @never_clone_insert_slice_ops(%arg0 : tensor<?xi32>, %arg1 : tensor<?xi32>,
    %arg2 : index, %arg3 : index, %arg4 : index, %update : tensor<?x?xf32>,
    %original : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tensor.insert_slice %arg0 into %arg1[0] [%arg2] [1]: tensor<?xi32> into tensor<?xi32>
  %1 = flow.dispatch.region -> (tensor<?x?xf32>{%arg3, %arg4}) {
    %2 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true)
        ins(%update, %0 : tensor<?x?xf32>, tensor<?xi32>)
        outs(%original: tensor<?x?xf32>) {
      ^bb0(%b0: f32, %b1: f32):
        %1 = arith.addf %b0, %b1 : f32
        iree_linalg_ext.yield %1 : f32
    } -> tensor<?x?xf32>
    flow.return %2 : tensor<?x?xf32>
  }
  util.return %1 : tensor<?x?xf32>
}
// CHECK-LABEL: @never_clone_insert_slice_ops
//       CHECK:   tensor.insert_slice
//       CHECK:   %[[DISPATCH:.+]] = flow.dispatch.region
//   CHECK-NOT:   tensor.insert_slice
//       CHECK:   return %[[DISPATCH]]

util.func @clone_gather_elementwise(%source : tensor<2x2x2048xi32>,
                                   %indices : tensor<2xi32>) -> tensor<2048xi32> {
  %empty = tensor.empty() : tensor<2048xi32>
  %result = iree_linalg_ext.gather dimension_map = [0, 1]
                          ins(%source, %indices : tensor<2x2x2048xi32>, tensor<2xi32>)
                          outs(%empty: tensor<2048xi32>) -> tensor<2048xi32>
  %dispatch = flow.dispatch.region -> (tensor<2048xi32>) {
    %cst = arith.constant 2 : i32
    %generic = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%result : tensor<2048xi32>) outs(%empty: tensor<2048xi32>) {
      ^bb0(%arg0: i32, %arg1: i32):
        %0 = arith.muli %arg0, %cst : i32
        linalg.yield %0 : i32
    } -> tensor<2048xi32>
    flow.return %generic : tensor<2048xi32>
  }
  util.return %dispatch : tensor<2048xi32>
}
// CHECK-LABEL: func public @clone_gather_elementwise
// CHECK-SAME:     %[[SOURCE:.+]]: tensor<2x2x2048xi32>
// CHECK-SAME:     %[[INDICES:.+]]: tensor<2xi32>
//      CHECK:   %[[DISPATCH:.+]] = flow.dispatch.region
//      CHECK:     %[[GATHER:.+]] = iree_linalg_ext.gather
// CHECK-SAME:         ins(%[[SOURCE]], %[[INDICES]] :
//      CHECK:     %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:         ins(%[[GATHER]] :
//      CHECK:     flow.return %[[GENERIC]]
//      CHECK:   util.return %[[DISPATCH]]

// -----

util.func public @clone_linalg_ext_gather_attention(
  %arg0: tensor<?x32x2x32x8x128xf16>, %arg1: tensor<8x4x1x128xf16>,
  %arg2: tensor<8x4x1x?x32xf16>, %arg3: index,
  %arg4: tensor<1x?xi64>, %arg5: index) -> tensor<8x4x1x128xf16> {
  %0 = tensor.empty(%arg5) : tensor<1x?x32x8x128xf16>
  %extracted_slice = tensor.extract_slice %arg0[0, 31, 1, 0, 0, 0] [%arg3, 1, 1, 32, 8, 128] [1, 1, 1, 1, 1, 1] : tensor<?x32x2x32x8x128xf16> to tensor<?x32x8x128xf16>
  %extracted_slice_0 = tensor.extract_slice %arg0[0, 31, 0, 0, 0, 0] [%arg3, 1, 1, 32, 8, 128] [1, 1, 1, 1, 1, 1] : tensor<?x32x2x32x8x128xf16> to tensor<?x32x8x128xf16>
  %1 = iree_linalg_ext.gather dimension_map = [0]
    ins(%extracted_slice_0, %arg4 : tensor<?x32x8x128xf16>, tensor<1x?xi64>)
    outs(%0 : tensor<1x?x32x8x128xf16>) -> tensor<1x?x32x8x128xf16>
  %2 = iree_linalg_ext.gather dimension_map = [0]
    ins(%extracted_slice, %arg4 : tensor<?x32x8x128xf16>, tensor<1x?xi64>)
    outs(%0 : tensor<1x?x32x8x128xf16>) -> tensor<1x?x32x8x128xf16>

  %3 = flow.dispatch.region -> (tensor<8x4x1x128xf16>) {
    %4 = tensor.empty() : tensor<8x4x1x128xf16>
    %cst = arith.constant 8.837890e-02 : f16
    %5 = iree_linalg_ext.attention {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d4)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d5, d6, d7, d0, d4)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d5, d6, d7, d0, d3)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> ()>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d5, d6, d7)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3)>]}
      ins(%arg1, %1, %2, %cst, %arg2 : tensor<8x4x1x128xf16>, tensor<1x?x32x8x128xf16>, tensor<1x?x32x8x128xf16>, f16, tensor<8x4x1x?x32xf16>)
      outs(%4 : tensor<8x4x1x128xf16>) {
    ^bb0(%arg6: f32):
      iree_linalg_ext.yield %arg6 : f32
    } -> tensor<8x4x1x128xf16>
    flow.return %5 : tensor<8x4x1x128xf16>
  }
  util.return %3 : tensor<8x4x1x128xf16>
}
// CHECK-LABEL: func public @clone_linalg_ext_gather_attention
// CHECK-SAME:     %[[ARG0:.+]]: tensor<?x32x2x32x8x128xf16>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<8x4x1x128xf16>
//      CHECK:   %[[DISPATCH:.+]] = flow.dispatch.region
//  CHECK-DAG:     %[[EXTRACTK:.+]] = tensor.extract_slice %[[ARG0]][0, 31, 0, 0, 0, 0]
//  CHECK-DAG:     %[[EXTRACTV:.+]] = tensor.extract_slice %[[ARG0]][0, 31, 1, 0, 0, 0]
//  CHECK-DAG:     %[[GATHERK:.+]] = iree_linalg_ext.gather
// CHECK-SAME:         ins(%[[EXTRACTK]]
//  CHECK-DAG:     %[[GATHERV:.+]] = iree_linalg_ext.gather
// CHECK-SAME:         ins(%[[EXTRACTV]]
//      CHECK:     %[[ATTN:.+]] = iree_linalg_ext.attention
// CHECK-SAME:       ins(%[[ARG1]], %[[GATHERK]], %[[GATHERV]]
//      CHECK:     flow.return %[[ATTN]]
//      CHECK:   util.return %[[DISPATCH]]

// -----

// Don't clone 'gather-like' operations (only `iree_linalg_ext.gather`) is supported.
util.func public @dont_clone_gather_like(%arg0: tensor<4x1x4xi64>, %arg1: tensor<16384x16x32x128xf16>, %arg2: f16, %arg3: i64, %arg4: tensor<4x4x16x32x128xf16>, %arg5: tensor<4x1x32x1x128xf16>) -> tensor<4x32x1x128xf16> {
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
        affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d7)>]}
      ins(%arg5, %1, %2, %arg2 : tensor<4x1x32x1x128xf16>, tensor<4x1x4x16x32x128xf16>, tensor<4x1x4x16x32x128xf16>, f16)
      outs(%3 : tensor<4x1x32x1x128xf16>) {
      ^bb0(%score: f16):
        iree_linalg_ext.yield %score: f16
    } -> tensor<4x1x32x1x128xf16>
    flow.return %5 : tensor<4x1x32x1x128xf16>
  }
  %collapsed = tensor.collapse_shape %4 [[0, 1], [2], [3], [4]] : tensor<4x1x32x1x128xf16> into tensor<4x32x1x128xf16>
  util.return %collapsed : tensor<4x32x1x128xf16>
}
// CHECK-LABEL:  util.func public @dont_clone_gather_like
//   CHECK-DAG:    %[[DISPATCHK:.+]] = flow.dispatch.region
//   CHECK-DAG:    %[[DISPATCHV:.+]] = flow.dispatch.region
//       CHECK:    %[[DISPATCH:.+]] = flow.dispatch.region
//       CHECK:      %[[ATTENTION:.+]] = iree_linalg_ext.attention
//       CHECK:        ins({{.*}}, %[[DISPATCHK]], %[[DISPATCHV]]
//       CHECK:      flow.return %[[ATTENTION]]
