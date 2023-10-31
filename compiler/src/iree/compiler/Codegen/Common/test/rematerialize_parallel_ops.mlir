// RUN: iree-opt -iree-codegen-rematerialize-parallel-ops --split-input-file %s | FileCheck %s

func.func @merged_reduction_parallel(%0: tensor<1x40960xf32>, %1: tensor<1xf32>, %7: tensor<1xf32>)
  -> tensor<1x40960xf32> {
   %res = flow.dispatch.region -> (tensor<1x40960xf32>) {
     %2 = tensor.empty() : tensor<1x40960xf32>
     %cst = arith.constant -3.40282347E+38 : f32
     %8 = linalg.generic
     {indexing_maps = [
         affine_map<(d0, d1) -> (d0, d1)>,
         affine_map<(d0, d1) -> (d0)>,
         affine_map<(d0, d1) -> (d0, d1)>],
         iterator_types = ["parallel", "parallel"]}
         ins(%0, %1 : tensor<1x40960xf32>, tensor<1xf32>)
         outs(%2 : tensor<1x40960xf32>) {
       ^bb0(%in: f32, %in_2: f32, %out: f32):
         %10 = arith.subf %in, %in_2 : f32
         %11 = math.exp %10 : f32
         linalg.yield %11 : f32
       } -> (tensor<1x40960xf32>)
     %9 = linalg.generic {
         indexing_maps = [
             affine_map<(d0, d1) -> (d0, d1)>,
             affine_map<(d0, d1) -> (d0)>,
             affine_map<(d0, d1) -> (d0, d1)>],
             iterator_types = ["parallel", "parallel"]}
             ins(%8, %7 : tensor<1x40960xf32>, tensor<1xf32>)
             outs(%2 : tensor<1x40960xf32>) attrs = {foo = "foo"} {
       ^bb0(%in: f32, %in_2: f32, %out: f32):
         %10 = arith.divf %cst, %in_2 : f32
         %11 = arith.mulf %in, %10 : f32
         linalg.yield %11 : f32
       } -> tensor<1x40960xf32>
     flow.return %9 : tensor<1x40960xf32>
   }
   return %res : tensor<1x40960xf32>
}


//   CHECK-LABEL: func.func @merged_reduction_parallel
//         CHECK:   %{{.+}} = linalg.generic
//    CHECK-SAME:       attrs = {foo = "foo"}
//         CHECK:     arith.subf
//    CHECK-NEXT:     math.exp
//    CHECK-NEXT:     arith.divf
//    CHECK-NEXT:     arith.mulf
//    CHECK-NEXT:     linalg.yield %{{.+}} : f32
//         CHECK:   } -> tensor<1x40960xf32>

// -----

func.func @softmax(%7 : tensor<16x32x4096xf32>) -> tensor<16x32x4096xf32> {
  %res = flow.dispatch.region -> (tensor<16x32x4096xf32>) {
    %cst = arith.constant -3.40282347E+38 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %8 = tensor.empty() : tensor<16x32xf32>
    %6 = tensor.empty() : tensor<16x32x4096xf32>
    %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%7 : tensor<16x32x4096xf32>) outs(%9 : tensor<16x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %16 = arith.maximumf %in, %out : f32
      linalg.yield %16 : f32
    } -> tensor<16x32xf32>
    %11 = tensor.empty() : tensor<16x32x4096xf32>
    %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%7, %10 : tensor<16x32x4096xf32>, tensor<16x32xf32>) outs(%11 : tensor<16x32x4096xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %16 = arith.subf %in, %in_2 : f32
      %17 = math.exp %16 : f32
      linalg.yield %17 : f32
    } -> tensor<16x32x4096xf32>
    %13 = linalg.fill ins(%cst_0 : f32) outs(%8 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%12 : tensor<16x32x4096xf32>) outs(%13 : tensor<16x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %16 = arith.addf %in, %out : f32
      linalg.yield %16 : f32
    } -> tensor<16x32xf32>
    %15 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%12, %14 : tensor<16x32x4096xf32>, tensor<16x32xf32>) outs(%6 : tensor<16x32x4096xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %16 = arith.divf %cst_1, %in_2 : f32
      %17 = arith.mulf %in, %16 : f32
      linalg.yield %17 : f32
    } -> tensor<16x32x4096xf32>
    flow.return %15 : tensor<16x32x4096xf32>
  }
  return %res : tensor<16x32x4096xf32>
}
//      CHECK: func @softmax(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<16x32x4096xf32>)
//  CHECK-DAG:   %[[CST0:.+]] = arith.constant 0.0
//      CHECK:   %[[MAXF:.+]] = linalg.generic
// CHECK-SAME:       ["parallel", "parallel", "reduction"]
// CHECK-SAME:       ins(%[[ARG0]] :
//      CHECK:   %[[FILL0:.+]] = linalg.fill
// CHECK-SAME:       ins(%[[CST0]] :
//      CHECK:   %[[EXPF:.+]] = linalg.generic
// CHECK-SAME:       ["parallel", "parallel", "reduction"]
// CHECK-SAME:       ins(%[[ARG0]], %[[MAXF]] :
//      CHECK:   %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:       ["parallel", "parallel", "parallel"]
// CHECK-SAME:       ins(%[[ARG0]], %[[MAXF]], %[[EXPF]] :
//      CHECK:   return %[[RESULT]]

// -----

func.func @no_rematerialize_scalar_ops(%arg0 : tensor<f32>) -> tensor<f32> {
  %res = flow.dispatch.region -> (tensor<f32>) {
    %0 = tensor.empty() : tensor<f32>
    %1 = linalg.generic {
        indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>],
        iterator_types = []}
        ins(%arg0: tensor<f32>) outs(%0 : tensor<f32>) {
      ^bb0(%b0 : f32, %b1 : f32):
        %2 = arith.addf %b0, %b0: f32
        linalg.yield %2: f32
    } -> tensor<f32>
    %3 = linalg.generic {
        indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>],
        iterator_types = []}
        ins(%1: tensor<f32>) outs(%0 : tensor<f32>) {
      ^bb0(%b0 : f32, %b1 : f32):
        %4 = arith.mulf %b0, %b0: f32
        linalg.yield %4: f32
    } -> tensor<f32>
    %5 = linalg.generic {
        indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>],
        iterator_types = []}
        ins(%1, %3 : tensor<f32>, tensor<f32>) outs(%0 : tensor<f32>) {
      ^bb0(%b0 : f32, %b1 : f32, %b2 : f32):
        %6 = arith.addf %b0, %b1: f32
        linalg.yield %6: f32
    } -> tensor<f32>
    flow.return %5 : tensor<f32>
  }
  return %res : tensor<f32>
}
// CHECK-LABEL: func @no_rematerialize_scalar_ops(
//       CHECK:   linalg.generic
//       CHECK:   linalg.generic
//       CHECK:   linalg.generic
