// RUN: iree-opt -iree-codegen-rematerialize-parallel-ops %s | FileCheck %s

func.func @merged_reduction_parallel(%0: tensor<1x40960xf32>, %1: tensor<1xf32>, %7: tensor<1xf32>)
  -> tensor<1x40960xf32> {
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
            outs(%2 : tensor<1x40960xf32>) {
      ^bb0(%in: f32, %in_2: f32, %out: f32):
        %10 = arith.divf %cst, %in_2 : f32
        %11 = arith.mulf %in, %10 : f32
        linalg.yield %11 : f32
      } -> tensor<1x40960xf32>
   return %9 : tensor<1x40960xf32>
}


//   CHECK-LABEL: func.func @merged_reduction_parallel
//         CHECK:   %{{.+}} = linalg.generic
//         CHECK:     arith.subf
//    CHECK-NEXT:     math.exp
//    CHECK-NEXT:     arith.divf
//    CHECK-NEXT:     arith.mulf
//    CHECK-NEXT:     linalg.yield %{{.+}} : f32
//         CHECK:   } -> tensor<1x40960xf32>
