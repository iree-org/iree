// RUN: iree-opt --split-input-file --iree-preprocessing-transpose-extract-concat-pass %s | FileCheck %s

#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map20 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
util.func public @jit_eval_174(%1104 :  tensor<1024x7x7x2xi8>) -> tensor<1024x7x7x2xi8> {
    %cst_9 = arith.constant 2.00 : f32
    %cst_4 = arith.constant 4.00 : f32
    %cst_0 = arith.constant 0.00 : f32
    %cst_1 = arith.constant 1.00 : f32

    %1015 = tensor.empty() : tensor<1024x7x7x2xf32>
    %1105 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1104 : tensor<1024x7x7x2xi8>) outs(%1015 : tensor<1024x7x7x2xf32>) {
    ^bb0(%in: i8, %out: f32):
      %3555 = arith.extsi %in : i8 to i32
      %3556 = arith.sitofp %3555 : i32 to f32
      %3557 = arith.mulf %3556, %cst_9 : f32
      linalg.yield %3557 : f32
    } -> tensor<1024x7x7x2xf32>

    %cst_218 = arith.constant dense<1.000000e+00> : tensor<f32>

    %1020 = tensor.empty() : tensor<1024x7x7x1xi8>
    %1022 = tensor.empty() : tensor<1024x7x7x1xf32>
    %extracted_slice_466 = tensor.extract_slice %1105[0, 0, 0, 0] [1024, 7, 7, 1] [1, 1, 1, 1] : tensor<1024x7x7x2xf32> to tensor<1024x7x7x1xf32>
    %extracted_slice_467 = tensor.extract_slice %1105[0, 0, 0, 1] [1024, 7, 7, 1] [1, 1, 1, 1] : tensor<1024x7x7x2xf32> to tensor<1024x7x7x1xf32>

     %1106 = linalg.generic
     {indexing_maps = [#map20, #map2],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%extracted_slice_466 : tensor<1024x7x7x1xf32>)
      outs(%1020 : tensor<1024x7x7x1xi8>) {
    ^bb0(%in: f32, %out: i8):
      %3555 = arith.divf %in, %cst_9 : f32
      %3562 = arith.fptosi %3555 : f32 to i8
      linalg.yield %3562 : i8
    } -> tensor<1024x7x7x1xi8>

    %1108 = linalg.generic
      {indexing_maps = [#map20, #map2],
       iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%extracted_slice_467 : tensor<1024x7x7x1xf32>)
      outs(%1020 : tensor<1024x7x7x1xi8>) {
    ^bb0(%in: f32, %out: i8):
      %3555 = arith.divf %in, %cst_9 : f32
      %3562 = arith.fptosi %3555 : f32 to i8
      linalg.yield %3562 : i8
    } -> tensor<1024x7x7x1xi8>


    %concat_468 = tensor.concat dim(3) %1108, %1106 : (tensor<1024x7x7x1xi8>, tensor<1024x7x7x1xi8>) -> tensor<1024x7x7x2xi8>
    util.return %concat_468: tensor<1024x7x7x2xi8>
}


// CHECK-LABEL: util.func public @jit_eval_174
// CHECK:         %[[PRODUCER:.+]] = linalg.generic
// CHECK:         %[[TRANSPOSE1:.+]] = linalg.transpose
// CHECK-SAME:      ins(%[[PRODUCER]] : tensor<1024x7x7x2xf32>)
// CHECK-SAME:      outs(%[[EMPTY1:.+]] : tensor<2x1024x7x7xf32>)
// CHECK-DAG:         %[[SLICE1:.+]] = tensor.extract_slice %[[TRANSPOSE1]]
// CHECK-SAME       tensor<2x1024x7x7xf32> to tensor<1x1024x7x7xf32>
// CHECK-DAG:         %[[SLICE2:.+]] = tensor.extract_slice %[[TRANSPOSE1]]
// CHECK-SAME       tensor<2x1024x7x7xf32> to tensor<1x1024x7x7xf32>
// CHECK:         %[[GEN1:.+]] = linalg.generic
// CHECK-SAME:      ins(%[[SLICE1]] : tensor<1x1024x7x7xf32>)
// CHECK:         %[[GEN2:.+]] = linalg.generic
// CHECK-SAME:      ins(%[[SLICE2]] : tensor<1x1024x7x7xf32>)
// CHECK:         %[[CONCAT:.+]] = tensor.concat dim(0)
// CHECK:         %[[TRANSPOSE2:.+]] = linalg.transpose
// CHECK-SAME:      ins(%[[CONCAT]] : tensor<2x1024x7x7xi8>)
// CHECK-SAME:      outs(%[[EMPTY2:.+]] : tensor<1024x7x7x2xi8>)

// -----

util.func public @trivial_case(%arg0:  tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32> {
    %extracted1 = tensor.extract_slice %arg0[0, 0, 0, 0] [2, 3, 4, 3] [1, 1, 1, 1] : tensor<2x3x4x5xf32> to tensor<2x3x4x3xf32>
    %extracted2 = tensor.extract_slice %arg0[0, 0, 0, 3] [2, 3, 4, 2] [1, 1, 1, 1] : tensor<2x3x4x5xf32> to tensor<2x3x4x2xf32>
    %concat = tensor.concat dim(3) %extracted1, %extracted2 : (tensor<2x3x4x3xf32>, tensor<2x3x4x2xf32>) -> tensor<2x3x4x5xf32>
    util.return %concat : tensor<2x3x4x5xf32>
}


// CHECK-LABEL: util.func public @trivial_case
// CHECK:         linalg.transpose
// CHECK:         linalg.transpose

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> ()>
util.func public @multi_arg_generic(%arg0:  tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32> {
    %cst_9 = arith.constant 2.00 : f32
    %cst_4 = arith.constant 4.00 : f32

    %cst_220 = arith.constant dense<2.000000e+00> : tensor<f32>

    %extracted1 = tensor.extract_slice %arg0[0, 0, 0, 0] [2, 3, 4, 3] [1, 1, 1, 1] : tensor<2x3x4x5xf32> to tensor<2x3x4x3xf32>
    %extracted2 = tensor.extract_slice %arg0[0, 0, 0, 3] [2, 3, 4, 2] [1, 1, 1, 1] : tensor<2x3x4x5xf32> to tensor<2x3x4x2xf32>

    %empty1 = tensor.empty() : tensor<2x3x4x3xf32>
    %empty2 = tensor.empty() : tensor<2x3x4x2xf32>
    %generic1 = linalg.generic {
      indexing_maps = [#map, #map3, #map],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%extracted1, %cst_220: tensor<2x3x4x3xf32>,  tensor<f32>)
      outs(%empty1: tensor<2x3x4x3xf32>){
      ^bb0(%in: f32, %in2: f32, %out: f32):
        %res = arith.mulf %in, %in2 : f32
        linalg.yield %res : f32
    } -> tensor<2x3x4x3xf32>

    %generic2 = linalg.generic {
      indexing_maps = [#map, #map],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%extracted2: tensor<2x3x4x2xf32>)
      outs(%empty2: tensor<2x3x4x2xf32>){
      ^bb0(%in: f32, %out: f32):
        %res = arith.mulf %in, %cst_9 : f32
        linalg.yield %res : f32
    } -> tensor<2x3x4x2xf32>

    %concat = tensor.concat dim(3) %generic1, %generic2: (tensor<2x3x4x3xf32>, tensor<2x3x4x2xf32>) -> tensor<2x3x4x5xf32>
    util.return %concat : tensor<2x3x4x5xf32>
}

// CHECK-LABEL: util.func public @multi_arg_generic
// CHECK:         linalg.transpose
// CHECK:         linalg.transpose
