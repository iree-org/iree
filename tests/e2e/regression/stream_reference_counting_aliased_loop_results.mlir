// RUN: iree-run-mlir --device=local-task --Xcompiler,iree-hal-target-device=local --Xcompiler,iree-hal-local-target-device-backends=llvm-cpu --Xcompiler,iree-llvmcpu-target-cpu=generic --input=1x240x320x3xi8=0 %s | FileCheck %s

// End-to-end regression test for the Stream AutomaticReferenceCounting (ARC)
// pass emitting a double `stream.resource.dealloca` for control-flow results
// that path-dependently alias one another.
//
// The loop carries two transient resources as iter_args
// (a scalar counter and an accumulator) and, inside a data-dependent scf.if,
// yields values that alias those same iter_args (the `else` branch forwards
// them unchanged; the `then` branch selects between the incoming accumulator
// and a slice-updated copy of it). ARC previously failed to notice that sibling
// results of the same multi-result op (here the scf.if) can alias, and
// inserted a dealloca for each loop result independently.
// Leading to: FAILED_PRECONDITION; transient buffer has not been committed


#map = affine_map<(d0, d1, d2) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<() -> ()>
func.func @main(%arg0: tensor<?x240x320x3xi8>) -> tensor<?x6xi8> {
  %c100_i32 = arith.constant 100 : i32
  %cst = arith.constant dense<1> : tensor<i32>
  %cst_0 = arith.constant dense<0> : tensor<i32>
  %c4420 = arith.constant 4420 : index
  %c0 = arith.constant 0 : index
  %cst_1 = arith.constant dense<0> : tensor<1x4420x1xi32>
  %c1 = arith.constant 1 : index
  %0 = tensor.empty() : tensor<4420xi32>
  %1 = tensor.empty() : tensor<i32>
  %2 = tensor.empty() : tensor<4420x6xi8>
  %3 = tensor.empty() : tensor<4420xi1>
  %expanded = tensor.expand_shape %3 [[0, 1, 2]] output_shape [1, 4420, 1] : tensor<4420xi1> into tensor<1x4420x1xi1>
  // Loop carries two aliasable transients: %arg2 (counter) and %arg3 (accumulator).
  %4:2 = scf.for %arg1 = %c0 to %c4420 step %c1 iter_args(%arg2 = %cst_0, %arg3 = %cst_1) -> (tensor<i32>, tensor<1x4420x1xi32>) {
    %8 = arith.index_cast %arg1 : index to i32
    %from_elements = tensor.from_elements %8 : tensor<1x1xi32>
    %expanded_5 = tensor.expand_shape %0 [[0, 1, 2]] output_shape [1, 4420, 1] : tensor<4420xi32> into tensor<1x4420x1xi32>
    %9 = tensor.empty() : tensor<1x1x1xi32>
    %10 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%from_elements : tensor<1x1xi32>) outs(%9 : tensor<1x1x1xi32>) {
    ^bb0(%in: i32, %out: i32):
      %14 = arith.index_cast %in : i32 to index
      %extracted_9 = tensor.extract %expanded_5[%c0, %14, %c0] : tensor<1x4420x1xi32>
      linalg.yield %extracted_9 : i32
    } -> tensor<1x1x1xi32>
    %collapsed_6 = tensor.collapse_shape %10 [[0], [1, 2]] : tensor<1x1x1xi32> into tensor<1x1xi32>
    %11 = tensor.empty() : tensor<1x1x1xi1>
    %12 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_6 : tensor<1x1xi32>) outs(%11 : tensor<1x1x1xi1>) {
    ^bb0(%in: i32, %out: i1):
      %14 = arith.index_cast %in : i32 to index
      %extracted_9 = tensor.extract %expanded[%c0, %14, %c0] : tensor<1x4420x1xi1>
      linalg.yield %extracted_9 : i1
    } -> tensor<1x1x1xi1>
    %collapsed_7 = tensor.collapse_shape %12 [[0, 1, 2]] : tensor<1x1x1xi1> into tensor<1xi1>
    %extracted_8 = tensor.extract %collapsed_7[%c0] : tensor<1xi1>
    // Sibling results %13#0 / %13#1 path-dependently alias %arg2 / %arg3.
    %13:2 = scf.if %extracted_8 -> (tensor<i32>, tensor<1x4420x1xi32>) {
      %expanded_9 = tensor.expand_shape %arg2 [] output_shape [1, 1] : tensor<i32> into tensor<1x1xi32>
      %extracted_10 = tensor.extract %expanded_9[%c0, %c0] : tensor<1x1xi32>
      %14 = arith.index_cast %extracted_10 : i32 to index
      %inserted_slice = tensor.insert_slice %10 into %arg3[0, %14, 0] [1, 1, 1] [1, 1, 1] : tensor<1x1x1xi32> into tensor<1x4420x1xi32>
      %15 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = []} ins(%arg2, %cst : tensor<i32>, tensor<i32>) outs(%1 : tensor<i32>) {
      ^bb0(%in: i32, %in_12: i32, %out: i32):
        %18 = arith.addi %in, %in_12 : i32
        linalg.yield %18 : i32
      } -> tensor<i32>
      %extracted_11 = tensor.extract %15[] : tensor<i32>
      %16 = arith.cmpi sgt, %extracted_11, %c100_i32 : i32
      %17 = arith.select %16, %arg3, %inserted_slice : tensor<1x4420x1xi32>
      scf.yield %15, %17 : tensor<i32>, tensor<1x4420x1xi32>
    } else {
      scf.yield %arg2, %arg3 : tensor<i32>, tensor<1x4420x1xi32>
    }
    scf.yield %13#0, %13#1 : tensor<i32>, tensor<1x4420x1xi32>
  }
  %collapsed = tensor.collapse_shape %4#1 [[0, 1, 2]] : tensor<1x4420x1xi32> into tensor<4420xi32>
  %extracted = tensor.extract %4#0[] : tensor<i32>
  %5 = arith.index_cast %extracted : i32 to index
  %extracted_slice = tensor.extract_slice %collapsed[0] [%5] [1] : tensor<4420xi32> to tensor<?xi32>
  %expanded_2 = tensor.expand_shape %2 [[0, 1], [2]] output_shape [1, 4420, 6] : tensor<4420x6xi8> into tensor<1x4420x6xi8>
  %expanded_3 = tensor.expand_shape %extracted_slice [[0, 1]] output_shape [1, %5] : tensor<?xi32> into tensor<1x?xi32>
  %dim = tensor.dim %expanded_3, %c1 : tensor<1x?xi32>
  %6 = tensor.empty(%dim) : tensor<1x?x6xi8>
  %7 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_3 : tensor<1x?xi32>) outs(%6 : tensor<1x?x6xi8>) {
  ^bb0(%in: i32, %out: i8):
    %8 = arith.index_cast %in : i32 to index
    %9 = linalg.index 2 : index
    %extracted_5 = tensor.extract %expanded_2[%c0, %8, %9] : tensor<1x4420x6xi8>
    linalg.yield %extracted_5 : i8
  } -> tensor<1x?x6xi8>
  %collapsed_4 = tensor.collapse_shape %7 [[0, 1], [2]] : tensor<1x?x6xi8> into tensor<?x6xi8>
  return %collapsed_4 : tensor<?x6xi8>
}
// The module must run to completion (pre-fix this aborted during hal.fence.await
// with "transient buffer has not been committed").
// CHECK: EXEC @main
// CHECK: result[0]: hal.buffer_view
