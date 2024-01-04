// RUN: iree-opt  --iree-codegen-generic-vectorization %s | FileCheck %s

// Verify that FoldExtractSliceIntoTransferRead won't fold tensor.extract_slice
// in cases where there are more than 1 users of the result. In this case, if
// folding was to take place, tensor.extract_slice would be needed anyway (so
// folding wouldn't be beneficial).
//
// This pattern is included in e.g. GenericVectorization and that's the pass
// that's used for testing.

// CHECK-LABEL: pipeline()
// CHECK:     scf.for
// CHECK:        scf.for
// CHECK:          tensor.extract_slice
// CHECK:          scf.for {{.*}} {
// CHECK-NEXT:            %[[EXTRACTED:.*]] = tensor.extract_slice
// CHECK-NEXT:            %[[LHS:.*]] = vector.transfer_read
// CHECK-NEXT:            %[[RHS:.*]] = vector.transfer_read
// CHECK-NEXT:            %[[OUT:.*]] = vector.transfer_read %[[EXTRACTED]]
// CHECK-NEXT:            %[[CONTRACT:.*]] = vector.contract {{.*}} %[[LHS]], %[[RHS]], %[[OUT]]
// CHECK-NEXT:            %[[SLICE:.*]] = vector.transfer_write %[[CONTRACT]], %[[EXTRACTED]]
// CHECK-NEXT:            tensor.insert_slice %[[SLICE]]
// CHECK:         }
// CHECK-NEXT:    tensor.insert_slice

#map = affine_map<()[s0] -> (-(1024 mod s0) + 1024)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @pipeline() {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1024x1024xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1024x1024xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<1024x1024xf32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1024x1024xf32>> -> tensor<1024x1024xf32>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1024x1024xf32>> -> tensor<1024x1024xf32>
    %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1] : !flow.dispatch.tensor<readwrite:tensor<1024x1024xf32>> -> tensor<1024x1024xf32>
    %6 = vector.vscale
    %7 = arith.muli %6, %c16 : index
    %8 = scf.for %arg0 = %c0 to %c1024 step %c8 iter_args(%arg1 = %5) -> (tensor<1024x1024xf32>) {
      %9 = affine.apply #map()[%7]
      %10 = scf.for %arg2 = %c0 to %9 step %7 iter_args(%arg3 = %arg1) -> (tensor<1024x1024xf32>) {
        %extracted_slice = tensor.extract_slice %arg3[%arg0, %arg2] [8, %7] [1, 1] : tensor<1024x1024xf32> to tensor<8x?xf32>
        %11 = scf.for %arg4 = %c0 to %c1024 step %c1 iter_args(%arg5 = %extracted_slice) -> (tensor<8x?xf32>) {
          %extracted_slice_0 = tensor.extract_slice %arg5[0, 0] [8, %7] [1, 1] : tensor<8x?xf32> to tensor<8x?xf32>
          %12 = vector.transfer_read %3[%arg0, %arg4], %cst {in_bounds = [true, true]} : tensor<1024x1024xf32>, vector<8x1xf32>
          %13 = vector.transfer_read %4[%arg4, %arg2], %cst {in_bounds = [true, true]} : tensor<1024x1024xf32>, vector<1x[16]xf32>
          %14 = vector.transfer_read %extracted_slice_0[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<8x?xf32>, vector<8x[16]xf32>
          %15 = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %12, %13, %14 : vector<8x1xf32>, vector<1x[16]xf32> into vector<8x[16]xf32>
          %16 = vector.transfer_write %15, %extracted_slice_0[%c0, %c0] {in_bounds = [true, true]} : vector<8x[16]xf32>, tensor<8x?xf32>
          %inserted_slice_1 = tensor.insert_slice %16 into %arg5[0, 0] [8, %7] [1, 1] : tensor<8x?xf32> into tensor<8x?xf32>
          scf.yield %inserted_slice_1 : tensor<8x?xf32>
        }
        %inserted_slice = tensor.insert_slice %11 into %arg3[%arg0, %arg2] [8, %7] [1, 1] : tensor<8x?xf32> into tensor<1024x1024xf32>
        scf.yield %inserted_slice : tensor<1024x1024xf32>
      }
      scf.yield %10 : tensor<1024x1024xf32>
    }
    flow.dispatch.tensor.store %8, %2, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1] : tensor<1024x1024xf32> -> !flow.dispatch.tensor<readwrite:tensor<1024x1024xf32>>
    return
  }
}
