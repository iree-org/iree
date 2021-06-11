// RUN: iree-opt %s -iree-codegen-optimize-vector-transfer -split-input-file | IreeFileCheck %s

func @fold_subtensor_write_insert(%src : tensor<16x16xf32>, %vec : vector<4x4xf32>, %x : index, %y : index) -> tensor<16x16xf32> {
    %c0 = constant 0 : index 
    %0 = subtensor %src[%x, %y] [4, 4] [1, 1] : tensor<16x16xf32> to tensor<4x4xf32>
    %1 = vector.transfer_write %vec, %0[%c0, %c0] {in_bounds = [true, true]}  : vector<4x4xf32> , tensor<4x4xf32>
    %2 = subtensor_insert %1 into %src[%x, %y] [4, 4] [1, 1] : tensor<4x4xf32> into tensor<16x16xf32>
    return %2 : tensor<16x16xf32>
} 
// CHECK-LABEL: fold_subtensor_write_insert
//  CHECK-SAME:   %[[SRC:.+]]: tensor<16x16xf32>
//  CHECK-SAME:   %[[VEC:.+]]: vector<4x4xf32>
//  CHECK-SAME:   %[[X:.+]]: index, %[[Y:.+]]: index
//       CHECK:   %[[RESULT:.+]] = vector.transfer_write %[[VEC]], %[[SRC]][%[[X]], %[[Y]]]
//       CHECK:   return %[[RESULT]]
// -----

func @dontfold_subtensor_write_insert(%src : tensor<16x16xf32>, %vec : vector<4x4xf32>, %x : index, %y : index) -> tensor<16x16xf32> {
    %c0 = constant 0 : index 
    %0 = subtensor %src[%y, %x] [4, 4] [1, 1] : tensor<16x16xf32> to tensor<4x4xf32>
    %1 = vector.transfer_write %vec, %0[%c0, %c0] {in_bounds = [true, true]}  : vector<4x4xf32> , tensor<4x4xf32>
    %2 = subtensor_insert %1 into %src[%x, %y] [4, 4] [1, 1] : tensor<4x4xf32> into tensor<16x16xf32>
    return %2 : tensor<16x16xf32>
} 
// CHECK-LABEL: dontfold_subtensor_write_insert
//  CHECK-SAME:   %[[SRC:.+]]: tensor<16x16xf32>
//  CHECK-SAME:   %[[VEC:.+]]: vector<4x4xf32>
//  CHECK-SAME:   %[[X:.+]]: index, %[[Y:.+]]: index
//       CHECK:   %[[RESULT:.+]] = subtensor_insert 
//  CHECK-SAME:         into %[[SRC]]
//       CHECK:   return %[[RESULT]]
