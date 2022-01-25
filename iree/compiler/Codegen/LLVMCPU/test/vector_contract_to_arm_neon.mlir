// RUN: iree-opt -iree-llvmcpu-vector-contract-custom-kernels='aarch64 dotprod intrinsics' %s | FileCheck %s

// CHECK-LABEL:   @vector_i8i8i32matmul_to_aarch64_asm_vec_dot(
// CHECK-SAME:          %[[LHS:[^:[:space:]]+]]
// CHECK-SAME:          %[[RHS:[^:[:space:]]+]]
// CHECK-SAME:          %[[ACC:[^:[:space:]]+]]
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant dense<0> : vector<4x4xi8>
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant dense<0> : vector<8x8xi32>
// CHECK-DAG:       %[[VAL_5:.*]] = vector.extract %[[ACC]][0] : vector<8x8xi32>
// CHECK-DAG:       %[[VAL_6:.*]] = vector.extract_strided_slice %[[VAL_5]] {offsets = [0], sizes = [4], strides = [1]} : vector<8xi32> to vector<4xi32>
// CHECK-DAG:       %[[VAL_7:.*]] = vector.extract_strided_slice %[[VAL_5]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xi32> to vector<4xi32>
// CHECK-DAG:       %[[VAL_8:.*]] = vector.extract %[[ACC]][1] : vector<8x8xi32>
// CHECK-DAG:       %[[VAL_9:.*]] = vector.extract_strided_slice %[[VAL_8]] {offsets = [0], sizes = [4], strides = [1]} : vector<8xi32> to vector<4xi32>
// CHECK-DAG:       %[[VAL_10:.*]] = vector.extract_strided_slice %[[VAL_8]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xi32> to vector<4xi32>
// CHECK-DAG:       %[[VAL_11:.*]] = vector.extract %[[ACC]][2] : vector<8x8xi32>
// CHECK-DAG:       %[[VAL_12:.*]] = vector.extract_strided_slice %[[VAL_11]] {offsets = [0], sizes = [4], strides = [1]} : vector<8xi32> to vector<4xi32>
// CHECK-DAG:       %[[VAL_13:.*]] = vector.extract_strided_slice %[[VAL_11]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xi32> to vector<4xi32>
// CHECK-DAG:       %[[VAL_14:.*]] = vector.extract %[[ACC]][3] : vector<8x8xi32>
// CHECK-DAG:       %[[VAL_15:.*]] = vector.extract_strided_slice %[[VAL_14]] {offsets = [0], sizes = [4], strides = [1]} : vector<8xi32> to vector<4xi32>
// CHECK-DAG:       %[[VAL_16:.*]] = vector.extract_strided_slice %[[VAL_14]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xi32> to vector<4xi32>
// CHECK-DAG:       %[[VAL_17:.*]] = vector.extract %[[ACC]][4] : vector<8x8xi32>
// CHECK-DAG:       %[[VAL_18:.*]] = vector.extract_strided_slice %[[VAL_17]] {offsets = [0], sizes = [4], strides = [1]} : vector<8xi32> to vector<4xi32>
// CHECK-DAG:       %[[VAL_19:.*]] = vector.extract_strided_slice %[[VAL_17]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xi32> to vector<4xi32>
// CHECK-DAG:       %[[VAL_20:.*]] = vector.extract %[[ACC]][5] : vector<8x8xi32>
// CHECK-DAG:       %[[VAL_21:.*]] = vector.extract_strided_slice %[[VAL_20]] {offsets = [0], sizes = [4], strides = [1]} : vector<8xi32> to vector<4xi32>
// CHECK-DAG:       %[[VAL_22:.*]] = vector.extract_strided_slice %[[VAL_20]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xi32> to vector<4xi32>
// CHECK-DAG:       %[[VAL_23:.*]] = vector.extract %[[ACC]][6] : vector<8x8xi32>
// CHECK-DAG:       %[[VAL_24:.*]] = vector.extract_strided_slice %[[VAL_23]] {offsets = [0], sizes = [4], strides = [1]} : vector<8xi32> to vector<4xi32>
// CHECK-DAG:       %[[VAL_25:.*]] = vector.extract_strided_slice %[[VAL_23]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xi32> to vector<4xi32>
// CHECK-DAG:       %[[VAL_26:.*]] = vector.extract %[[ACC]][7] : vector<8x8xi32>
// CHECK-DAG:       %[[VAL_27:.*]] = vector.extract_strided_slice %[[VAL_26]] {offsets = [0], sizes = [4], strides = [1]} : vector<8xi32> to vector<4xi32>
// CHECK-DAG:       %[[VAL_28:.*]] = vector.extract_strided_slice %[[VAL_26]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xi32> to vector<4xi32>
// CHECK-DAG:       %[[VAL_29:.*]] = vector.extract_strided_slice %[[LHS]] {offsets = [0, 0], sizes = [4, 4], strides = [1, 1]} : vector<8x4xi8> to vector<4x4xi8>
// CHECK-DAG:       %[[VAL_30:.*]] = vector.extract_strided_slice %[[LHS]] {offsets = [4, 0], sizes = [4, 4], strides = [1, 1]} : vector<8x4xi8> to vector<4x4xi8>
// CHECK-DAG:       %[[VAL_31:.*]] = vector.extract_strided_slice %[[RHS]] {offsets = [0, 0], sizes = [4, 4], strides = [1, 1]} : vector<8x4xi8> to vector<4x4xi8>
// CHECK-DAG:       %[[VAL_32:.*]] = vector.extract_strided_slice %[[RHS]] {offsets = [4, 0], sizes = [4, 4], strides = [1, 1]} : vector<8x4xi8> to vector<4x4xi8>
// CHECK-DAG:       %[[VAL_33:.*]] = vector.shuffle %[[VAL_29]], %[[VAL_3]] [0, 0, 0, 0] : vector<4x4xi8>, vector<4x4xi8>
// CHECK-DAG:       %[[VAL_34:.*]] = arm_neon.2d.sdot %[[VAL_6]], %[[VAL_31]], %[[VAL_33]] : vector<4x4xi8>, vector<4x4xi8> to vector<4xi32>
// CHECK-DAG:       %[[VAL_35:.*]] = vector.shuffle %[[VAL_29]], %[[VAL_3]] [0, 0, 0, 0] : vector<4x4xi8>, vector<4x4xi8>
// CHECK-DAG:       %[[VAL_36:.*]] = arm_neon.2d.sdot %[[VAL_7]], %[[VAL_32]], %[[VAL_35]] : vector<4x4xi8>, vector<4x4xi8> to vector<4xi32>
// CHECK-DAG:       %[[VAL_37:.*]] = vector.shuffle %[[VAL_29]], %[[VAL_3]] [1, 1, 1, 1] : vector<4x4xi8>, vector<4x4xi8>
// CHECK-DAG:       %[[VAL_38:.*]] = arm_neon.2d.sdot %[[VAL_9]], %[[VAL_31]], %[[VAL_37]] : vector<4x4xi8>, vector<4x4xi8> to vector<4xi32>
// CHECK-DAG:       %[[VAL_39:.*]] = vector.shuffle %[[VAL_29]], %[[VAL_3]] [1, 1, 1, 1] : vector<4x4xi8>, vector<4x4xi8>
// CHECK-DAG:       %[[VAL_40:.*]] = arm_neon.2d.sdot %[[VAL_10]], %[[VAL_32]], %[[VAL_39]] : vector<4x4xi8>, vector<4x4xi8> to vector<4xi32>
// CHECK-DAG:       %[[VAL_41:.*]] = vector.shuffle %[[VAL_29]], %[[VAL_3]] [2, 2, 2, 2] : vector<4x4xi8>, vector<4x4xi8>
// CHECK-DAG:       %[[VAL_42:.*]] = arm_neon.2d.sdot %[[VAL_12]], %[[VAL_31]], %[[VAL_41]] : vector<4x4xi8>, vector<4x4xi8> to vector<4xi32>
// CHECK-DAG:       %[[VAL_43:.*]] = vector.shuffle %[[VAL_29]], %[[VAL_3]] [2, 2, 2, 2] : vector<4x4xi8>, vector<4x4xi8>
// CHECK-DAG:       %[[VAL_44:.*]] = arm_neon.2d.sdot %[[VAL_13]], %[[VAL_32]], %[[VAL_43]] : vector<4x4xi8>, vector<4x4xi8> to vector<4xi32>
// CHECK-DAG:       %[[VAL_45:.*]] = vector.shuffle %[[VAL_29]], %[[VAL_3]] [3, 3, 3, 3] : vector<4x4xi8>, vector<4x4xi8>
// CHECK-DAG:       %[[VAL_46:.*]] = arm_neon.2d.sdot %[[VAL_15]], %[[VAL_31]], %[[VAL_45]] : vector<4x4xi8>, vector<4x4xi8> to vector<4xi32>
// CHECK-DAG:       %[[VAL_47:.*]] = vector.shuffle %[[VAL_29]], %[[VAL_3]] [3, 3, 3, 3] : vector<4x4xi8>, vector<4x4xi8>
// CHECK-DAG:       %[[VAL_48:.*]] = arm_neon.2d.sdot %[[VAL_16]], %[[VAL_32]], %[[VAL_47]] : vector<4x4xi8>, vector<4x4xi8> to vector<4xi32>
// CHECK-DAG:       %[[VAL_49:.*]] = vector.shuffle %[[VAL_30]], %[[VAL_3]] [0, 0, 0, 0] : vector<4x4xi8>, vector<4x4xi8>
// CHECK-DAG:       %[[VAL_50:.*]] = arm_neon.2d.sdot %[[VAL_18]], %[[VAL_31]], %[[VAL_49]] : vector<4x4xi8>, vector<4x4xi8> to vector<4xi32>
// CHECK-DAG:       %[[VAL_51:.*]] = vector.shuffle %[[VAL_30]], %[[VAL_3]] [0, 0, 0, 0] : vector<4x4xi8>, vector<4x4xi8>
// CHECK-DAG:       %[[VAL_52:.*]] = arm_neon.2d.sdot %[[VAL_19]], %[[VAL_32]], %[[VAL_51]] : vector<4x4xi8>, vector<4x4xi8> to vector<4xi32>
// CHECK-DAG:       %[[VAL_53:.*]] = vector.shuffle %[[VAL_30]], %[[VAL_3]] [1, 1, 1, 1] : vector<4x4xi8>, vector<4x4xi8>
// CHECK-DAG:       %[[VAL_54:.*]] = arm_neon.2d.sdot %[[VAL_21]], %[[VAL_31]], %[[VAL_53]] : vector<4x4xi8>, vector<4x4xi8> to vector<4xi32>
// CHECK-DAG:       %[[VAL_55:.*]] = vector.shuffle %[[VAL_30]], %[[VAL_3]] [1, 1, 1, 1] : vector<4x4xi8>, vector<4x4xi8>
// CHECK-DAG:       %[[VAL_56:.*]] = arm_neon.2d.sdot %[[VAL_22]], %[[VAL_32]], %[[VAL_55]] : vector<4x4xi8>, vector<4x4xi8> to vector<4xi32>
// CHECK-DAG:       %[[VAL_57:.*]] = vector.shuffle %[[VAL_30]], %[[VAL_3]] [2, 2, 2, 2] : vector<4x4xi8>, vector<4x4xi8>
// CHECK-DAG:       %[[VAL_58:.*]] = arm_neon.2d.sdot %[[VAL_24]], %[[VAL_31]], %[[VAL_57]] : vector<4x4xi8>, vector<4x4xi8> to vector<4xi32>
// CHECK-DAG:       %[[VAL_59:.*]] = vector.shuffle %[[VAL_30]], %[[VAL_3]] [2, 2, 2, 2] : vector<4x4xi8>, vector<4x4xi8>
// CHECK-DAG:       %[[VAL_60:.*]] = arm_neon.2d.sdot %[[VAL_25]], %[[VAL_32]], %[[VAL_59]] : vector<4x4xi8>, vector<4x4xi8> to vector<4xi32>
// CHECK-DAG:       %[[VAL_61:.*]] = vector.shuffle %[[VAL_30]], %[[VAL_3]] [3, 3, 3, 3] : vector<4x4xi8>, vector<4x4xi8>
// CHECK-DAG:       %[[VAL_62:.*]] = arm_neon.2d.sdot %[[VAL_27]], %[[VAL_31]], %[[VAL_61]] : vector<4x4xi8>, vector<4x4xi8> to vector<4xi32>
// CHECK-DAG:       %[[VAL_63:.*]] = vector.shuffle %[[VAL_30]], %[[VAL_3]] [3, 3, 3, 3] : vector<4x4xi8>, vector<4x4xi8>
// CHECK-DAG:       %[[VAL_64:.*]] = arm_neon.2d.sdot %[[VAL_28]], %[[VAL_32]], %[[VAL_63]] : vector<4x4xi8>, vector<4x4xi8> to vector<4xi32>
// CHECK-DAG:       %[[VAL_65:.*]] = vector.insert_strided_slice %[[VAL_34]], %[[VAL_4]] {offsets = [0, 0], strides = [1]} : vector<4xi32> into vector<8x8xi32>
// CHECK-DAG:       %[[VAL_66:.*]] = vector.insert_strided_slice %[[VAL_36]], %[[VAL_65]] {offsets = [0, 4], strides = [1]} : vector<4xi32> into vector<8x8xi32>
// CHECK-DAG:       %[[VAL_67:.*]] = vector.insert_strided_slice %[[VAL_38]], %[[VAL_66]] {offsets = [1, 0], strides = [1]} : vector<4xi32> into vector<8x8xi32>
// CHECK-DAG:       %[[VAL_68:.*]] = vector.insert_strided_slice %[[VAL_40]], %[[VAL_67]] {offsets = [1, 4], strides = [1]} : vector<4xi32> into vector<8x8xi32>
// CHECK-DAG:       %[[VAL_69:.*]] = vector.insert_strided_slice %[[VAL_42]], %[[VAL_68]] {offsets = [2, 0], strides = [1]} : vector<4xi32> into vector<8x8xi32>
// CHECK-DAG:       %[[VAL_70:.*]] = vector.insert_strided_slice %[[VAL_44]], %[[VAL_69]] {offsets = [2, 4], strides = [1]} : vector<4xi32> into vector<8x8xi32>
// CHECK-DAG:       %[[VAL_71:.*]] = vector.insert_strided_slice %[[VAL_46]], %[[VAL_70]] {offsets = [3, 0], strides = [1]} : vector<4xi32> into vector<8x8xi32>
// CHECK-DAG:       %[[VAL_72:.*]] = vector.insert_strided_slice %[[VAL_48]], %[[VAL_71]] {offsets = [3, 4], strides = [1]} : vector<4xi32> into vector<8x8xi32>
// CHECK-DAG:       %[[VAL_73:.*]] = vector.insert_strided_slice %[[VAL_50]], %[[VAL_72]] {offsets = [4, 0], strides = [1]} : vector<4xi32> into vector<8x8xi32>
// CHECK-DAG:       %[[VAL_74:.*]] = vector.insert_strided_slice %[[VAL_52]], %[[VAL_73]] {offsets = [4, 4], strides = [1]} : vector<4xi32> into vector<8x8xi32>
// CHECK-DAG:       %[[VAL_75:.*]] = vector.insert_strided_slice %[[VAL_54]], %[[VAL_74]] {offsets = [5, 0], strides = [1]} : vector<4xi32> into vector<8x8xi32>
// CHECK-DAG:       %[[VAL_76:.*]] = vector.insert_strided_slice %[[VAL_56]], %[[VAL_75]] {offsets = [5, 4], strides = [1]} : vector<4xi32> into vector<8x8xi32>
// CHECK-DAG:       %[[VAL_77:.*]] = vector.insert_strided_slice %[[VAL_58]], %[[VAL_76]] {offsets = [6, 0], strides = [1]} : vector<4xi32> into vector<8x8xi32>
// CHECK-DAG:       %[[VAL_78:.*]] = vector.insert_strided_slice %[[VAL_60]], %[[VAL_77]] {offsets = [6, 4], strides = [1]} : vector<4xi32> into vector<8x8xi32>
// CHECK-DAG:       %[[VAL_79:.*]] = vector.insert_strided_slice %[[VAL_62]], %[[VAL_78]] {offsets = [7, 0], strides = [1]} : vector<4xi32> into vector<8x8xi32>
// CHECK-DAG:       %[[VAL_80:.*]] = vector.insert_strided_slice %[[VAL_64]], %[[VAL_79]] {offsets = [7, 4], strides = [1]} : vector<4xi32> into vector<8x8xi32>
// CHECK:           return %[[VAL_80]]
func @vector_i8i8i32matmul_to_aarch64_asm_vec_dot(
    %lhs: vector<8x4xi8>,
    %rhs: vector<8x4xi8>,
    %acc: vector<8x8xi32>) -> vector<8x8xi32> {
  %lhs_wide = arith.extsi %lhs : vector<8x4xi8> to vector<8x4xi32>
  %rhs_wide = arith.extsi %rhs : vector<8x4xi8> to vector<8x4xi32>
  %res = vector.contract {
      indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d2)>,
          affine_map<(d0, d1, d2) -> (d1, d2)>,
          affine_map<(d0, d1, d2) -> (d0, d1)>
      ], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>
  } %lhs_wide, %rhs_wide, %acc : vector<8x4xi32>, vector<8x4xi32> into vector<8x8xi32>
  return %res : vector<8x8xi32>
}
