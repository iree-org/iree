// RUN: iree-opt -pass-pipeline="hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target{use-lowering-pipeline='func(iree-llvmcpu-vectorization)'}))" -split-input-file %s | IreeFileCheck %s
// RUN: iree-opt -pass-pipeline="hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target{use-lowering-pipeline='func(iree-llvmcpu-vectorization{promote-workgroup-to-full-tiles}),cse'}))" -split-input-file %s | IreeFileCheck %s -check-prefix=CHECK-PROMOTED

#config = {nativeVectorSize = [4, 4, 4], tileSizes = [[64, 64], [32, 32, 32], [4, 4, 4]]}
hal.executable @dynamic_matmul attributes {sym_visibility = "private"} {
  hal.interface @io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.variant @llvm, target="llvm" {
    hal.executable.entry_point @matmul_128x128x128 attributes {
      interface = @io,
      ordinal = 0 : index
    }
    module {
      func @matmul_128x128x128() {
        %c0 = constant 0 : index
        %c128 = constant 128 : index
        %0 = hal.interface.binding.subspan @io::@arg0[%c0] : memref<128x128xf32>
        %1 = hal.interface.binding.subspan @io::@arg1[%c0] : memref<128x128xf32>
        %2 = hal.interface.binding.subspan @io::@ret0[%c0] : memref<128x128xf32>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %3 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_y]
        %4 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_count_y]
        scf.for %arg0 = %3 to %c128 step %4 {
          %5 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_x]
          %6 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_count_x]
          scf.for %arg1 = %5 to %c128 step %6 {
            %7 = memref.subview %0[%arg0, 0] [64, 128] [1, 1] : memref<128x128xf32> to memref<64x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>>
            %8 = memref.subview %1[0, %arg1] [128, 64] [1, 1] : memref<128x128xf32> to memref<128x64xf32, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>>
            %9 = memref.subview %2[%arg0, %arg1] [64, 64] [1, 1] : memref<128x128xf32> to memref<64x64xf32, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>>
            linalg.matmul {__internal_linalg_transform__ = "workgroup", lowering.config = #config} ins(%7, %8 : memref<64x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>>, memref<128x64xf32, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>>) outs(%9 : memref<64x64xf32, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>>)
          }
        }
        return
      }
    }
  }
}
// CHECK-LABEL: func @matmul_128x128x128
//   CHECK-DAG:   %[[ARG0:.+]] = hal.interface.binding.subspan @io::@arg0
//   CHECK-DAG:   %[[ARG1:.+]] = hal.interface.binding.subspan @io::@arg1
//   CHECK-DAG:   %[[RET0:.+]] = hal.interface.binding.subspan @io::@ret0
//   CHECK-DAG:   %[[START:.+]] = constant 0 : index
//   CHECK-DAG:   %[[WORGKROUP_SIZE:.+]] = constant 64
//   CHECK-DAG:   %[[VECTOR_SIZE:.+]] = constant 4
//   CHECK-DAG:   %[[L1_SIZE:.+]] = constant 32
//   CHECK-DAG:   %[[KDIM_SIZE:.+]] = constant 128
//       CHECK:   scf.for
//       CHECK:     scf.for
//       CHECK:       scf.for {{.*}} = %[[START]] to %[[WORGKROUP_SIZE]] step %[[L1_SIZE]] {
//       CHECK:         scf.for {{.*}} = %[[START]] to %[[WORGKROUP_SIZE]] step %[[L1_SIZE]] {
//       CHECK:           scf.for {{.*}} = %[[START]] to %[[KDIM_SIZE]] step %[[L1_SIZE]] {
//       CHECK:             scf.for {{.*}} = %[[START]] to %[[L1_SIZE]] step %[[VECTOR_SIZE]] {
//       CHECK:               scf.for {{.*}} = %[[START]] to %[[L1_SIZE]] step %[[VECTOR_SIZE]] {
//       CHECK:                 %[[VEC_C_0:.+]] = vector.transfer_read %[[RET0]]
//       CHECK:                 %[[VEC_C_1:.+]] = vector.transfer_read %[[RET0]]
//       CHECK:                 %[[VEC_C_2:.+]] = vector.transfer_read %[[RET0]]
//       CHECK:                 %[[VEC_C_3:.+]] = vector.transfer_read %[[RET0]]
//       CHECK:                 scf.for {{.*}} = %[[START]] to %[[L1_SIZE]] step %[[VECTOR_SIZE]]
//       CHECK:                   %[[VEC_A_0:.+]] = vector.transfer_read %[[ARG0]]
//       CHECK:                   %[[VEC_A_1:.+]] = vector.transfer_read %[[ARG0]]
//       CHECK:                   %[[VEC_A_2:.+]] = vector.transfer_read %[[ARG0]]
//       CHECK:                   %[[VEC_A_3:.+]] = vector.transfer_read %[[ARG0]]
//       CHECK:                   %[[VEC_B_0:.+]] = vector.transfer_read %[[ARG1]]
//       CHECK:                   %[[VEC_b_1:.+]] = vector.transfer_read %[[ARG1]]
//       CHECK:                   %[[VEC_B_2:.+]] = vector.transfer_read %[[ARG1]]
//       CHECK:                   %[[VEC_B_3:.+]] = vector.transfer_read %[[ARG1]]

//     CHECK-PROMOTED: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 * 64)>
//     CHECK-PROMOTED: #[[MAP1:.+]] = affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>
//     CHECK-PROMOTED: func @matmul_128x128x128
// CHECK-PROMOTED-DAG:   %[[KDIM_SIZE:.+]] = constant 128 : index
// CHECK-PROMOTED-DAG:   %[[WORGKROUP_SIZE:.+]] = constant 64 : index
// CHECK-PROMOTED-DAG:   %[[VECTOR_SIZE:.+]] = constant 4 : index
// CHECK-PROMOTED-DAG:   %[[L1_SIZE:.+]] = constant 32 : index
// CHECK-PROMOTED-DAG:   %[[START:.+]] = constant 0 : index
// CHECK-PROMOTED-DAG:   %[[C1:.+]] = constant 1 : index
// CHECK-PROMOTED-DAG:   %[[C1:.+]] = constant 2 : index
// CHECK-PROMOTED-DAG:   %[[C1:.+]] = constant 3 : index
// CHECK-PROMOTED-DAG:   %[[C_PROMOTED_TILE:.+]] = memref.alloca() : memref<64x64xf32>
// CHECK-PROMOTED-DAG:   %[[B_PROMOTED_TILE:.+]] = memref.alloca() : memref<128x64xf32>
// CHECK-PROMOTED-DAG:   %[[A_PROMOTED_TILE:.+]] = memref.alloca() : memref<64x128xf32>
// CHECK-PROMOTED-DAG:   %[[ARG0:.+]] = hal.interface.binding.subspan @io::@arg0
// CHECK-PROMOTED-DAG:   %[[ARG1:.+]] = hal.interface.binding.subspan @io::@arg1
// CHECK-PROMOTED-DAG:   %[[RET0:.+]] = hal.interface.binding.subspan @io::@ret0
//     CHECK-PROMOTED:   scf.for %[[IV0:.+]] =
//     CHECK-PROMOTED:     scf.for %[[IV1:.+]] =
// CHECK-PROMOTED-DAG:       %[[A_TILE:.+]] = memref.subview %[[ARG0]][%[[IV0]], 0] [64, 128]
// CHECK-PROMOTED-DAG:       %[[B_TILE:.+]] = memref.subview %[[ARG1]][0, %[[IV1]]] [128, 64]
// CHECK-PROMOTED-DAG:       %[[C_TILE:.+]] = memref.subview %[[RET0]][%[[IV0]], %[[IV1]]] [64, 64]
// CHECK-PROMOTED-DAG:       %[[A_PROMOTED_TILE_VIEW:.+]] = memref.subview %[[A_PROMOTED_TILE]][0, 0] [64, 128]
// CHECK-PROMOTED_DAG:       linalg.fill(%{{.+}}, %[[A_PROMOTED_TILE]])
// CHECK-PROMOTED-DAG:       %[[B_PROMOTED_TILE_VIEW:.+]] = memref.subview %[[B_PROMOTED_TILE]][0, 0] [128, 64]
// CHECK-PROMOTED_DAG:       linalg.fill(%{{.+}}, %[[B_PROMOTED_TILE]])
// CHECK-PROMOTED-DAG:       %[[C_PROMOTED_TILE_VIEW:.+]] = memref.subview %[[C_PROMOTED_TILE]][0, 0] [64, 64]
// CHECK-PROMOTED_DAG:       linalg.fill(%{{.+}}, %[[C_PROMOTED_TILE]])
//     CHECK-PROMOTED:       linalg.copy(%[[A_TILE]], %[[A_PROMOTED_TILE_VIEW]])
//     CHECK-PROMOTED:       linalg.copy(%[[B_TILE]], %[[B_PROMOTED_TILE_VIEW]])
//     CHECK-PROMOTED:       linalg.copy(%[[C_TILE]], %[[C_PROMOTED_TILE_VIEW]])
//     CHECK-PROMOTED:       scf.for {{.*}} = %[[START]] to %[[WORGKROUP_SIZE]] step %[[L1_SIZE]] {
//     CHECK-PROMOTED:         scf.for {{.*}} = %[[START]] to %[[WORGKROUP_SIZE]] step %[[L1_SIZE]] {
//     CHECK-PROMOTED:           scf.for {{.*}} = %[[START]] to %[[KDIM_SIZE]] step %[[L1_SIZE]] {
//     CHECK-PROMOTED:             scf.for {{.*}} = %[[START]] to %[[L1_SIZE]] step %[[VECTOR_SIZE]] {
//     CHECK-PROMOTED:               scf.for {{.*}} = %[[START]] to %[[L1_SIZE]] step %[[VECTOR_SIZE]] {
//     CHECK-PROMOTED:                 %[[VEC_C_0:.+]] = vector.transfer_read %[[C_PROMOTED_TILE]]
//     CHECK-PROMOTED:                 %[[VEC_C_1:.+]] = vector.transfer_read %[[C_PROMOTED_TILE]]
//     CHECK-PROMOTED:                 %[[VEC_C_2:.+]] = vector.transfer_read %[[C_PROMOTED_TILE]]
//     CHECK-PROMOTED:                 %[[VEC_C_3:.+]] = vector.transfer_read %[[C_PROMOTED_TILE]]

// -----

#config = {nativeVectorSize = [4, 4, 4], tileSizes = [[64, 64], [32, 32, 32], [4, 4, 4]]}
hal.executable @matmul_i8_i8_i32 attributes {sym_visibility = "private"} {
  hal.interface @io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.variant @llvm, target="llvm" {
    hal.executable.entry_point @matmul_i8_i8_i32_128x128x128 attributes {
      interface = @io,
      ordinal = 0 : index
    }
    module {
      func @matmul_i8_i8_i32_128x128x128() {
        %c0 = constant 0 : index
        %c128 = constant 128 : index
        %0 = hal.interface.binding.subspan @io::@arg0[%c0] : memref<128x128xi8>
        %1 = hal.interface.binding.subspan @io::@arg1[%c0] : memref<128x128xi8>
        %2 = hal.interface.binding.subspan @io::@ret0[%c0] : memref<128x128xi32>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %3 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_y]
        %4 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_count_y]
        scf.for %arg0 = %3 to %c128 step %4 {
          %5 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_x]
          %6 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_count_x]
          scf.for %arg1 = %5 to %c128 step %6 {
            %7 = memref.subview %0[%arg0, 0] [64, 128] [1, 1] : memref<128x128xi8> to memref<64x128xi8, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>>
            %8 = memref.subview %1[0, %arg1] [128, 64] [1, 1] : memref<128x128xi8> to memref<128x64xi8, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>>
            %9 = memref.subview %2[%arg0, %arg1] [64, 64] [1, 1] : memref<128x128xi32> to memref<64x64xi32, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>>
            linalg.matmul_i8_i8_i32 {__internal_linalg_transform__ = "workgroup", lowering.config = #config} ins(%7, %8 : memref<64x128xi8, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>>, memref<128x64xi8, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>>) outs(%9 : memref<64x64xi32, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>>)
          }
        }
        return
      }
    }
  }
}
// CHECK-LABEL: func @matmul_i8_i8_i32_128x128x128
//   CHECK-DAG:   %[[ARG0:.+]] = hal.interface.binding.subspan @io::@arg0
//   CHECK-DAG:   %[[ARG1:.+]] = hal.interface.binding.subspan @io::@arg1
//   CHECK-DAG:   %[[RET0:.+]] = hal.interface.binding.subspan @io::@ret0
//   CHECK-DAG:   %[[START:.+]] = constant 0 : index
//   CHECK-DAG:   %[[WORGKROUP_SIZE:.+]] = constant 64
//   CHECK-DAG:   %[[VECTOR_SIZE:.+]] = constant 4
//   CHECK-DAG:   %[[L1_SIZE:.+]] = constant 32
//   CHECK-DAG:   %[[KDIM_SIZE:.+]] = constant 128
//       CHECK:   scf.for
//       CHECK:     scf.for
//       CHECK:       scf.for {{.*}} = %[[START]] to %[[WORGKROUP_SIZE]] step %[[L1_SIZE]] {
//       CHECK:         scf.for {{.*}} = %[[START]] to %[[WORGKROUP_SIZE]] step %[[L1_SIZE]] {
//       CHECK:           scf.for {{.*}} = %[[START]] to %[[KDIM_SIZE]] step %[[L1_SIZE]] {
//       CHECK:             scf.for {{.*}} = %[[START]] to %[[L1_SIZE]] step %[[VECTOR_SIZE]] {
//       CHECK:               scf.for {{.*}} = %[[START]] to %[[L1_SIZE]] step %[[VECTOR_SIZE]] {
//       CHECK:                 %[[VEC_C_0:.+]] = vector.transfer_read %[[RET0]]
//       CHECK:                 %[[VEC_C_1:.+]] = vector.transfer_read %[[RET0]]
//       CHECK:                 %[[VEC_C_2:.+]] = vector.transfer_read %[[RET0]]
//       CHECK:                 %[[VEC_C_3:.+]] = vector.transfer_read %[[RET0]]
//       CHECK:                   scf.for {{.*}} = %[[START]] to %[[L1_SIZE]] step %[[VECTOR_SIZE]]
//       CHECK:                     %[[VEC_A_0:.+]] = vector.transfer_read %[[ARG0]]
//       CHECK:                     %[[VEC_A_1:.+]] = vector.transfer_read %[[ARG0]]
//       CHECK:                     %[[VEC_A_2:.+]] = vector.transfer_read %[[ARG0]]
//       CHECK:                     %[[VEC_A_3:.+]] = vector.transfer_read %[[ARG0]]
//       CHECK:                     %[[VEC_B_0:.+]] = vector.transfer_read %[[ARG1]]
//       CHECK:                     %[[VEC_b_1:.+]] = vector.transfer_read %[[ARG1]]
//       CHECK:                     %[[VEC_B_2:.+]] = vector.transfer_read %[[ARG1]]
//       CHECK:                     %[[VEC_B_3:.+]] = vector.transfer_read %[[ARG1]]
