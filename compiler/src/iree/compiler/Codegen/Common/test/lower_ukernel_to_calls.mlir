// RUN: iree-opt --iree-codegen-lower-ukernel-ops-to-calls -split-input-file --verify-diagnostics --cse %s | FileCheck %s

func.func @ukernel_generic_scalar_types(%arg0: i32, %arg1 : f64, %arg2 : index, %arg3 : memref<f32>) {
  iree_codegen.ukernel.generic "scalar_fn" ins(%arg0, %arg1, %arg2 : i32, f64, index) outs(%arg3 : memref<f32>)
  return
}
//      CHECK: func.func private @scalar_fn(i32, f64, index, memref<f32>, index)
//      CHECK: func.func @ukernel_generic_scalar_types
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: i32
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: f64
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: memref<f32>
//      CHECK:   %[[BASE:.+]], %[[OFFSET:.+]] = memref.extract_strided_metadata %[[ARG3]]
//      CHECK:   call @scalar_fn(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[BASE]], %[[OFFSET]])

// -----

func.func @ukernel_generic_memref_1D(%arg0 : memref<?xf32, strided<[1], offset: ?>>) {
  iree_codegen.ukernel.generic "test1d" outs(%arg0 : memref<?xf32, strided<[1], offset: ?>>)
  return
}
//      CHECK: func.func private @test1d(memref<f32>, index, index)
//      CHECK: func.func @ukernel_generic_memref_1D
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: memref<?xf32, strided<[1], offset: ?>>
//      CHECK:   %[[BASE:.+]], %[[OFFSET:.+]], %[[SIZE:.+]], %[[STRIDES:.+]] = memref.extract_strided_metadata %[[ARG0]]
//      CHECK:   call @test1d(%[[BASE]], %[[OFFSET]], %[[STRIDES]])

// -----

func.func @ukernel_generic_memref_1D_strided_outer_dims_0(%arg0 : memref<?xf32, strided<[1], offset: ?>>) {
  iree_codegen.ukernel.generic "test1d" outs(%arg0 : memref<?xf32, strided<[1], offset: ?>>) strided_outer_dims(0)
  return
}
//      CHECK: func.func private @test1d(memref<f32>, index)
//      CHECK: func.func @ukernel_generic_memref_1D_strided_outer_dims_0
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: memref<?xf32, strided<[1], offset: ?>>
//      CHECK:   %[[BASE:.+]], %[[OFFSET:.+]], %[[SIZE:.+]], %[[STRIDES:.+]] = memref.extract_strided_metadata %[[ARG0]]
//      CHECK:   call @test1d(%[[BASE]], %[[OFFSET]])

// -----

func.func @ukernel_generic_memref_3D(%arg0 : memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>>) {
  iree_codegen.ukernel.generic "test3d" outs(%arg0 : memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>>)
  return
}
//      CHECK: func.func private @test3d(memref<f32>, index, index, index, index)
//      CHECK: func.func @ukernel_generic_memref_3D
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>>
//      CHECK:   %[[BASE:.+]], %[[OFFSET:.+]], %[[SIZE:.+]]:3, %[[STRIDES:.+]]:3 = memref.extract_strided_metadata %[[ARG0]]
//      CHECK:   call @test3d(%[[BASE]], %[[OFFSET]], %[[STRIDES]]#0, %[[STRIDES]]#1, %[[STRIDES]]#2)

// -----

func.func @ukernel_generic_memref_3D_strided_outer_dims_2(%arg0 : memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>>) {
  iree_codegen.ukernel.generic "test3d" outs(%arg0 : memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>>) strided_outer_dims(2)
  return
}
//      CHECK: func.func private @test3d(memref<f32>, index, index, index)
//      CHECK: func.func @ukernel_generic_memref_3D_strided_outer_dims_2
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>>
//      CHECK:   %[[BASE:.+]], %[[OFFSET:.+]], %[[SIZE:.+]]:3, %[[STRIDES:.+]]:3 = memref.extract_strided_metadata %[[ARG0]]
//      CHECK:   call @test3d(%[[BASE]], %[[OFFSET]], %[[STRIDES]]#0, %[[STRIDES]]#1)


// -----

func.func @ukernel_generic_return_value(%arg0 : tensor<f32>) -> tensor<f32> {
  // expected-error @+1 {{failed to lower micro kernel operation to function call}}
  %0 = iree_codegen.ukernel.generic "err" outs(%arg0 : tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

func.func @ukernel_generic(%arg0 : memref<?x?xf32>, %arg1 : memref<?x?xf32>,
    %arg2 : memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %dim_1 = memref.dim %arg0, %c0 : memref<?x?xf32>
  %dim_2 = memref.dim %arg1, %c1 : memref<?x?xf32>
  %dim_3 = memref.dim %arg2, %c1 : memref<?x?xf32>
  iree_codegen.ukernel.generic "vmvx.matmul.f32f32f32" ins(%arg0, %arg1 : memref<?x?xf32>, memref<?x?xf32>) outs(%arg2 : memref<?x?xf32>) (%dim_1, %dim_2, %dim_3, %c0_i32 : index, index, index, i32) strided_outer_dims(1)
  return
}
// CHECK-LABEL: func.func private @vmvx.matmul.f32f32f32
//  CHECK-SAME:     (memref<f32>, index, index, memref<f32>, index, index,
//  CHECK-SAME:     memref<f32>, index, index, index, index, index, i32)
// CHECK-LABEL: func.func @ukernel_generic
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: memref<?x?xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: memref<?x?xf32>
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: memref<?x?xf32>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C0_I32:.+]] = arith.constant 0 : i32
//   CHECK-DAG:   %[[D0:.+]] = memref.dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = memref.dim %[[ARG1]], %[[C1]]
//   CHECK-DAG:   %[[D2:.+]] = memref.dim %[[ARG2]], %[[C1]]
//       CHECK:   %[[BASE0:.+]], %[[OFFSET0:.+]], %[[SIZE0:.+]]:2, %[[STRIDES0:.+]]:2 = memref.extract_strided_metadata %[[ARG0]]
//       CHECK:   %[[BASE1:.+]], %[[OFFSET1:.+]], %[[SIZE1:.+]]:2, %[[STRIDES1:.+]]:2 = memref.extract_strided_metadata %[[ARG1]]
//       CHECK:   %[[BASE2:.+]], %[[OFFSET2:.+]], %[[SIZE2:.+]]:2, %[[STRIDES2:.+]]:2 = memref.extract_strided_metadata %[[ARG2]]
//       CHECK:   call @vmvx.matmul.f32f32f32(%[[BASE0]], %[[OFFSET0]], %[[STRIDES0]]#0
//  CHECK-SAME:       %[[BASE1]], %[[OFFSET1]], %[[STRIDES1]]#0
//  CHECK-SAME:       %[[BASE2]], %[[OFFSET2]], %[[STRIDES2]]#0
//  CHECK-SAME:       %[[D0]], %[[D1]], %[[D2]], %[[C0_I32]])

// -----

func.func @ukernel_mmt4d_f32f32f32(%lhs : memref<?x?x?x?xf32>, %rhs : memref<?x?x?x?xf32>,
    %acc : memref<?x?x?x?xf32>, %m : index, %n : index, %k : index, %m0 : i32, %n0 : i32, %k0 : i32, %flags : i32) {
  iree_codegen.ukernel.generic "vmvx.mmt4d.f32f32f32" ins(%lhs, %rhs : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>)
      outs(%acc : memref<?x?x?x?xf32>) (%m, %n, %k, %m0, %n0, %k0, %flags : index, index, index, i32, i32, i32, i32) strided_outer_dims(1)
  return
}
// CHECK-LABEL: func.func private @vmvx.mmt4d.f32f32f32
//  CHECK-SAME:     (memref<f32>, index, index, memref<f32>, index, index,
//  CHECK-SAME:     memref<f32>, index, index, index, index, index, i32, i32, i32, i32)
// CHECK-LABEL: func.func @ukernel_mmt4d_f32f32f32(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: memref<?x?x?x?xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: memref<?x?x?x?xf32>
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: memref<?x?x?x?xf32>
//  CHECK-SAME:     %[[M:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[N:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[K:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[M0:[a-zA-Z0-9]+]]: i32
//  CHECK-SAME:     %[[N0:[a-zA-Z0-9]+]]: i32
//  CHECK-SAME:     %[[K0:[a-zA-Z0-9]+]]: i32
//  CHECK-SAME:     %[[FLAGS:[a-zA-Z0-9]+]]: i32
//       CHECK:   %[[BASE0:.+]], %[[OFFSET0:.+]], %[[SIZE0:.+]]:4, %[[STRIDES0:.+]]:4 = memref.extract_strided_metadata %[[ARG0]]
//       CHECK:   %[[BASE1:.+]], %[[OFFSET1:.+]], %[[SIZE1:.+]]:4, %[[STRIDES1:.+]]:4 = memref.extract_strided_metadata %[[ARG1]]
//       CHECK:   %[[BASE2:.+]], %[[OFFSET2:.+]], %[[SIZE2:.+]]:4, %[[STRIDES2:.+]]:4 = memref.extract_strided_metadata %[[ARG2]]
//       CHECK:   call @vmvx.mmt4d.f32f32f32(%[[BASE0]], %[[OFFSET0]], %[[STRIDES0]]#0
//  CHECK-SAME:       %[[BASE1]], %[[OFFSET1]], %[[STRIDES1]]#0
//  CHECK-SAME:       %[[BASE2]], %[[OFFSET2]], %[[STRIDES2]]#0
//  CHECK-SAME:       %[[M]], %[[N]], %[[K]],
//  CHECK-SAME:       %[[M0]], %[[N0]], %[[K0]], %[[FLAGS]])

// -----

func.func @ukernel_mmt4d_i8i8i32(%lhs : memref<?x?x?x?xi8>, %rhs : memref<?x?x?x?xi8>,
    %acc : memref<?x?x?x?xi32>, %m : index, %n : index, %k : index, %m0 : i32, %n0 : i32, %k0 : i32, %flags : i32) {
  iree_codegen.ukernel.generic "vmvx.mmt4d.i8i8i32" ins(%lhs, %rhs : memref<?x?x?x?xi8>, memref<?x?x?x?xi8>)
      outs(%acc : memref<?x?x?x?xi32>) (%m, %n, %k, %m0, %n0, %k0, %flags : index, index, index, i32, i32, i32, i32) strided_outer_dims(1)
  return
}
// CHECK-LABEL: func.func private @vmvx.mmt4d.i8i8i32
//  CHECK-SAME:     (memref<i8>, index, index, memref<i8>, index, index,
//  CHECK-SAME:     memref<i32>, index, index, index, index, index, i32, i32, i32, i32)
// CHECK-LABEL: func.func @ukernel_mmt4d_i8i8i32(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: memref<?x?x?x?xi8>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: memref<?x?x?x?xi8>
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: memref<?x?x?x?xi32>
//  CHECK-SAME:     %[[M:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[N:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[K:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[M0:[a-zA-Z0-9]+]]: i32
//  CHECK-SAME:     %[[N0:[a-zA-Z0-9]+]]: i32
//  CHECK-SAME:     %[[K0:[a-zA-Z0-9]+]]: i32
//  CHECK-SAME:     %[[FLAGS:[a-zA-Z0-9]+]]: i32
//       CHECK:   %[[BASE0:.+]], %[[OFFSET0:.+]], %[[SIZE0:.+]]:4, %[[STRIDES0:.+]]:4 = memref.extract_strided_metadata %[[ARG0]]
//       CHECK:   %[[BASE1:.+]], %[[OFFSET1:.+]], %[[SIZE1:.+]]:4, %[[STRIDES1:.+]]:4 = memref.extract_strided_metadata %[[ARG1]]
//       CHECK:   %[[BASE2:.+]], %[[OFFSET2:.+]], %[[SIZE2:.+]]:4, %[[STRIDES2:.+]]:4 = memref.extract_strided_metadata %[[ARG2]]
//       CHECK:   call @vmvx.mmt4d.i8i8i32(%[[BASE0]], %[[OFFSET0]], %[[STRIDES0]]#0
//  CHECK-SAME:       %[[BASE1]], %[[OFFSET1]], %[[STRIDES1]]#0
//  CHECK-SAME:       %[[BASE2]], %[[OFFSET2]], %[[STRIDES2]]#0
//  CHECK-SAME:       %[[M]], %[[N]], %[[K]],
//  CHECK-SAME:       %[[M0]], %[[N0]], %[[K0]], %[[FLAGS]])