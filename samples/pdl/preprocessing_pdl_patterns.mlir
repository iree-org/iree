// RUN: iree-compile --iree-preprocessing-pdl-spec-filename=%p/linalg_pdl.mlir %s --compile-to=preprocessing | FileCheck %s

// CHECK-LABEL:   stream.executable private @mlp_external_executable
//       CHECK:   stream.executable.export public @mlp_external_entry_point
//       CHECK:   builtin.module
//       CHECK:     func.func private @mlp_external
//  CHECK-SAME:         (memref<f32>, index, memref<f32>, index, memref<f32>, index, i32, i32, i32)
//  CHECK-SAME:         attributes {llvm.bareptr = [true]}
//       CHECK:     func.func @mlp_external_entry_point
//  CHECK-SAME:         %[[ARG0:[a-zA-Z0-9]+]]: !stream.binding
//  CHECK-SAME:         %[[ARG1:[a-zA-Z0-9]+]]: !stream.binding
//  CHECK-SAME:         %[[ARG2:[a-zA-Z0-9]+]]: !stream.binding
//  CHECK-SAME:         %[[ARG3:[a-zA-Z0-9]+]]: i32
//  CHECK-SAME:         %[[ARG4:[a-zA-Z0-9]+]]: i32
//  CHECK-SAME:         %[[ARG5:[a-zA-Z0-9]+]]: i32
//  CHECK-SAME:         %[[ARG6:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:         %[[ARG7:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:         %[[ARG8:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:         %[[ARG9:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:         %[[ARG10:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:         %[[ARG11:[a-zA-Z0-9]+]]: index
//       CHECK:       %[[C0:.+]] = arith.constant 0 : index
//       CHECK:       %[[STREAM0:.+]] = stream.binding.subspan %[[ARG0]][%[[C0]]] : !stream.binding -> memref<?x?xf32>{%[[ARG6]], %[[ARG7]]}
//  CHECK-NEXT:       %[[STREAM0_BASE:[a-zA-Z0-9_]+]],
//  CHECK-SAME:             = memref.extract_strided_metadata %[[STREAM0]]
//       CHECK:       %[[STREAM1:.+]] = stream.binding.subspan %[[ARG1]][%[[C0]]] : !stream.binding -> memref<?x?xf32>{%[[ARG8]], %[[ARG9]]}
//  CHECK-NEXT:       %[[STREAM1_BASE:[a-zA-Z0-9_]+]],
//  CHECK-SAME:             = memref.extract_strided_metadata %[[STREAM1]]
//       CHECK:       %[[STREAM2:.+]] = stream.binding.subspan %[[ARG2]][%[[C0]]] : !stream.binding -> memref<?x?xf32>{%[[ARG10]], %[[ARG11]]}
//  CHECK-NEXT:       %[[STREAM2_BASE:[a-zA-Z0-9_]+]],
//  CHECK-SAME:             = memref.extract_strided_metadata %[[STREAM2]]
//       CHECK:       call @mlp_external
//  CHECK-SAME:           %[[STREAM0_BASE]], %[[C0]], %[[STREAM1_BASE]], %[[C0]], %[[STREAM2_BASE]], %[[C0]], %[[ARG3]], %[[ARG4]], %[[ARG5]]

//       CHECK:     util.func public @mlp_invocation
//  CHECK-SAME:         (%[[ARG0:[a-zA-Z0-9]+]]: !hal.buffer_view, %[[ARG1:[a-zA-Z0-9]+]]: !hal.buffer_view)
//   CHECK-DAG:       %[[MDIM0:.+]] = hal.buffer_view.dim<%[[ARG0]] : !hal.buffer_view>[0] : index
//   CHECK-DAG:       %[[MDIM1:.+]] = hal.buffer_view.dim<%[[ARG0]] : !hal.buffer_view>[1] : index
//   CHECK-DAG:       %[[LHS:.+]] = hal.tensor.import %[[ARG0]] "input0" : !hal.buffer_view -> tensor<?x?xf32>{%[[MDIM0]], %[[MDIM1]]}
//   CHECK-DAG:       %[[NDIM0:.+]] = hal.buffer_view.dim<%[[ARG1]] : !hal.buffer_view>[0] : index
//   CHECK-DAG:       %[[NDIM1:.+]] = hal.buffer_view.dim<%[[ARG1]] : !hal.buffer_view>[1] : index
//   CHECK-DAG:       %[[RHS:.+]] = hal.tensor.import %[[ARG1]] "input1" : !hal.buffer_view -> tensor<?x?xf32>{%[[NDIM0]], %[[NDIM1]]}
//   CHECK-DAG:       %[[M_I32:.+]] = arith.index_cast %[[MDIM0]] : index to i32
//   CHECK-DAG:       %[[N_I32:.+]] = arith.index_cast %[[NDIM1]] : index to i32
//   CHECK-DAG:       %[[K_I32:.+]] = arith.index_cast %[[MDIM1]] : index to i32
//       CHECK:       %[[RESULT:.+]] = flow.dispatch
//  CHECK-SAME:           @mlp_external_executable::@mlp_external_entry_point
//  CHECK-SAME:           (%[[LHS]], %[[RHS]], %[[M_I32]], %[[N_I32]], %[[K_I32]], %[[MDIM0]], %[[MDIM1]], %[[NDIM0]], %[[NDIM1]], %[[MDIM0]], %[[NDIM1]])
//       CHECK:       linalg.generic
//  CHECK-SAME:           ins(%[[RESULT]] :

#x86_64_target = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 32 : index,
  target_triple = "x86_64-none-elf"
}>

// The target devices that the program will run on. We can compile and run with
// multiple targets, but this example is maintaining an implicit requirement
// that the custom kernel being spliced in is supported by the target device,
// hence we only support llvm-cpu here.
#cpu_target = #hal.device.target<"llvm-cpu", [
  #x86_64_target
]>

#map = affine_map<(d0, d1) -> (d0, d1)>
module @example attributes {hal.device.targets = [#cpu_target]} {

  func.func @mlp_invocation(%lhs: tensor<?x?xf32>,
                            %rhs: tensor<?x?xf32>) -> (tensor<?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.0 : f32
    %dim0 = tensor.dim %lhs, %c0 : tensor<?x?xf32>
    %dim1 = tensor.dim %rhs, %c1 : tensor<?x?xf32>
    %empty = tensor.empty(%dim0, %dim1) : tensor<?x?xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
    %matmul = linalg.matmul ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
        outs(%fill : tensor<?x?xf32>) -> tensor<?x?xf32>
    %relu = linalg.generic {
        indexing_maps = [#map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%matmul : tensor<?x?xf32>) outs(%empty : tensor<?x?xf32>) {
      ^bb0(%b0 : f32, %b1 : f32):
        %0 = arith.maximumf %b0, %cst : f32
        linalg.yield %0 : f32
      } -> tensor<?x?xf32>
    %neg = linalg.generic {
        indexing_maps = [#map, #map],
        iterator_types  = ["parallel", "parallel"]}
        ins(%relu : tensor<?x?xf32>) outs(%empty : tensor<?x?xf32>) {
      ^bb0(%b0 : f32, %b1 : f32):
        %0 = arith.negf %b0 : f32
        linalg.yield %0 : f32
    } -> tensor<?x?xf32>
    return %neg : tensor<?x?xf32>
  }
}  // module
