// RUN: iree-opt -pass-pipeline="hal.executable(hal.executable.variant(iree-codegen-distribute-to-workgroups{tile-sizes=3,5})),cse" %s | FileCheck %s

#executable_layout = #hal.executable.layout<push_constants = 3, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @simple_distribute {
  hal.executable.variant @llvm, target = <"llvm", "embedded-elf-arm_64", {}> {
    hal.executable.export @simple_distribute layout(#executable_layout)
    builtin.module {
      func.func @simple_distribute() {
        %m = hal.interface.constant.load[0] : index
        %n = hal.interface.constant.load[1] : index
        %k = hal.interface.constant.load[2] : index
        %cst = arith.constant 0.0 : f32
        %lhs_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:?x?xf32>{%m, %k}
        %rhs_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:?x?xf32>{%k, %n}
        %result_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:?x?xf32>{%m, %n}
        %lhs = flow.dispatch.tensor.load %lhs_binding, offsets = [0, 0], sizes = [%m, %k], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:?x?xf32>{%m, %k} -> tensor<?x?xf32>
        %rhs = flow.dispatch.tensor.load %rhs_binding, offsets = [0, 0], sizes = [%k, %n], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:?x?xf32>{%k, %n} -> tensor<?x?xf32>
        %init = linalg.init_tensor [%m, %n] : tensor<?x?xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
        %gemm = linalg.matmul ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>) outs(%fill : tensor<?x?xf32>) -> tensor<?x?xf32>
        flow.dispatch.tensor.store %gemm, %result_binding, offsets = [0, 0], sizes = [%m, %n], strides = [1, 1]
            : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:?x?xf32>{%m, %n}
        return
      }
    }
  }
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 5)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 3)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<()[s0, s1] -> (5, s0 - s1 * 5)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<()[s0, s1] -> (3, s0 - s1 * 3)>
//  CHECK-DAG: #[[MAP4:.+]] = affine_map<()[s0] -> (s0 * 3)>
//  CHECK-DAG: #[[MAP5:.+]] = affine_map<()[s0] -> (s0 * 5)>
//      CHECK: hal.executable.export public @simple_distribute
// CHECK-NEXT:   ^bb0(%{{.+}}: !hal.device, %[[ARG1:[a-zA-Z0-9]+]]: index, %[[ARG2:[a-zA-Z0-9]+]]: index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[NWG_X:.+]] = affine.apply #[[MAP0]]()[%[[ARG1]]]
//  CHECK-DAG:   %[[NWG_Y:.+]] = affine.apply #[[MAP1]]()[%[[ARG2]]]
//      CHECK:   hal.return %[[NWG_X]], %[[NWG_Y]], %[[C1]]
//      CHECK: func.func @simple_distribute()
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load[0] : index
//  CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load[1] : index
//  CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load[2] : index
//  CHECK-DAG:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
//  CHECK-DAG:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
//  CHECK-DAG:   %[[RESULT_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(2)
//  CHECK-DAG:   %[[WGID_X:.+]] = hal.interface.workgroup.id[0]
//  CHECK-DAG:   %[[TS_X:.+]] = affine.min #[[MAP2]]()[%[[N]], %[[WGID_X]]]
//  CHECK-DAG:   %[[WGID_Y:.+]] = hal.interface.workgroup.id[1]
//  CHECK-DAG:   %[[TS_Y:.+]] = affine.min #[[MAP3]]()[%[[M]], %[[WGID_Y]]]
//  CHECK-DAG:   %[[OFFSET_Y:.+]] = affine.apply #[[MAP4]]()[%[[WGID_Y]]]
//  CHECK-DAG:   %[[LHS_TILE:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
// CHECK-SAME:       offsets = [%[[OFFSET_Y]], %[[C0]]], sizes = [%[[TS_Y]], %[[K]]], strides = [%[[C1]], %[[C1]]]
//  CHECK-DAG:   %[[OFFSET_X:.+]] = affine.apply #[[MAP5]]()[%[[WGID_X]]]
//  CHECK-DAG:   %[[RHS_TILE:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
// CHECK-SAME:       offsets = [%[[C0]], %[[OFFSET_X]]], sizes = [%[[K]], %[[TS_X]]], strides = [%[[C1]], %[[C1]]]
//  CHECK-DAG:   %[[INIT_TILE:.+]] = linalg.init_tensor [%[[TS_Y]], %[[TS_X]]]
//      CHECK:   %[[FILL_TILE:.+]] = linalg.fill
// CHECK-SAME:       outs(%[[INIT_TILE]] :
//      CHECK:   %[[GEMM_TILE:.+]] = linalg.matmul ins(%[[LHS_TILE]], %[[RHS_TILE]] :
// CHECK-SAME:       outs(%[[FILL_TILE]] :
//      CHECK:   flow.dispatch.tensor.store %[[GEMM_TILE]], %[[RESULT_BINDING]]
// CHECK-SAME:       offsets = [%[[OFFSET_Y]], %[[OFFSET_X]]], sizes = [%[[TS_Y]], %[[TS_X]]], strides = [%[[C1]], %[[C1]]]
