// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-codegen-tile-and-distribute-to-workgroups)), canonicalize, cse)' --split-input-file %s | FileCheck %s

#config = #iree_codegen.lowering_config<tile_sizes = [[64, 64, 0], [16, 4, 0], [0, 0, 64]]>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>
  ]>
]>
#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "aarch64-unknown-unknown-eabi-elf"
}>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
hal.executable private @matmul_tensors {
  hal.executable.variant public @llvm, target = #executable_target_embedded_elf_arm_64_ {
    hal.executable.export public @matmul_tensors layout(#pipeline_layout) attributes {translation_info = #translation} {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_tensors() {
        %0 = hal.interface.constant.load[0] : index
        %1 = hal.interface.constant.load[1] : index
        %2 = hal.interface.constant.load[2] : index
        %3 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %2}
        %4 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %1}
        %5 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1}
        %6 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
        %7 = flow.dispatch.tensor.load %3, offsets = [0, 0], sizes = [%0, %2], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %2} -> tensor<?x?xf32>
        %8 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [%2, %1], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %1} -> tensor<?x?xf32>
        %9 = flow.dispatch.tensor.load %5, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1} -> tensor<?x?xf32>
        %10 = linalg.matmul {lowering_config = #config}
            ins(%7, %8 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%9 : tensor<?x?xf32>) -> tensor<?x?xf32>
        flow.dispatch.tensor.store %10, %6, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1]
            : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
        return
      }
    }
  }
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 64)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 * 64)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 64)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: hal.executable.export public @matmul_tensors
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
// CHECK-NEXT:   (%[[DEVICE:.+]]: !hal.device,
// CHECK-SAME:    %[[WORKLOAD_M:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_N:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_K:[a-zA-Z0-9_]+]]: index)
//  CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:    %[[D0:.+]] = affine.apply #[[MAP0]]()[%[[WORKLOAD_M]]]
//  CHECK-DAG:    %[[D1:.+]] = affine.apply #[[MAP0]]()[%[[WORKLOAD_N]]]
//      CHECK:    hal.return %[[D1]], %[[D0]], %[[C1]] : index, index, index
//      CHECK: func.func @matmul_tensors()
//  CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load[0]
//  CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load[1]
//  CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load[2]
//  CHECK-DAG:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
//  CHECK-DAG:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
//  CHECK-DAG:   %[[INIT_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(2)
//  CHECK-DAG:   %[[OUT_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(3)
//  CHECK-DAG:   %[[WG_ID_X:.+]] = hal.interface.workgroup.id[0]
//  CHECK-DAG:   %[[WG_COUNT_X:.+]] = hal.interface.workgroup.count[0]
//  CHECK-DAG:   %[[WG_ID_Y:.+]] = hal.interface.workgroup.id[1]
//  CHECK-DAG:   %[[WG_COUNT_Y:.+]] = hal.interface.workgroup.count[1]
//  CHECK-DAG:   %[[LB_Y:.+]] = affine.apply #[[MAP1]]()[%[[WG_ID_Y]]]
//  CHECK-DAG:   %[[STEP_Y:.+]] = affine.apply #[[MAP1]]()[%[[WG_COUNT_Y]]]
//      CHECK:   scf.for %[[IV0:.+]] = %[[LB_Y]] to %[[M]] step %[[STEP_Y]]
//  CHECK-DAG:     %[[TILESIZE_M:.+]] = affine.min #[[MAP2]](%[[IV0]])[%[[M]]]
//  CHECK-DAG:     %[[LB_X:.+]] = affine.apply #[[MAP1]]()[%[[WG_ID_X]]]
//  CHECK-DAG:     %[[STEP_X:.+]] = affine.apply #[[MAP1]]()[%[[WG_COUNT_X]]]
//      CHECK:     scf.for %[[IV1:.+]] = %[[LB_X]] to %[[N]] step %[[STEP_X]]
//  CHECK-DAG:       %[[TILESIZE_N:.+]] = affine.min #[[MAP2]](%[[IV1]])[%[[N]]]
//  CHECK-DAG:       %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]], offsets = [%[[IV0]], 0], sizes = [%[[TILESIZE_M]], %[[K]]]
//  CHECK-DAG:       %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]], offsets = [0, %[[IV1]]], sizes = [%[[K]], %[[TILESIZE_N]]]
//  CHECK-DAG:       %[[INIT:.+]] = flow.dispatch.tensor.load %[[INIT_BINDING]], offsets = [%[[IV0]], %[[IV1]]], sizes = [%[[TILESIZE_M]], %[[TILESIZE_N]]]
//      CHECK:       %[[GEMM:.+]] = linalg.matmul
// CHECK-SAME:           ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:           outs(%[[INIT]] :
//      CHECK:       flow.dispatch.tensor.store %[[GEMM]], %[[OUT_BINDING]]
// CHECK-SAME:           offsets = [%[[IV0]], %[[IV1]]], sizes = [%[[TILESIZE_M]], %[[TILESIZE_N]]]

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[64, 64], [1, 4], [0, 0]]>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "x86_64-unknown-linux-gnu"
}>
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
hal.executable private @add {
  hal.executable.variant public @llvm, target = #executable_target_embedded_elf_x86_64_ {
    hal.executable.export public @add layout(#pipeline_layout) attributes {translation_info = #translation} {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @add() {
        %0 = hal.interface.constant.load[0] : index
        %1 = hal.interface.constant.load[1] : index
        %2 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1}
        %3 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%1}
        %4 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
        %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1} -> tensor<?x?xf32>
        %6 = flow.dispatch.tensor.load %3, offsets = [0], sizes = [%1], strides = [1]
            : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%1} -> tensor<?xf32>
        %7 = tensor.empty(%0, %1) : tensor<?x?xf32>
        %8 = linalg.generic {
            indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]}
            ins(%5, %6 : tensor<?x?xf32>, tensor<?xf32>) outs(%7 : tensor<?x?xf32>)
            attrs =  {lowering_config = #config} {
        ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
          %9 = arith.addf %arg0, %arg1 : f32
          linalg.yield %9 : f32
        } -> tensor<?x?xf32>
        flow.dispatch.tensor.store %8, %4, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1]
            : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
        return
      }
    }
  }
}
//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0] -> (s0 ceildiv 64)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: hal.executable private @add
//      CHECK: hal.executable.export public @add
// CHECK-SAME:  translation_info = #[[TRANSLATION]]
// CHECK-NEXT:   (%[[DEVICE:.+]]: !hal.device,
// CHECK-SAME:    %[[WORKLOAD_0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_1:[a-zA-Z0-9_]+]]: index)
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[D0:.+]] = affine.apply #[[MAP]]()[%[[WORKLOAD_0]]]
//  CHECK-DAG:   %[[D1:.+]] = affine.apply #[[MAP]]()[%[[WORKLOAD_1]]]
//      CHECK:   hal.return %[[D1]], %[[D0]], %[[C1]] : index, index, index
//      CHECK: func.func @add()
//      CHECK:   scf.for %[[IV0:.+]] =
//      CHECK:     scf.for %[[IV1:.+]] =
//      CHECK:       %[[RESULT:.+]] = linalg.generic
//      CHECK:       flow.dispatch.tensor.store %[[RESULT]], %{{.+}}, offsets = [%[[IV0]], %[[IV1]]]

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[0, 64, 64, 64], [1, 1, 1, 4], [0, 0, 0, 0]]>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>
  ]>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "x86_64-unknown-linux-gnu"}>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
hal.executable private @add4D {
  hal.executable.variant public @llvm, target = #executable_target_embedded_elf_x86_64_ {
    hal.executable.export public @add4D layout(#pipeline_layout) attributes {translation_info = #translation} {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index, %arg4 :index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3, %arg4
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @add4D() {
        %0 = hal.interface.constant.load[0] : index
        %1 = hal.interface.constant.load[1] : index
        %2 = hal.interface.constant.load[2] : index
        %3 = hal.interface.constant.load[3] : index
        %4 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(32)
            : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3}
        %5 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(32)
            : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3}
        %6 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(32)
            : !flow.dispatch.tensor<writeonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3}
        %7 = flow.dispatch.tensor.load %4, offsets = [0, 0, 0, 0], sizes = [%0, %1, %2, %3], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3} -> tensor<?x?x?x?xf32>
        %8 = flow.dispatch.tensor.load %5, offsets = [0, 0, 0, 0], sizes = [%0, %1, %2, %3], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3} -> tensor<?x?x?x?xf32>
        %9 = tensor.empty(%0, %1, %2, %3) : tensor<?x?x?x?xf32>
        %10 = linalg.generic {
            indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
            ins(%7, %8 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) outs(%9 : tensor<?x?x?x?xf32>) attrs =  {lowering_config = #config} {
        ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
          %11 = arith.addf %arg0, %arg1 : f32
          linalg.yield %11 : f32
        } -> tensor<?x?x?x?xf32>
        flow.dispatch.tensor.store %10, %6, offsets = [0, 0, 0, 0], sizes = [%0, %1, %2, %3], strides = [1, 1, 1, 1]
            : tensor<?x?x?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3}
        return
      }
    }
  }
}
//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0] -> (s0 ceildiv 64)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: hal.executable.export public @add4D
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
// CHECK-NEXT:   (%[[DEVICE:.+]]: !hal.device,
// CHECK-SAME:    %[[WORKLOAD_0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_3:[a-zA-Z0-9_]+]]: index)
//  CHECK-DAG:    %[[D0:.+]] = affine.apply #[[MAP]]()[%[[WORKLOAD_1]]
//  CHECK-DAG:    %[[D1:.+]] = affine.apply #[[MAP]]()[%[[WORKLOAD_2]]]
//  CHECK-DAG:    %[[D2:.+]] = affine.apply #[[MAP]]()[%[[WORKLOAD_3]]]
//      CHECK:    hal.return %[[D2]], %[[D1]], %[[D0]] : index, index, index
//      CHECK: func.func @add4D()
//      CHECK:   scf.for %[[IV0:.+]] =
//      CHECK:     scf.for %[[IV1:.+]] =
//      CHECK:       scf.for %[[IV2:.+]] =
//  CHECK-NOT:         scf.for
//      CHECK:         %[[GENERIC:.+]] = linalg.generic
//      CHECK:         flow.dispatch.tensor.store %[[GENERIC]], %{{.+}}, offsets = [0, %[[IV0]], %[[IV1]], %[[IV2]]]

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[2, 64, 64, 64], [1, 1, 1, 4], [0, 0, 0, 0]]>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>
  ]>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "x86_64-unknown-linux-gnu"}>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
hal.executable private @add_distribute4D {
  hal.executable.variant public @llvm, target = #executable_target_embedded_elf_x86_64_ {
    hal.executable.export public @add_distribute4D layout(#pipeline_layout) attributes {translation_info = #translation} {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index, %arg4 :index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3, %arg4
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @add_distribute4D() {
        %0 = hal.interface.constant.load[0] : index
        %1 = hal.interface.constant.load[1] : index
        %2 = hal.interface.constant.load[2] : index
        %3 = hal.interface.constant.load[3] : index
        %4 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(32)
            : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3}
        %5 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(32)
            : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3}
        %6 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(32)
            : !flow.dispatch.tensor<writeonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3}
        %7 = flow.dispatch.tensor.load %4, offsets = [0, 0, 0, 0], sizes = [%0, %1, %2, %3], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3} -> tensor<?x?x?x?xf32>
        %8 = flow.dispatch.tensor.load %5, offsets = [0, 0, 0, 0], sizes = [%0, %1, %2, %3], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3} -> tensor<?x?x?x?xf32>
        %9 = tensor.empty(%0, %1, %2, %3) : tensor<?x?x?x?xf32>
        %10 = linalg.generic {
            indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
            ins(%7, %8 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) outs(%9 : tensor<?x?x?x?xf32>) attrs =  {lowering_config = #config} {
        ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
          %11 = arith.addf %arg0, %arg1 : f32
          linalg.yield %11 : f32
        } -> tensor<?x?x?x?xf32>
        flow.dispatch.tensor.store %10, %6, offsets = [0, 0, 0, 0], sizes = [%0, %1, %2, %3], strides = [1, 1, 1, 1]
            : tensor<?x?x?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3}
        return
      }
    }
  }
}
//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0] -> (s0 ceildiv 64)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0, s1] -> ((s0 ceildiv 64) * (s1 ceildiv 2))>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<()[s0] -> (s0 * 2)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<()[s0] -> (s0 * 64)>

//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: hal.executable.export public @add_distribute4D
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
// CHECK-NEXT:   (%[[DEVICE:.+]]: !hal.device,
// CHECK-SAME:    %[[WORKLOAD_0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_3:[a-zA-Z0-9_]+]]: index)
//  CHECK-DAG:    %[[D0:.+]] = affine.apply #[[MAP1]]()[%[[WORKLOAD_1]], %[[WORKLOAD_0]]
//  CHECK-DAG:    %[[D1:.+]] = affine.apply #[[MAP]]()[%[[WORKLOAD_2]]]
//  CHECK-DAG:    %[[D2:.+]] = affine.apply #[[MAP]]()[%[[WORKLOAD_3]]]
//      CHECK:    hal.return %[[D2]], %[[D1]], %[[D0]] : index, index, index
//      CHECK: func.func @add_distribute4D()
//  CHECK-DAG:   %[[D0:.+]] = hal.interface.constant.load[0] : index
//  CHECK-DAG:   %[[D1:.+]] = hal.interface.constant.load[1] : index
//  CHECK-DAG:   %[[D2:.+]] = hal.interface.constant.load[2] : index
//  CHECK-DAG:   %[[D3:.+]] = hal.interface.constant.load[3] : index
//  CHECK-DAG:   %[[IDX:.+]] = hal.interface.workgroup.id[0] : index
//  CHECK-DAG:   %[[IDY:.+]] = hal.interface.workgroup.id[1] : index
//  CHECK-DAG:   %[[IDZ:.+]] = hal.interface.workgroup.id[2] : index
//  CHECK-DAG:   %[[NUMT:.+]] = affine.apply #[[MAP]]()[%[[D1]]]
//  CHECK-DAG:   %[[ID3:.+]] = arith.remui %[[IDZ]], %[[NUMT]] : index
//  CHECK-DAG:   %[[ID4:.+]] = arith.divui %[[IDZ]], %[[NUMT]] : index
//  CHECK-DAG:   %[[L4:.+]] = affine.apply #[[MAP2]]()[%[[ID4]]]
//      CHECK:   scf.for %[[IV0:.+]] = %[[L4]]
//      CHECK:     %[[L3:.+]] = affine.apply #[[MAP3]]()[%[[ID3]]]
//      CHECK:     scf.for %[[IV1:.+]] = %[[L3]]
//      CHECK:       scf.for %[[IV2:.+]] =
//      CHECK:         scf.for %[[IV3:.+]] =
//  CHECK-NOT:         scf.for
//      CHECK:         %[[GENERIC:.+]] = linalg.generic
//      CHECK:         flow.dispatch.tensor.store %[[GENERIC]], %{{.+}}, offsets = [%[[IV0]], %[[IV1]], %[[IV2]], %[[IV3]]]


// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[2, 64, 0, 64], [1, 1, 1, 4], [0, 0, 0, 0]]>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>
  ]>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "x86_64-unknown-linux-gnu"}>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
hal.executable private @add_distribute4D_zero_tile_size {
  hal.executable.variant public @llvm, target = #executable_target_embedded_elf_x86_64_ {
    hal.executable.export public @add_distribute4D_zero_tile_size layout(#pipeline_layout) attributes {translation_info = #translation} {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index, %arg4 :index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3, %arg4
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @add_distribute4D_zero_tile_size() {
        %0 = hal.interface.constant.load[0] : index
        %1 = hal.interface.constant.load[1] : index
        %2 = hal.interface.constant.load[2] : index
        %3 = hal.interface.constant.load[3] : index
        %4 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(32)
            : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3}
        %5 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(32)
            : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3}
        %6 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(32)
            : !flow.dispatch.tensor<writeonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3}
        %7 = flow.dispatch.tensor.load %4, offsets = [0, 0, 0, 0], sizes = [%0, %1, %2, %3], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3} -> tensor<?x?x?x?xf32>
        %8 = flow.dispatch.tensor.load %5, offsets = [0, 0, 0, 0], sizes = [%0, %1, %2, %3], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3} -> tensor<?x?x?x?xf32>
        %9 = tensor.empty(%0, %1, %2, %3) : tensor<?x?x?x?xf32>
        %10 = linalg.generic {
            indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
            ins(%7, %8 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) outs(%9 : tensor<?x?x?x?xf32>) attrs =  {lowering_config = #config} {
        ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
          %11 = arith.addf %arg0, %arg1 : f32
          linalg.yield %11 : f32
        } -> tensor<?x?x?x?xf32>
        flow.dispatch.tensor.store %10, %6, offsets = [0, 0, 0, 0], sizes = [%0, %1, %2, %3], strides = [1, 1, 1, 1]
            : tensor<?x?x?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3}
        return
      }
    }
  }
}
//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0] -> (s0 ceildiv 64)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 2)>

//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: hal.executable.export public @add_distribute4D_zero_tile_size
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
// CHECK-NEXT:   (%[[DEVICE:.+]]: !hal.device,
// CHECK-SAME:    %[[WORKLOAD_0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_3:[a-zA-Z0-9_]+]]: index)
//  CHECK-DAG:    %[[D0:.+]] = affine.apply #[[MAP1]]()[%[[WORKLOAD_0]]]
//  CHECK-DAG:    %[[D1:.+]] = affine.apply #[[MAP]]()[%[[WORKLOAD_1]]]
//  CHECK-DAG:    %[[D2:.+]] = affine.apply #[[MAP]]()[%[[WORKLOAD_3]]]
//      CHECK:    hal.return %[[D2]], %[[D1]], %[[D0]] : index, index, index
//      CHECK: func.func @add_distribute4D_zero_tile_size()


// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[1, 64, 64, 0], [1, 16, 4, 0], [0, 0, 0, 64]]>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "aarch64-unknown-unknown-eabi-elf"}>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
hal.executable private @batch_matmul_tensors {
  hal.executable.variant public @llvm, target = #executable_target_embedded_elf_arm_64_ {
    hal.executable.export public @batch_matmul_tensors layout(#pipeline_layout) attributes {translation_info = #translation} {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index, %arg4 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3, %arg4
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @batch_matmul_tensors() {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.constant.load[0] : index
        %1 = hal.interface.constant.load[1] : index
        %2 = hal.interface.constant.load[2] : index
        %3 = hal.interface.constant.load[3] : index
        %4 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(32)
            : !flow.dispatch.tensor<readonly:tensor<?x?x?xf32>>{%0, %1, %3}
        %5 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(32)
            : !flow.dispatch.tensor<readonly:tensor<?x?x?xf32>>{%0, %3, %2}
        %6 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(32)
            : !flow.dispatch.tensor<writeonly:tensor<?x?x?xf32>>{%0, %1, %2}
        %7 = flow.dispatch.tensor.load %4, offsets = [0, 0, 0], sizes = [%0, %1, %3], strides = [1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?x?xf32>>{%0, %1, %3} -> tensor<?x?x?xf32>
        %8 = flow.dispatch.tensor.load %5, offsets = [0, 0, 0], sizes = [%0, %3, %2], strides = [1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?x?xf32>>{%0, %3, %2} -> tensor<?x?x?xf32>
        %9 = tensor.empty(%0, %1, %2) : tensor<?x?x?xf32>
        %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
        %11 = linalg.batch_matmul {lowering_config = #config}
            ins(%7, %8 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%10 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
        flow.dispatch.tensor.store %11, %6, offsets = [0, 0, 0], sizes = [%0, %1, %2], strides = [1, 1, 1]
            : tensor<?x?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?x?xf32>>{%0, %1, %2}
        return
      }
    }
  }
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 64)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: hal.executable.export public @batch_matmul_tensors
// CHECK-NEXT:   (%[[DEVICE:.+]]: !hal.device,
// CHECK-SAME:    %[[WORKLOAD_0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_3:[a-zA-Z0-9_]+]]: index)
//  CHECK-DAG:   %[[D0:.+]] = affine.apply #[[MAP0]]()[%[[WORKLOAD_1]]]
//  CHECK-DAG:   %[[D1:.+]] = affine.apply #[[MAP0]]()[%[[WORKLOAD_2]]]
//      CHECK:   hal.return %[[D1]], %[[D0]], %[[WORKLOAD_0]]
//      CHECK: func.func @batch_matmul_tensors()
//      CHECK:   scf.for %[[IV0:.+]] =
//      CHECK:     scf.for %[[IV1:.+]] =
//      CHECK:       scf.for %[[IV2:.+]] =
//  CHECK-NOT:         scf.for
//      CHECK:         %[[BATCH_GEMM:.+]] = linalg.batch_matmul
//      CHECK:         flow.dispatch.tensor.store %[[BATCH_GEMM]]
// CHECK-SAME:             offsets = [%[[IV0]], %[[IV1]], %[[IV2]]], sizes = [1, %{{.+}}, %{{.+}}]

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[32, 16, 0], [16, 8, 0], [0, 0, 2]]>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64">
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
hal.executable private @preset_config_matmul_tensors {
  hal.executable.variant public @system_elf_x86_64, target = #executable_target_system_elf_x86_64_ {
    hal.executable.export public @preset_config layout(#pipeline_layout) attributes {translation_info = #translation} {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @preset_config() {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<128x256xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<256x512xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:tensor<128x512xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 256], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<128x256xf32>> -> tensor<128x256xf32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 512], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<256x512xf32>> -> tensor<256x512xf32>
        %5 = tensor.empty() : tensor<128x512xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<128x512xf32>) -> tensor<128x512xf32>
        %7 = linalg.matmul {lowering_config = #config}
            ins(%3, %4 : tensor<128x256xf32>, tensor<256x512xf32>) outs(%6 : tensor<128x512xf32>) -> tensor<128x512xf32>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [128, 512], strides = [1, 1]
            : tensor<128x512xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x512xf32>>
        return
      }
    }
  }
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 * 32)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 * 16)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: hal.executable.export public @preset_config
// CHECK-NEXT:   (%[[DEVICE:.+]]: !hal.device,
// CHECK-SAME:    %[[WORKLOAD_0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_2:[a-zA-Z0-9_]+]]: index)
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
//  CHECK-DAG:   %[[C32:.+]] = arith.constant 32 : index
//      CHECK:   hal.return %[[C32]], %[[C4]], %[[C1]]
//      CHECK: func.func @preset_config()
//      CHECK:   scf.for %[[IV0:.+]] =
//      CHECK:     scf.for %[[IV1:.+]] =
//  CHECK-DAG:       %[[LHS:.+]] = flow.dispatch.tensor.load %{{.+}}, offsets = [%[[IV0]], 0], sizes = [32, 256]
//  CHECK-DAG:       %[[RHS:.+]] = flow.dispatch.tensor.load %{{.+}}, offsets = [0, %[[IV1]]], sizes = [256, 16]
//  CHECK-DAG:       %[[INIT:.+]] = tensor.empty
//  CHECK-DAG:       %[[FILL:.+]] = linalg.fill
// CHECK-SAME:           outs(%[[INIT]] :
//  CHECK-DAG:       %[[GEMM:.+]] = linalg.matmul
// CHECK-SAME:           outs(%[[FILL]] :
//      CHECK:       flow.dispatch.tensor.store %[[GEMM]]
// CHECK-SAME:           offsets = [%[[IV0]], %[[IV1]]], sizes = [32, 16]

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[64, 64]]>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64">
#translation = #iree_codegen.translation_info<CPUBufferOpsTileAndVectorize>
hal.executable public @copy_op {
  hal.executable.variant public @system_elf_x86_64, target = #executable_target_system_elf_x86_64_ {
    hal.executable.export public @copy_op layout(#pipeline_layout) attributes {translation_info = #translation} {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @copy_op() {
        %source_size_y = hal.interface.constant.load[0] : index
        %source_size_x = hal.interface.constant.load[1] : index
        %dest_size_y = hal.interface.constant.load[2] : index
        %dest_size_x = hal.interface.constant.load[3] : index
        %source_offset_y = hal.interface.constant.load[4] : index
        %source_offset_x = hal.interface.constant.load[5] : index
        %dest_offset_y = hal.interface.constant.load[6] : index
        %dest_offset_x = hal.interface.constant.load[7] : index
        %slice_size_y = hal.interface.constant.load[8] : index
        %slice_size_x = hal.interface.constant.load[9] : index
        %source = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<?x?xi32>{%source_size_y, %source_size_x}
        %dest = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<?x?xi32>{%dest_size_y, %dest_size_x}
        %source_subview = memref.subview %source[%source_offset_y, %source_offset_x] [%slice_size_y, %slice_size_x] [1, 1] : memref<?x?xi32> to memref<?x?xi32, strided<[?, ?], offset : ?>>
        %dest_subview = memref.subview %dest[%dest_offset_y, %dest_offset_x] [%slice_size_y, %slice_size_x] [1, 1] : memref<?x?xi32> to memref<?x?xi32, strided<[?, ?], offset : ?>>
        linalg.generic {
            indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
            iterator_types = ["parallel", "parallel"]}
            ins(%source_subview : memref<?x?xi32, strided<[?, ?], offset : ?>>)
            outs(%dest_subview : memref<?x?xi32, strided<[?, ?], offset : ?>>)
            attrs = {lowering_config = #config} {
          ^bb0(%arg0: i32, %arg1: i32):
            linalg.yield %arg0 : i32
          }
        return
      }
    }
  }
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 64)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 * 64)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 64)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUBufferOpsTileAndVectorize>
//      CHECK: hal.executable.export public @copy_op
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
// CHECK-NEXT:   (%[[DEVICE:.+]]: !hal.device,
// CHECK-SAME:    %[[WORKLOAD_0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_1:[a-zA-Z0-9_]+]]: index)
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[D0:.+]] = affine.apply #[[MAP0]]()[%[[WORKLOAD_0]]]
//  CHECK-DAG:   %[[D1:.+]] = affine.apply #[[MAP0]]()[%[[WORKLOAD_1]]]
//      CHECK:   hal.return %[[D1]], %[[D0]], %[[C1]]
//      CHECK: func.func @copy_op()
//  CHECK-DAG:   %[[SOURCE_SIZE_Y:.+]] = hal.interface.constant.load[0] : index
//  CHECK-DAG:   %[[SOURCE_SIZE_X:.+]] = hal.interface.constant.load[1] : index
//  CHECK-DAG:   %[[DEST_SIZE_Y:.+]] = hal.interface.constant.load[2] : index
//  CHECK-DAG:   %[[DEST_SIZE_X:.+]] = hal.interface.constant.load[3] : index
//  CHECK-DAG:   %[[SOURCE_OFFSET_Y:.+]] = hal.interface.constant.load[4] : index
//  CHECK-DAG:   %[[SOURCE_OFFSET_X:.+]] = hal.interface.constant.load[5] : index
//  CHECK-DAG:   %[[DEST_OFFSET_Y:.+]] = hal.interface.constant.load[6] : index
//  CHECK-DAG:   %[[DEST_OFFSET_X:.+]] = hal.interface.constant.load[7] : index
//  CHECK-DAG:   %[[SLICE_SIZE_Y:.+]] = hal.interface.constant.load[8] : index
//  CHECK-DAG:   %[[SLICE_SIZE_X:.+]] = hal.interface.constant.load[9] : index
//  CHECK-DAG:   %[[SOURCE_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
//  CHECK-DAG:   %[[DEST_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
//  CHECK-DAG:   %[[SOURCE:.+]] = memref.subview %[[SOURCE_BINDING]][%[[SOURCE_OFFSET_Y]], %[[SOURCE_OFFSET_X]]]
//  CHECK-DAG:   %[[DEST:.+]] = memref.subview %[[DEST_BINDING]][%[[DEST_OFFSET_Y]], %[[DEST_OFFSET_X]]]
//  CHECK-DAG:   %[[WG_ID_X:.+]] = hal.interface.workgroup.id[0]
//  CHECK-DAG:   %[[WG_COUNT_X:.+]] = hal.interface.workgroup.count[0]
//  CHECK-DAG:   %[[WG_ID_Y:.+]] = hal.interface.workgroup.id[1]
//  CHECK-DAG:   %[[WG_COUNT_Y:.+]] = hal.interface.workgroup.count[1]
//  CHECK-DAG:   %[[LB_Y:.+]] = affine.apply #[[MAP1]]()[%[[WG_ID_Y]]]
//  CHECK-DAG:   %[[STEP_Y:.+]] = affine.apply #[[MAP1]]()[%[[WG_COUNT_Y]]]
//      CHECK:   scf.for %[[IV0:.+]] = %[[LB_Y]] to %[[SLICE_SIZE_Y]] step %[[STEP_Y]]
//  CHECK-DAG      %[[TILESIZE_Y:.+]] = affine.min #[[MAP2]](%[[IV0]])[%[[SLICE_SIZE_Y]]]
//  CHECK-DAG:     %[[LB_X:.+]] = affine.apply #[[MAP1]]()[%[[WG_ID_X]]]
//  CHECK-DAG:     %[[STEP_X:.+]] = affine.apply #[[MAP1]]()[%[[WG_COUNT_X]]]
//      CHECK:     scf.for %[[IV1:.+]] = %[[LB_X]] to %[[SLICE_SIZE_X]] step %[[STEP_X]]
//  CHECK-DAG:       %[[TILESIZE_X:.+]] = affine.min #[[MAP2]](%[[IV1]])[%[[SLICE_SIZE_X]]]
//  CHECK-DAG:       %[[SOURCE_SUBVIEW:.+]] = memref.subview %[[SOURCE]][%[[IV0]], %[[IV1]]]
//  CHECK-DAG:       %[[DEST_SUBVIEW:.+]] = memref.subview %[[DEST]][%[[IV0]], %[[IV1]]]
//      CHECK:       linalg.generic
// CHECK-SAME:           ins(%[[SOURCE_SUBVIEW]] :
// CHECK-SAME:           outs(%[[DEST_SUBVIEW]] :

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[64]]>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64">
#translation = #iree_codegen.translation_info<CPUDefault>
hal.executable private @static_1d_fft_stage2 {
  hal.executable.variant public @system_elf_x86_64, target = #executable_target_system_elf_x86_64_ {
    hal.executable.export public @static_1d_fft_stage2 layout(#pipeline_layout) attributes {translation_info = #translation} {
    ^bb0(%arg0: !hal.device, %arg1: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @static_1d_fft_stage2() {
        %c2 = arith.constant 2 : index
        %cst = arith.constant dense<[1.000000e+00, 6.12323426E-17]> : tensor<2xf32>
        %cst_0 = arith.constant dense<[-0.000000e+00, -1.000000e+00]> : tensor<2xf32>
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readwrite:tensor<32xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<readwrite:tensor<32xf32>>
        %2 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [32], strides = [1]
            : !flow.dispatch.tensor<readwrite:tensor<32xf32>> -> tensor<32xf32>
        %3 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [32], strides = [1]
            : !flow.dispatch.tensor<readwrite:tensor<32xf32>> -> tensor<32xf32>
        %4:2 = iree_linalg_ext.fft {__internal_linalg_transform__ = "workgroup", lowering_config = #config}
            ins(%c2, %cst, %cst_0 : index, tensor<2xf32>, tensor<2xf32>) outs(%2, %3 : tensor<32xf32>, tensor<32xf32>) : tensor<32xf32>, tensor<32xf32>
        flow.dispatch.tensor.store %4#0, %0, offsets = [0], sizes = [32], strides = [1]
            : tensor<32xf32> -> !flow.dispatch.tensor<readwrite:tensor<32xf32>>
        flow.dispatch.tensor.store %4#1, %1, offsets = [0], sizes = [32], strides = [1]
            : tensor<32xf32> -> !flow.dispatch.tensor<readwrite:tensor<32xf32>>
        return
      }
    }
  }
}
//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0] -> (s0 ceildiv 64)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDefault>
//      CHECK: hal.executable private @static_1d_fft_stage2
//      CHECK: hal.executable.export public @static_1d_fft_stage2
// CHECK-SAME:  translation_info = #[[TRANSLATION]]
// CHECK-NEXT:   (%[[DEVICE:.+]]: !hal.device,
// CHECK-SAME:    %[[WORKLOAD_0:[a-zA-Z0-9_]+]]: index)
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[D0:.+]] = affine.apply #[[MAP]]()[%[[WORKLOAD_0]]]
//      CHECK:   hal.return %[[D0]], %[[C1]], %[[C1]] : index, index, index
//      CHECK: func.func @static_1d_fft_stage2()
//      CHECK:   scf.for %[[IV0:.+]] =
//      CHECK:     %[[RESULT:.+]]:2 = iree_linalg_ext.fft
//  CHECK-DAG:     flow.dispatch.tensor.store %[[RESULT]]#0, %{{.+}}, offsets = [%[[IV0]]]
//  CHECK-DAG:     flow.dispatch.tensor.store %[[RESULT]]#1, %{{.+}}, offsets = [%[[IV0]]]

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[64, 64, 64]]>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64">
#translation = #iree_codegen.translation_info<CPUDefault>
hal.executable private @static_3d_fft_stage3 {
  hal.executable.variant public @system_elf_x86_64, target = #executable_target_system_elf_x86_64_ {
    hal.executable.export public @static_3d_fft_stage3 layout(#pipeline_layout) attributes {translation_info = #translation} {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @static_3d_fft_stage3() {
        %c3 = arith.constant 3 : index
        %cst = arith.constant dense<[1.000000e+00, 0.707106769, 6.12323426E-17, -0.707106769]> : tensor<4xf32>
        %cst_0 = arith.constant dense<[-0.000000e+00, -0.707106769, -1.000000e+00, -0.707106769]> : tensor<4xf32>
        %0 = bufferization.to_memref %cst_0 : memref<4xf32>
        %1 = bufferization.to_memref %cst : memref<4xf32>
        %2 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<64x128x32xf32>
        %3 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<64x128x32xf32>
        iree_linalg_ext.fft {lowering_config = #config}
            ins(%c3, %1, %0 : index, memref<4xf32>, memref<4xf32>) outs(%2, %3 : memref<64x128x32xf32>, memref<64x128x32xf32>)
        return
      }
    }
  }
}
//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0] -> (s0 ceildiv 64)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDefault>
//      CHECK: hal.executable private @static_3d_fft_stage3
//      CHECK: hal.executable.export public @static_3d_fft_stage3
// CHECK-SAME:  translation_info = #[[TRANSLATION]]
// CHECK-NEXT:   (%[[DEVICE:.+]]: !hal.device,
// CHECK-SAME:    %[[WORKLOAD_0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_2:[a-zA-Z0-9_]+]]: index)
//  CHECK-DAG:   %[[D0:.+]] = affine.apply #[[MAP]]()[%[[WORKLOAD_0]]]
//  CHECK-DAG:   %[[D1:.+]] = affine.apply #[[MAP]]()[%[[WORKLOAD_1]]]
//  CHECK-DAG:   %[[D2:.+]] = affine.apply #[[MAP]]()[%[[WORKLOAD_2]]]
//      CHECK:   hal.return %[[D2]], %[[D1]], %[[D0]] : index, index, index
//      CHECK: func.func @static_3d_fft_stage3()
//      CHECK:   scf.for %[[IV0:.+]] =
//      CHECK:     scf.for %[[IV1:.+]] =
//      CHECK:       scf.for %[[IV2:.+]] =
//  CHECK-DAG:         %[[SUBVIEW1:.+]] = memref.subview %{{.+}}[%[[IV0]], %[[IV1]], %[[IV2]]]
//  CHECK-DAG:         %[[CAST1:.+]] = memref.cast %[[SUBVIEW1]]
//  CHECK-DAG:         %[[SUBVIEW2:.+]] = memref.subview %{{.+}}[%[[IV0]], %[[IV1]], %[[IV2]]]
//  CHECK-DAG:         %[[CAST2:.+]] = memref.cast %[[SUBVIEW2]]
//      CHECK:         iree_linalg_ext.fft
// CHECK-SAME:             outs(%[[CAST1]], %[[CAST2]] :

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[64, 64, 0], [1, 4, 0], [0, 0, 4]]>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64">
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
hal.executable private @outs_fusion {
  hal.executable.variant public @system_elf_x86_64, target = #executable_target_system_elf_x86_64_ {
    hal.executable.export public @outs_fusion_fn layout(#pipeline_layout) attributes {translation_info = #translation} {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @outs_fusion_fn() {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.constant.load[0] : index
        %1 = hal.interface.constant.load[1] : index
        %2 = hal.interface.constant.load[2] : index
        %3 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %2}
        %4 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %1}
        %5 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
        %6 = tensor.empty(%0, %1) : tensor<?x?xf32>
        %7 = linalg.generic {
            indexing_maps = [#map0], iterator_types = ["parallel", "parallel"]} outs(%6 : tensor<?x?xf32>) {
        ^bb0(%arg0: f32):
          linalg.yield %cst : f32
        } -> tensor<?x?xf32>
        %8 = flow.dispatch.tensor.load %3, offsets = [0, 0], sizes = [%0, %2], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %2} -> tensor<?x?xf32>
        %9 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [%2, %1], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %1} -> tensor<?x?xf32>
        %10 = linalg.generic {
            indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]}
            ins(%8, %9 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%7 : tensor<?x?xf32>) attrs =  {lowering_config = #config} {
        ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
          %11 = arith.mulf %arg0, %arg1 : f32
          linalg.yield %11 : f32
        } -> tensor<?x?xf32>
        flow.dispatch.tensor.store %10, %5, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1]
            : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
        return
      }
    }
  }
}
//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0] -> (s0 ceildiv 64)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: hal.executable private @outs_fusion
//      CHECK: hal.executable.export public @outs_fusion_fn
// CHECK-SAME:  translation_info = #[[TRANSLATION]]
// CHECK-NEXT:   (%[[DEVICE:.+]]: !hal.device,
// CHECK-SAME:    %[[WORKLOAD_0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_2:[a-zA-Z0-9_]+]]: index)
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[D0:.+]] = affine.apply #[[MAP]]()[%[[WORKLOAD_0]]]
//  CHECK-DAG:   %[[D1:.+]] = affine.apply #[[MAP]]()[%[[WORKLOAD_1]]]
//      CHECK:   hal.return %[[D1]], %[[D0]], %[[C1]] : index, index, index
//      CHECK: func.func @outs_fusion_fn
//      CHECK:   scf.for %[[IV0:.+]] =
//      CHECK:     scf.for %[[IV1:.+]] =
//      CHECK:       %[[INIT:.+]] = tensor.empty
//      CHECK:       %[[FILL:.+]] = linalg.generic
// CHECK-SAME:           outs(%[[INIT]] :
//      CHECK:       %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:           outs(%[[FILL]] :
//      CHECK:       flow.dispatch.tensor.store %[[GENERIC]], %{{.+}}, offsets = [%[[IV0]], %[[IV1]]]

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[0, 64, 64, 64, 0, 0, 0]]>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "x86_64-unknown-linux-gnu"}>
#translation = #iree_codegen.translation_info<CPUDefault>
hal.executable private @conv {
  hal.executable.variant public @system_elf_x86_64, target = #executable_target_system_elf_x86_64_ {
    hal.executable.export public @conv layout(#pipeline_layout) attributes {translation_info = #translation} {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index, %arg4 : index, %arg5 : index, %arg6 : index, %arg7 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @conv() {
        %0 = hal.interface.constant.load[0] : index
        %1 = hal.interface.constant.load[1] : index
        %2 = hal.interface.constant.load[2] : index
        %3 = hal.interface.constant.load[3] : index
        %4 = hal.interface.constant.load[4] : index
        %5 = hal.interface.constant.load[5] : index
        %6 = hal.interface.constant.load[6] : index
        %7 = hal.interface.constant.load[7] : index
        %8 = hal.interface.constant.load[8] : index
        %9 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3}
        %10 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%4, %5, %3, %6}
        %11 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
            : !flow.dispatch.tensor<readwrite:tensor<?x?x?x?xf32>>{%0, %7, %8, %6}
        %12 = flow.dispatch.tensor.load %9, offsets = [0, 0, 0, 0], sizes = [%0, %1, %2, %3], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3} -> tensor<?x?x?x?xf32>
        %13 = flow.dispatch.tensor.load %10, offsets = [0, 0, 0, 0], sizes = [%4, %5, %3, %6], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%4, %5, %3, %6} -> tensor<?x?x?x?xf32>
        %14 = flow.dispatch.tensor.load %11, offsets = [0, 0, 0, 0], sizes = [%0, %7, %8, %6], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readwrite:tensor<?x?x?x?xf32>>{%0, %7, %8, %6} -> tensor<?x?x?x?xf32>
        %15 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, lowering_config = #config, strides = dense<1> : tensor<2xi64>}
            ins(%12, %13 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) outs(%14 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
        flow.dispatch.tensor.store %15, %11, offsets = [0, 0, 0, 0], sizes = [%0, %7, %8, %6], strides = [1, 1, 1, 1]
            : tensor<?x?x?x?xf32> -> !flow.dispatch.tensor<readwrite:tensor<?x?x?x?xf32>>{%0, %7, %8, %6}
        return
      }
    }
  }
}
//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0] -> (s0 ceildiv 64)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDefault>
//      CHECK: hal.executable private @conv
//      CHECK: hal.executable.export public @conv
// CHECK-SAME:  translation_info = #[[TRANSLATION]]
// CHECK-NEXT:   (%[[DEVICE:.+]]: !hal.device,
// CHECK-SAME:    %[[WORKLOAD_0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_3:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_4:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_5:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_6:[a-zA-Z0-9_]+]]: index)
//  CHECK-DAG:   %[[D0:.+]] = affine.apply #[[MAP]]()[%[[WORKLOAD_1]]]
//  CHECK-DAG:   %[[D1:.+]] = affine.apply #[[MAP]]()[%[[WORKLOAD_2]]]
//  CHECK-DAG:   %[[D2:.+]] = affine.apply #[[MAP]]()[%[[WORKLOAD_3]]]
//      CHECK:   hal.return %[[D2]], %[[D1]], %[[D0]] : index, index, index
//      CHECK: func.func @conv()
//      CHECK:   scf.for %[[IV0:.+]] =
//      CHECK:     scf.for %[[IV1:.+]] =
//      CHECK:       scf.for %[[IV2:.+]] =
//  CHECK-DAG:         %[[INPUT:.+]] = flow.dispatch.tensor.load %{{.+}}, offsets = [0, %[[IV0]], %[[IV1]], 0]
//  CHECK-DAG:         %[[FILTER:.+]] = flow.dispatch.tensor.load %{{.+}}, offsets = [0, 0, 0, %[[IV2]]]
//  CHECK-DAG:         %[[INIT:.+]] = flow.dispatch.tensor.load %{{.+}}, offsets = [0, %[[IV0]], %[[IV1]], %[[IV2]]]
//      CHECK:         %[[RESULT:.+]] = linalg.conv_2d_nhwc_hwcf
//      CHECK:         flow.dispatch.tensor.store %[[RESULT]], %{{.+}}, offsets = [0, %[[IV0]], %[[IV1]], %[[IV2]]]

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[0, 20, 40, 48, 0, 0]]>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "x86_64-unknown-linux-gnu"}>
#translation = #iree_codegen.translation_info<CPUDefault>
hal.executable private @conv_static {
  hal.executable.variant public @system_elf_x86_64, target = #executable_target_system_elf_x86_64_ {
    hal.executable.export public @conv_static layout(#pipeline_layout) attributes {translation_info = #translation} {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index, %arg4 : index, %arg5 : index, %arg6 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3, %arg4, %arg5, %arg6
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @conv_static() {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<1x161x161x96xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<3x3x96xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:tensor<1x80x80x96xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 161, 161, 96], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<1x161x161x96xf32>> -> tensor<1x161x161x96xf32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [3, 3, 96], strides = [1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<3x3x96xf32>> -> tensor<3x3x96xf32>
        %5 = tensor.empty() : tensor<1x80x80x96xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x80x80x96xf32>) -> tensor<1x80x80x96xf32>
        %7 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, lowering_config = #config, strides = dense<2> : tensor<2xi64>}
            ins(%3, %4 : tensor<1x161x161x96xf32>, tensor<3x3x96xf32>) outs(%6 : tensor<1x80x80x96xf32>) -> tensor<1x80x80x96xf32>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [1, 80, 80, 96], strides = [1, 1, 1, 1]
            : tensor<1x80x80x96xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x80x80x96xf32>>
        return
      }
    }
  }
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 * 20)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 * 40)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<()[s0] -> (s0 * 48)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDefault>
//      CHECK: hal.executable private @conv_static
//      CHECK: hal.executable.export public @conv_static
// CHECK-SAME:  translation_info = #[[TRANSLATION]]
// CHECK-NEXT:   (%[[DEVICE:.+]]: !hal.device,
// CHECK-SAME:    %[[WORKLOAD_0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_3:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_4:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_5:[a-zA-Z0-9_]+]]: index)
//  CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
//  CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//      CHECK:   hal.return %[[C2]], %[[C2]], %[[C4]] : index, index, index
//      CHECK: func.func @conv_static()
//      CHECK:   scf.for %[[IV0:.+]] =
//      CHECK:     scf.for %[[IV1:.+]] =
//      CHECK:       scf.for %[[IV2:.+]] =
//      CHECK:         %[[INIT:.+]] = tensor.empty
//      CHECK:         %[[FILL:.+]] = linalg.fill
// CHECK-SAME:             outs(%[[INIT]] :
//      CHECK:         %[[RESULT:.+]] = linalg.depthwise_conv_2d_nhwc_hwc
// CHECK-SAME:             outs(%[[FILL]] :
//      CHECK:         flow.dispatch.tensor.store %[[RESULT]], %{{.+}}, offsets = [0, %[[IV0]], %[[IV1]], %[[IV2]]], sizes = [1, 20, 40, 48]

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[16, 32], [16, 16], [0, 0]]>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 64 : index,
  target_triple = "x86_64-pc-linux-gnu"}>
#map0 = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
hal.executable private @generic_static {
  hal.executable.variant public @system_elf_x86_64, target = #executable_target_system_elf_x86_64_ {
    hal.executable.export public @generic_static layout(#pipeline_layout) attributes {translation_info = #translation} {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @generic_static() {
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<96x16xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:tensor<16x96xf32>>
        %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [96, 16], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<96x16xf32>> -> tensor<96x16xf32>
        %3 = tensor.empty() : tensor<16x96xf32>
        %4 = linalg.generic {
            indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]}
            ins(%2 : tensor<96x16xf32>) outs(%3 : tensor<16x96xf32>) attrs =  {lowering_config = #config} {
        ^bb0(%arg0: f32, %arg1: f32):
          linalg.yield %arg0 : f32
        } -> tensor<16x96xf32>
        flow.dispatch.tensor.store %4, %1, offsets = [0, 0], sizes = [16, 96], strides = [1, 1]
            : tensor<16x96xf32> -> !flow.dispatch.tensor<writeonly:tensor<16x96xf32>>
        return
      }
    }
  }
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 * 16)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 * 32)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: hal.executable private @generic_static
//      CHECK: hal.executable.export public @generic_static
// CHECK-SAME:  translation_info = #[[TRANSLATION]]
// CHECK-NEXT:   (%[[DEVICE:.+]]: !hal.device,
// CHECK-SAME:    %[[WORKLOAD_0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_1:[a-zA-Z0-9_]+]]: index)
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
//      CHECK:   hal.return %[[C3]], %[[C1]], %[[C1]] : index, index, index
//      CHECK: func.func @generic_static()
//      CHECK:   scf.for %[[IV0:.+]] =
//      CHECK:     scf.for %[[IV1:.+]] =
//      CHECK:       %[[RESULT:.+]] = linalg.generic
//      CHECK:       flow.dispatch.tensor.store %[[RESULT]], %{{.+}}, offsets = [%[[IV0]], %[[IV1]]]

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[28, 8, 0], [4, 4, 0], [0, 0, 60]]>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#executable_target_system_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "system-elf-arm_64", {
  data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "aarch64-none-linux-android30"}>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
hal.executable private @matmul_static {
  hal.executable.variant public @system_elf_arm_64, target = #executable_target_system_elf_arm_64_ {
    hal.executable.export public @matmul_static layout(#pipeline_layout) attributes {translation_info = #translation} {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_static() {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<196x240xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<240x40xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:tensor<196x40xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [196, 240], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<196x240xf32>> -> tensor<196x240xf32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [240, 40], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<240x40xf32>> -> tensor<240x40xf32>
        %5 = tensor.empty() : tensor<196x40xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<196x40xf32>) -> tensor<196x40xf32>
        %7 = linalg.matmul {lowering_config = #config}
            ins(%3, %4 : tensor<196x240xf32>, tensor<240x40xf32>) outs(%6 : tensor<196x40xf32>) -> tensor<196x40xf32>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [196, 40], strides = [1, 1]
            : tensor<196x40xf32> -> !flow.dispatch.tensor<writeonly:tensor<196x40xf32>>
        return
      }
    }
  }
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 * 28)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 * 8)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: hal.executable private @matmul_static
//      CHECK: hal.executable.export public @matmul_static
// CHECK-SAME:  translation_info = #[[TRANSLATION]]
// CHECK-NEXT:   (%[[DEVICE:.+]]: !hal.device,
// CHECK-SAME:    %[[WORKLOAD_0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_2:[a-zA-Z0-9_]+]]: index)
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C7:.+]] = arith.constant 7 : index
//  CHECK-DAG:   %[[C5:.+]] = arith.constant 5 : index
//      CHECK:   hal.return %[[C5]], %[[C7]], %[[C1]] : index, index, index

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[0, 1, 7, 64, 0, 0]]>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#executable_target_system_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "system-elf-arm_64", {
  data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "aarch64-none-linux-android30"}>
#translation = #iree_codegen.translation_info<CPUDefault>
hal.executable private @restrict_num_workgroups {
  hal.executable.variant public @system_elf_arm_64, target = #executable_target_system_elf_arm_64_ {
    hal.executable.export public @restrict_num_workgroups layout(#pipeline_layout) attributes {translation_info = #translation} {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 :index, %arg4 : index, %arg5 : index, %arg6 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3, %arg4, %arg5, %arg6
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @restrict_num_workgroups() {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<1x11x11x576xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<5x5x576xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:tensor<1x7x7x576xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 11, 11, 576], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<1x11x11x576xf32>> -> tensor<1x11x11x576xf32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [5, 5, 576], strides = [1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<5x5x576xf32>> -> tensor<5x5x576xf32>
        %5 = tensor.empty() : tensor<1x7x7x576xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x7x7x576xf32>) -> tensor<1x7x7x576xf32>
        %7 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, lowering_config = #config, strides = dense<1> : tensor<2xi64>}
            ins(%3, %4 : tensor<1x11x11x576xf32>, tensor<5x5x576xf32>) outs(%6 : tensor<1x7x7x576xf32>) -> tensor<1x7x7x576xf32>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [1, 7, 7, 576], strides = [1, 1, 1, 1]
            : tensor<1x7x7x576xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x7x7x576xf32>>
        return
      }
    }
  }
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 * 7)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 * 64)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDefault>
//      CHECK: hal.executable private @restrict_num_workgroups
//      CHECK: hal.executable.export public @restrict_num_workgroups
// CHECK-SAME:  translation_info = #[[TRANSLATION]]
// CHECK-NEXT:   (%[[DEVICE:.+]]: !hal.device,
// CHECK-SAME:    %[[WORKLOAD_0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_3:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_4:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_5:[a-zA-Z0-9_]+]]: index)
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C7:.+]] = arith.constant 7 : index
//  CHECK-DAG:   %[[C9:.+]] = arith.constant 9 : index
//      CHECK:   hal.return %[[C9]], %[[C1]], %[[C7]] : index, index, index

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[4, 0, 0], [4, 0, 0], [0, 1, 4]]>
#pipeline_layout = #hal.pipeline.layout<push_constants = 4, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "x86_64-unknown-unknown-eabi-elf"}>
#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0)>
#map2 = affine_map<(d0) -> (d0)>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
hal.executable private @reduction {
  hal.executable.variant public @reduction, target = #executable_target_embedded_elf_x86_64_ {
    hal.executable.export public @reduction ordinal(0) layout(#pipeline_layout) attributes {translation_info = #translation} {
    ^bb0(%arg0: !hal.device, %arg1: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @reduction(%arg0 : !flow.dispatch.tensor<readonly:tensor<7x7x2048xf32>>,
          %arg1 : !flow.dispatch.tensor<writeonly:tensor<7xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 1.000000e+01 : f32
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [7, 7, 2048], strides = [1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<7x7x2048xf32>> -> tensor<7x7x2048xf32>
        %1 = tensor.empty() : tensor<7xf32>
        %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<7xf32>) -> tensor<7xf32>
        %3 = linalg.generic {
            indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction", "reduction"]}
            ins(%0 : tensor<7x7x2048xf32>) outs(%2 : tensor<7xf32>) attrs =  {lowering_config = #config} {
        ^bb0(%arg2: f32, %arg3: f32):
          %5 = arith.addf %arg2, %arg3 : f32
          linalg.yield %5 : f32
        } -> tensor<7xf32>
        %4 = linalg.generic {
            indexing_maps = [#map2, #map2], iterator_types = ["parallel"]}
            ins(%3 : tensor<7xf32>) outs(%1 : tensor<7xf32>) {
        ^bb0(%arg2: f32, %arg3: f32):
          %5 = arith.divf %arg2, %cst_0 : f32
          linalg.yield %5 : f32
        } -> tensor<7xf32>
        flow.dispatch.tensor.store %4, %arg1, offsets = [0], sizes = [7], strides = [1]
            : tensor<7xf32> -> !flow.dispatch.tensor<writeonly:tensor<7xf32>>
        return
      }
    }
  }
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 * 4)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: hal.executable private @reduction
//      CHECK: hal.executable.export public @reduction
// CHECK-SAME:  translation_info = #[[TRANSLATION]]
// CHECK-NEXT:   (%[[DEVICE:.+]]: !hal.device,
// CHECK-SAME:    %[[WORKLOAD:[a-zA-Z0-9_]+]]: index)
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//      CHECK:   hal.return %[[C2]], %[[C1]], %[[C1]] : index, index, index
//      CHECK: func.func @reduction
//      CHECK:   scf.for %[[IV0:.+]] =
//      CHECK:     %[[INIT:.+]] = tensor.empty
//      CHECK:     %[[FILL:.+]] = linalg.fill
// CHECK-SAME:         outs(%[[INIT]] :
//      CHECK:     %[[REDUCE:.+]] = linalg.generic
// CHECK-SAME:         outs(%[[FILL]] :
//      CHECK:     %[[GENERIC:.+]] = linalg.generic
//      CHECK:     flow.dispatch.tensor.store %[[GENERIC]], %{{.+}}, offsets = [%[[IV0]]]

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[64, 0, 0], [8, 0, 0], [0, 0, 16]]>
#pipeline_layout = #hal.pipeline.layout<push_constants = 4, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "x86_64-unknown-unknown-eabi-elf"}>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
hal.executable private @gemm_unit_N {
  hal.executable.variant public @embedded_elf_x86_64, target = #executable_target_embedded_elf_x86_64_ {
    hal.executable.export public @gemm_unit_N ordinal(0) layout(#pipeline_layout) attributes {translation_info = #translation} {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @gemm_unit_N() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.constant.load[0] : index
        %1 = hal.interface.constant.load[1] : index
        %2 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(32) offset(%c0)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1}
        %3 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(32) offset(%c0)
            : !flow.dispatch.tensor<readonly:tensor<?x1xf32>>{%1}
        %4 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(32) offset(%c0)
            : !flow.dispatch.tensor<readwrite:tensor<?x1xf32>>{%0}
        %5 = flow.dispatch.tensor.load %3, offsets = [0, 0], sizes = [%1, 1], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x1xf32>>{%1} -> tensor<?x1xf32>
        %6 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1} -> tensor<?x?xf32>
        %7 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [%0, 1], strides = [1, 1]
            : !flow.dispatch.tensor<readwrite:tensor<?x1xf32>>{%0} -> tensor<?x1xf32>
        %8 = linalg.matmul {lowering_config = #config}
            ins(%6, %5 : tensor<?x?xf32>, tensor<?x1xf32>) outs(%7 : tensor<?x1xf32>) -> tensor<?x1xf32>
        flow.dispatch.tensor.store %8, %4, offsets = [0, 0], sizes = [%0, 1], strides = [1, 1]
            : tensor<?x1xf32> -> !flow.dispatch.tensor<readwrite:tensor<?x1xf32>>{%0}
        return
      }
    }
  }
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 64)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: hal.executable private @gemm_unit_N
//      CHECK: hal.executable.export public @gemm_unit_N
// CHECK-SAME:  translation_info = #[[TRANSLATION]]
// CHECK-NEXT:   (%[[DEVICE:.+]]: !hal.device,
// CHECK-SAME:    %[[WORKLOAD_0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_2:[a-zA-Z0-9_]+]]: index)
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[D0:.+]] = affine.apply #[[MAP0]]()[%[[WORKLOAD_0]]]
//      CHECK:   hal.return %[[D0]], %[[C1]], %[[C1]] : index, index, index
//      CHECK: func.func @gemm_unit_N()
//  CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load[0]
//  CHECK-DAG:   %[[WG_ID_X:.+]] = hal.interface.workgroup.id[0]
//  CHECK-DAG:   %[[WG_COUNT_X:.+]] = hal.interface.workgroup.count[0]
//  CHECK-DAG:   %[[LB:.+]] = affine.apply #[[MAP1]]()[%[[WG_ID_X]]]
//  CHECK-DAG:   %[[STEP:.+]] = affine.apply #[[MAP1]]()[%[[WG_COUNT_X]]]
//      CHECK:   scf.for %[[IV0:.+]] = %[[LB]] to %[[M]] step %[[STEP]]
//  CHECK-NOT:     scf.for
//      CHECK:     %[[GEMM:.+]] = linalg.matmul
//      CHECK:     flow.dispatch.tensor.store %[[GEMM]],
// CHECK-SAME:         offsets = [%[[IV0]], 0]

// -----
#config = #iree_codegen.lowering_config<tile_sizes = [[0, 0, 0], [0, 0, 0], [0, 0, 16]]>
#pipeline_layout = #hal.pipeline.layout<push_constants = 4, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "x86_64-unknown-unknown-eabi-elf"}>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
hal.executable private @gemm_unit_M_unit_N {
  hal.executable.variant public @embedded_elf_x86_64, target = #executable_target_embedded_elf_x86_64_ {
    hal.executable.export public @gemm_unit_M_unit_N ordinal(0) layout(#pipeline_layout) attributes {translation_info = #translation} {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @gemm_unit_M_unit_N() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.constant.load[0] : index
        %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(32) offset(%c0)
            : !flow.dispatch.tensor<readonly:tensor<1x?xf32>>{%0}
        %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(32) offset(%c0)
            : !flow.dispatch.tensor<readonly:tensor<?x1xf32>>{%0}
        %3 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(32) offset(%c0)
            : !flow.dispatch.tensor<readwrite:tensor<1x1xf32>>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1, %0], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<1x?xf32>>{%0} -> tensor<1x?xf32>
        %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%0, 1], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x1xf32>>{%0} -> tensor<?x1xf32>
        %6 = flow.dispatch.tensor.load %3, offsets = [0, 0], sizes = [1, 1], strides = [1, 1]
            : !flow.dispatch.tensor<readwrite:tensor<1x1xf32>> -> tensor<1x1xf32>
        %7 = linalg.matmul {lowering_config = #config}
            ins(%4, %5 : tensor<1x?xf32>, tensor<?x1xf32>) outs(%6 : tensor<1x1xf32>) -> tensor<1x1xf32>
        flow.dispatch.tensor.store %7, %3, offsets = [0, 0], sizes = [1, 1], strides = [1, 1]
            : tensor<1x1xf32> -> !flow.dispatch.tensor<readwrite:tensor<1x1xf32>>
        return
      }
    }
  }
}
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: hal.executable private @gemm_unit_M_unit_N
//      CHECK: hal.executable.export public @gemm_unit_M_unit_N
// CHECK-SAME:  translation_info = #[[TRANSLATION]]
// CHECK-NEXT:   (%[[DEVICE:.+]]: !hal.device,
// CHECK-SAME:    %[[WORKLOAD_0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_2:[a-zA-Z0-9_]+]]: index)
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//      CHECK:   hal.return %[[C1]], %[[C1]], %[[C1]] : index, index, index
//      CHECK: func.func @gemm_unit_M_unit_N()
//  CHECK-NOT:   scf.for
//      CHECK:   %[[GEMM:.+]] = linalg.matmul
//      CHECK:   flow.dispatch.tensor.store %[[GEMM]], %{{.+}}, offsets = [0, 0]

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[0, 0, 0, 0, 64, 64, 0, 64], [0, 1, 0, 0, 1, 1, 0, 4], [0, 0, 0, 0, 0, 0, 0, 0]]>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "x86_64-unknown-linux-gnu"}>
#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5, d6, d7)>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
hal.executable private @generic_unit_dims {
  hal.executable.variant public @llvm, target = #executable_target_embedded_elf_x86_64_ {
    hal.executable.export public @generic_unit_dims layout(#pipeline_layout) attributes {translation_info = #translation} {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index, %arg4 : index, %arg5 : index, %arg6 : index, %arg7 : index, %arg8 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @generic_unit_dims() {
        %0 = hal.interface.constant.load[0] : index
        %1 = hal.interface.constant.load[1] : index
        %2 = hal.interface.constant.load[2] : index
        %3 = hal.interface.constant.load[3] : index
        %4 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<1x?x1x1x?x?x1x?xf32>>{%0, %1, %2, %3}
        %5 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:tensor<1x?x1x1x?x?x1x?xf32>>{%0, %1, %2, %3}
        %6 = flow.dispatch.tensor.load %4, offsets = [0, 0, 0, 0, 0, 0, 0, 0], sizes = [1, %0, 1, 1, %1, %2, 1, %3], strides = [1, 1, 1, 1, 1, 1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<1x?x1x1x?x?x1x?xf32>>{%0, %1, %2, %3} -> tensor<1x?x1x1x?x?x1x?xf32>
        %7 = tensor.empty(%0, %1, %2, %3) : tensor<1x?x1x1x?x?x1x?xf32>
        %8 = linalg.generic {
            indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]}
            ins(%6 : tensor<1x?x1x1x?x?x1x?xf32>) outs(%7 : tensor<1x?x1x1x?x?x1x?xf32>) attrs =  {lowering_config = #config} {
        ^bb0(%arg0: f32, %arg1: f32):
          %9 = arith.addf %arg0, %arg0 : f32
          linalg.yield %9 : f32
        } -> tensor<1x?x1x1x?x?x1x?xf32>
        flow.dispatch.tensor.store %8, %5, offsets = [0, 0, 0, 0, 0, 0, 0, 0], sizes = [1, %0, 1, 1, %1, %2, 1, %3], strides = [1, 1, 1, 1, 1, 1, 1, 1]
            : tensor<1x?x1x1x?x?x1x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x?x1x1x?x?x1x?xf32>>{%0, %1, %2, %3}
        return
      }
    }
  }
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 64)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: hal.executable private @generic_unit_dims
//      CHECK: hal.executable.export public @generic_unit_dims
// CHECK-SAME:  translation_info = #[[TRANSLATION]]
// CHECK-NEXT:   (%[[DEVICE:.+]]: !hal.device,
// CHECK-SAME:    %[[WORKLOAD_0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_3:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_4:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_5:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_6:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_7:[a-zA-Z0-9_]+]]: index)
//  CHECK-DAG:   %[[D0:.+]] = affine.apply #[[MAP0]]()[%[[WORKLOAD_4]]]
//  CHECK-DAG:   %[[D1:.+]] = affine.apply #[[MAP0]]()[%[[WORKLOAD_5]]]
//  CHECK-DAG:   %[[D2:.+]] = affine.apply #[[MAP0]]()[%[[WORKLOAD_7]]]
//      CHECK:   hal.return %[[D2]], %[[D1]], %[[D0]] : index, index, index
//      CHECK: func.func @generic_unit_dims()
//      CHECK:   scf.for %[[IV0:.+]] =
//      CHECK:     scf.for %[[IV1:.+]] =
//      CHECK:       scf.for %[[IV2:.+]] =
//      CHECK:         %[[GENERIC:.+]] = linalg.generic
//      CHECK:         flow.dispatch.tensor.store %[[GENERIC]],
// CHECK-SAME:             offsets = [0, 0, 0, 0, %[[IV0]], %[[IV1]], 0, %[[IV2]]]

// -----
#config = #iree_codegen.lowering_config<tile_sizes = [[0], [0], [4]]>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "x86_64-unknown-linux-gnu"}>
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
hal.executable private @reduce_to_scalar {
  hal.executable.variant public @llvm, target = #executable_target_embedded_elf_x86_64_ {
    hal.executable.export public @reduce_to_scalar layout(#pipeline_layout) attributes {translation_info = #translation} {
    ^bb0(%arg0: !hal.device, %arg1: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @reduce_to_scalar() {
        %0 = hal.interface.constant.load[0] : index
        %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%0}
        %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<readwrite:tensor<f32>>
        %3 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [%0], strides = [1]
            : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%0} -> tensor<?xf32>
        %4 = flow.dispatch.tensor.load %2, offsets = [], sizes = [], strides = []
            : !flow.dispatch.tensor<readwrite:tensor<f32>> -> tensor<f32>
        %5 = linalg.generic {
            indexing_maps = [#map0, #map1], iterator_types = ["reduction"]}
            ins(%3 : tensor<?xf32>) outs(%4 : tensor<f32>) attrs =  {lowering_config = #config} {
        ^bb0(%arg0: f32, %arg1: f32):
          %6 = arith.addf %arg0, %arg1 : f32
          linalg.yield %6 : f32
        } -> tensor<f32>
        flow.dispatch.tensor.store %5, %2, offsets = [], sizes = [], strides = []
            : tensor<f32> -> !flow.dispatch.tensor<readwrite:tensor<f32>>
        return
      }
    }
  }
}
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: hal.executable private @reduce_to_scalar
//      CHECK: hal.executable.export public @reduce_to_scalar
// CHECK-SAME:  translation_info = #[[TRANSLATION]]
// CHECK-NEXT:   (%[[DEVICE:.+]]: !hal.device,
// CHECK-SAME:    %[[WORKLOAD_0:[a-zA-Z0-9_]+]]: index)
//      CHECK:   %[[C1:.+]] = arith.constant 1 : index
//      CHECK:   hal.return %[[C1]], %[[C1]], %[[C1]] : index, index, index
//      CHECK: func.func @reduce_to_scalar()
//  CHECK-NOT:   scf.for

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[]]>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "x86_64-unknown-linux-gnu"}>
#map = affine_map<() -> ()>
#translation = #iree_codegen.translation_info<CPUDefault>
hal.executable private @scalar {
  hal.executable.variant public @llvm, target = #executable_target_embedded_elf_x86_64_ {
    hal.executable.export public @scalar layout(#pipeline_layout) attributes {translation_info = #translation} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @scalar() {
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<f32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:tensor<f32>>
        %2 = flow.dispatch.tensor.load %0, offsets = [], sizes = [], strides = []
            : !flow.dispatch.tensor<readonly:tensor<f32>> -> tensor<f32>
        %3 = flow.dispatch.tensor.load %1, offsets = [], sizes = [], strides = []
            : !flow.dispatch.tensor<writeonly:tensor<f32>> -> tensor<f32>
        %4 = linalg.generic {
            indexing_maps = [#map, #map], iterator_types = []}
            ins(%2 : tensor<f32>) outs(%3 : tensor<f32>)
            attrs = {lowering_config = #config} {
        ^bb0(%arg0: f32, %arg1: f32):
          %5 = arith.addf %arg0, %arg1 : f32
          linalg.yield %5 : f32
        } -> tensor<f32>
        flow.dispatch.tensor.store %4, %1, offsets = [], sizes = [], strides = []
            : tensor<f32> -> !flow.dispatch.tensor<writeonly:tensor<f32>>
        return
      }
    }
  }
}
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDefault>
//      CHECK: hal.executable private @scalar
//      CHECK: hal.executable.export public @scalar
// CHECK-SAME:  translation_info = #[[TRANSLATION]]
// CHECK-NEXT:   (%[[DEVICE:.+]]: !hal.device)
//      CHECK:   %[[C1:.+]] = arith.constant 1 : index
//      CHECK:   hal.return %[[C1]], %[[C1]], %[[C1]] : index, index, index
//      CHECK: func.func @scalar()
//  CHECK-NOT:   scf.for

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[2], [2], [0]]>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "aarch64-unknown-unknown-eabi-elf"
}>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
hal.executable private @rank_reduced_slice {
  hal.executable.variant public @llvm, target = #executable_target_embedded_elf_arm_64_ {
    hal.executable.export public @rank_reduced_slice layout(#pipeline_layout) attributes {translation_info = #translation} {
    ^bb0(%arg0: !hal.device, %arg1: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @rank_reduced_slice() {
        %in_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<5x40xf32>>
        %out_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:tensor<10xf32>>
        %in = flow.dispatch.tensor.load %in_binding, offsets = [3, 10], sizes = [1, 10], strides = [2, 1]
            : !flow.dispatch.tensor<readonly:tensor<5x40xf32>> -> tensor<10xf32>
        %out = tensor.empty() : tensor<10xf32>
        %val = linalg.generic {
            indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
            iterator_types = ["parallel"]}
            ins(%in : tensor<10xf32>) outs(%out : tensor<10xf32>) attrs =  {lowering_config = #config} {
          ^bb0(%b0 : f32, %b1 : f32):
            %0 = arith.addf %b0, %b0 : f32
            linalg.yield %0 : f32
          } -> tensor<10xf32>
        flow.dispatch.tensor.store %val, %out_binding, offsets = [0], sizes = [10], strides = [1]
            : tensor<10xf32> -> !flow.dispatch.tensor<writeonly:tensor<10xf32>>
        return
      }
    }
  }
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 * 2)>
//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0] -> (s0 + 10)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: hal.executable.export public @rank_reduced_slice
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
// CHECK-NEXT:   %[[WORKLOAD:[a-zA-Z0-9]+]]: index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1
//  CHECK-DAG:   %[[C5:.+]] = arith.constant 5
//      CHECK:   hal.return %[[C5]], %[[C1]], %[[C1]]
//      CHECK: func.func @rank_reduced_slice()
//  CHECK-DAG:   %[[SRC_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
// CHECK-SAME:       : !flow.dispatch.tensor<readonly:tensor<5x40xf32>>
//  CHECK-DAG:   %[[DST_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
// CHECK-SAME:       : !flow.dispatch.tensor<writeonly:tensor<10xf32>>
//      CHECK:   scf.for %[[IV0:.+]] =
//      CHECK:     %[[OFFSET:.+]] = affine.apply #[[MAP]]()[%[[IV0]]]
//      CHECK:     %[[SRC_TILE:.+]] = flow.dispatch.tensor.load %[[SRC_BINDING]]
// CHECK-SAME:         offsets = [3, %[[OFFSET]]], sizes = [1, 2], strides = [2, 1]
//      CHECK:     linalg.generic
// CHECK-SAME:         ins(%[[SRC_TILE]] :

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[32, 64, 0], [8, 32, 0], [0, 0, 16]], tile_interchange = [[1, 0, 2], [], []]>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>
  ]>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "x86_64-unknown-linux-gnu"}>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
hal.executable private @matmul_interchange {
  hal.executable.variant public @llvm, target = #executable_target_embedded_elf_x86_64_ {
    hal.executable.export public @matmul_interchange layout(#pipeline_layout) attributes {translation_info = #translation} {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_interchange() {
        %0 = hal.interface.constant.load[0] : index
        %1 = hal.interface.constant.load[1] : index
        %2 = hal.interface.constant.load[2] : index
        %3 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %2}
        %4 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %1}
        %5 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1}
        %6 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
        %7 = flow.dispatch.tensor.load %3, offsets = [0, 0], sizes = [%0, %2], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %2} -> tensor<?x?xf32>
        %8 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [%2, %1], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %1} -> tensor<?x?xf32>
        %9 = flow.dispatch.tensor.load %5, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1} -> tensor<?x?xf32>
        %10 = linalg.matmul {lowering_config = #config}
            ins(%7, %8 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%9 : tensor<?x?xf32>) -> tensor<?x?xf32>
        flow.dispatch.tensor.store %10, %6, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1]
            : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
        return
      }
    }
  }
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 32)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 64)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: hal.executable.export public @matmul_interchange
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
// CHECK-NEXT:   (%[[DEVICE:.+]]: !hal.device,
// CHECK-SAME:    %[[WORKLOAD_0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_2:[a-zA-Z0-9_]+]]: index)
//  CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:    %[[D0:.+]] = affine.apply #[[MAP0]]()[%[[WORKLOAD_0]]]
//  CHECK-DAG:    %[[D1:.+]] = affine.apply #[[MAP1]]()[%[[WORKLOAD_1]]]
//      CHECK:    hal.return %[[D0]], %[[D1]], %[[C1]] : index, index, index
//      CHECK: func.func @matmul_interchange()
//  CHECK-DAG:   %[[D0:.+]] = hal.interface.constant.load[0] : index
//  CHECK-DAG:   %[[D1:.+]] = hal.interface.constant.load[1] : index
//      CHECK:   scf.for %{{.+}} = %{{.+}} to %[[D1]] step %{{.+}} {
//      CHECK:     scf.for %{{.+}} = %{{.+}} to %[[D0]] step %{{.+}} {

// -----

hal.executable private @no_compute {
  hal.executable.variant public @embedded_elf_x86_64, target = <"llvm-cpu", "embedded-elf-x86_64", {}> {
    hal.executable.export public @no_compute ordinal(0) layout(#hal.pipeline.layout<push_constants = 5, sets = [<0, bindings = [<0, storage_buffer>, <1, storage_buffer>]>]>) attributes {translation_info = #iree_codegen.translation_info<CPUDefault>} {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @no_compute() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = hal.interface.constant.load[2] : i32
        %3 = hal.interface.constant.load[3] : i32
        %4 = hal.interface.constant.load[4] : i32
        %5 = arith.index_cast %0 : i32 to index
        %6 = arith.index_cast %1 : i32 to index
        %7 = arith.index_cast %2 : i32 to index
        %8 = arith.index_cast %3 : i32 to index
        %9 = arith.index_cast %4 : i32 to index
        %10 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<?x?x?xf32>{%5, %6, %7}
        memref.assume_alignment %10, 64 : memref<?x?x?xf32>
        %11 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<1x?x?xf32>{%8, %9}
        memref.assume_alignment %11, 64 : memref<1x?x?xf32>
        return
      }
    }
  }
}
//      CHECK: hal.executable.export public @no_compute
// CHECK-NEXT: ^bb0
// CHECK-NEXT:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-NEXT:   hal.return %[[C1]], %[[C1]], %[[C1]]

// -----

hal.executable private @tile_multiuse_producer {
  hal.executable.variant public @embedded_elf_x86_64, target = <"llvm-cpu", "embedded-elf_x86_64", {}> {
    hal.executable.export public @tile_multiuse_producer ordinal(0) layout (#hal.pipeline.layout<
      push_constants = 0, sets = [<0, bindings = [
          <0, storage_buffer, ReadOnly>, <1, storage_buffer>, <2, storage_buffer>, <3, storage_buffer>]>]>)
      attributes {translation_info = #iree_codegen.translation_info<CPUDoubleTilingExpert>} {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @tile_multiuse_producer() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 1.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
            : !flow.dispatch.tensor<readonly:tensor<12x128x128xf32>>
        %s0 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
            : !flow.dispatch.tensor<writeonly:tensor<12x128x128xf32>>
        %s1 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0)
            : !flow.dispatch.tensor<writeonly:tensor<12x128xf32>>
        %s2 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0)
            : !flow.dispatch.tensor<writeonly:tensor<12x128xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [12, 128, 128], strides = [1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<12x128x128xf32>> -> tensor<12x128x128xf32>
        %5 = tensor.empty() : tensor<12x128x128xf32>
        %6 = tensor.empty() : tensor<12x128xf32>
        %1 = linalg.fill ins(%cst : f32) outs(%6 : tensor<12x128xf32>) -> tensor<12x128xf32>
        %8 = linalg.generic {
            indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
            iterator_types = ["parallel", "parallel", "reduction"]}
            ins(%3 : tensor<12x128x128xf32>) outs(%1 : tensor<12x128xf32>) {
          ^bb0(%arg0: f32, %arg1: f32):
            %11 = arith.maxf %arg0, %arg1 : f32
            linalg.yield %11 : f32
          } -> tensor<12x128xf32>
        %7 = linalg.fill ins(%cst : f32) outs(%6 : tensor<12x128xf32>) -> tensor<12x128xf32>
        %9:2 = linalg.generic {
            indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>,
                             affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
            iterator_types = ["parallel", "parallel", "reduction"]}
            ins(%3, %8 : tensor<12x128x128xf32>, tensor<12x128xf32>)
            outs(%5, %7 : tensor<12x128x128xf32>, tensor<12x128xf32>)
            attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[4, 32, 0], [1, 4, 0], [0, 0, 4]]>} {
          ^bb0(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32):
            %11 = arith.subf %arg0, %arg1 : f32
            %12 = math.exp %11 : f32
            %13 = arith.addf %12, %arg3 : f32
            linalg.yield %12, %13 : f32, f32
          } -> (tensor<12x128x128xf32>, tensor<12x128xf32>)
        %10 = linalg.generic {
            indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>,
                             affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%9#0, %9#1 : tensor<12x128x128xf32>, tensor<12x128xf32>) outs(%5 : tensor<12x128x128xf32>) {
          ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
            %11 = arith.divf %cst_0, %arg1 : f32
            %12 = arith.mulf %arg0, %11 : f32
            linalg.yield %12 : f32
          } -> tensor<12x128x128xf32>
        flow.dispatch.tensor.store %10, %s0, offsets = [0, 0, 0], sizes = [12, 128, 128], strides = [1, 1, 1]
            : tensor<12x128x128xf32> -> !flow.dispatch.tensor<writeonly:tensor<12x128x128xf32>>
        flow.dispatch.tensor.store %9#1, %s1, offsets = [0, 0], sizes = [12, 128], strides = [1, 1]
            : tensor<12x128xf32> -> !flow.dispatch.tensor<writeonly:tensor<12x128xf32>>
        flow.dispatch.tensor.store %8, %s2, offsets = [0, 0], sizes = [12, 128], strides = [1, 1]
            : tensor<12x128xf32> -> !flow.dispatch.tensor<writeonly:tensor<12x128xf32>>
        return
      }
    }
  }
}
// CHECK-LABEL: func @tile_multiuse_producer()
//   CHECK-DAG:     %[[SRC_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
//   CHECK-DAG:     %[[RESULT_BINDING0:.+]] = hal.interface.binding.subspan set(0) binding(1)
//   CHECK-DAG:     %[[RESULT_BINDING1:.+]] = hal.interface.binding.subspan set(0) binding(2)
//   CHECK-DAG:     %[[RESULT_BINDING2:.+]] = hal.interface.binding.subspan set(0) binding(3)
//       CHECK:     scf.for %[[IV0:.+]] =
//       CHECK:       scf.for %[[IV1:.+]] =
//       CHECK:         %[[SRC:.+]] = flow.dispatch.tensor.load %[[SRC_BINDING]], offsets = [%[[IV0]], %[[IV1]], 0]
//       CHECK:         %[[INIT0:.+]] = tensor.empty() : tensor<4x32xf32>
//       CHECK:         %[[FILL:.+]] = linalg.fill
//  CHECK-SAME:             outs(%[[INIT0]] :
//       CHECK:         %[[GENERIC0:.+]] = linalg.generic
//  CHECK-SAME:             ins(%[[SRC]] :
//  CHECK-SAME:             outs(%[[FILL]] :
//       CHECK:         %[[INIT1:.+]] = tensor.empty() : tensor<4x32x128xf32>
//       CHECK:         %[[GENERIC1:.+]]:2 = linalg.generic
//  CHECK-SAME:             ins(%[[SRC]], %[[GENERIC0]] :
//  CHECK-SAME:             outs(%[[INIT1]], %[[FILL]]
//       CHECK:         %[[GENERIC2:.+]] = linalg.generic
//  CHECK-SAME:             ins(%[[GENERIC1]]#0, %[[GENERIC1]]#1 :
//   CHECK-DAG:         flow.dispatch.tensor.store %[[GENERIC2]], %[[RESULT_BINDING0]], offsets = [%[[IV0]], %[[IV1]], 0]
//   CHECK-DAG:         flow.dispatch.tensor.store %[[GENERIC1]]#1, %[[RESULT_BINDING1]], offsets = [%[[IV0]], %[[IV1]]]
//   CHECK-DAG:         flow.dispatch.tensor.store %[[GENERIC0]], %[[RESULT_BINDING2]], offsets = [%[[IV0]], %[[IV1]]]

// -----

hal.executable private @no_tile {
  hal.executable.variant public @embedded_elf_x86_64, target = <"llvm-cpu", "embedded-elf-x86_64", {}> {
    hal.executable.export public @no_tile ordinal(0) layout(#hal.pipeline.layout<
        push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>, <3, storage_buffer>]>]>)
        attributes {translation_info = #iree_codegen.translation_info<CPUDefault>} {
    ^bb0(%arg0: !hal.device, %arg1: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @no_tile() {
        %c0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<10xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<10xi32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<3xf32>>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c64) : !flow.dispatch.tensor<readwrite:tensor<3xi32>>
        %4 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [10], strides = [1] : !flow.dispatch.tensor<readonly:tensor<10xf32>> -> tensor<10xf32>
        %5 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [10], strides = [1] : !flow.dispatch.tensor<readonly:tensor<10xi32>> -> tensor<10xi32>
        %6 = flow.dispatch.tensor.load %2, offsets = [0], sizes = [3], strides = [1] : !flow.dispatch.tensor<readwrite:tensor<3xf32>> -> tensor<3xf32>
        %7 = flow.dispatch.tensor.load %3, offsets = [0], sizes = [3], strides = [1] : !flow.dispatch.tensor<readwrite:tensor<3xi32>> -> tensor<3xi32>
        %8:2 = iree_linalg_ext.topk {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0]]>} dimension(0) ins(%4, %5 : tensor<10xf32>, tensor<10xi32>) outs(%6, %7 : tensor<3xf32>, tensor<3xi32>) {
        ^bb0(%arg0: f32, %arg1: f32):
          %9 = arith.cmpf ogt, %arg0, %arg1 : f32
          iree_linalg_ext.yield %9 : i1
        } -> tensor<3xf32>, tensor<3xi32>
        flow.dispatch.tensor.store %8#0, %2, offsets = [0], sizes = [3], strides = [1] : tensor<3xf32> -> !flow.dispatch.tensor<readwrite:tensor<3xf32>>
        flow.dispatch.tensor.store %8#1, %3, offsets = [0], sizes = [3], strides = [1] : tensor<3xi32> -> !flow.dispatch.tensor<readwrite:tensor<3xi32>>
        return
      }
    }
  }
}
// CHECK-LABEL: func @no_tile()
//   CHECK-NOT: scf.for
//       CHECK: iree_linalg_ext.topk

// -----

hal.executable private @pack_lowering {
  hal.executable.variant public @embedded_elf_x86_64, target = <"llvm-cpu", "embedded-elf-x86_64", {}> {
    hal.executable.export public @gemm_lhs_pack ordinal(0)
        layout(#hal.pipeline.layout<push_constants = 0,
            sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>)
        attributes {translation_info = #iree_codegen.translation_info<CPUDataTiling>} {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_set_encoding_op %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @gemm_lhs_pack() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
            : !flow.dispatch.tensor<readonly:tensor<100x250xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
            : !flow.dispatch.tensor<writeonly:tensor<14x64x8x4xf32>>
        %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [100, 250], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<100x250xf32>> -> tensor<100x250xf32>
        %3 = tensor.empty() : tensor<14x64x8x4xf32>
        %4 = tensor.pack %2 padding_value(%cst : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 4] into %3
            {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 64]]>}
            : tensor<100x250xf32> -> tensor<14x64x8x4xf32>
        flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0, 0], sizes = [14, 64, 8, 4], strides = [1, 1, 1, 1]
            : tensor<14x64x8x4xf32> -> !flow.dispatch.tensor<writeonly:tensor<14x64x8x4xf32>>
        return
      }
    }
  }
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> ((s0 ceildiv 8) ceildiv 64)
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> ((s0 ceildiv 4) ceildiv 64)
//      CHECK: hal.executable.export public @gemm_lhs_pack
// CHECK-NEXT:   %[[ARG0:.+]]: !hal.device
// CHECK-SAME:   %[[ARG1:.+]]: index,
// CHECK-SAME:   %[[ARG2:.+]]: index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[W0:.+]] = affine.apply #[[MAP0]]()[%[[ARG1]]]
//  CHECK-DAG:   %[[W1:.+]] = affine.apply #[[MAP1]]()[%[[ARG2]]]
//      CHECK:   hal.return %[[W1]], %[[W0]], %[[C1]]

// -----

hal.executable private @pack_lowering {
  hal.executable.variant public @embedded_elf_x86_64, target = <"llvm-cpu", "embedded-elf-x86_64", {}> {
    hal.executable.export public @gemm_rhs_transpose_pack ordinal(0)
        layout(#hal.pipeline.layout<push_constants = 0,
            sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>)
        attributes {translation_info = #iree_codegen.translation_info<CPUDataTiling>} {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_set_encoding_op %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @gemm_rhs_transpose_pack() {
        %c0 = arith.constant 0 : index
        %c114688 = arith.constant 114688 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
            : !flow.dispatch.tensor<readonly:tensor<250x500xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c114688)
            : !flow.dispatch.tensor<writeonly:tensor<64x64x8x4xf32>>
        %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [250, 500], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<250x500xf32>> -> tensor<250x500xf32>
        %3 = tensor.empty() : tensor<64x64x8x4xf32>
        %4 = tensor.pack %2 padding_value(%cst : f32) outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [8, 4] into %3
            {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 64]]>}
            : tensor<250x500xf32> -> tensor<64x64x8x4xf32>
        flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0, 0], sizes = [64, 64, 8, 4], strides = [1, 1, 1, 1]
            : tensor<64x64x8x4xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x64x8x4xf32>>
        return
      }
    }
  }
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> ((s0 ceildiv 8) ceildiv 64)
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> ((s0 ceildiv 4) ceildiv 64)
//      CHECK: hal.executable.export public @gemm_rhs_transpose_pack
// CHECK-NEXT:   %[[ARG0:.+]]: !hal.device
// CHECK-SAME:   %[[ARG1:.+]]: index,
// CHECK-SAME:   %[[ARG2:.+]]: index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[W0:.+]] = affine.apply #[[MAP0]]()[%[[ARG2]]]
//  CHECK-DAG:   %[[W1:.+]] = affine.apply #[[MAP1]]()[%[[ARG1]]]
//      CHECK:   hal.return %[[W1]], %[[W0]], %[[C1]]

// -----

hal.executable private @clone_index_computations {
  hal.executable.variant public @embedded_elf_x86_64, target = <"llvm-cpu", "embedded-elf-x86_64", {}> {
    hal.executable.export public @clone_index_computations ordinal(0) layout(
        #hal.pipeline.layout<push_constants = 4, sets = [
            <0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>)
        attributes {translation_info = #iree_codegen.translation_info<CPUDataTiling>} {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_set_encoding_op %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @clone_index_computations() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = hal.interface.constant.load[2] : i32
        %3 = hal.interface.constant.load[3] : i32
        %4 = arith.index_castui %0 : i32 to index
        %5 = arith.index_castui %1 : i32 to index
        %6 = arith.index_castui %2 : i32 to index
        %7 = arith.index_castui %3 : i32 to index
        %8 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%4, %5}
        %9 = affine.apply affine_map<()[s0] -> (s0 ceildiv 8)>()[%6]
        %10 = affine.apply affine_map<()[s0] -> (s0 ceildiv 4)>()[%7]
        %11 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
            : !flow.dispatch.tensor<writeonly:tensor<?x?x8x4xf32>>{%9, %10}
        %12 = flow.dispatch.tensor.load %8, offsets = [0, 0], sizes = [%4, %5], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%4, %5} -> tensor<?x?xf32>
        %13 = affine.apply affine_map<()[s0, s1] -> (-s0 + s1 + (s0 ceildiv 16) * 16)>()[%4, %4]
        %14 = affine.apply affine_map<()[s0, s1] -> (-s0 + s1 + (s0 ceildiv 16) * 16)>()[%5, %5]
        %15 = affine.apply affine_map<()[s0] -> (s0 ceildiv 8)>()[%13]
        %16 = affine.apply affine_map<()[s0] -> (s0 ceildiv 4)>()[%14]
        %17 = tensor.empty(%15, %16) : tensor<?x?x8x4xf32>
        %18 = tensor.pack %12 padding_value(%cst : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 4] into %17
            {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 64]]>}
            : tensor<?x?xf32> -> tensor<?x?x8x4xf32>
        %19 = affine.apply affine_map<()[s0] -> (s0 ceildiv 8)>()[%6]
        %20 = affine.apply affine_map<()[s0] -> (s0 ceildiv 4)>()[%7]
        flow.dispatch.tensor.store %18, %11, offsets = [0, 0, 0, 0], sizes = [%19, %20, 8, 4], strides = [1, 1, 1, 1]
            : tensor<?x?x8x4xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?x8x4xf32>>{%19, %20}
        return
      }
    }
  }
}
//   CHECK-DAG: #[[MAP7:.+]] = affine_map<(d0)[s0] -> (-d0 + (s0 ceildiv 16) * 2, 64)>
//   CHECK-DAG: #[[MAP8:.+]] = affine_map<(d0)[s0] -> (-d0 + (s0 ceildiv 16) * 4, 64)>
//       CHECK: func @clone_index_computations()
//       CHECK:   scf.for
//       CHECK:     %[[SIZE_Y:.+]] = affine.min #[[MAP7]](%{{.+}})[%{{.+}}]
//       CHECK:     scf.for
//       CHECK:       %[[SIZE_X:.+]] = affine.min #[[MAP8]](%{{.+}})[%{{.+}}]
//       CHECK:       flow.dispatch.tensor.store
//  CHECK-SAME:           sizes = [%[[SIZE_Y]], %[[SIZE_X]], 8, 4]

// -----

hal.executable private @dynamic_unpack {
  hal.executable.variant public @embedded_elf_x86_64, target = <"llvm-cpu", "embedded-elf-x86_64", {}> {
    hal.executable.export public @dynamic_unpack ordinal(0) layout(
        #hal.pipeline.layout<push_constants = 4, sets = [
            <0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>)
        attributes {translation_info = #iree_codegen.translation_info<CPUDataTiling>} {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      %c1 = arith.constant 1 : index
      %0 = affine.apply affine_map<()[s0] -> (s0 ceildiv 64)>()[%arg1]
      %1 = affine.apply affine_map<()[s0] -> (s0 ceildiv 64)>()[%arg2]
      hal.return %1, %0, %c1 : index, index, index
    }
    builtin.module {
      func.func @dynamic_unpack() {
        %c131072 = arith.constant 131072 : index
        %c0 = arith.constant 0 : index
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = hal.interface.constant.load[2] : i32
        %3 = hal.interface.constant.load[3] : i32
        %4 = arith.index_castui %0 : i32 to index
        %5 = arith.index_castui %1 : i32 to index
        %6 = arith.index_castui %2 : i32 to index
        %7 = arith.index_castui %3 : i32 to index
        %8 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<?x?x32x16xi32>>{%4, %5}
        %9 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c131072) : !flow.dispatch.tensor<writeonly:tensor<?x?xi32>>{%6, %7}
        %10 = flow.dispatch.tensor.load %8, offsets = [0, 0, 0, 0], sizes = [%4, %5, 32, 16], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?x32x16xi32>>{%4, %5} -> tensor<?x?x32x16xi32>
        %11 = tensor.empty(%6, %7) : tensor<?x?xi32>
        %12 = tensor.unpack %10 inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %11
          {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 64]]>}
          : tensor<?x?x32x16xi32> -> tensor<?x?xi32>
        flow.dispatch.tensor.store %12, %9, offsets = [0, 0], sizes = [%6, %7], strides = [1, 1] : tensor<?x?xi32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xi32>>{%6, %7}
        return
      }
    }
  }
}
// CHECK-LABEL: func.func @dynamic_unpack
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             tensor.unpack

// -----

hal.executable private @dynamic_unpack_dynamic_tile {
  hal.executable.variant public @embedded_elf_x86_64, target = <"llvm-cpu", "embedded-elf-x86_64", {}> {
    hal.executable.export public @dynamic_unpack_dynamic_tile ordinal(0) layout(
        #hal.pipeline.layout<push_constants = 4, sets = [
            <0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>)
        attributes {translation_info = #iree_codegen.translation_info<CPUDataTiling>} {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      %c1 = arith.constant 1 : index
      %0 = affine.apply affine_map<()[s0] -> (s0 ceildiv 64)>()[%arg1]
      %1 = affine.apply affine_map<()[s0] -> (s0 ceildiv 64)>()[%arg2]
      hal.return %1, %0, %c1 : index, index, index
    }
    builtin.module {
      func.func @dynamic_unpack_dynamic_tile() {
        %c131072 = arith.constant 131072 : index
        %c0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c32 = arith.constant 32 : index
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = hal.interface.constant.load[2] : i32
        %3 = hal.interface.constant.load[3] : i32
        %4 = arith.index_castui %0 : i32 to index
        %5 = arith.index_castui %1 : i32 to index
        %6 = arith.index_castui %2 : i32 to index
        %7 = arith.index_castui %3 : i32 to index
        %8 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xi32>>{%4, %5, %c32, %c16}
        %9 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c131072) : !flow.dispatch.tensor<writeonly:tensor<?x?xi32>>{%6, %7}
        %10 = flow.dispatch.tensor.load %8, offsets = [0, 0, 0, 0], sizes = [%4, %5, %c32, %c16], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xi32>>{%4, %5, %c32, %c16} -> tensor<?x?x?x?xi32>
        %11 = tensor.empty(%6, %7) : tensor<?x?xi32>
        %12 = tensor.unpack %10 inner_dims_pos = [0, 1] inner_tiles = [%c32, %c16] into %11
          {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 64]]>}
          : tensor<?x?x?x?xi32> -> tensor<?x?xi32>
        flow.dispatch.tensor.store %12, %9, offsets = [0, 0], sizes = [%6, %7], strides = [1, 1] : tensor<?x?xi32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xi32>>{%6, %7}
        return
      }
    }
  }
}
// CHECK-LABEL: func.func @dynamic_unpack_dynamic_tile
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             tensor.unpack

// -----

hal.executable private @unpack_elem {
  hal.executable.variant public @embedded_elf_arm_64, target = <"llvm-cpu", "embedded-elf-arm_64", {}> {
    hal.executable.export public @unpack_elem ordinal(0) layout(
        #hal.pipeline.layout<push_constants = 0, sets = [
            <0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>)
        attributes {translation_info = #iree_codegen.translation_info<CPUDataTiling>} {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      %c1 = arith.constant 1 : index
      %0 = affine.apply affine_map<()[s0] -> (s0 ceildiv 64)>()[%arg1]
      %1 = affine.apply affine_map<()[s0] -> (s0 ceildiv 64)>()[%arg2]
      hal.return %1, %0, %c1 : index, index, index
    }
    builtin.module {
      func.func @unpack_elem() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<16x48x8x8xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<128x384xf32>>
        %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [16, 48, 8, 8], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x48x8x8xf32>> -> tensor<16x48x8x8xf32>
        %3 = tensor.empty() : tensor<128x384xf32>
        %4 = tensor.unpack %2 inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %3 {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 64]]>} : tensor<16x48x8x8xf32> -> tensor<128x384xf32>
        %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<128x384xf32>) outs(%3 : tensor<128x384xf32>) {
        ^bb0(%in: f32, %out: f32):
          %6 = arith.addf %in, %in : f32
          linalg.yield %6 : f32
        } -> tensor<128x384xf32>
        flow.dispatch.tensor.store %5, %1, offsets = [0, 0], sizes = [128, 384], strides = [1, 1] : tensor<128x384xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x384xf32>>
        return
      }
    }
  }
}

// CHECK-LABEL: func.func @unpack_elem
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             tensor.unpack
// CHECK:             linalg.generic

// -----

hal.executable private @dynamic_unpack_fusion {
  hal.executable.variant public @vmvx_bytecode_fb, target = <"vmvx", "vmvx-bytecode-fb", {ukernels = true}> {
    hal.executable.export public @dynamic_unpack_fusion ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) attributes {translation_info = #iree_codegen.translation_info<VMVXDefault>} {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @dynamic_unpack_fusion() {
        %c200960 = arith.constant 200960 : index
        %c1003776 = arith.constant 1003776 : index
        %c1053952 = arith.constant 1053952 : index
        %c0 = arith.constant 0 : index
        %c-30_i32 = arith.constant -30 : i32
        %c-128_i32 = arith.constant -128 : i32
        %c30720_i32 = arith.constant 30720 : i32
        %cst = arith.constant dense<[-918, -4433, 87, -234, -21393, 7738, 529, -8835, -16817, -375, -199, 572, 5082, 15569, -186, 4955]> : tensor<16xi32>
        %c12544 = arith.constant 12544 : index
        %c16 = arith.constant 16 : index
        %0:2 = vmvx.query_tile_sizes sizes(%c12544, %c16) flags(1245184) -> index, index
        %1 = affine.apply affine_map<()[s0] -> (12544 ceildiv s0)>()[%0#0]
        %2 = affine.apply affine_map<()[s0] -> (16 ceildiv s0)>()[%0#1]
        %3 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c200960) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xi32>>{%1, %2, %0#0, %0#1}
        %4 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c1003776) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<12544xi32>>
        %5 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c1053952) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16xi32>>
        %6 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<12544x16xi32>>
        %7:2 = vmvx.query_tile_sizes sizes(%c12544, %c16) flags(1245184) -> index, index
        %8 = affine.apply affine_map<()[s0] -> (12544 ceildiv s0)>()[%7#0]
        %9 = affine.apply affine_map<()[s0] -> (16 ceildiv s0)>()[%7#1]
        %10 = flow.dispatch.tensor.load %3, offsets = [0, 0, 0, 0], sizes = [%8, %9, %7#0, %7#1], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xi32>>{%8, %9, %7#0, %7#1} -> tensor<?x?x?x?xi32>
        %11 = flow.dispatch.tensor.load %4, offsets = [0], sizes = [12544], strides = [1] : !flow.dispatch.tensor<readonly:tensor<12544xi32>> -> tensor<12544xi32>
        %12 = flow.dispatch.tensor.load %5, offsets = [0], sizes = [16], strides = [1] : !flow.dispatch.tensor<readonly:tensor<16xi32>> -> tensor<16xi32>
        %13 = tensor.empty() : tensor<12544x16xi32>
        %14 = tensor.empty() : tensor<12544x16xi32>
        %15:2 = vmvx.query_tile_sizes sizes(%c12544, %c16) flags(1245184) -> index, index
        %16 = tensor.unpack %10 inner_dims_pos = [0, 1] inner_tiles = [%15#0, %15#1] into %14 {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[16, 16]]>} : tensor<?x?x?x?xi32> -> tensor<12544x16xi32>
        %17 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%cst, %16, %11, %12 : tensor<16xi32>, tensor<12544x16xi32>, tensor<12544xi32>, tensor<16xi32>) outs(%13 : tensor<12544x16xi32>) {
        ^bb0(%in: i32, %in_0: i32, %in_1: i32, %in_2: i32, %out: i32):
          %18 = arith.muli %in_1, %c-30_i32 : i32
          %19 = arith.subi %in_0, %18 : i32
          %20 = arith.muli %in_2, %c-128_i32 : i32
          %21 = arith.subi %19, %20 : i32
          %22 = arith.addi %21, %c30720_i32 : i32
          %23 = arith.addi %in, %22 : i32
          linalg.yield %23 : i32
        } -> tensor<12544x16xi32>
        flow.dispatch.tensor.store %17, %6, offsets = [0, 0], sizes = [12544, 16], strides = [1, 1] : tensor<12544x16xi32> -> !flow.dispatch.tensor<writeonly:tensor<12544x16xi32>>
        return
      }
    }
  }
}
// CHECK-LABEL: func.func @dynamic_unpack_fusion
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             tensor.unpack
// CHECK:             tensor.extract_slice
// CHECK:             linalg.generic

// -----

hal.executable private @elem_pack {
  hal.executable.variant public @embedded_elf_arm_64, target = <"llvm-cpu", "embedded-elf-arm_64", {cpu = "generic", cpu_features = "", data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-unknown-unknown-eabi-elf"}> {
    hal.executable.export public @elem_pack ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>, <3, storage_buffer>, <4, storage_buffer>, <5, storage_buffer>]>]>) attributes {translation_info = #iree_codegen.translation_info<CPUDataTiling>}
    builtin.module {
      func.func @elem_pack() {
        %c1339392 = arith.constant 1339392 : index
        %c0 = arith.constant 0 : index
        %c823296 = arith.constant 823296 : index
        %c825344 = arith.constant 825344 : index
        %c786432 = arith.constant 786432 : index
        %c1572864 = arith.constant 1572864 : index
        %c2359296 = arith.constant 2359296 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c1339392) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x2x512xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c786432) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<384x512xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<384x512xf32>>
        %3 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<384xi32>>
        %4 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c823296) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<512xf32>>
        %5 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c825344) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<512xf32>>
        %6 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<48x512x8x1xf32>>
        %7 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) alignment(64) offset(%c1572864) : !flow.dispatch.tensor<writeonly:tensor<384x512xf32>>
        %8 = hal.interface.binding.subspan set(0) binding(5) type(storage_buffer) alignment(64) offset(%c2359296) : !flow.dispatch.tensor<writeonly:tensor<384x512xf32>>
        %9 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [1, 2, 512], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x2x512xf32>> -> tensor<1x2x512xf32>
        %10 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [384, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<384x512xf32>> -> tensor<384x512xf32>
        %11 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [384, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<384x512xf32>> -> tensor<384x512xf32>
        %12 = flow.dispatch.tensor.load %3, offsets = [0], sizes = [384], strides = [1] : !flow.dispatch.tensor<readonly:tensor<384xi32>> -> tensor<384xi32>
        %13 = flow.dispatch.tensor.load %4, offsets = [0], sizes = [512], strides = [1] : !flow.dispatch.tensor<readonly:tensor<512xf32>> -> tensor<512xf32>
        %14 = flow.dispatch.tensor.load %5, offsets = [0], sizes = [512], strides = [1] : !flow.dispatch.tensor<readonly:tensor<512xf32>> -> tensor<512xf32>
        %15 = tensor.empty() : tensor<384x512xf32>
        %16:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%10, %11, %12, %13, %14 : tensor<384x512xf32>, tensor<384x512xf32>, tensor<384xi32>, tensor<512xf32>, tensor<512xf32>) outs(%15, %15 : tensor<384x512xf32>, tensor<384x512xf32>) {
        ^bb0(%in: f32, %in_0: f32, %in_1: i32, %in_2: f32, %in_3: f32, %out: f32, %out_4: f32):
          %19 = linalg.index 1 : index
          %20 = arith.addf %in, %cst : f32
          %21 = arith.index_cast %in_1 : i32 to index
          %extracted = tensor.extract %9[%c0, %21, %19] : tensor<1x2x512xf32>
          %22 = arith.addf %20, %in_0 : f32
          %23 = arith.addf %22, %extracted : f32
          %24 = arith.mulf %23, %in_2 : f32
          %25 = arith.addf %24, %in_3 : f32
          linalg.yield %23, %25 : f32, f32
        } -> (tensor<384x512xf32>, tensor<384x512xf32>)
        %17 = tensor.empty() : tensor<48x512x8x1xf32>
        %18 = tensor.pack %16#0 inner_dims_pos = [0, 1] inner_tiles = [8, 1] into %17 {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[8, 64]]>} : tensor<384x512xf32> -> tensor<48x512x8x1xf32>
        flow.dispatch.tensor.store %18, %6, offsets = [0, 0, 0, 0], sizes = [48, 512, 8, 1], strides = [1, 1, 1, 1] : tensor<48x512x8x1xf32> -> !flow.dispatch.tensor<writeonly:tensor<48x512x8x1xf32>>
        flow.dispatch.tensor.store %16#0, %7, offsets = [0, 0], sizes = [384, 512], strides = [1, 1] : tensor<384x512xf32> -> !flow.dispatch.tensor<writeonly:tensor<384x512xf32>>
        flow.dispatch.tensor.store %16#1, %8, offsets = [0, 0], sizes = [384, 512], strides = [1, 1] : tensor<384x512xf32> -> !flow.dispatch.tensor<writeonly:tensor<384x512xf32>>
        return
      }
    }
  }
}
// CHECK-LABEL: func.func @elem_pack
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             %[[ELEM:.+]]:2 = linalg.generic
// CHECK:             %[[PACK:.+]] = tensor.pack
// CHECK-DAG:         flow.dispatch.tensor.store %[[PACK]], {{.*}} sizes = [8, 64, 8, 1]
// CHECK-DAG:         flow.dispatch.tensor.store %[[ELEM]]#0, {{.*}} sizes = [64, 64]
// CHECK-DAG:         flow.dispatch.tensor.store %[[ELEM]]#1, {{.*}} sizes = [64, 64]
