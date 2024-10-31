// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-codegen-tile-and-distribute-to-workgroups, canonicalize)), cse)))' --split-input-file %s | FileCheck %s
// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-codegen-tile-and-distribute-to-workgroups{max-workgroup-parallel-dims=1}, canonicalize)), cse)))' --split-input-file %s | FileCheck %s -check-prefix=CHECKW
// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-codegen-tile-and-distribute-to-workgroups{distribution-method=2})), canonicalize, cse)))' --split-input-file %s | FileCheck %s -check-prefix=NO-LOOP
#config = #iree_codegen.lowering_config<tile_sizes = [[64, 64, 0], [16, 4, 0], [0, 0, 64]]>
#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "aarch64-none-elf"
}>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
hal.executable private @matmul_tensors {
  hal.executable.variant public @llvm target(#executable_target_embedded_elf_arm_64_) {
    hal.executable.export public @matmul_tensors layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_tensors() attributes {translation_info = #translation} {
        %cl_0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
        %cl_1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
        %cl_2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
        %0 = flow.dispatch.workload.ordinal %cl_0, 0 : index
        %1 = flow.dispatch.workload.ordinal %cl_1, 1 : index
        %2 = flow.dispatch.workload.ordinal %cl_2, 2 : index
        %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %2}
        %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %1}
        %5 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1}
        %6 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3)
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
// CHECK-NEXT:   (%[[DEVICE:.+]]: !hal.device,
// CHECK-SAME:    %[[WORKLOAD_M:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_N:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_K:[a-zA-Z0-9_]+]]: index)
//  CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:    %[[D0:.+]] = affine.apply #[[MAP0]]()[%[[WORKLOAD_M]]]
//  CHECK-DAG:    %[[D1:.+]] = affine.apply #[[MAP0]]()[%[[WORKLOAD_N]]]
//      CHECK:    hal.return %[[D1]], %[[D0]], %[[C1]] : index, index, index
//      CHECK: func.func @matmul_tensors()
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
//  CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(0)
//  CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(1)
//  CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(2)
//  CHECK-DAG:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//  CHECK-DAG:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//  CHECK-DAG:   %[[INIT_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//  CHECK-DAG:   %[[OUT_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(3)
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
#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
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
  hal.executable.variant public @llvm target(#executable_target_embedded_elf_x86_64_) {
    hal.executable.export public @add layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @add() attributes {translation_info = #translation} {
        %cl_0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
        %cl_1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
        %0 = flow.dispatch.workload.ordinal %cl_0, 0 : index
        %1 = flow.dispatch.workload.ordinal %cl_1, 1 : index
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1}
        %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1)
            : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%1}
        %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2)
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
// CHECK-NEXT:   (%[[DEVICE:.+]]: !hal.device,
// CHECK-SAME:    %[[WORKLOAD_0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_1:[a-zA-Z0-9_]+]]: index)
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[D0:.+]] = affine.apply #[[MAP]]()[%[[WORKLOAD_0]]]
//  CHECK-DAG:   %[[D1:.+]] = affine.apply #[[MAP]]()[%[[WORKLOAD_1]]]
//      CHECK:   hal.return %[[D1]], %[[D0]], %[[C1]] : index, index, index
//      CHECK: func.func @add()
// CHECK-SAME:  translation_info = #[[TRANSLATION]]
//      CHECK:   scf.for %[[IV0:.+]] =
//      CHECK:     scf.for %[[IV1:.+]] =
//      CHECK:       %[[RESULT:.+]] = linalg.generic
//      CHECK:       flow.dispatch.tensor.store %[[RESULT]], %{{.+}}, offsets = [%[[IV0]], %[[IV1]]]

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[0, 64, 64, 64], [1, 1, 1, 4], [0, 0, 0, 0]]>
#pipeline_layout = #hal.pipeline.layout<constants = 4, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "x86_64-unknown-linux-gnu"}>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
hal.executable private @add4D {
  hal.executable.variant public @llvm target(#executable_target_embedded_elf_x86_64_) {
    hal.executable.export public @add4D layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index, %arg4 :index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1, %arg2, %arg3, %arg4
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @add4D() attributes {translation_info = #translation} {
        %cl_0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
        %cl_1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
        %cl_2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
        %cl_3 = hal.interface.constant.load layout(#pipeline_layout) ordinal(3) : index
        %0 = flow.dispatch.workload.ordinal %cl_0, 0 : index
        %1 = flow.dispatch.workload.ordinal %cl_1, 1 : index
        %2 = flow.dispatch.workload.ordinal %cl_2, 2 : index
        %3 = flow.dispatch.workload.ordinal %cl_3, 3  : index
        %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(32)
            : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3}
        %5 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(32)
            : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3}
        %6 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(32)
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
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
//      CHECK:   scf.for %[[IV0:.+]] =
//      CHECK:     scf.for %[[IV1:.+]] =
//      CHECK:       scf.for %[[IV2:.+]] =
//  CHECK-NOT:         scf.for
//      CHECK:         %[[GENERIC:.+]] = linalg.generic
//      CHECK:         flow.dispatch.tensor.store %[[GENERIC]], %{{.+}}, offsets = [0, %[[IV0]], %[[IV1]], %[[IV2]]]

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[2, 64, 64, 64], [1, 1, 1, 4], [0, 0, 0, 0]]>
#pipeline_layout = #hal.pipeline.layout<constants = 4, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "x86_64-unknown-linux-gnu"}>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
hal.executable private @add_distribute4D {
  hal.executable.variant public @llvm target(#executable_target_embedded_elf_x86_64_) {
    hal.executable.export public @add_distribute4D layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index, %arg4 :index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1, %arg2, %arg3, %arg4
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @add_distribute4D() attributes {translation_info = #translation} {
        %cl_0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
        %cl_1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
        %cl_2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
        %cl_3 = hal.interface.constant.load layout(#pipeline_layout) ordinal(3) : index
        %0 = flow.dispatch.workload.ordinal %cl_0, 0 : index
        %1 = flow.dispatch.workload.ordinal %cl_1, 1 : index
        %2 = flow.dispatch.workload.ordinal %cl_2, 2 : index
        %3 = flow.dispatch.workload.ordinal %cl_3, 3 : index
        %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(32)
            : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3}
        %5 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(32)
            : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3}
        %6 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(32)
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
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<()[s0, s1] -> ((s0 floordiv (s1 ceildiv 64)) * 2)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<()[s0] -> ((s0 ceildiv 2) * 2)>
//  CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 2)>
//  CHECK-DAG: #[[MAP5:.+]] = affine_map<()[s0, s1] -> ((s0 mod (s1 ceildiv 64)) * 64)>
//  CHECK-DAG: #[[MAP6:.+]] = affine_map<()[s0] -> ((s0 ceildiv 64) * 64)>
//  CHECK-DAG: #[[MAP7:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 64)>
//  CHECK-DAG: #[[MAP8:.+]] = affine_map<()[s0] -> (s0 * 64)>
//  CHECK-DAG: #[[MAP9:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: hal.executable.export public @add_distribute4D
// CHECK-NEXT:   (%[[DEVICE:.+]]: !hal.device,
// CHECK-SAME:    %[[WORKLOAD_0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_3:[a-zA-Z0-9_]+]]: index)
//  CHECK-DAG:    %[[D0:.+]] = affine.apply #[[MAP1]]()[%[[WORKLOAD_1]], %[[WORKLOAD_0]]
//  CHECK-DAG:    %[[D1:.+]] = affine.apply #[[MAP]]()[%[[WORKLOAD_2]]]
//  CHECK-DAG:    %[[D2:.+]] = affine.apply #[[MAP]]()[%[[WORKLOAD_3]]]
//      CHECK:    hal.return %[[D2]], %[[D1]], %[[D0]] : index, index, index
//      CHECK:    func.func @add_distribute4D()
// CHECK-SAME:    translation_info = #[[TRANSLATION]]
//  CHECK-DAG:      %[[D0:.*]] = hal.interface.constant.load layout({{.+}}) ordinal(0) : index
//  CHECK-DAG:      %[[D1:.*]] = hal.interface.constant.load layout({{.+}}) ordinal(1) : index
//  CHECK-DAG:      %[[D2:.*]] = hal.interface.constant.load layout({{.+}}) ordinal(2) : index
//  CHECK-DAG:      %[[D3:.*]] = hal.interface.constant.load layout({{.+}}) ordinal(3) : index
//  CHECK-DAG:      %[[D4:.*]] = hal.interface.binding.subspan layout({{.+}}) binding(0) alignment(32) : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%[[D0]], %[[D1]], %[[D2]], %[[D3]]}
//  CHECK-DAG:      %[[D5:.*]] = hal.interface.binding.subspan layout({{.+}}) binding(1) alignment(32) : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%[[D0]], %[[D1]], %[[D2]], %[[D3]]}
//  CHECK-DAG:      %[[D6:.*]] = hal.interface.binding.subspan layout({{.+}}) binding(2) alignment(32) : !flow.dispatch.tensor<writeonly:tensor<?x?x?x?xf32>>{%[[D0]], %[[D1]], %[[D2]], %[[D3]]}
//      CHECK:      %[[WORKGROUP_ID_X:.*]] = hal.interface.workgroup.id[0] : index
//      CHECK:      %[[WORKGROUP_COUNT_X:.*]] = hal.interface.workgroup.count[0] : index
//      CHECK:      %[[WORKGROUP_ID_Y:.*]] = hal.interface.workgroup.id[1] : index
//      CHECK:      %[[WORKGROUP_COUNT_Y:.*]] = hal.interface.workgroup.count[1] : index
//      CHECK:      %[[WORKGROUP_ID_Z:.*]] = hal.interface.workgroup.id[2] : index
//  CHECK-DAG:      %[[D7:.*]] = affine.apply #map2(){{\[}}%[[WORKGROUP_ID_Z]], %[[D1]]]
//  CHECK-DAG:      %[[D8:.*]] = affine.apply #map3(){{\[}}%[[D0]]]
//      CHECK:      scf.for %[[ARG0:.*]] = %[[D7]] to %[[D0]] step %[[D8]] {
//  CHECK-DAG:        %[[D9:.*]] = affine.min #map4(%[[ARG0]]){{\[}}%[[D0]]]
//  CHECK-DAG:        %[[D10:.*]] = affine.apply #map5(){{\[}}%[[WORKGROUP_ID_Z]], %[[D1]]]
//  CHECK-DAG:        %[[D11:.*]] = affine.apply #map6(){{\[}}%[[D1]]]
//      CHECK:        scf.for %[[ARG1:.*]] = %[[D10]] to %[[D1]] step %[[D11]] {
//  CHECK-DAG:          %[[D12:.*]] = affine.min #map7(%[[ARG1]]){{\[}}%[[D1]]]
//  CHECK-DAG:          %[[D13:.*]] = affine.apply #map8(){{\[}}%[[WORKGROUP_ID_Y]]]
//  CHECK-DAG:          %[[D14:.*]] = affine.apply #map8(){{\[}}%[[WORKGROUP_COUNT_Y]]]
//      CHECK:          scf.for %[[ARG2:.*]] = %[[D13]] to %[[D2]] step %[[D14]] {
//  CHECK-DAG:            %[[D15:.*]] = affine.min #map7(%[[ARG2]]){{\[}}%[[D2]]]
//  CHECK-DAG:            %[[D16:.*]] = affine.apply #map8(){{\[}}%[[WORKGROUP_ID_X]]]
//  CHECK-DAG:            %[[D17:.*]] = affine.apply #map8(){{\[}}%[[WORKGROUP_COUNT_X]]]
//      CHECK:            scf.for %[[ARG3:.*]] = %[[D16]] to %[[D3]] step %[[D17]] {
//      CHECK:              %[[D18:.*]] = affine.min #map7(%[[ARG3]]){{\[}}%[[D3]]]
//      CHECK:              %[[D19:.*]] = flow.dispatch.tensor.load %[[D4]], offsets = {{\[}}%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]]], sizes = {{\[}}%[[D9]], %[[D12]], %[[D15]], %[[D18]]], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%[[D0]], %[[D1]], %[[D2]], %[[D3]]} -> tensor<?x?x?x?xf32>
//      CHECK:              %[[D20:.*]] = flow.dispatch.tensor.load %[[D5]], offsets = {{\[}}%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]]], sizes = {{\[}}%[[D9]], %[[D12]], %[[D15]], %[[D18]]], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%[[D0]], %[[D1]], %[[D2]], %[[D3]]} -> tensor<?x?x?x?xf32>
//      CHECK:              %[[D21:.*]] = tensor.empty(%[[D9]], %[[D12]], %[[D15]], %[[D18]]) : tensor<?x?x?x?xf32>
//      CHECK:              %[[D22:.*]] = linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[D19]], %[[D20]] : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) outs(%[[D21]] : tensor<?x?x?x?xf32>) attrs =  {lowering_config = #config} {
//      CHECK:              ^bb0(%[[IN:.*]]: f32, %[[IN_0:.*]]: f32, %[[OUT:.*]]: f32):
//      CHECK:                %[[D23:.*]] = arith.addf %[[IN]], %[[IN_0]] : f32
//      CHECK:                linalg.yield %[[D23]] : f32
//      CHECK:              } -> tensor<?x?x?x?xf32>
//      CHECK:              flow.dispatch.tensor.store %[[D22:.*]], %[[D6]], offsets = {{\[}}%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]]], sizes = {{\[}}%[[D9]], %[[D12]], %[[D15]], %[[D18]]], strides = [1, 1, 1, 1] : tensor<?x?x?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?x?x?xf32>>{%[[D0]], %[[D1]], %[[D2]], %[[D3]]}

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[2, 64, 0, 64], [1, 1, 1, 4], [0, 0, 0, 0]]>
#pipeline_layout = #hal.pipeline.layout<constants = 4, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "x86_64-unknown-linux-gnu"}>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
hal.executable private @add_distribute4D_zero_tile_size {
  hal.executable.variant public @llvm target(#executable_target_embedded_elf_x86_64_) {
    hal.executable.export public @add_distribute4D_zero_tile_size layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index, %arg4 :index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1, %arg2, %arg3, %arg4
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @add_distribute4D_zero_tile_size() attributes {translation_info = #translation} {
        %cl_0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
        %cl_1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
        %cl_2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
        %cl_3 = hal.interface.constant.load layout(#pipeline_layout) ordinal(3) : index
        %0 = flow.dispatch.workload.ordinal %cl_0, 0 : index
        %1 = flow.dispatch.workload.ordinal %cl_1, 1 : index
        %2 = flow.dispatch.workload.ordinal %cl_2, 2 : index
        %3 = flow.dispatch.workload.ordinal %cl_3, 3 : index
        %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(32)
            : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3}
        %5 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(32)
            : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3}
        %6 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(32)
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
// CHECK-SAME:   translation_info = #[[TRANSLATION]]

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[1, 64, 64, 0], [1, 16, 4, 0], [0, 0, 0, 64]]>
#pipeline_layout = #hal.pipeline.layout<constants = 4, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "aarch64-none-elf"}>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
hal.executable private @batch_matmul_tensors {
  hal.executable.variant public @llvm target(#executable_target_embedded_elf_arm_64_) {
    hal.executable.export public @batch_matmul_tensors layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index, %arg4 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1, %arg2, %arg3, %arg4
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @batch_matmul_tensors() attributes {translation_info = #translation} {
        %cst = arith.constant 0.000000e+00 : f32
        %cl_0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
        %cl_1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
        %cl_2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
        %cl_3 = hal.interface.constant.load layout(#pipeline_layout) ordinal(3) : index
        %0 = flow.dispatch.workload.ordinal %cl_0, 0 : index
        %1 = flow.dispatch.workload.ordinal %cl_1, 1 : index
        %2 = flow.dispatch.workload.ordinal %cl_2, 2 : index
        %3 = flow.dispatch.workload.ordinal %cl_3, 3 : index
        %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(32)
            : !flow.dispatch.tensor<readonly:tensor<?x?x?xf32>>{%0, %1, %3}
        %5 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(32)
            : !flow.dispatch.tensor<readonly:tensor<?x?x?xf32>>{%0, %3, %2}
        %6 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(32)
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
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64">
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
hal.executable private @preset_config_matmul_tensors {
  hal.executable.variant public @system_elf_x86_64 target(#executable_target_system_elf_x86_64_) {
    hal.executable.export public @preset_config layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @preset_config() attributes {translation_info = #translation} {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0)
            : !flow.dispatch.tensor<readonly:tensor<128x256xf32>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1)
            : !flow.dispatch.tensor<readonly:tensor<256x512xf32>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2)
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
//      CHECK: hal.executable.export public @preset_config
// CHECK-NEXT:   (%[[DEVICE:.+]]: !hal.device)
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
#pipeline_layout = #hal.pipeline.layout<constants = 10, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64">
#translation = #iree_codegen.translation_info<CPUBufferOpsTileAndVectorize>
hal.executable public @copy_op {
  hal.executable.variant public @system_elf_x86_64 target(#executable_target_system_elf_x86_64_) {
    hal.executable.export public @copy_op layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3: index, %arg4 : index, %arg5: index, %arg6 : index, %arg7: index, %arg8 : index, %arg9: index, %arg10: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @copy_op() attributes {translation_info = #translation} {
        %cl_0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
        %cl_1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
        %cl_2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
        %cl_3 = hal.interface.constant.load layout(#pipeline_layout) ordinal(3) : index
        %cl_4 = hal.interface.constant.load layout(#pipeline_layout) ordinal(4) : index
        %cl_5 = hal.interface.constant.load layout(#pipeline_layout) ordinal(5) : index
        %cl_6 = hal.interface.constant.load layout(#pipeline_layout) ordinal(6) : index
        %cl_7 = hal.interface.constant.load layout(#pipeline_layout) ordinal(7) : index
        %cl_8 = hal.interface.constant.load layout(#pipeline_layout) ordinal(8) : index
        %cl_9 = hal.interface.constant.load layout(#pipeline_layout) ordinal(9) : index
        %source_size_y = flow.dispatch.workload.ordinal %cl_0, 0: index
        %source_size_x = flow.dispatch.workload.ordinal %cl_1, 1: index
        %dest_size_y = flow.dispatch.workload.ordinal %cl_2, 2: index
        %dest_size_x = flow.dispatch.workload.ordinal %cl_3, 3: index
        %source_offset_y = flow.dispatch.workload.ordinal %cl_4, 4: index
        %source_offset_x = flow.dispatch.workload.ordinal %cl_5, 5: index
        %dest_offset_y = flow.dispatch.workload.ordinal %cl_6, 6: index
        %dest_offset_x = flow.dispatch.workload.ordinal %cl_7, 7: index
        %slice_size_y = flow.dispatch.workload.ordinal %cl_8, 8: index
        %slice_size_x = flow.dispatch.workload.ordinal %cl_9, 9: index
        %source = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<?x?xi32>{%source_size_y, %source_size_x}
        %dest = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : memref<?x?xi32>{%dest_size_y, %dest_size_x}
        %source_subview = memref.subview %source[%source_offset_y, %source_offset_x] [%slice_size_y, %slice_size_x] [1, 1] : memref<?x?xi32> to memref<?x?xi32, strided<[?, 1], offset : ?>>
        %dest_subview = memref.subview %dest[%dest_offset_y, %dest_offset_x] [%slice_size_y, %slice_size_x] [1, 1] : memref<?x?xi32> to memref<?x?xi32, strided<[?, 1], offset : ?>>
        linalg.generic {
            indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
            iterator_types = ["parallel", "parallel"]}
            ins(%source_subview : memref<?x?xi32, strided<[?, 1], offset : ?>>)
            outs(%dest_subview : memref<?x?xi32, strided<[?, 1], offset : ?>>)
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
//      CHECK:   %[[C1:.+]] = arith.constant 1 : index
//      CHECK:   %[[D0:.+]] = affine.apply #[[MAP0]]()[%{{.+}}]
//      CHECK:   %[[D1:.+]] = affine.apply #[[MAP0]]()[%{{.+}}]
//      CHECK:   hal.return %[[D1]], %[[D0]], %[[C1]]
//      CHECK: func.func @copy_op()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//  CHECK-DAG:   %[[SOURCE_SIZE_Y:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(0) : index
//  CHECK-DAG:   %[[SOURCE_SIZE_X:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(1) : index
//  CHECK-DAG:   %[[DEST_SIZE_Y:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(2) : index
//  CHECK-DAG:   %[[DEST_SIZE_X:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(3) : index
//  CHECK-DAG:   %[[SOURCE_OFFSET_Y:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(4) : index
//  CHECK-DAG:   %[[SOURCE_OFFSET_X:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(5) : index
//  CHECK-DAG:   %[[DEST_OFFSET_Y:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(6) : index
//  CHECK-DAG:   %[[DEST_OFFSET_X:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(7) : index
//  CHECK-DAG:   %[[SLICE_SIZE_Y:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(8) : index
//  CHECK-DAG:   %[[SLICE_SIZE_X:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(9) : index
//  CHECK-DAG:   %[[SOURCE_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//  CHECK-DAG:   %[[DEST_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
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

#config = #iree_codegen.lowering_config<tile_sizes = [[15]]>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64">
#translation = #iree_codegen.translation_info<CPUDefault>
hal.executable private @static_1d_fft_stage2 {
  hal.executable.variant public @system_elf_x86_64 target(#executable_target_system_elf_x86_64_) {
    hal.executable.export public @static_1d_fft_stage2 layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @static_1d_fft_stage2() attributes {translation_info = #translation} {
        %c2 = arith.constant 2 : index
        %cst = arith.constant dense<[1.000000e+00, 6.12323426E-17]> : tensor<2xf32>
        %cst_0 = arith.constant dense<[-0.000000e+00, -1.000000e+00]> : tensor<2xf32>
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0)
            : !flow.dispatch.tensor<readwrite:tensor<32xf32>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1)
            : !flow.dispatch.tensor<readwrite:tensor<32xf32>>
        %2 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [32], strides = [1]
            : !flow.dispatch.tensor<readwrite:tensor<32xf32>> -> tensor<32xf32>
        %3 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [32], strides = [1]
            : !flow.dispatch.tensor<readwrite:tensor<32xf32>> -> tensor<32xf32>
        %4:2 = iree_linalg_ext.fft {lowering_config = #config}
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
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDefault>
//      CHECK: hal.executable private @static_1d_fft_stage2
//      CHECK: hal.executable.export public @static_1d_fft_stage2
//  CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//      CHECK:   hal.return %[[C3]], %[[C1]], %[[C1]] : index, index, index
//      CHECK: func.func @static_1d_fft_stage2()
// CHECK-SAME:  translation_info = #[[TRANSLATION]]
//      CHECK:   scf.for %[[IV0:.+]] =
//      CHECK:     %[[RESULT:.+]]:2 = iree_linalg_ext.fft
//  CHECK-DAG:     flow.dispatch.tensor.store %[[RESULT]]#0, %{{.+}}, offsets = [%[[IV0]]]
//  CHECK-DAG:     flow.dispatch.tensor.store %[[RESULT]]#1, %{{.+}}, offsets = [%[[IV0]]]

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[64, 64, 64]]>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64">
#translation = #iree_codegen.translation_info<CPUDefault>
hal.executable private @static_3d_fft_stage3 {
  hal.executable.variant public @system_elf_x86_64 target(#executable_target_system_elf_x86_64_) {
    hal.executable.export public @static_3d_fft_stage3 layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @static_3d_fft_stage3() attributes {translation_info = #translation} {
        %c3 = arith.constant 3 : index
        %cst = arith.constant dense<[1.000000e+00, 0.707106769, 6.12323426E-17, -0.707106769]> : tensor<4xf32>
        %cst_0 = arith.constant dense<[-0.000000e+00, -0.707106769, -1.000000e+00, -0.707106769]> : tensor<4xf32>
        %0 = bufferization.to_memref %cst_0 : memref<4xf32>
        %1 = bufferization.to_memref %cst : memref<4xf32>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<64x128x32xf32>
        %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : memref<64x128x32xf32>
        iree_linalg_ext.fft {lowering_config = #config}
            ins(%c3, %1, %0 : index, memref<4xf32>, memref<4xf32>) outs(%2, %3 : memref<64x128x32xf32>, memref<64x128x32xf32>)
        return
      }
    }
  }
}
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDefault>
//      CHECK: hal.executable private @static_3d_fft_stage3
//      CHECK: hal.executable.export public @static_3d_fft_stage3
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//      CHECK:   hal.return %[[C1]], %[[C2]], %[[C1]] : index, index, index
//      CHECK: func.func @static_3d_fft_stage3()
// CHECK-SAME:  translation_info = #[[TRANSLATION]]
//      CHECK:   scf.for %[[IV0:.+]] =
//      CHECK:     scf.for %[[IV1:.+]] =
//  CHECK-DAG:         %[[SUBVIEW1:.+]] = memref.subview %{{.+}}[0, %[[IV0]], %[[IV1]]]
//  CHECK-DAG:         %[[CAST1:.+]] = memref.cast %[[SUBVIEW1]]
//  CHECK-DAG:         %[[SUBVIEW2:.+]] = memref.subview %{{.+}}[0, %[[IV0]], %[[IV1]]]
//  CHECK-DAG:         %[[CAST2:.+]] = memref.cast %[[SUBVIEW2]]
//      CHECK:         iree_linalg_ext.fft
// CHECK-SAME:             outs(%[[CAST1]], %[[CAST2]] :

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[64, 64, 0], [1, 4, 0], [0, 0, 4]]>
#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64">
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
hal.executable private @outs_fusion {
  hal.executable.variant public @system_elf_x86_64 target(#executable_target_system_elf_x86_64_) {
    hal.executable.export public @outs_fusion_fn layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @outs_fusion_fn() attributes {translation_info = #translation} {
        %cst = arith.constant 0.000000e+00 : f32
        %cl_0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
        %cl_1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
        %cl_2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
        %0 = flow.dispatch.workload.ordinal %cl_0, 0 : index
        %1 = flow.dispatch.workload.ordinal %cl_1, 1 : index
        %2 = flow.dispatch.workload.ordinal %cl_2, 2 : index
        %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %2}
        %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %1}
        %5 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2)
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
// CHECK-NEXT:   (%[[DEVICE:.+]]: !hal.device,
// CHECK-SAME:    %[[WORKLOAD_0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_2:[a-zA-Z0-9_]+]]: index)
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[D0:.+]] = affine.apply #[[MAP]]()[%[[WORKLOAD_0]]]
//  CHECK-DAG:   %[[D1:.+]] = affine.apply #[[MAP]]()[%[[WORKLOAD_1]]]
//      CHECK:   hal.return %[[D1]], %[[D0]], %[[C1]] : index, index, index
//      CHECK: func.func @outs_fusion_fn
// CHECK-SAME:  translation_info = #[[TRANSLATION]]
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
#pipeline_layout = #hal.pipeline.layout<constants = 9, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "x86_64-unknown-linux-gnu"}>
#translation = #iree_codegen.translation_info<CPUDefault>
hal.executable private @conv {
  hal.executable.variant public @system_elf_x86_64 target(#executable_target_system_elf_x86_64_) {
    hal.executable.export public @conv layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index, %arg4 : index, %arg5 : index, %arg6 : index, %arg7 : index, %arg8 : index, %arg9 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @conv() attributes {translation_info = #translation} {
        %cl_0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
        %cl_1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
        %cl_2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
        %cl_3 = hal.interface.constant.load layout(#pipeline_layout) ordinal(3) : index
        %cl_4 = hal.interface.constant.load layout(#pipeline_layout) ordinal(4) : index
        %cl_5 = hal.interface.constant.load layout(#pipeline_layout) ordinal(5) : index
        %cl_6 = hal.interface.constant.load layout(#pipeline_layout) ordinal(6) : index
        %cl_7 = hal.interface.constant.load layout(#pipeline_layout) ordinal(7) : index
        %cl_8 = hal.interface.constant.load layout(#pipeline_layout) ordinal(8) : index
        %0 = flow.dispatch.workload.ordinal %cl_0, 0 : index
        %1 = flow.dispatch.workload.ordinal %cl_1, 1 : index
        %2 = flow.dispatch.workload.ordinal %cl_2, 2 : index
        %3 = flow.dispatch.workload.ordinal %cl_3, 3 : index
        %4 = flow.dispatch.workload.ordinal %cl_4, 4 : index
        %5 = flow.dispatch.workload.ordinal %cl_5, 5 : index
        %6 = flow.dispatch.workload.ordinal %cl_6, 6 : index
        %7 = flow.dispatch.workload.ordinal %cl_7, 7 : index
        %8 = flow.dispatch.workload.ordinal %cl_8, 8 : index
        %9 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0)
            : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3}
        %10 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1)
            : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%4, %5, %3, %6}
        %11 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2)
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
// CHECK-SAME:  translation_info = #[[TRANSLATION]]
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
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "x86_64-unknown-linux-gnu"}>
#translation = #iree_codegen.translation_info<CPUDefault>
hal.executable private @conv_static {
  hal.executable.variant public @system_elf_x86_64 target(#executable_target_system_elf_x86_64_) {
    hal.executable.export public @conv_static layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index, %arg4 : index, %arg5 : index, %arg6 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3, %arg4, %arg5, %arg6
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @conv_static() attributes {translation_info = #translation} {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0)
            : !flow.dispatch.tensor<readonly:tensor<1x161x161x96xf32>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1)
            : !flow.dispatch.tensor<readonly:tensor<3x3x96xf32>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2)
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
// CHECK-SAME:  translation_info = #[[TRANSLATION]]
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
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 64 : index,
  target_triple = "x86_64-pc-linux-gnu"}>
#map0 = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
hal.executable private @generic_static {
  hal.executable.variant public @system_elf_x86_64 target(#executable_target_system_elf_x86_64_) {
    hal.executable.export public @generic_static layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @generic_static() attributes {translation_info = #translation} {
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0)
            : !flow.dispatch.tensor<readonly:tensor<96x16xf32>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1)
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
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 * 32)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: hal.executable private @generic_static
//      CHECK: hal.executable.export public @generic_static
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
//      CHECK:   hal.return %[[C3]], %[[C1]], %[[C1]] : index, index, index
//      CHECK: func.func @generic_static()
// CHECK-SAME:  translation_info = #[[TRANSLATION]]
//      CHECK:   scf.for %[[IV0:.+]] =
//      CHECK:     %[[RESULT:.+]] = linalg.generic
//      CHECK:     flow.dispatch.tensor.store %[[RESULT]], %{{.+}}, offsets = [0, %[[IV0]]]

//  NO-LOOP-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 * 32)>
//      NO-LOOP: func.func @generic_static()
//  NO-LOOP-DAG:   %[[IDX:.+]] = hal.interface.workgroup.id[0] : index
//  NO-LOOP-DAG:   %[[OFFX:.+]] = affine.apply #[[MAP1]]()[%[[IDX]]]
//  NO-LOOP-NOT:   scf.for
//      NO-LOOP:   %[[RESULT:.+]] = linalg.generic
//      NO-LOOP:   -> tensor<16x32xf32>
//      NO-LOOP:   flow.dispatch.tensor.store %[[RESULT]], %{{.+}}, offsets = [0, %[[OFFX]]]

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[28, 8, 0], [4, 4, 0], [0, 0, 60]]>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_system_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "system-elf-arm_64", {
  data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "aarch64-none-linux-android30"}>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
hal.executable private @matmul_static {
  hal.executable.variant public @system_elf_arm_64 target(#executable_target_system_elf_arm_64_) {
    hal.executable.export public @matmul_static layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_static() attributes {translation_info = #translation} {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0)
            : !flow.dispatch.tensor<readonly:tensor<196x240xf32>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1)
            : !flow.dispatch.tensor<readonly:tensor<240x40xf32>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2)
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
//      CHECK: hal.executable private @matmul_static
//      CHECK: hal.executable.export public @matmul_static
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C7:.+]] = arith.constant 7 : index
//  CHECK-DAG:   %[[C5:.+]] = arith.constant 5 : index
//      CHECK:   hal.return %[[C5]], %[[C7]], %[[C1]] : index, index, index

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[0, 1, 7, 64, 0, 0]]>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_system_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "system-elf-arm_64", {
  data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "aarch64-none-linux-android30"}>
#translation = #iree_codegen.translation_info<CPUDefault>
hal.executable private @restrict_num_workgroups {
  hal.executable.variant public @system_elf_arm_64 target(#executable_target_system_elf_arm_64_) {
    hal.executable.export public @restrict_num_workgroups layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @restrict_num_workgroups() attributes {translation_info = #translation} {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0)
            : !flow.dispatch.tensor<readonly:tensor<1x11x11x576xf32>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1)
            : !flow.dispatch.tensor<readonly:tensor<5x5x576xf32>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2)
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
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 * 64)>
//      CHECK: hal.executable private @restrict_num_workgroups
//      CHECK: hal.executable.export public @restrict_num_workgroups
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C7:.+]] = arith.constant 7 : index
//  CHECK-DAG:   %[[C9:.+]] = arith.constant 9 : index
//      CHECK:   hal.return %[[C9]], %[[C7]], %[[C1]] : index, index, index

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[4, 0, 0], [4, 0, 0], [0, 1, 4]]>
#pipeline_layout = #hal.pipeline.layout<constants = 4, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "x86_64-none-elf"}>
#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0)>
#map2 = affine_map<(d0) -> (d0)>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
hal.executable private @reduction {
  hal.executable.variant public @reduction target(#executable_target_embedded_elf_x86_64_) {
    hal.executable.export public @reduction ordinal(0) layout(#pipeline_layout) attributes {translation_info = #translation} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @reduction(%arg0 : !flow.dispatch.tensor<readonly:tensor<7x7x2048xf32>>,
          %arg1 : !flow.dispatch.tensor<writeonly:tensor<7xf32>>) attributes {translation_info = #translation} {
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
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//      CHECK:   hal.return %[[C2]], %[[C1]], %[[C1]] : index, index, index
//      CHECK: func.func @reduction
// CHECK-SAME:  translation_info = #[[TRANSLATION]]
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
#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "x86_64-none-elf"}>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
hal.executable private @gemm_unit_N {
  hal.executable.variant public @embedded_elf_x86_64 target(#executable_target_embedded_elf_x86_64_) {
    hal.executable.export public @gemm_unit_N ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @gemm_unit_N() attributes {translation_info = #translation} {
        %c0 = arith.constant 0 : index
        %cl_0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
        %cl_1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
        %0 = flow.dispatch.workload.ordinal %cl_0, 0 : index
        %1 = flow.dispatch.workload.ordinal %cl_1, 1 : index
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(32) offset(%c0)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1}
        %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(32) offset(%c0)
            : !flow.dispatch.tensor<readonly:tensor<?x1xf32>>{%1}
        %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(32) offset(%c0)
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
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 * 64)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: hal.executable private @gemm_unit_N
//      CHECK: hal.executable.export public @gemm_unit_N
// CHECK-NEXT:   (%[[DEVICE:.+]]: !hal.device,
// CHECK-SAME:    %[[WORKLOAD_0:[a-zA-Z0-9_]+]]: index,
// CHECK-SAME:    %[[WORKLOAD_1:[a-zA-Z0-9_]+]]: index)
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[D0:.+]] = affine.apply #[[MAP0]]()[%[[WORKLOAD_0]]
//      CHECK:   hal.return %[[D0]], %[[C1]], %[[C1]] : index, index, index
//      CHECK: func.func @gemm_unit_N()
// CHECK-SAME:  translation_info = #[[TRANSLATION]]
//  CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load layout(#pipeline_layout) ordinal(0)
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
#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "x86_64-none-elf"}>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
hal.executable private @gemm_unit_M_unit_N {
  hal.executable.variant public @embedded_elf_x86_64 target(#executable_target_embedded_elf_x86_64_) {
    hal.executable.export public @gemm_unit_M_unit_N ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @gemm_unit_M_unit_N() attributes {translation_info = #translation} {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(32) offset(%c0)
            : !flow.dispatch.tensor<readonly:tensor<1x?xf32>>{%0}
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(32) offset(%c0)
            : !flow.dispatch.tensor<readonly:tensor<?x1xf32>>{%0}
        %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(32) offset(%c0)
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
// CHECK-NEXT:   (%[[DEVICE:.+]]: !hal.device)
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//      CHECK:   hal.return %[[C1]], %[[C1]], %[[C1]] : index, index, index
//      CHECK: func.func @gemm_unit_M_unit_N()
// CHECK-SAME:  translation_info = #[[TRANSLATION]]
//  CHECK-NOT:   scf.for
//      CHECK:   %[[GEMM:.+]] = linalg.matmul
//      CHECK:   flow.dispatch.tensor.store %[[GEMM]], %{{.+}}, offsets = [0, 0]

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[0, 0, 0, 0, 64, 64, 0, 64], [0, 1, 0, 0, 1, 1, 0, 4], [0, 0, 0, 0, 0, 0, 0, 0]]>
#pipeline_layout = #hal.pipeline.layout<constants = 4, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "x86_64-unknown-linux-gnu"}>
#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5, d6, d7)>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
hal.executable private @generic_unit_dims {
  hal.executable.variant public @llvm target(#executable_target_embedded_elf_x86_64_) {
    hal.executable.export public @generic_unit_dims layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index, %arg4 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1, %arg2, %arg3, %arg4
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @generic_unit_dims() attributes {translation_info = #translation} {
        %cl_0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
        %cl_1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
        %cl_2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
        %cl_3 = hal.interface.constant.load layout(#pipeline_layout) ordinal(3) : index
        %0 = flow.dispatch.workload.ordinal %cl_0, 0 : index
        %1 = flow.dispatch.workload.ordinal %cl_1, 1 : index
        %2 = flow.dispatch.workload.ordinal %cl_2, 2 : index
        %3 = flow.dispatch.workload.ordinal %cl_3, 3 : index
        %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0)
            : !flow.dispatch.tensor<readonly:tensor<1x?x1x1x?x?x1x?xf32>>{%0, %1, %2, %3}
        %5 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1)
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
// CHECK-NEXT:   (%[[DEVICE:.+]]: !hal.device,
// CHECK-SAME:    %[[WORKLOAD_0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_3:[a-zA-Z0-9_]+]]: index)
//  CHECK-DAG:   %[[D0:.+]] = affine.apply #[[MAP0]]()[%[[WORKLOAD_1]]]
//  CHECK-DAG:   %[[D1:.+]] = affine.apply #[[MAP0]]()[%[[WORKLOAD_2]]]
//  CHECK-DAG:   %[[D2:.+]] = affine.apply #[[MAP0]]()[%[[WORKLOAD_3]]]
//      CHECK:   hal.return %[[D2]], %[[D1]], %[[D0]] : index, index, index
//      CHECK: func.func @generic_unit_dims()
// CHECK-SAME:  translation_info = #[[TRANSLATION]]
//      CHECK:   scf.for %[[IV0:.+]] =
//      CHECK:     scf.for %[[IV1:.+]] =
//      CHECK:       scf.for %[[IV2:.+]] =
//      CHECK:         %[[GENERIC:.+]] = linalg.generic
//      CHECK:         flow.dispatch.tensor.store %[[GENERIC]],
// CHECK-SAME:             offsets = [0, 0, 0, 0, %[[IV0]], %[[IV1]], 0, %[[IV2]]]

// -----
#config = #iree_codegen.lowering_config<tile_sizes = [[0], [0], [4]]>
#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "x86_64-unknown-linux-gnu"}>
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
hal.executable private @reduce_to_scalar {
  hal.executable.variant public @llvm target(#executable_target_embedded_elf_x86_64_) {
    hal.executable.export public @reduce_to_scalar layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @reduce_to_scalar() attributes {translation_info = #translation} {
        %cl_0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
        %0 = flow.dispatch.workload.ordinal %cl_0, 0 : index
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0)
            : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%0}
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1)
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
// CHECK-NEXT:   (%[[DEVICE:.+]]: !hal.device,
// CHECK-SAME:    %[[WORKLOAD_0:[a-zA-Z0-9_]+]]: index)
//      CHECK:   %[[C1:.+]] = arith.constant 1 : index
//      CHECK:   hal.return %[[C1]], %[[C1]], %[[C1]] : index, index, index
//      CHECK: func.func @reduce_to_scalar()
// CHECK-SAME:  translation_info = #[[TRANSLATION]]
//  CHECK-NOT:   scf.for

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[]]>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "x86_64-unknown-linux-gnu"}>
#map = affine_map<() -> ()>
#translation = #iree_codegen.translation_info<CPUDefault>
hal.executable private @scalar {
  hal.executable.variant public @llvm target(#executable_target_embedded_elf_x86_64_) {
    hal.executable.export public @scalar layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @scalar() attributes {translation_info = #translation} {
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0)
            : !flow.dispatch.tensor<readonly:tensor<f32>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1)
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
// CHECK-NEXT:   (%[[DEVICE:.+]]: !hal.device)
//      CHECK:   %[[C1:.+]] = arith.constant 1 : index
//      CHECK:   hal.return %[[C1]], %[[C1]], %[[C1]] : index, index, index
//      CHECK: func.func @scalar()
// CHECK-SAME:  translation_info = #[[TRANSLATION]]
//  CHECK-NOT:   scf.for

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[2], [2], [0]]>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "aarch64-none-elf"
}>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
hal.executable private @rank_reduced_slice {
  hal.executable.variant public @llvm target(#executable_target_embedded_elf_arm_64_) {
    hal.executable.export public @rank_reduced_slice layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @rank_reduced_slice() attributes {translation_info = #translation} {
        %in_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(0)
            : !flow.dispatch.tensor<readonly:tensor<5x40xf32>>
        %out_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(1)
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
// CHECK-NEXT:   %[[WORKLOAD:[a-zA-Z0-9]+]]: index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1
//  CHECK-DAG:   %[[C5:.+]] = arith.constant 5
//      CHECK:   hal.return %[[C5]], %[[C1]], %[[C1]]
//      CHECK: func.func @rank_reduced_slice()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//  CHECK-DAG:   %[[SRC_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
// CHECK-SAME:       : !flow.dispatch.tensor<readonly:tensor<5x40xf32>>
//  CHECK-DAG:   %[[DST_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
// CHECK-SAME:       : !flow.dispatch.tensor<writeonly:tensor<10xf32>>
//      CHECK:   scf.for %[[IV0:.+]] =
//      CHECK:     %[[OFFSET:.+]] = affine.apply #[[MAP]]()[%[[IV0]]]
//      CHECK:     %[[SRC_TILE:.+]] = flow.dispatch.tensor.load %[[SRC_BINDING]]
// CHECK-SAME:         offsets = [3, %[[OFFSET]]], sizes = [1, 2], strides = [2, 1]
//      CHECK:     linalg.generic
// CHECK-SAME:         ins(%[[SRC_TILE]] :

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [{sizes=[32, 64, 0], interchange=[1, 0, 2]}, [8, 32, 0], [0, 0, 16]]>
#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "x86_64-unknown-linux-gnu"}>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
hal.executable private @matmul_interchange {
  hal.executable.variant public @llvm target(#executable_target_embedded_elf_x86_64_) {
    hal.executable.export public @matmul_interchange layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_interchange() attributes {translation_info = #translation} {
        %cl_0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
        %cl_1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
        %cl_2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
        %0 = flow.dispatch.workload.ordinal %cl_0, 0 : index
        %1 = flow.dispatch.workload.ordinal %cl_1, 1 : index
        %2 = flow.dispatch.workload.ordinal %cl_2, 2 : index
        %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %2}
        %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %1}
        %5 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1}
        %6 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3)
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
// CHECK-NEXT:   (%[[DEVICE:.+]]: !hal.device,
// CHECK-SAME:    %[[WORKLOAD_0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_2:[a-zA-Z0-9_]+]]: index)
//  CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:    %[[D0:.+]] = affine.apply #[[MAP0]]()[%[[WORKLOAD_0]]]
//  CHECK-DAG:    %[[D1:.+]] = affine.apply #[[MAP1]]()[%[[WORKLOAD_1]]]
//      CHECK:    hal.return %[[D0]], %[[D1]], %[[C1]] : index, index, index
//      CHECK: func.func @matmul_interchange()
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
//  CHECK-DAG:   %[[D0:.+]] = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
//  CHECK-DAG:   %[[D1:.+]] = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
//      CHECK:   scf.for %{{.+}} = %{{.+}} to %[[D1]] step %{{.+}} {
//      CHECK:     scf.for %{{.+}} = %{{.+}} to %[[D0]] step %{{.+}} {

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 5, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @no_compute {
  hal.executable.variant public @embedded_elf_x86_64 target(<"llvm-cpu", "embedded-elf-x86_64", {}>) {
    hal.executable.export public @no_compute ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index, %arg4 : index, %arg5 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1, %arg2, %arg3, %arg4, %arg5
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @no_compute() attributes {translation_info = #iree_codegen.translation_info<CPUDefault>} {
        %c0 = arith.constant 0 : index
        %cl_0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
        %cl_1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
        %cl_2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : i32
        %cl_3 = hal.interface.constant.load layout(#pipeline_layout) ordinal(3) : i32
        %cl_4 = hal.interface.constant.load layout(#pipeline_layout) ordinal(4) : i32
        %0 = arith.index_cast %cl_0 : i32 to index
        %1 = arith.index_cast %cl_1 : i32 to index
        %2 = arith.index_cast %cl_2 : i32 to index
        %3 = arith.index_cast %cl_3 : i32 to index
        %4 = arith.index_cast %cl_4 : i32 to index
        %5 = flow.dispatch.workload.ordinal %0, 0 : index
        %6 = flow.dispatch.workload.ordinal %1, 1 : index
        %7 = flow.dispatch.workload.ordinal %2, 2 : index
        %8 = flow.dispatch.workload.ordinal %3, 3 : index
        %9 = flow.dispatch.workload.ordinal %4, 4 : index
        %10 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : memref<?x?x?xf32>{%5, %6, %7}
        memref.assume_alignment %10, 64 : memref<?x?x?xf32>
        %11 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : memref<1x?x?xf32>{%8, %9}
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

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @tile_multiuse_producer {
  hal.executable.variant public @embedded_elf_x86_64 target(<"llvm-cpu", "embedded-elf_x86_64", {}>) {
    hal.executable.export public @tile_multiuse_producer ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @tile_multiuse_producer() attributes {translation_info = #iree_codegen.translation_info<CPUDoubleTilingExpert>} {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 1.000000e+00 : f32
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
            : !flow.dispatch.tensor<readonly:tensor<12x128x128xf32>>
        %s0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
            : !flow.dispatch.tensor<writeonly:tensor<12x128x128xf32>>
        %s1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
            : !flow.dispatch.tensor<writeonly:tensor<12x128xf32>>
        %s2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0)
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
            %11 = arith.maximumf %arg0, %arg1 : f32
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
//   CHECK-DAG:     %[[SRC_BINDING:.+]] = hal.interface.binding.subspan layout(#pipeline_layout) binding(0)
//   CHECK-DAG:     %[[RESULT_BINDING0:.+]] = hal.interface.binding.subspan layout(#pipeline_layout) binding(1)
//   CHECK-DAG:     %[[RESULT_BINDING1:.+]] = hal.interface.binding.subspan layout(#pipeline_layout) binding(2)
//   CHECK-DAG:     %[[RESULT_BINDING2:.+]] = hal.interface.binding.subspan layout(#pipeline_layout) binding(3)
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

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @no_tile {
  hal.executable.variant public @embedded_elf_x86_64 target(<"llvm-cpu", "embedded-elf-x86_64", {}>) {
    hal.executable.export public @no_tile ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @no_tile() attributes {translation_info = #iree_codegen.translation_info<CPUDefault>} {
        %c0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<10xf32>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<10xi32>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<3xf32>>
        %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c64) : !flow.dispatch.tensor<readwrite:tensor<3xi32>>
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

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @pack_lowering {
  hal.executable.variant public @embedded_elf_x86_64 target(<"llvm-cpu", "embedded-elf-x86_64", {}>) {
    hal.executable.export public @gemm_lhs_pack ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @gemm_lhs_pack() attributes {translation_info = #iree_codegen.translation_info<CPUDataTiling>} {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
            : !flow.dispatch.tensor<readonly:tensor<100x250xf32>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
            : !flow.dispatch.tensor<writeonly:tensor<14x64x8x4xf32>>
        %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [100, 250], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<100x250xf32>> -> tensor<100x250xf32>
        %3 = tensor.empty() : tensor<14x64x8x4xf32>
        %4 = tensor.pack %2 padding_value(%cst : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 4] into %3
            {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[12, 12]]>}
            : tensor<100x250xf32> -> tensor<14x64x8x4xf32>
        flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0, 0], sizes = [14, 64, 8, 4], strides = [1, 1, 1, 1]
            : tensor<14x64x8x4xf32> -> !flow.dispatch.tensor<writeonly:tensor<14x64x8x4xf32>>
        return
      }
    }
  }
}
//      CHECK: hal.executable.export public @gemm_lhs_pack
// CHECK-NEXT:   %[[ARG0:.+]]: !hal.device
//  CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//  CHECK-DAG:   %[[C6:.+]] = arith.constant 6 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//      CHECK:   hal.return %[[C6]], %[[C2]], %[[C1]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @pack_lowering {
  hal.executable.variant public @embedded_elf_x86_64 target(<"llvm-cpu", "embedded-elf-x86_64", {}>) {
    hal.executable.export public @gemm_rhs_transpose_pack ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @gemm_rhs_transpose_pack() attributes {translation_info = #iree_codegen.translation_info<CPUDataTiling>} {
        %c0 = arith.constant 0 : index
        %c114688 = arith.constant 114688 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
            : !flow.dispatch.tensor<readonly:tensor<250x500xf32>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c114688)
            : !flow.dispatch.tensor<writeonly:tensor<64x64x8x4xf32>>
        %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [250, 500], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<250x500xf32>> -> tensor<250x500xf32>
        %3 = tensor.empty() : tensor<64x64x8x4xf32>
        %4 = tensor.pack %2 padding_value(%cst : f32) outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [8, 4] into %3
            {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[12, 14]]>}
            : tensor<250x500xf32> -> tensor<64x64x8x4xf32>
        flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0, 0], sizes = [64, 64, 8, 4], strides = [1, 1, 1, 1]
            : tensor<64x64x8x4xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x64x8x4xf32>>
        return
      }
    }
  }
}
//      CHECK: hal.executable.export public @gemm_rhs_transpose_pack
//  CHECK-DAG:   %[[C6:.+]] = arith.constant 6 : index
//  CHECK-DAG:   %[[C5:.+]] = arith.constant 5 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//      CHECK:   hal.return %[[C5]], %[[C6]], %[[C1]]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 4, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @clone_index_computations {
  hal.executable.variant public @embedded_elf_x86_64 target(<"llvm-cpu", "embedded-elf-x86_64", {}>) {
    hal.executable.export public @clone_index_computations ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3 : index, %arg4 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1, %arg2, %arg3, %arg4
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @clone_index_computations() attributes {translation_info = #iree_codegen.translation_info<CPUDataTiling>} {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %cl_0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
        %cl_1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
        %cl_2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : i32
        %cl_3 = hal.interface.constant.load layout(#pipeline_layout) ordinal(3) : i32
        %0 = arith.index_castui %cl_0 : i32 to index
        %1 = arith.index_castui %cl_1 : i32 to index
        %2 = arith.index_castui %cl_2 : i32 to index
        %3 = arith.index_castui %cl_3 : i32 to index
        %4 = flow.dispatch.workload.ordinal %0, 0 : index
        %5 = flow.dispatch.workload.ordinal %1, 1 : index
        %6 = flow.dispatch.workload.ordinal %2, 2 : index
        %7 = flow.dispatch.workload.ordinal %3, 3 : index
        %8 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%4, %5}
        %9 = affine.apply affine_map<()[s0] -> (s0 ceildiv 8)>()[%6]
        %10 = affine.apply affine_map<()[s0] -> (s0 ceildiv 4)>()[%7]
        %11 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
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

#pipeline_layout = #hal.pipeline.layout<constants = 4, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @dynamic_unpack {
  hal.executable.variant public @embedded_elf_x86_64 target(<"llvm-cpu", "embedded-elf-x86_64", {}>) {
    hal.executable.export public @dynamic_unpack ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index, %arg4: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1, %arg2, %arg3, %arg4
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @dynamic_unpack() attributes {translation_info = #iree_codegen.translation_info<CPUDataTiling>} {
        %c131072 = arith.constant 131072 : index
        %c0 = arith.constant 0 : index
        %cl_0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
        %cl_1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
        %cl_2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : i32
        %cl_3 = hal.interface.constant.load layout(#pipeline_layout) ordinal(3) : i32
        %0 = arith.index_castui %cl_0 : i32 to index
        %1 = arith.index_castui %cl_1 : i32 to index
        %2 = arith.index_castui %cl_2 : i32 to index
        %3 = arith.index_castui %cl_3 : i32 to index
        %4 = flow.dispatch.workload.ordinal %0, 0 : index
        %5 = flow.dispatch.workload.ordinal %1, 1 : index
        %6 = flow.dispatch.workload.ordinal %2, 2 : index
        %7 = flow.dispatch.workload.ordinal %3, 3 : index
        %8 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<?x?x32x16xi32>>{%4, %5}
        %9 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c131072) : !flow.dispatch.tensor<writeonly:tensor<?x?xi32>>{%6, %7}
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

#pipeline_layout = #hal.pipeline.layout<constants = 6, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @dynamic_unpack_dynamic_tile {
  hal.executable.variant public @embedded_elf_x86_64 target(<"llvm-cpu", "embedded-elf-x86_64", {}>) {
    hal.executable.export public @dynamic_unpack_dynamic_tile ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index, %arg4: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1, %arg2, %arg3, %arg4
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @dynamic_unpack_dynamic_tile() attributes {translation_info = #iree_codegen.translation_info<CPUDataTiling>} {
        %c131072 = arith.constant 131072 : index
        %c0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c32 = arith.constant 32 : index
        %cl_0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
        %cl_1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
        %cl_2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : i32
        %cl_3 = hal.interface.constant.load layout(#pipeline_layout) ordinal(3) : i32
        %cl_4 = hal.interface.constant.load layout(#pipeline_layout) ordinal(4) : i32
        %cl_5 = hal.interface.constant.load layout(#pipeline_layout) ordinal(5) : i32
        %0 = arith.index_castui %cl_0 : i32 to index
        %1 = arith.index_castui %cl_1 : i32 to index
        %2 = arith.index_castui %cl_2 : i32 to index
        %3 = arith.index_castui %cl_3 : i32 to index
        %tile0 = arith.index_castui %cl_4 : i32 to index
        %tile1 = arith.index_castui %cl_5 : i32 to index
        %4 = flow.dispatch.workload.ordinal %0, 0 : index
        %5 = flow.dispatch.workload.ordinal %1, 1 : index
        %6 = flow.dispatch.workload.ordinal %2, 2 : index
        %7 = flow.dispatch.workload.ordinal %3, 3 : index
        %8 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xi32>>{%4, %5, %c32, %c16}
        %9 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c131072) : !flow.dispatch.tensor<writeonly:tensor<?x?xi32>>{%6, %7}
        %10 = flow.dispatch.tensor.load %8, offsets = [0, 0, 0, 0], sizes = [%4, %5, %c32, %c16], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xi32>>{%4, %5, %c32, %c16} -> tensor<?x?x?x?xi32>
        %11 = tensor.empty(%6, %7) : tensor<?x?xi32>
        %12 = tensor.unpack %10 inner_dims_pos = [0, 1] inner_tiles = [%tile0, %tile1] into %11
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

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @unpack_elem {
  hal.executable.variant public @embedded_elf_arm_64 target(<"llvm-cpu", "embedded-elf-arm_64", {}>) {
    hal.executable.export public @unpack_elem ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @unpack_elem() attributes {translation_info = #iree_codegen.translation_info<CPUDataTiling>} {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<16x48x8x8xf32>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<128x384xf32>>
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

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
hal.executable private @dynamic_unpack_fusion {
  hal.executable.variant public @vmvx_bytecode_fb target(<"vmvx", "vmvx-bytecode-fb", {ukernels = true}>) {
    hal.executable.export public @dynamic_unpack_fusion ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @dynamic_unpack_fusion() attributes {translation_info = #iree_codegen.translation_info<VMVXDefault>} {
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
        %0:2 = iree_codegen.query_tile_sizes tensor<12544x16xi32, #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>> -> index, index
        %1 = affine.apply affine_map<()[s0] -> (12544 ceildiv s0)>()[%0#0]
        %2 = affine.apply affine_map<()[s0] -> (16 ceildiv s0)>()[%0#1]
        %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c200960) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xi32>>{%1, %2, %0#0, %0#1}
        %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c1003776) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<12544xi32>>
        %5 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c1053952) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16xi32>>
        %6 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<12544x16xi32>>
        %10 = flow.dispatch.tensor.load %3, offsets = [0, 0, 0, 0], sizes = [%1, %2, %0#0, %0#1], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xi32>>{%1, %2, %0#0, %0#1} -> tensor<?x?x?x?xi32>
        %11 = flow.dispatch.tensor.load %4, offsets = [0], sizes = [12544], strides = [1] : !flow.dispatch.tensor<readonly:tensor<12544xi32>> -> tensor<12544xi32>
        %12 = flow.dispatch.tensor.load %5, offsets = [0], sizes = [16], strides = [1] : !flow.dispatch.tensor<readonly:tensor<16xi32>> -> tensor<16xi32>
        %13 = tensor.empty() : tensor<12544x16xi32>
        %14 = tensor.empty() : tensor<12544x16xi32>
        %16 = tensor.unpack %10 inner_dims_pos = [0, 1] inner_tiles = [%0#0, %0#1] into %14 {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[16, 16]]>} : tensor<?x?x?x?xi32> -> tensor<12544x16xi32>
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
// CHECK:           tensor.unpack
// CHECK:           tensor.extract_slice
// CHECK:           linalg.generic

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @elem_pack {
  hal.executable.variant public @embedded_elf_arm_64 target(<"llvm-cpu", "embedded-elf-arm_64", {cpu = "generic", cpu_features = "", data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>) {
    hal.executable.export public @elem_pack ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @elem_pack() attributes {translation_info = #iree_codegen.translation_info<CPUDataTiling>} {
        %c1339392 = arith.constant 1339392 : index
        %c0 = arith.constant 0 : index
        %c823296 = arith.constant 823296 : index
        %c825344 = arith.constant 825344 : index
        %c786432 = arith.constant 786432 : index
        %c1572864 = arith.constant 1572864 : index
        %c2359296 = arith.constant 2359296 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c1339392) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x2x512xf32>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c786432) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<384x512xf32>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<384x512xf32>>
        %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<384xi32>>
        %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c823296) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<512xf32>>
        %5 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c825344) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<512xf32>>
        %6 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<48x512x8x1xf32>>
        %7 = hal.interface.binding.subspan layout(#pipeline_layout) binding(4) alignment(64) offset(%c1572864) : !flow.dispatch.tensor<writeonly:tensor<384x512xf32>>
        %8 = hal.interface.binding.subspan layout(#pipeline_layout) binding(5) alignment(64) offset(%c2359296) : !flow.dispatch.tensor<writeonly:tensor<384x512xf32>>
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

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @scatter {
  hal.executable.variant public @cuda_nvptx_fb target(<"cuda", "cuda-nvptx-fb">) {
    hal.executable.export public @scatter ordinal(0)
    layout(#hal.pipeline.layout<bindings = [
      #hal.pipeline.binding<storage_buffer, ReadOnly>,
      #hal.pipeline.binding<storage_buffer>
    ]>)
    {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @scatter() attributes {translation_info = #iree_codegen.translation_info<LLVMGPUDistribute workgroup_size = [1, 1, 1]>} {
        %c228075520 = arith.constant 228075520 : index
        %c251668480 = arith.constant 251668480 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c228075520) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<5898240xf32>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c251668480) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<5898240x4xi32>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1x640x48x48xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [5898240], strides = [1] : !flow.dispatch.tensor<readonly:tensor<5898240xf32>> -> tensor<5898240xf32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [5898240, 4], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<5898240x4xi32>> -> tensor<5898240x4xi32>
        %5 = tensor.empty() : tensor<1x640x48x48xf32>
        %6 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[]]>} ins(%cst : f32) outs(%5 : tensor<1x640x48x48xf32>) -> tensor<1x640x48x48xf32>
        %7 = iree_linalg_ext.scatter {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[]]>} dimension_map = [0, 1, 2, 3] unique_indices(false) ins(%3, %4 : tensor<5898240xf32>, tensor<5898240x4xi32>) outs(%6 : tensor<1x640x48x48xf32>) {
        ^bb0(%arg0: f32, %arg1: f32):
          %8 = arith.addf %arg1, %arg0 : f32
          iree_linalg_ext.yield %8 : f32
        } -> tensor<1x640x48x48xf32>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [1, 640, 48, 48], strides = [1, 1, 1, 1] : tensor<1x640x48x48xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x640x48x48xf32>>
        return
      }
    }
  }
}
// CHECK-LABEL: func.func @scatter
//       CHECK:   iree_linalg_ext.scatter

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @collapse_workgroups_dispatch_dispatch_0 {
  hal.executable.variant public @cuda_nvptx_fb target(<"cuda", "cuda-nvptx-fb">) {
    hal.executable.export public @collapse_workgroups_dispatch_dispatch_0_generic_1024x128x16x64 ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @collapse_workgroups_dispatch_dispatch_0_generic_1024x128x16x64() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1024x16x128x64xf32>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1024x128x16x64xf32>>
        %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1024, 16, 128, 64], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1024x16x128x64xf32>> -> tensor<1024x16x128x64xf32>
        %3 = tensor.empty() : tensor<1024x128x16x64xf32>
        %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<1024x16x128x64xf32>) outs(%3 : tensor<1024x128x16x64xf32>) attrs = {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 1, 64]]>} {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<1024x128x16x64xf32>
        flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0, 0], sizes = [1024, 128, 16, 64], strides = [1, 1, 1, 1] : tensor<1024x128x16x64xf32> -> !flow.dispatch.tensor<writeonly:tensor<1024x128x16x64xf32>>
        return
      }
    }
  }
}

// CHECKW-LABEL:   hal.executable private @collapse_workgroups_dispatch_dispatch_0 {
//       CHECKW:           hal.executable.variant public @cuda_nvptx_fb
//       CHECKW:             hal.executable.export public @collapse_workgroups_dispatch_dispatch_0_generic_1024x128x16x64 ordinal(0) layout({{.+}}) {
//       CHECKW:             ^bb0(%[[ARG0:.*]]: !hal.device):
//   CHECKW-DAG:               %[[C2097152:.*]] = arith.constant 2097152 : index
//   CHECKW-DAG:               %[[C1:.*]] = arith.constant 1 : index
//       CHECKW:               hal.return %[[C2097152]], %[[C1]], %[[C1]] : index, index, index
//       CHECKW:             }

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[64, 64, 0], [16, 4, 0], [0, 0, 64]]>
#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "aarch64-none-elf"
}>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
hal.executable private @matmul_tensors {
  hal.executable.variant public @llvm target(#executable_target_embedded_elf_arm_64_) {
    hal.executable.export public @matmul_tensor_count_from_dag_root layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_tensor_count_from_dag_root() attributes {translation_info = #translation} {
        %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
        %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
        %2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
        %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %2}
        %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %1}
        %5 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1}
        %6 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3)
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
//      CHECK: hal.executable.export public @matmul_tensor_count_from_dag_root
// CHECK-NEXT:   (%[[DEVICE:.+]]: !hal.device,
// CHECK-SAME:    %[[WORKLOAD_M:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_N:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[WORKLOAD_K:[a-zA-Z0-9_]+]]: index)
//  CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:    %[[D0:.+]] = affine.apply #[[MAP0]]()[%[[WORKLOAD_M]]]
//  CHECK-DAG:    %[[D1:.+]] = affine.apply #[[MAP0]]()[%[[WORKLOAD_N]]]
//      CHECK:    hal.return %[[D1]], %[[D0]], %[[C1]] : index, index, index

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[64, 64, 0], [16, 4, 0], [0, 0, 64]]>
#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
#map = affine_map<()[s0] -> (s0 ceildiv 64)>
#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
module {
  hal.executable private @matmul_tensors {
    hal.executable.variant public @llvm target(#executable_target_embedded_elf_arm_64_) {
      hal.executable.export public @matmul_already_distributed layout(#pipeline_layout) {
      ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
        %c1 = arith.constant 1 : index
        %0 = affine.apply #map()[%arg1]
        %1 = affine.apply #map()[%arg2]
        hal.return %1, %0, %c1 : index, index, index
      }
      builtin.module {
        func.func @matmul_already_distributed() attributes {translation_info = #translation} {
          %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
          %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
          %2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
          %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %2}
          %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %1}
          %5 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1}
          %6 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
          %workgroup_id_x = hal.interface.workgroup.id[0] : index
          %workgroup_count_x = hal.interface.workgroup.count[0] : index
          %workgroup_id_y = hal.interface.workgroup.id[1] : index
          %workgroup_count_y = hal.interface.workgroup.count[1] : index
          %13 = flow.dispatch.tensor.load %3, offsets = [%workgroup_id_y, 0], sizes = [%0, %2], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %2} -> tensor<?x?xf32>
          %14 = flow.dispatch.tensor.load %4, offsets = [0, %workgroup_id_x], sizes = [%2, %1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %1} -> tensor<?x?xf32>
          %15 = flow.dispatch.tensor.load %5, offsets = [%workgroup_id_y, %workgroup_id_x], sizes = [%0, %1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1} -> tensor<?x?xf32>
          %16 = linalg.matmul {lowering_config = #config} ins(%13, %14 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
          flow.dispatch.tensor.store %16, %6, offsets = [%workgroup_id_y, %workgroup_id_x], sizes = [%0, %1], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
          return
        }
      }
    }
  }
}

// CHECK-LABEL: func.func @matmul_already_distributed
// CHECK:         %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
// CHECK:         %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
// CHECK:         %[[OUT_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(3)
// CHECK-NOT:     scf.for
// CHECK:         %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]], offsets = [%workgroup_id_y, 0]
// CHECK:         %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]], offsets = [0, %workgroup_id_x]
// CHECK:         %[[MATMUL:.+]] = linalg.matmul {{.*}} ins(%[[LHS]], %[[RHS]]
// CHECK-DAG:     flow.dispatch.tensor.store %[[MATMUL]], %[[OUT_BINDING]]

// -----

// Check that the distribution avoids distributing unit-trip count loops.

#pipeline_layout = #hal.pipeline.layout<constants = 6, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @avoid_unit_range_distribute {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @avoid_unit_range_distribute ordinal(0) layout(#pipeline_layout) attributes {subgroup_size = 32 : index, translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute>, workgroup_size = [32 : index, 1 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @avoid_unit_range_distribute() {
        %c0 = arith.constant 0 : index
        %c32_i64 = arith.constant 32 : i64
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
        %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
        %2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : i32
        %3 = hal.interface.constant.load layout(#pipeline_layout) ordinal(3) : i32
        %4 = hal.interface.constant.load layout(#pipeline_layout) ordinal(4) : i32
        %5 = hal.interface.constant.load layout(#pipeline_layout) ordinal(5) : i32
        %6 = arith.extui %0 : i32 to i64
        %7 = arith.extui %1 : i32 to i64
        %8 = arith.shli %7, %c32_i64 : i64
        %9 = arith.ori %6, %8 : i64
        %10 = arith.index_castui %9 : i64 to index
        %11 = arith.extui %2 : i32 to i64
        %12 = arith.extui %3 : i32 to i64
        %13 = arith.shli %12, %c32_i64 : i64
        %14 = arith.ori %11, %13 : i64
        %15 = arith.index_castui %14 : i64 to index
        %16 = arith.extui %4 : i32 to i64
        %17 = arith.extui %5 : i32 to i64
        %18 = arith.shli %17, %c32_i64 : i64
        %19 = arith.ori %16, %18 : i64
        %20 = arith.index_castui %19 : i64 to index
        %21 = flow.dispatch.workload.ordinal %15, 0 : index
        %22 = flow.dispatch.workload.ordinal %20, 1 : index
        %23 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x?x?x16x16xf16>>{%21, %22}
        %24 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%10) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x?x8x16x16xf16>>{%22}
        %25 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<32x?x16x8x16xf16>>{%22}
        %26 = flow.dispatch.tensor.load %23, offsets = [0, 0, 0, 0, 0], sizes = [32, %21, %22, 16, 16], strides = [1, 1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x?x?x16x16xf16>>{%21, %22} -> tensor<32x?x?x16x16xf16>
        %27 = flow.dispatch.tensor.load %24, offsets = [0, 0, 0, 0, 0], sizes = [32, %22, 8, 16, 16], strides = [1, 1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x?x8x16x16xf16>>{%22} -> tensor<32x?x8x16x16xf16>
        %28 = tensor.empty(%22) : tensor<32x?x16x8x16xf16>
        %29 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 16, 1, 16, 1, 16]]>} ins(%cst : f16) outs(%28 : tensor<32x?x16x8x16xf16>) -> tensor<32x?x16x8x16xf16>
        %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d5, d2, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d5, d3, d6, d4)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%26, %27 : tensor<32x?x?x16x16xf16>, tensor<32x?x8x16x16xf16>) outs(%29 : tensor<32x?x16x8x16xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 16, 1, 16, 1, 16]]>} {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
          %31 = arith.mulf %in, %in_0 : f16
          %32 = arith.addf %out, %31 : f16
          linalg.yield %32 : f16
        } -> tensor<32x?x16x8x16xf16>
        flow.dispatch.tensor.store %30, %25, offsets = [0, 0, 0, 0, 0], sizes = [32, %22, 16, 8, 16], strides = [1, 1, 1, 1, 1] : tensor<32x?x16x8x16xf16> -> !flow.dispatch.tensor<writeonly:tensor<32x?x16x8x16xf16>>{%22}
        return
      }
    }
  }
}
// CHECK-LABEL: func.func @avoid_unit_range_distribute()
//   CHECK-DAG:   %[[WGID_X:.+]] = hal.interface.workgroup.id[0]
//   CHECK-DAG:   %[[WGCOUNT_X:.+]] = hal.interface.workgroup.count[0]
//   CHECK-DAG:   %[[WGID_Y:.+]] = hal.interface.workgroup.id[1]
//   CHECK-DAG:   %[[WGCOUNT_Y:.+]] = hal.interface.workgroup.count[1]
//   CHECK-DAG:   %[[WGID_Z:.+]] = hal.interface.workgroup.id[2]
//   CHECK-DAG:   %[[WGCOUNT_Z:.+]] = hal.interface.workgroup.count[2]
//       CHECK:   scf.for %{{.+}} = %[[WGID_Z]] to %{{.+}} step %[[WGCOUNT_Z]]
//       CHECK:     scf.for %{{.+}} = %[[WGID_Y]] to %{{.+}} step %[[WGCOUNT_Y]]
//       CHECK:       scf.for %{{.+}} = %[[WGID_X]] to %{{.+}} step %[[WGCOUNT_X]]
//       CHECK:         tensor.empty() : tensor<1x1x16x1x16xf16>


// -----

// Check that the distribution avoids distributing unit-trip count loops.

#pipeline_layout = #hal.pipeline.layout<constants = 6, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @set_size_to_tilesize_when_divisible {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @set_size_to_tilesize_when_divisible ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @set_size_to_tilesize_when_divisible() attributes {translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute workgroup_size = [128, 1, 1] subgroup_size = 32, {mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mma_layout<WMMA_F32_16x16x16_F16>, subgroup_m_count = 1, subgroup_n_count = 4>}>} {
        %c0 = arith.constant 0 : index
        %c32_i64 = arith.constant 32 : i64
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
        %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
        %2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : i32
        %3 = hal.interface.constant.load layout(#pipeline_layout) ordinal(3) : i32
        %4 = hal.interface.constant.load layout(#pipeline_layout) ordinal(4) : i32
        %5 = hal.interface.constant.load layout(#pipeline_layout) ordinal(5) : i32
        %6 = arith.extui %0 : i32 to i64
        %7 = arith.extui %1 : i32 to i64
        %8 = arith.shli %7, %c32_i64 : i64
        %9 = arith.ori %6, %8 : i64
        %10 = arith.index_castui %9 : i64 to index
        %11 = arith.extui %2 : i32 to i64
        %12 = arith.extui %3 : i32 to i64
        %13 = arith.shli %12, %c32_i64 : i64
        %14 = arith.ori %11, %13 : i64
        %15 = arith.index_castui %14 : i64 to index
        %16 = arith.extui %4 : i32 to i64
        %17 = arith.extui %5 : i32 to i64
        %18 = arith.shli %17, %c32_i64 : i64
        %19 = arith.ori %16, %18 : i64
        %20 = arith.index_castui %19 : i64 to index
        %21 = flow.dispatch.workload.ordinal %20, 1 : index
        %22 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4096x32x128xf16>>
        %23 = flow.dispatch.workload.ordinal %21, 2 : index
        %24 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<?x16x32x128xf16>>{%21}
        %25 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%10) : !flow.dispatch.tensor<writeonly:tensor<?x16x4096xf16>>{%23}
        %26 = flow.dispatch.workload.ordinal %15, 0 : index
        %27 = flow.dispatch.tensor.load %24, offsets = [0, 0, 0, 0], sizes = [%21, 16, 32, 128], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x16x32x128xf16>>{%21} -> tensor<?x16x32x128xf16>
        %28 = flow.dispatch.tensor.load %22, offsets = [0, 0, 0], sizes = [4096, 32, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x32x128xf16>> -> tensor<4096x32x128xf16>
        %29 = affine.apply affine_map<()[s0] -> (s0 ceildiv 16)>()[%26]
        %30 = tensor.empty(%29) : tensor<?x16x4096xf16>
        %31 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 16, 128, 1, 128]]>} ins(%cst : f16) outs(%30 : tensor<?x16x4096xf16>) -> tensor<?x16x4096xf16>
        %32 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%27, %28 : tensor<?x16x32x128xf16>, tensor<4096x32x128xf16>) outs(%31 : tensor<?x16x4096xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 16, 128, 1, 128]]>} {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
          %33 = arith.mulf %in, %in_0 : f16
          %34 = arith.addf %33, %out : f16
          linalg.yield %34 : f16
        } -> tensor<?x16x4096xf16>
        flow.dispatch.tensor.store %32, %25, offsets = [0, 0, 0], sizes = [%23, 16, 4096], strides = [1, 1, 1] : tensor<?x16x4096xf16> -> !flow.dispatch.tensor<writeonly:tensor<?x16x4096xf16>>{%23}
        return
      }
    }
  }
}

//  NO-LOOP-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 * 128)>
//      NO-LOOP: func.func @set_size_to_tilesize_when_divisible()
//  NO-LOOP-DAG:   %[[IDX_X:.+]] = hal.interface.workgroup.id[0] : index
//  NO-LOOP-DAG:   %[[IDX_Y:.+]] = hal.interface.workgroup.id[1] : index
//  NO-LOOP-DAG:   %[[OFFX:.+]] = affine.apply #[[MAP1]]()[%[[IDX_X]]]
//  NO-LOOP-NOT:   scf.for
//      NO-LOOP:   %[[RESULT:.+]] = linalg.generic
//      NO-LOOP:   -> tensor<1x16x128xf16>
//      NO-LOOP:   flow.dispatch.tensor.store %[[RESULT]], %{{.+}}, offsets = [%[[IDX_Y]], 0, %[[OFFX]]]

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[32, 16, 0], [16, 8, 0], [0, 0, 2]]>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64">
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
hal.executable private @reshape_matmul_tensors {
  hal.executable.variant public @system_elf_x86_64 target(#executable_target_system_elf_x86_64_) {
    hal.executable.export public @reshape_matmul layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @reshape_matmul() attributes {translation_info = #translation} {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0)
            : !flow.dispatch.tensor<readonly:tensor<64x2x256xf32>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1)
            : !flow.dispatch.tensor<readonly:tensor<256x512xf32>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2)
            : !flow.dispatch.tensor<writeonly:tensor<128x512xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [64, 2, 256], strides = [1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<64x2x256xf32>> -> tensor<64x2x256xf32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 512], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<256x512xf32>> -> tensor<256x512xf32>
        %collapsed = tensor.collapse_shape %3 [[0, 1], [2]] : tensor<64x2x256xf32> into tensor<128x256xf32>
        %5 = tensor.empty() : tensor<128x512xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<128x512xf32>) -> tensor<128x512xf32>
        %7 = linalg.matmul {lowering_config = #config}
            ins(%collapsed, %4 : tensor<128x256xf32>, tensor<256x512xf32>) outs(%6 : tensor<128x512xf32>) -> tensor<128x512xf32>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [128, 512], strides = [1, 1]
            : tensor<128x512xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x512xf32>>
        return
      }
    }
  }
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 * 32)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 * 16)>
//      CHECK: hal.executable.export public @reshape_matmul
// CHECK-NEXT:   (%[[DEVICE:.+]]: !hal.device)
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
//  CHECK-DAG:   %[[C32:.+]] = arith.constant 32 : index
//      CHECK:   hal.return %[[C32]], %[[C4]], %[[C1]]
//      CHECK: func.func @reshape_matmul()
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
