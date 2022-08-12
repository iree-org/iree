// RUN: iree-opt --iree-codegen-specialize-workgroup-distribution --pass-pipeline='hal.executable(hal.executable.variant(iree-codegen-tile-and-distribute-to-workgroups)), canonicalize, cse' --split-input-file %s | FileCheck %s

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
      %x, %y, %z = flow.dispatch.default_workgroup_count %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_tensors() {
        %cst = arith.constant 0.000000e+00 : f32
        %3 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:123x456xf32>
        %4 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:456x789xf32>
        %5 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:123x789xf32>
        %7 = flow.dispatch.tensor.load %3, offsets = [0, 0], sizes = [123, 456], strides = [1, 1] : !flow.dispatch.tensor<readonly:123x456xf32> -> tensor<123x456xf32>
        %8 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [456, 789], strides = [1, 1] : !flow.dispatch.tensor<readonly:456x789xf32> -> tensor<456x789xf32>
        %init = linalg.init_tensor [123, 789] : tensor<123x789xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<123x789xf32>) -> tensor<123x789xf32>
        %10 = linalg.matmul {lowering_config = #config} ins(%7, %8 : tensor<123x456xf32>, tensor<456x789xf32>) outs(%fill : tensor<123x789xf32>) -> tensor<123x789xf32>
        flow.dispatch.tensor.store %10, %5, offsets = [0, 0], sizes = [123, 789], strides = [1, 1] : tensor<123x789xf32> -> !flow.dispatch.tensor<writeonly:123x789xf32>
        return
      }
    }
  }
}

// CHECK: func.func @matmul_tensors()
// CHECK: %[[CMP0:.+]] = arith.cmpi eq, %{{.+}}, %c64 : index
// CHECK: %[[CMP1:.+]] = arith.cmpi eq, %{{.+}}, %c64 : index
// CHECK: %[[COND:.+]] = arith.andi %[[CMP0]], %[[CMP1]] : i1
// CHECK: scf.if %[[COND]] {
// CHECK:   scf.for
// CHECK:     scf.for 
// CHECK:       linalg.matmul
// CHECK-SAME:                ins(%{{.+}}, %{{.+}} : tensor<64x456xf32>, tensor<456x64xf32>) outs(%{{.+}} : tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK: } else {
// CHECK:   scf.for
// CHECK:     scf.for 
// CHECK:       linalg.matmul
// CHECK-SAME:                ins(%{{.+}}, %{{.+}} : tensor<?x456xf32>, tensor<456x?xf32>) outs(%{{.+}} : tensor<?x?xf32>) -> tensor<?x?xf32>

// -----

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
hal.executable private @add_tensors {
  hal.executable.variant public @llvm, target = #executable_target_embedded_elf_arm_64_ {
    hal.executable.export public @add_tensors layout(#pipeline_layout) attributes {translation_info = #translation} {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index):
      %x, %y, %z = flow.dispatch.default_workgroup_count %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @add_tensors() {
        %cst = arith.constant 0.000000e+00 : f32
        %3 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:123x789xf32>
        %4 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:123x789xf32>
        %5 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:123x789xf32>
        %7 = flow.dispatch.tensor.load %3, offsets = [0, 0], sizes = [123, 789], strides = [1, 1] : !flow.dispatch.tensor<readonly:123x789xf32> -> tensor<123x789xf32>
        %8 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [123, 789], strides = [1, 1] : !flow.dispatch.tensor<readonly:123x789xf32> -> tensor<123x789xf32>
        %init = linalg.init_tensor [123, 789] : tensor<123x789xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<123x789xf32>) -> tensor<123x789xf32>
        %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]}
            ins(%7, %8: tensor<123x789xf32>, tensor<123x789xf32>)
            outs(%fill : tensor<123x789xf32>)
            attrs = {lowering_config = #config} {
          ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
            %11 = arith.addf %arg0, %arg1 : f32
            linalg.yield %11 : f32
          } -> tensor<123x789xf32>
        flow.dispatch.tensor.store %10, %5, offsets = [0, 0], sizes = [123, 789], strides = [1, 1] : tensor<123x789xf32> -> !flow.dispatch.tensor<writeonly:123x789xf32>
        return
      }
    }
  }
}

// CHECK: func.func @add_tensors()
// CHECK: %[[CMP0:.+]] = arith.cmpi eq, %{{.+}}, %c64 : index
// CHECK: %[[CMP1:.+]] = arith.cmpi eq, %{{.+}}, %c64 : index
// CHECK: %[[COND:.+]] = arith.andi %[[CMP0]], %[[CMP1]] : i1
// CHECK: scf.if %[[COND]] {
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       linalg.generic
// CHECK-SAME:                ins(%{{.+}}, %{{.+}} : tensor<64x64xf32>, tensor<64x64xf32>) outs(%{{.+}} : tensor<64x64xf32>)
// CHECK: } else {
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       linalg.generic
// CHECK-SAME:                ins(%{{.+}}, %{{.+}} : tensor<?x?xf32>, tensor<?x?xf32>) outs(%{{.+}} : tensor<?x?xf32>)
