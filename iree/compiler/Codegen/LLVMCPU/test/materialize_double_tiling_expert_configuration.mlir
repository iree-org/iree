// RUN: iree-opt --iree-codegen-use-double-tiling-expert -pass-pipeline='hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target{test-lowering-configuration=true}))' -cse -canonicalize -split-input-file %s | FileCheck %s

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_x86  {
  hal.executable.variant public @embedded_elf_x86_64, target = #hal.executable.target<
    "llvm",
    "embedded-elf-x86_64", {
      data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
      native_vector_size = 16 : index,
      target_triple = "x86_64-unknown-unknown-eabi-elf"
    }> {
    hal.executable.entry_point public @matmul_x86 layout(#executable_layout)
    builtin.module {
      func @matmul_x86() {
        %c128 = arith.constant 128 : index
        %c384 = arith.constant 384 : index
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:384x512xf32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:512x128xf32>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:384x128xf32>
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %3 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_y, %workgroup_size_y]
        %4 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_y, %workgroup_size_y]
        scf.for %arg0 = %3 to %c384 step %4 {
          %5 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
          %6 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
          scf.for %arg1 = %5 to %c128 step %6 {
            %7 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 384)>(%arg0)[%workgroup_size_y]
            %8 = flow.dispatch.tensor.load %0, offsets = [%arg0, 0], sizes = [%7, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:384x512xf32> -> tensor<?x512xf32>
            %9 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 128)>(%arg1)[%workgroup_size_x]
            %10 = flow.dispatch.tensor.load %1, offsets = [0, %arg1], sizes = [512, %9], strides = [1, 1] : !flow.dispatch.tensor<readonly:512x128xf32> -> tensor<512x?xf32>
            %11 = affine.min affine_map<(d0)[s0] -> (-d0 + 384, s0)>(%arg0)[%workgroup_size_y]
            %12 = affine.min affine_map<(d0)[s0] -> (-d0 + 128, s0)>(%arg1)[%workgroup_size_x]
            %13 = linalg.init_tensor [%11, %12] : tensor<?x?xf32>
            %14 = linalg.fill(%cst, %13) : f32, tensor<?x?xf32> -> tensor<?x?xf32>
            %15 = linalg.matmul ins(%8, %10 : tensor<?x512xf32>, tensor<512x?xf32>) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
            flow.dispatch.tensor.store %15, %2, offsets = [%arg0, %arg1], sizes = [%7, %9], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:384x128xf32>
          }
        }
        return
      }
    }
  }
}

//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"CPUDoubleTilingExpert", workload_per_wg = [64, 64]>
//  CHECK-DAG: #[[CONFIG:.+]] =  #iree_codegen.lowering.config<tile_sizes = [{{\[}}], [288, 128, 512], [9, 32, 16]], native_vector_size = [9, 32, 16]>
//  CHECK:       linalg.matmul {lowering.config = #[[CONFIG]]}
