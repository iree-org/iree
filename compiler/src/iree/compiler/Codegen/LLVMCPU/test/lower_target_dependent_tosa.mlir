// RUN: iree-opt --pass-pipeline='hal.executable(hal.executable.variant(builtin.module(func.func(iree-llvmcpu-lower-target-dependent-tosa))))' %s | FileCheck %s

// Test LLVMCPULowerTargetDependentTosa pass.

#device_target_llvm_cpu = #hal.device.target<"llvm-cpu", {executable_targets = [#hal.executable.target<"llvm-cpu", "system-elf-riscv_64", {cpu_features = "+m,+a,+f,+d,+v", data_layout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128", native_vector_size = 64 : index, target_triple = "riscv64"}>], legacy_sync}>
#executable_target_system_elf_riscv_64_ = #hal.executable.target<"llvm-cpu", "system-elf-riscv_64", {cpu_features = "+m,+a,+f,+d,+v", data_layout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128", native_vector_size = 64 : index, target_triple = "riscv64"}>
#map0 = affine_map<()[s0] -> (s0 ceildiv 4)>
#map1 = affine_map<()[s0] -> (s0 ceildiv 64)>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>
#translation = #iree_codegen.translation_info<CPUDoubleTilingPeelingExpert workload_per_wg = [64, 64, 4]>
module attributes {hal.device.targets = [#device_target_llvm_cpu]} {
  hal.executable private @apply_scale_double_round {
    hal.executable.variant public @system_elf_riscv_64, target = #executable_target_system_elf_riscv_64_ {
      hal.executable.export public @apply_scale_double_round ordinal(0) layout(#pipeline_layout) attributes {translation_info = #translation} {
      ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index, %arg4: index):
        %0 = affine.apply #map0()[%arg1]
        %1 = affine.apply #map1()[%arg2]
        %2 = affine.apply #map1()[%arg4]
        hal.return %2, %1, %0 : index, index, index
      }
      builtin.module {
        func.func @apply_scale_double_round(%arg0: vector<8xi32>) -> vector<8xi32> {
          %cst = arith.constant dense<1559761830> : vector<8xi32>
          %cst_0 = arith.constant dense<50> : vector<8xi8>
          %0 = "tosa.apply_scale"(%arg0, %cst, %cst_0) {double_round = true} : (vector<8xi32>, vector<8xi32>, vector<8xi8>) -> vector<8xi32>
          return %0 : vector<8xi32>
        }
      }
    }
  }
}

// CHECK-LABEL: @apply_scale_double_round
//   CHECK-NOT: tosa.apply_scale

