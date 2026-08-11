// RUN: iree-opt --iree-codegen-llvmcpu-configuration-pipeline --iree-codegen-llvmcpu-lowering-pipeline='enable-native-bf16-converts=true' --split-input-file %s | FileCheck %s --check-prefixes=COMMON,NATIVE
// RUN: iree-opt --iree-codegen-llvmcpu-configuration-pipeline --iree-codegen-llvmcpu-lowering-pipeline --split-input-file %s | FileCheck %s --check-prefixes=COMMON,EMULATED

// Verifies the enable-native-bf16-converts pipeline option. The target backend
// derives this option from the target's cpu features (Zfbfmin + Zvfbfmin on
// RISC-V). Arithmetic is promoted to f32 either way, there is no non-widening
// bf16 arithmetic (the only bf16-input arithmetic instruction, Zvfbfwma's
// `vfwmaccbf16`, widens to f32). What the option controls is the conversions
// around the promotion. When it's set, bf16 storage is kept and the promotion's
// extf/truncf survive to the LLVM dialect as fpext/fptrunc, which select as
// native conversion instructions, otherwise bf16 storage becomes i16 and the
// conversions are expanded into shift/round-bias integer sequences.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

#executable_target_embedded_elf_riscv_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {cpu_features = "+m,+a,+f,+d,+c,+v,+zfbfmin,+zvfbfmin", data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", native_vector_size = 16 : index, target_triple = "riscv64-unknown-unknown-eabi-elf"}>
builtin.module {
  func.func @bf16_add() attributes {hal.executable.target = #executable_target_embedded_elf_riscv_64_} {
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024xbf16>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024xbf16>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024xbf16>>
    %lhs = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0], sizes = [1024], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024xbf16>> -> tensor<1024xbf16>
    %rhs = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0], sizes = [1024], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024xbf16>> -> tensor<1024xbf16>
    %init = tensor.empty() : tensor<1024xbf16>
    %add = linalg.add ins(%lhs, %rhs : tensor<1024xbf16>, tensor<1024xbf16>) outs(%init : tensor<1024xbf16>) -> tensor<1024xbf16>
    iree_tensor_ext.dispatch.tensor.store %add, %2, offsets = [0], sizes = [1024], strides = [1] : tensor<1024xbf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024xbf16>>
    return
  }
}
// Native bf16 add: bf16 loads, fpext/fptrunc around the add, then bf16 store.
// Emulated: the add is performed in f32 on values reconstructed from i16
// storage, and the result is rounded back with the shift/bias sequence.
// COMMON-LABEL: llvm.func @bf16_add
//      NATIVE:    %[[LHS:.+]] = llvm.load {{.*}} -> bf16
//      NATIVE:    %[[LHSF:.+]] = llvm.fpext %[[LHS]] : bf16 to f32
//      NATIVE:    %[[RHS:.+]] = llvm.load {{.*}} -> bf16
//      NATIVE:    %[[RHSF:.+]] = llvm.fpext %[[RHS]] : bf16 to f32
//    EMULATED:    %[[LHS:.+]] = llvm.load {{.*}} -> i16
//    EMULATED:    %[[LHSZ:.+]] = llvm.zext %[[LHS]] : i16 to i32
//    EMULATED:    %[[LHSS:.+]] = llvm.shl %[[LHSZ]], {{.*}} : i32
//    EMULATED:    %[[LHSF:.+]] = llvm.bitcast %[[LHSS]] : i32 to f32
//    EMULATED:    %[[RHS:.+]] = llvm.load {{.*}} -> i16
//    EMULATED:    %[[RHSZ:.+]] = llvm.zext %[[RHS]] : i16 to i32
//    EMULATED:    %[[RHSS:.+]] = llvm.shl %[[RHSZ]], {{.*}} : i32
//    EMULATED:    %[[RHSF:.+]] = llvm.bitcast %[[RHSS]] : i32 to f32
//      COMMON:    %[[SUM:.+]] = llvm.fadd %[[LHSF]], %[[RHSF]] {{.*}} : f32
//      NATIVE:    %[[RES:.+]] = llvm.fptrunc %[[SUM]] : f32 to bf16
//      NATIVE:    llvm.store %[[RES]], {{.*}} : bf16
//  NATIVE-NOT:    llvm.lshr
//  NATIVE-NOT:    llvm.shl
//    EMULATED:    llvm.fcmp "une" %[[SUM]], %[[SUM]]
//    EMULATED:    llvm.lshr
//    EMULATED:    llvm.store {{.*}} i16

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

#executable_target_embedded_elf_riscv_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {cpu_features = "+m,+a,+f,+d,+c,+v,+zfbfmin,+zvfbfmin", data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", native_vector_size = 16 : index, target_triple = "riscv64-unknown-unknown-eabi-elf"}>
builtin.module {
  func.func @bf16_truncf() attributes {hal.executable.target = #executable_target_embedded_elf_riscv_64_} {
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024xf32>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024xbf16>>
    %in = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0], sizes = [1024], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024xf32>> -> tensor<1024xf32>
    %init = tensor.empty() : tensor<1024xbf16>
    %trunc = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
        ins(%in : tensor<1024xf32>) outs(%init : tensor<1024xbf16>) {
      ^bb0(%a: f32, %out: bf16):
        %t = arith.truncf %a : f32 to bf16
        linalg.yield %t : bf16
    } -> tensor<1024xbf16>
    iree_tensor_ext.dispatch.tensor.store %trunc, %1, offsets = [0], sizes = [1024], strides = [1] : tensor<1024xbf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024xbf16>>
    return
  }
}
// Both paths load the f32 input the same way. Native: the cast stays a single
// fptrunc (selectable as one narrowing convert) and stores bf16. Emulated: the
// round-to-nearest-even shift/bias sequence, storing i16.
// COMMON-LABEL: llvm.func @bf16_truncf
//        COMMON:   %[[IN:.+]] = llvm.load {{.*}} -> vector<{{[0-9]+}}xf32>
//        NATIVE:   %[[RES:.+]] = llvm.fptrunc %[[IN]] : vector<{{[0-9]+}}xf32> to vector<{{[0-9]+}}xbf16>
//        NATIVE:   llvm.store %[[RES]], {{.*}} : vector<{{[0-9]+}}xbf16>
//    NATIVE-NOT:   llvm.lshr
//  EMULATED-NOT:   llvm.fptrunc {{.*}} to vector<{{[0-9]+}}xbf16>
//      EMULATED:   llvm.fcmp "une" %[[IN]], %[[IN]]
//      EMULATED:   llvm.lshr
//      EMULATED:   llvm.store {{.*}} vector<{{[0-9]+}}xi16>
