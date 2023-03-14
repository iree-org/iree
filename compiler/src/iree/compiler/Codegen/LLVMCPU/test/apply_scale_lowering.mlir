// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-convert-to-llvm))))' %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
#executable_target_embedded_elf_riscv_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {
  cpu_features = "+m,+a,+f,+d,+c",
  data_layout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128",
  native_vector_size = 512 : index,
  target_triple = "riscv64-unknown-unknown-eabi-elf"
}>
#map = affine_map<()[s0] -> (s0 ceildiv 2)>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
hal.executable private @apply_scale_no_vector_feature {
  hal.executable.variant public @embedded_elf_riscv_64, target = #executable_target_embedded_elf_riscv_64_ {
    hal.executable.export public @apply_scale_no_vector_feature ordinal(0) layout(#pipeline_layout) attributes {translation_info = #translation} {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
      %c1 = arith.constant 1 : index
      %0 = affine.apply #map()[%arg1]
      hal.return %0, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @apply_scale_no_vector_feature() {
        %cst = arith.constant dense<19689> : vector<2xi32>
        %cst_0 = arith.constant dense<15> : vector<2xi8>
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<2xi32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<2xi32>
        %2 = vector.load %0[%c0] : memref<2xi32>, vector<2xi32>
        %3 = "tosa.apply_scale"(%2, %cst, %cst_0) {double_round = false} : (vector<2xi32>, vector<2xi32>, vector<2xi8>) -> vector<2xi32>
        vector.store %3, %1[%c0] : memref<2xi32>, vector<2xi32>
        return
      }
    }
  }
}

// 64-bit lowering is used by default if no vector features are provided.
// TODO(diegocaballero): We shouldn't vectorize the code if no vector features
// are provided.
// CHECK-LABEL: llvm.func @apply_scale_no_vector_feature
//       CHECK:   %[[ADD:.*]] = llvm.add %{{.*}}, %{{.*}} : vector<2xi64>
//  CHECK-NEXT:   %[[SHR:.*]] = llvm.ashr %[[ADD]], %{{.*}} : vector<2xi64>
//  CHECK-NEXT:   llvm.trunc %[[SHR]] : vector<2xi64> to vector<2xi32>

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
#executable_target_embedded_elf_riscv_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {
  cpu_features = "+m,+a,+f,+d,+c,+zvl512b,+v",
  data_layout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128",
  native_vector_size = 512 : index,
  target_triple = "riscv64-unknown-unknown-eabi-elf"
}>
#map = affine_map<()[s0] -> (s0 ceildiv 2)>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
hal.executable private @apply_scale_v {
  hal.executable.variant public @embedded_elf_riscv_64, target = #executable_target_embedded_elf_riscv_64_ {
    hal.executable.export public @apply_scale_v ordinal(0) layout(#pipeline_layout) attributes {translation_info = #translation} {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
      %c1 = arith.constant 1 : index
      %0 = affine.apply #map()[%arg1]
      hal.return %0, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @apply_scale_v() {
        %cst = arith.constant dense<19689> : vector<2xi32>
        %cst_0 = arith.constant dense<15> : vector<2xi8>
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<2xi32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<2xi32>
        %2 = vector.load %0[%c0] : memref<2xi32>, vector<2xi32>
        %3 = "tosa.apply_scale"(%2, %cst, %cst_0) {double_round = false} : (vector<2xi32>, vector<2xi32>, vector<2xi8>) -> vector<2xi32>
        vector.store %3, %1[%c0] : memref<2xi32>, vector<2xi32>
        return
      }
    }
  }
}

// 64-bit lowering is used with '+v'.
// CHECK-LABEL: llvm.func @apply_scale_v
//       CHECK:   %[[ADD:.*]] = llvm.add %{{.*}}, %{{.*}} : vector<2xi64>
//  CHECK-NEXT:   %[[SHR:.*]] = llvm.ashr %[[ADD]], %{{.*}} : vector<2xi64>
//  CHECK-NEXT:   llvm.trunc %[[SHR]] : vector<2xi64> to vector<2xi32>

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
#executable_target_embedded_elf_riscv_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {
  cpu_features = "+m,+a,+f,+d,+c,+zvl512b,+zve64x",
  data_layout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128",
  native_vector_size = 512 : index,
  target_triple = "riscv64-unknown-unknown-eabi-elf"
}>
#map = affine_map<()[s0] -> (s0 ceildiv 2)>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
hal.executable private @apply_scale_zve64x {
  hal.executable.variant public @embedded_elf_riscv_64, target = #executable_target_embedded_elf_riscv_64_ {
    hal.executable.export public @apply_scale_zve64x ordinal(0) layout(#pipeline_layout) attributes {translation_info = #translation} {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
      %c1 = arith.constant 1 : index
      %0 = affine.apply #map()[%arg1]
      hal.return %0, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @apply_scale_zve64x() {
        %cst = arith.constant dense<19689> : vector<2xi32>
        %cst_0 = arith.constant dense<15> : vector<2xi8>
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<2xi32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<2xi32>
        %2 = vector.load %0[%c0] : memref<2xi32>, vector<2xi32>
        %3 = "tosa.apply_scale"(%2, %cst, %cst_0) {double_round = false} : (vector<2xi32>, vector<2xi32>, vector<2xi8>) -> vector<2xi32>
        vector.store %3, %1[%c0] : memref<2xi32>, vector<2xi32>
        return
      }
    }
  }
}

// 64-bit lowering is used with '+zve64x'.
// CHECK-LABEL: llvm.func @apply_scale_zve64x
//       CHECK:   %[[ADD:.*]] = llvm.add %{{.*}}, %{{.*}} : vector<2xi64>
//  CHECK-NEXT:   %[[SHR:.*]] = llvm.ashr %[[ADD]], %{{.*}} : vector<2xi64>
//  CHECK-NEXT:   llvm.trunc %[[SHR]] : vector<2xi64> to vector<2xi32>

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
#executable_target_embedded_elf_riscv_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {
  cpu_features = "+m,+a,+f,+d,+c,+zvl512b,+zve32x",
  data_layout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128",
  native_vector_size = 512 : index,
  target_triple = "riscv64-unknown-unknown-eabi-elf"
}>
#map = affine_map<()[s0] -> (s0 ceildiv 2)>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
hal.executable private @apply_scale_zve32x {
  hal.executable.variant public @embedded_elf_riscv_64, target = #executable_target_embedded_elf_riscv_64_ {
    hal.executable.export public @apply_scale_zve32x ordinal(0) layout(#pipeline_layout) attributes {translation_info = #translation} {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
      %c1 = arith.constant 1 : index
      %0 = affine.apply #map()[%arg1]
      hal.return %0, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @apply_scale_zve32x() {
        %cst = arith.constant dense<19689> : vector<2xi32>
        %cst_0 = arith.constant dense<15> : vector<2xi8>
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<2xi32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<2xi32>
        %2 = vector.load %0[%c0] : memref<2xi32>, vector<2xi32>
        %3 = "tosa.apply_scale"(%2, %cst, %cst_0) {double_round = false} : (vector<2xi32>, vector<2xi32>, vector<2xi8>) -> vector<2xi32>
        vector.store %3, %1[%c0] : memref<2xi32>, vector<2xi32>
        return
      }
    }
  }
}

// 32-bit lowering is used with '+zve32x'. Note that the 32-bit lowering
// generates 64-bit mul operations that are decomposed into 32-bit operations by
// the LLVM backend. The backend expects both the low half to be an `llvm.mul` op.
// CHECK-LABEL: llvm.func @apply_scale_zve32x
//   CHECK-DAG:   %[[RHS:.+]]    = llvm.mlir.constant(dense<19689> : vector<2xi32>) : vector<2xi32>
//   CHECK-DAG:   %[[RHSEXT:.+]] = llvm.mlir.constant(dense<19689> : vector<2xi64>) : vector<2xi64>
//       CHECK:   %[[LHS:.+]]    = llvm.load %{{.+}} {alignment = 4 : i64} : !llvm.ptr<vector<2xi32>>
//       CHECK:   %[[LHSEXT:.+]] = llvm.sext %[[LHS]] : vector<2xi32> to vector<2xi64>
//       CHECK:   %[[MULEXT:.*]] = llvm.mul %[[LHSEXT]], %[[RHSEXT]] : vector<2xi64>
//       CHECK:   %[[MULLOW:.*]] = llvm.mul %[[LHS]], %[[RHS]] : vector<2xi32>
//       CHECK:   %[[SHR:.*]]    = llvm.lshr %[[MULEXT]], %{{.*}} : vector<2xi64>
//  CHECK-NEXT:   llvm.trunc %[[SHR]] : vector<2xi64> to vector<2xi32>

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
#executable_target_embedded_elf_riscv_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {
  cpu_features = "+m,+a,+f,+d,+c,+zvl512b,+zve32f",
  data_layout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128",
  native_vector_size = 512 : index,
  target_triple = "riscv64-unknown-unknown-eabi-elf"
}>
#map = affine_map<()[s0] -> (s0 ceildiv 2)>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
hal.executable private @apply_scale_zve32f {
  hal.executable.variant public @embedded_elf_riscv_64, target = #executable_target_embedded_elf_riscv_64_ {
    hal.executable.export public @apply_scale_zve32f ordinal(0) layout(#pipeline_layout) attributes {translation_info = #translation} {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
      %c1 = arith.constant 1 : index
      %0 = affine.apply #map()[%arg1]
      hal.return %0, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @apply_scale_zve32f() {
        %cst = arith.constant dense<19689> : vector<2xi32>
        %cst_0 = arith.constant dense<15> : vector<2xi8>
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<2xi32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<2xi32>
        %2 = vector.load %0[%c0] : memref<2xi32>, vector<2xi32>
        %3 = "tosa.apply_scale"(%2, %cst, %cst_0) {double_round = false} : (vector<2xi32>, vector<2xi32>, vector<2xi8>) -> vector<2xi32>
        vector.store %3, %1[%c0] : memref<2xi32>, vector<2xi32>
        return
      }
    }
  }
}

// 32-bit lowering is used with '+zve32f'. Note that the 32-bit lowering
// generates 64-bit mul operations that are decomposed into 32-bit operations by
// the LLVM backend. The backend expects both the low half to be an `llvm.mul` op.
// CHECK-LABEL: llvm.func @apply_scale_zve32f
//   CHECK-DAG:   %[[RHS:.+]]    = llvm.mlir.constant(dense<19689> : vector<2xi32>) : vector<2xi32>
//   CHECK-DAG:   %[[RHSEXT:.+]] = llvm.mlir.constant(dense<19689> : vector<2xi64>) : vector<2xi64>
//       CHECK:   %[[LHS:.+]]    = llvm.load %{{.+}} {alignment = 4 : i64} : !llvm.ptr<vector<2xi32>>
//       CHECK:   %[[LHSEXT:.+]] = llvm.sext %[[LHS]] : vector<2xi32> to vector<2xi64>
//       CHECK:   %[[MULEXT:.*]] = llvm.mul %[[LHSEXT]], %[[RHSEXT]] : vector<2xi64>
//       CHECK:   %[[MULLOW:.*]] = llvm.mul %[[LHS]], %[[RHS]] : vector<2xi32>
//       CHECK:   %[[SHR:.*]]    = llvm.lshr %[[MULEXT]], %{{.*}} : vector<2xi64>
//  CHECK-NEXT:   llvm.trunc %[[SHR]] : vector<2xi64> to vector<2xi32>
