// Checks cumulative-budget promotion, original consumer types, and scratch packing.
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmcpu-promote-large-allocas))" %s | FileCheck %s
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmcpu-promote-large-allocas,iree-llvmcpu-assign-workgroup-local-memory))" %s | FileCheck %s --check-prefix=PACK

#target = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  max_stack_allocation_size = 32768 : i64
}>

// CHECK-LABEL: func.func @large(
// CHECK: %[[LOCAL:.*]] = memref.alloc() : memref<32769xi8, #iree_codegen.workgroup_local>
// CHECK-NEXT: %[[CAST:.*]] = memref.memory_space_cast %[[LOCAL]] : memref<32769xi8, #iree_codegen.workgroup_local> to memref<32769xi8>
// CHECK: memref.load %[[CAST]]
// PACK-LABEL: func.func @large(
// PACK: memref.alloc() : memref<32769xi8, #iree_codegen.workgroup_local>
// PACK: memref.memory_space_cast
// PACK: iree_codegen.dispatch_config @large workgroup_local_memory = 32769
func.func @large(%i: index) -> i8 attributes {hal.executable.target = #target} {
  %0 = memref.alloca() alignment = 64 : memref<32769xi8>
  %1 = memref.load %0[%i] : memref<32769xi8>
  return %1 : i8
}
iree_codegen.dispatch_config @large {
  %c1 = arith.constant 1 : index
  iree_codegen.yield %c1, %c1, %c1 : index, index, index
}

// CHECK-LABEL: func.func @exact_budget(
// CHECK: memref.alloca() : memref<32768xi8>
// CHECK-NOT: memref.alloc()
func.func @exact_budget(%i: index) -> i8 attributes {hal.executable.target = #target} {
  %0 = memref.alloca() : memref<32768xi8>
  %1 = memref.load %0[%i] : memref<32768xi8>
  return %1 : i8
}
iree_codegen.dispatch_config @exact_budget {
  %c1 = arith.constant 1 : index
  iree_codegen.yield %c1, %c1, %c1 : index, index, index
}

// CHECK-LABEL: func.func @cumulative(
// CHECK: memref.alloc() : memref<20000xi8, #iree_codegen.workgroup_local>
// CHECK: memref.alloc() : memref<20000xi8, #iree_codegen.workgroup_local>
// CHECK: memref.alloca() : memref<20000xi8>
// PACK: iree_codegen.dispatch_config @cumulative workgroup_local_memory = 40000
func.func @cumulative() attributes {hal.executable.target = #target} {
  %0 = memref.alloca() : memref<20000xi8>
  %1 = memref.alloca() : memref<20000xi8>
  %2 = memref.alloca() : memref<20000xi8>
  return
}
iree_codegen.dispatch_config @cumulative {
  %c1 = arith.constant 1 : index
  iree_codegen.yield %c1, %c1, %c1 : index, index, index
}

// CHECK-LABEL: func.func @strided(
// CHECK: %[[LOCAL:.*]] = memref.alloc() : memref<20000xf32, strided<[2], offset: 1>, #iree_codegen.workgroup_local>
// CHECK: %[[CAST:.*]] = memref.memory_space_cast %[[LOCAL]]
// CHECK: memref.subview %[[CAST]]
// PACK: iree_codegen.dispatch_config @strided workgroup_local_memory = 160000
func.func @strided(%i: index) -> f32 attributes {hal.executable.target = #target} {
  %0 = memref.alloca() : memref<20000xf32, strided<[2], offset: 1>>
  %1 = memref.subview %0[0] [10] [1] : memref<20000xf32, strided<[2], offset: 1>> to memref<10xf32, strided<[2], offset: 1>>
  %2 = memref.load %1[%i] : memref<10xf32, strided<[2], offset: 1>>
  return %2 : f32
}
iree_codegen.dispatch_config @strided {
  %c1 = arith.constant 1 : index
  iree_codegen.yield %c1, %c1, %c1 : index, index, index
}

// Dynamic allocas should not be promoted
//
// CHECK-LABEL: func.func @dynamic(
// CHECK: memref.alloca(%{{.*}}) : memref<?xi8>
// CHECK-NOT: memref.alloc()
func.func @dynamic(%size: index) {
  %0 = memref.alloca(%size) : memref<?xi8>
  return
}
iree_codegen.dispatch_config @dynamic {
  %c1 = arith.constant 1 : index
  iree_codegen.yield %c1, %c1, %c1 : index, index, index
}

// Scalable vector-based allocas should not be promoted
//
// CHECK-LABEL: func.func @scalable(
// CHECK: memref.alloca(%{{.*}}) : memref<?x65536xi8>
// CHECK-NOT: memref.alloc()
func.func @scalable() {
  %size = vector.vscale
  %0 = memref.alloca(%size) : memref<?x65536xi8>
  return
}
iree_codegen.dispatch_config @scalable {
  %c1 = arith.constant 1 : index
  iree_codegen.yield %c1, %c1, %c1 : index, index, index
}
