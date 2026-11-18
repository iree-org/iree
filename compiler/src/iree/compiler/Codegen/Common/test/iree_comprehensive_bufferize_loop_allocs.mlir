// RUN: iree-opt %s --split-input-file --pass-pipeline="builtin.module(func.func(iree-eliminate-empty-tensors{allow-return-allocs-from-loops=true},empty-tensor-to-alloc-tensor,iree-codegen-iree-comprehensive-bufferize{allow-return-allocs-from-loops=true}))" | FileCheck %s
// RUN: iree-opt %s --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-iree-comprehensive-bufferize{allow-return-allocs-from-loops=true}))" | FileCheck %s
// RUN: iree-opt %s --split-input-file --pass-pipeline="builtin.module(func.func(iree-eliminate-empty-tensors{allow-return-allocs-from-loops=true},empty-tensor-to-alloc-tensor,iree-codegen-iree-comprehensive-bufferize{allow-return-allocs-from-loops=true},ownership-based-buffer-deallocation,buffer-deallocation-simplification,bufferization-lower-deallocations))" | FileCheck %s --check-prefix=OWNED

// Opt-in permits fresh loop results. Ownership remains the caller's policy.
// CHECK-LABEL: func.func @for_yields_allocation(
// CHECK: %[[LOOP:.*]] = scf.for {{.*}} iter_args(%[[ITER:.*]] = {{.*}}) -> (memref<4xi32>) {
// CHECK:   %[[ALLOC:.*]] = memref.alloc() : memref<4xi32>
// CHECK:   linalg.generic {{.*}} ins(%[[ITER]] : memref<4xi32>) outs(%[[ALLOC]] : memref<4xi32>)
// CHECK:   scf.yield %[[ALLOC]] : memref<4xi32>
// CHECK: return %[[LOOP]] : memref<4xi32>
// OWNED-LABEL: func.func @for_yields_allocation(
// OWNED: %[[FALSE:.*]] = arith.constant false
// OWNED: %[[LOOP:.*]]:2 = scf.for {{.*}} iter_args(%[[ITER:.*]] = {{.*}}, %[[OWNS:.*]] = %[[FALSE]])
// OWNED:   %[[ALLOC:.*]] = memref.alloc()
// OWNED:   %[[BASE:[^, ]+]], {{.*}} = memref.extract_strided_metadata %[[ITER]]
// OWNED:   scf.if %[[OWNS]] {
// OWNED:     memref.dealloc %[[BASE]]
// OWNED:   scf.yield %[[ALLOC]], {{.*}} : memref<4xi32>, i1
// OWNED: %[[RESULT:.*]] = scf.if %[[LOOP]]#1
// OWNED:   scf.yield %[[LOOP]]#0
// OWNED: } else {
// OWNED:   bufferization.clone %[[LOOP]]#0
// OWNED: return %[[RESULT]] : memref<4xi32>
func.func @for_yields_allocation(%input: memref<4xi32>, %count: index) -> memref<4xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %tensor = bufferization.to_tensor %input restrict writable : memref<4xi32> to tensor<4xi32>
  %result = scf.for %i = %c0 to %count step %c1 iter_args(%iter = %tensor) -> tensor<4xi32> {
    %copy = bufferization.alloc_tensor() copy(%iter) : tensor<4xi32>
    scf.yield %copy : tensor<4xi32>
  }
  %buffer = bufferization.to_buffer %result : tensor<4xi32> to memref<4xi32>
  return %buffer : memref<4xi32>
}

// -----

// CHECK-LABEL: func.func @while_yields_allocation(
// CHECK: %[[LOOP:.*]] = scf.while
// CHECK:   %[[BEFORE:.*]] = memref.alloc()
// CHECK:   scf.condition({{.*}}) %[[BEFORE]] : memref<4xi32>
// CHECK: } do {
// CHECK:   %[[AFTER:.*]] = memref.alloc()
// CHECK:   scf.yield %[[AFTER]] : memref<4xi32>
// CHECK: return %[[LOOP]] : memref<4xi32>
// OWNED-LABEL: func.func @while_yields_allocation(
// OWNED: %[[FALSE:.*]] = arith.constant false
// OWNED: %[[LOOP:.*]]:2 = scf.while (%[[ITER:[^ ]+]] = {{.*}}, %[[OWNS:[^ ]+]] = %[[FALSE]])
// OWNED:   %[[BEFORE:.*]] = memref.alloc()
// OWNED:   %[[BASE:[^, ]+]], {{.*}} = memref.extract_strided_metadata %[[ITER]]
// OWNED:   scf.if %[[OWNS]] {
// OWNED:     memref.dealloc %[[BASE]]
// OWNED:   scf.condition({{.*}}) %[[BEFORE]], {{.*}} : memref<4xi32>, i1
// OWNED: } do {
// OWNED: ^bb0(%[[ITER2:[^:]+]]: memref<4xi32>, %[[OWNS2:[^:]+]]: i1):
// OWNED:   %[[AFTER:.*]] = memref.alloc()
// OWNED:   %[[BASE2:[^, ]+]], {{.*}} = memref.extract_strided_metadata %[[ITER2]]
// OWNED:   scf.if %[[OWNS2]] {
// OWNED:     memref.dealloc %[[BASE2]]
// OWNED:   scf.yield %[[AFTER]], {{.*}} : memref<4xi32>, i1
// OWNED: %[[RESULT:.*]] = scf.if %[[LOOP]]#1
// OWNED:   scf.yield %[[LOOP]]#0
// OWNED: } else {
// OWNED:   bufferization.clone %[[LOOP]]#0
// OWNED: return %[[RESULT]] : memref<4xi32>
func.func @while_yields_allocation(%input: memref<4xi32>, %condition: i1) -> memref<4xi32> {
  %tensor = bufferization.to_tensor %input restrict writable : memref<4xi32> to tensor<4xi32>
  %result = scf.while (%iter = %tensor) : (tensor<4xi32>) -> tensor<4xi32> {
    scf.condition(%condition) %iter : tensor<4xi32>
  } do {
  ^bb0(%iter: tensor<4xi32>):
    %copy = bufferization.alloc_tensor() copy(%iter) : tensor<4xi32>
    scf.yield %copy : tensor<4xi32>
  }
  %buffer = bufferization.to_buffer %result : tensor<4xi32> to memref<4xi32>
  return %buffer : memref<4xi32>
}

// -----

// CHECK-LABEL: func.func @for_preserves_alias(
// CHECK: %[[LOOP:.*]] = scf.for {{.*}} iter_args(%[[ITER:.*]] = {{.*}})
// CHECK-NOT: memref.alloc
// CHECK: scf.yield %[[ITER]] : memref<4xi32>
// CHECK: return %[[LOOP]] : memref<4xi32>
// OWNED-LABEL: func.func @for_preserves_alias(
// OWNED: bufferization.clone
// OWNED: return
func.func @for_preserves_alias(%input: memref<4xi32>, %count: index) -> memref<4xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %tensor = bufferization.to_tensor %input restrict writable : memref<4xi32> to tensor<4xi32>
  %result = scf.for %i = %c0 to %count step %c1 iter_args(%iter = %tensor) -> tensor<4xi32> {
    scf.yield %iter : tensor<4xi32>
  }
  %buffer = bufferization.to_buffer %result : tensor<4xi32> to memref<4xi32>
  return %buffer : memref<4xi32>
}

// -----

// CHECK-LABEL: func.func @copy_without_loop(
// CHECK: %[[ALLOC:.*]] = memref.alloc()
// CHECK: linalg.generic {{.*}} outs(%[[ALLOC]] : memref<4xi32>)
// CHECK: return %[[ALLOC]] : memref<4xi32>
// OWNED-LABEL: func.func @copy_without_loop(
// OWNED: %[[ALLOC:.*]] = memref.alloc()
// OWNED-NOT: memref.dealloc
// OWNED: return %[[ALLOC]] : memref<4xi32>
func.func @copy_without_loop(%input: memref<4xi32>) -> memref<4xi32> {
  %tensor = bufferization.to_tensor %input restrict writable : memref<4xi32> to tensor<4xi32>
  %copy = bufferization.alloc_tensor() copy(%tensor) : tensor<4xi32>
  %buffer = bufferization.to_buffer %copy : tensor<4xi32> to memref<4xi32>
  return %buffer : memref<4xi32>
}
