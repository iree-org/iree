// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(func.func(iree-llvmgpu-group-global-loads))' %s | FileCheck %s

// Two global loads separated by a pure (arith.addf) op that doesn't depend on
// either load. The pass hoists the second load to sit immediately after the
// first load.
// CHECK-LABEL: func.func @group_two_loads
// CHECK:         %[[L0:.+]] = vector.load %{{.+}}[%{{.+}}]
// CHECK-NEXT:    %[[L1:.+]] = vector.load %{{.+}}[%{{.+}}]
// CHECK-NEXT:    arith.addf
// CHECK-NEXT:    return %[[L0]], %[[L1]]
func.func @group_two_loads(%a: memref<256xf32, #amdgpu.address_space<fat_raw_buffer>>,
                           %b: memref<256xf32, #amdgpu.address_space<fat_raw_buffer>>,
                           %x: f32, %y: f32) -> (vector<4xf32>, vector<4xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %v0 = vector.load %a[%c0] : memref<256xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf32>
  %sum = arith.addf %x, %y : f32
  %v1 = vector.load %b[%c4] : memref<256xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf32>
  return %v0, %v1 : vector<4xf32>, vector<4xf32>
}

// -----

// The second load's index is computed in the gap, but it only depends on
// values available before the first load. The pass hoists that address
// computation before the first load so the global loads can be adjacent.
// CHECK-LABEL: func.func @hoists_independent_index_before_load_group
// CHECK:         %[[OFF1:.+]] = arith.addi
// CHECK-NEXT:    %[[L0:.+]] = vector.load %{{.+}}[%{{.+}}]
// CHECK-NEXT:    %[[L1:.+]] = vector.load %{{.+}}[%[[OFF1]]]
// CHECK-NEXT:    return %[[L0]], %[[L1]]
func.func @hoists_independent_index_before_load_group(%a: memref<256xf32, #amdgpu.address_space<fat_raw_buffer>>,
                                                      %off: index,
                                                      %stride: index) -> (vector<4xf32>, vector<4xf32>) {
  %v0 = vector.load %a[%off] : memref<256xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf32>
  %off1 = arith.addi %off, %stride : index
  %v1 = vector.load %a[%off1] : memref<256xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf32>
  return %v0, %v1 : vector<4xf32>, vector<4xf32>
}

// -----

// The second load reads from the same fat raw buffer that an intervening
// vector.store writes to. Hoisting the load above the store would
// change observable memory, so the pass must leave the loads in place.
// CHECK-LABEL: func.func @blocked_by_buffer_write
// CHECK:         vector.load
// CHECK-NEXT:    vector.store
// CHECK-NEXT:    vector.load
func.func @blocked_by_buffer_write(%a: memref<256xf32, #amdgpu.address_space<fat_raw_buffer>>,
                                   %v: vector<4xf32>) -> (vector<4xf32>, vector<4xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %v0 = vector.load %a[%c0] : memref<256xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf32>
  vector.store %v, %a[%c4] : memref<256xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf32>
  %v1 = vector.load %a[%c4] : memref<256xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf32>
  return %v0, %v1 : vector<4xf32>, vector<4xf32>
}

// -----

// Plain global memref store between two global loads is a global-memory write,
// so hoisting must be blocked.
// CHECK-LABEL: func.func @blocked_by_global_write
// CHECK:         vector.load
// CHECK-NEXT:    vector.store
// CHECK-NEXT:    vector.load
func.func @blocked_by_global_write(%a: memref<256xf32, #amdgpu.address_space<fat_raw_buffer>>,
                                   %g: memref<256xf32>,
                                   %v: vector<4xf32>) -> (vector<4xf32>, vector<4xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %v0 = vector.load %a[%c0] : memref<256xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf32>
  vector.store %v, %g[%c4] : memref<256xf32>, vector<4xf32>
  %v1 = vector.load %a[%c4] : memref<256xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf32>
  return %v0, %v1 : vector<4xf32>, vector<4xf32>
}

// -----

// A write to workgroup (shared) memory between the loads is not on the global
// address space, so hoisting is allowed.
// CHECK-LABEL: func.func @allowed_through_shared_write
// CHECK:         %[[L0:.+]] = vector.load %{{.+}}[%{{.+}}]
// CHECK-NEXT:    %[[L1:.+]] = vector.load %{{.+}}[%{{.+}}]
// CHECK-NEXT:    vector.store
// CHECK-NEXT:    return %[[L0]], %[[L1]]
func.func @allowed_through_shared_write(%a: memref<256xf32, #amdgpu.address_space<fat_raw_buffer>>,
                                        %s: memref<256xf32, #gpu.address_space<workgroup>>,
                                        %v: vector<4xf32>) -> (vector<4xf32>, vector<4xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %v0 = vector.load %a[%c0] : memref<256xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf32>
  vector.store %v, %s[%c4] : memref<256xf32, #gpu.address_space<workgroup>>, vector<4xf32>
  %v1 = vector.load %a[%c4] : memref<256xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf32>
  return %v0, %v1 : vector<4xf32>, vector<4xf32>
}

// -----

// The second load's index is computed in the gap from a value the first load
// produces. The pure address-computation ops (vector.extract, arith.index_cast,
// arith.muli) are transitive deps that must be hoisted along with the second
// load.
// CHECK-LABEL: func.func @hoists_index_dependencies
// CHECK:         %[[L0:.+]] = vector.load %{{.+}}[%{{.+}}]
// CHECK-NEXT:    vector.extract
// CHECK-NEXT:    arith.index_cast
// CHECK-NEXT:    arith.muli
// CHECK-NEXT:    %[[L1:.+]] = vector.load
// CHECK-NEXT:    return %[[L0]], %[[L1]]
func.func @hoists_index_dependencies(%a: memref<256xf32, #amdgpu.address_space<fat_raw_buffer>>,
                                     %b: memref<256xi32, #amdgpu.address_space<fat_raw_buffer>>)
    -> (vector<1xi32>, vector<4xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %v0 = vector.load %b[%c0] : memref<256xi32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xi32>
  %scalar = vector.extract %v0[0] : i32 from vector<1xi32>
  %idx = arith.index_cast %scalar : i32 to index
  %off = arith.muli %idx, %c4 : index
  %v1 = vector.load %a[%off] : memref<256xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf32>
  return %v0, %v1 : vector<1xi32>, vector<4xf32>
}

// -----

// vector.load from a plain memref is also a global load, so the pass groups
// these loads just like fat raw buffer loads.
// CHECK-LABEL: func.func @groups_plain_global_loads
// CHECK:         %[[L0:.+]] = vector.load %{{.+}}[%{{.+}}]
// CHECK-NEXT:    %[[L1:.+]] = vector.load %{{.+}}[%{{.+}}]
// CHECK-NEXT:    arith.addf
// CHECK-NEXT:    return %[[L0]], %[[L1]]
func.func @groups_plain_global_loads(%a: memref<256xf32>,
                                     %b: memref<256xf32>,
                                     %x: f32, %y: f32) -> (vector<4xf32>, vector<4xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %v0 = vector.load %a[%c0] : memref<256xf32>, vector<4xf32>
  %sum = arith.addf %x, %y : f32
  %v1 = vector.load %b[%c4] : memref<256xf32>, vector<4xf32>
  return %v0, %v1 : vector<4xf32>, vector<4xf32>
}

// -----

// memref.load is also a global load, so the pass groups it with other global
// loads.
// CHECK-LABEL: func.func @groups_memref_loads
// CHECK:         %[[L0:.+]] = memref.load %{{.+}}[%{{.+}}]
// CHECK-NEXT:    %[[L1:.+]] = memref.load %{{.+}}[%{{.+}}]
// CHECK-NEXT:    arith.addf
// CHECK-NEXT:    return %[[L0]], %[[L1]]
func.func @groups_memref_loads(%a: memref<256xf32>,
                               %b: memref<256xf32>,
                               %x: f32, %y: f32) -> (f32, f32) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %v0 = memref.load %a[%c0] : memref<256xf32>
  %sum = arith.addf %x, %y : f32
  %v1 = memref.load %b[%c4] : memref<256xf32>
  return %v0, %v1 : f32, f32
}

// -----

// Two adjacent global loads (no ops between them) — the pass should be a
// no-op and not perturb the IR.
// CHECK-LABEL: func.func @already_adjacent
// CHECK:         %[[L0:.+]] = vector.load %{{.+}}[%{{.+}}]
// CHECK-NEXT:    %[[L1:.+]] = vector.load %{{.+}}[%{{.+}}]
// CHECK-NEXT:    return %[[L0]], %[[L1]]
func.func @already_adjacent(%a: memref<256xf32, #amdgpu.address_space<fat_raw_buffer>>,
                            %b: memref<256xf32, #amdgpu.address_space<fat_raw_buffer>>)
    -> (vector<4xf32>, vector<4xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %v0 = vector.load %a[%c0] : memref<256xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf32>
  %v1 = vector.load %b[%c4] : memref<256xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf32>
  return %v0, %v1 : vector<4xf32>, vector<4xf32>
}

// -----

// Three global loads with pure ops scattered between them. All three should
// end up grouped at the position of the first load.
// CHECK-LABEL: func.func @group_three_loads
// CHECK:         %[[L0:.+]] = vector.load %{{.+}}[%{{.+}}]
// CHECK-NEXT:    %[[L1:.+]] = vector.load %{{.+}}[%{{.+}}]
// CHECK-NEXT:    %[[L2:.+]] = vector.load %{{.+}}[%{{.+}}]
// CHECK-NEXT:    arith.addf
// CHECK-NEXT:    arith.mulf
// CHECK-NEXT:    return %[[L0]], %[[L1]], %[[L2]]
func.func @group_three_loads(%a: memref<256xf32, #amdgpu.address_space<fat_raw_buffer>>,
                             %b: memref<256xf32, #amdgpu.address_space<fat_raw_buffer>>,
                             %c: memref<256xf32, #amdgpu.address_space<fat_raw_buffer>>,
                             %x: f32, %y: f32)
    -> (vector<4xf32>, vector<4xf32>, vector<4xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %v0 = vector.load %a[%c0] : memref<256xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf32>
  %sum = arith.addf %x, %y : f32
  %v1 = vector.load %b[%c4] : memref<256xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf32>
  %prod = arith.mulf %sum, %y : f32
  %v2 = vector.load %c[%c8] : memref<256xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf32>
  return %v0, %v1, %v2 : vector<4xf32>, vector<4xf32>, vector<4xf32>
}
