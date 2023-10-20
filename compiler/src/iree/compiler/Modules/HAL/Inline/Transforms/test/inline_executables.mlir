// RUN: iree-opt --split-input-file --iree-hal-inline-executables %s | FileCheck %s

// Tests that exported dispatch functions get placed into the module with
// wrapper functions that perform the dispatch and all dispatch sites are tagged
// with the wrapper function.

// CHECK-NOT: hal.executable
hal.executable private @ex {
  hal.executable.variant public @vmvx_ir target(<"vmvx-inline", "vmvx-ir">) {
    hal.executable.export public @dispatch_0 ordinal(0) layout(
         #hal.pipeline.layout<push_constants = 2,
                                sets = [
                                  <0, bindings = [
                                    <0, storage_buffer>,
                                    <1, storage_buffer>,
                                    <2, storage_buffer>
                                  ]>
                                ]>) {
    ^bb0(%arg0: !hal.device, %workload_x: index, %workload_y: index):
      %count_x = affine.apply affine_map<()[s0] -> (s0 ceildiv 4)>()[%workload_x]
      %count_y = affine.apply affine_map<()[s0] -> (s0 ceildiv 4)>()[%workload_y]
      %count_z = arith.constant 1 : index
      hal.return %count_x, %count_y, %count_z : index, index, index
    }
    builtin.module {
      util.global private @global_constant : !util.buffer
      util.initializer {
        %buffer_cst = util.buffer.constant : !util.buffer = dense<[1, 2, 3, 4, 5]> : tensor<5xi32>
        util.global.store %buffer_cst, @global_constant : !util.buffer
        util.initializer.return
      }
      func.func @dispatch_0(
          %local_memory: !util.buffer,
          %constants: !util.buffer,
          %bindings: !util.list<!util.buffer>,
          %workgroup_x: i32, %workgroup_y: i32, %workgroup_z: i32,
          %workgroup_size_x: i32, %workgroup_size_y: i32, %workgroup_size_z: i32,
          %workgroup_count_x: i32, %workgroup_count_y: i32, %workgroup_count_z: i32) {
        %c4 = arith.constant 4 : index

        // Unpack push constants:
        %constants_size = util.buffer.size %constants : !util.buffer
        %constant1_offset = arith.constant 4 : index
        %constant1_i32 = util.buffer.load %constants[%constant1_offset for %c4] : !util.buffer{%constants_size} -> i32
        %constant1_f32 = arith.sitofp %constant1_i32 : i32 to f32

        // Unpack buffer bindings:
        %c0 = arith.constant 0 : index
        %buffer0 = util.list.get %bindings[%c0] : !util.list<!util.buffer>
        %c1 = arith.constant 1 : index
        %buffer1 = util.list.get %bindings[%c1] : !util.list<!util.buffer>
        %c2 = arith.constant 2 : index
        %buffer2 = util.list.get %bindings[%c2] : !util.list<!util.buffer>
        %buffer0_size = util.buffer.size %buffer0 : !util.buffer
        %buffer1_size = util.buffer.size %buffer1 : !util.buffer
        %buffer2_size = util.buffer.size %buffer2 : !util.buffer

        // Test for global constants:
        %global_constant = util.global.load @global_constant : !util.buffer
        util.optimization_barrier %global_constant : !util.buffer

        %workgroup_x_idx = arith.index_cast %workgroup_x : i32 to index
        scf.for %i = %c0 to %workgroup_x_idx step %c1 {
          %idx = arith.muli %i, %c4 : index
          %lhs = util.buffer.load %buffer0[%idx for %c4] : !util.buffer{%buffer0_size} -> f32
          %rhs = util.buffer.load %buffer1[%idx for %c4] : !util.buffer{%buffer1_size} -> f32
          %mul = arith.mulf %lhs, %rhs : f32
          %scaled = arith.mulf %mul, %constant1_f32 : f32
          util.buffer.store %scaled, %buffer2[%idx for %c4] : f32 -> !util.buffer{%buffer2_size}
        }
        return
      }
    }
  }
}

// Ensures that we properly rename the globals we inline:
util.global private  @global_constant : i32

// CHECK: util.global private @global_constant_0 : !util.buffer
// CHECK: util.initializer
// CHECK:   %[[CONSTANT:.+]] = util.buffer.constant
// CHECK:   util.global.store %[[CONSTANT]], @global_constant

// Ensures that we properly rename the dispatch function we inline:
func.func private @dispatch_0()

// CHECK-LABEL: func private @dispatch_0_0
// CHECK-SAME: (%[[LOCAL_MEMORY:.+]]: !util.buffer, %[[CONSTANT0:.+]]: i32, %[[CONSTANT1:.+]]: i32,
// CHECK-SAME:  %[[BINDING0:.+]]: !util.buffer, %[[BINDING1:.+]]: !util.buffer, %[[BINDING2:.+]]: !util.buffer,
// CHECK-SAME:  %[[X:[a-z0-9]+]]: index, %[[Y:[a-z0-9]+]]: index, %[[Z:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[SIZE_XYZ:[a-z0-9]+]]: index, %[[SIZE_XYZ:[a-z0-9]+]]: index, %[[SIZE_XYZ:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[COUNT_X:[a-z0-9]+]]: index, %[[COUNT_Y:[a-z0-9]+]]: index, %[[COUNT_Z:[a-z0-9]+]]: index)

// Type conversion; most of these will fold away.
// CHECK: %[[X_I32:.+]] = arith.index_cast %[[X]]

// Push constant rewritten to use args:
// CHECK: %[[CONSTANT1_F32:.+]] = arith.sitofp %[[CONSTANT1]] : i32 to f32

// Bindings get changed to use args:
// CHECK: %[[BINDING0_SIZE:.+]] = util.buffer.size %[[BINDING0]]
// CHECK: %[[BINDING1_SIZE:.+]] = util.buffer.size %[[BINDING1]]
// CHECK: %[[BINDING2_SIZE:.+]] = util.buffer.size %[[BINDING2]]

// Globals get carried across:
// CHECK: %[[GLOBAL_CONSTANT:.+]] = util.global.load @global_constant_0 : !util.buffer
// CHECK: util.optimization_barrier %[[GLOBAL_CONSTANT]]

// CHECK: %[[X_IDX:.+]] = arith.index_cast %[[X_I32]]
// CHECK: scf.for %[[ELEMENT_INDEX:.+]] = %c0 to %[[X_IDX]]
// CHECK:   %[[ELEMENT_OFFSET:.+]] = arith.muli %[[ELEMENT_INDEX]]
// CHECK:   %[[LHS:.+]] = util.buffer.load %[[BINDING0]][%[[ELEMENT_OFFSET]] for {{.+}}] : !util.buffer{%[[BINDING0_SIZE]]} -> f32
// CHECK:   %[[RHS:.+]] = util.buffer.load %[[BINDING1]][%[[ELEMENT_OFFSET]] for {{.+}}] : !util.buffer{%[[BINDING1_SIZE]]} -> f32
// CHECK:   %[[MUL:.+]] = arith.mulf %[[LHS]], %[[RHS]] : f32
// CHECK:   %[[SCALED:.+]] = arith.mulf %[[MUL]], %[[CONSTANT1_F32]] : f32
// CHECK:   util.buffer.store %[[SCALED]], %[[BINDING2]][%[[ELEMENT_OFFSET]] for {{.+}}] : f32 -> !util.buffer{%[[BINDING2_SIZE]]}
// CHECK: return

// CHECK-LABEL: func private @__dispatch_ex_dispatch_0
// CHECK-SAME: (%[[WORKLOAD_X:.+]]: index, %[[WORKLOAD_Y:.+]]: index, %[[CONSTANT0:.+]]: i32, %[[CONSTANT1:.+]]: i32,
// CHECK-SAME:  %[[BINDING0:.+]]: !util.buffer, %[[BINDING1:.+]]: !util.buffer, %[[BINDING2:.+]]: !util.buffer,
// CHECK-SAME:  %[[OFFSET0:[a-z0-9]+]]: index, %[[OFFSET1:[a-z0-9]+]]: index, %[[OFFSET2:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[LENGTH0:.+]]: index, %[[LENGTH1:.+]]: index, %[[LENGTH2:.+]]: index)

// Inlined workgroup count calculation from the export op:
// CHECK:   %[[COUNT_X:.+]] = affine.apply {{.+}}[%[[WORKLOAD_X]]]
// CHECK:   %[[COUNT_Y:.+]] = affine.apply {{.+}}[%[[WORKLOAD_Y]]]
// CHECK:   %[[COUNT_Z:.+]] = arith.constant 1

// Local workgroup memory not currently used:
// CHECK:   %[[LOCAL_MEMORY:.+]] = util.null : !util.buffer

// Binding subspans as specified on the dispatch:
// CHECK:   %[[BINDING0_SIZE:.+]] = util.buffer.size %[[BINDING0]]
// CHECK:   %[[BINDING0_SPAN:.+]] = util.buffer.subspan %[[BINDING0]][%[[OFFSET0]]] : !util.buffer{%[[BINDING0_SIZE]]} -> !util.buffer{%[[LENGTH0]]}
// CHECK:   %[[BINDING1_SIZE:.+]] = util.buffer.size %[[BINDING1]]
// CHECK:   %[[BINDING1_SPAN:.+]] = util.buffer.subspan %[[BINDING1]][%[[OFFSET1]]] : !util.buffer{%[[BINDING1_SIZE]]} -> !util.buffer{%[[LENGTH1]]}
// CHECK:   %[[BINDING2_SIZE:.+]] = util.buffer.size %[[BINDING2]]
// CHECK:   %[[BINDING2_SPAN:.+]] = util.buffer.subspan %[[BINDING2]][%[[OFFSET2]]] : !util.buffer{%[[BINDING2_SIZE]]} -> !util.buffer{%[[LENGTH2]]}

// Workgroup XYZ loop:
// CHECK:   %[[SIZE_XYZ:.+]] = arith.constant 1
// CHECK:   scf.for %[[Z:.+]] = %c0 to %[[COUNT_Z]]
// CHECK:     scf.for %[[Y:.+]] = %c0 to %[[COUNT_Y]]
// CHECK:       scf.for %[[X:.+]] = %c0 to %[[COUNT_X]]
// CHECK:          func.call @dispatch_0_0(
// CHECK-SAME:         %[[LOCAL_MEMORY]],
// CHECK-SAME:         %[[CONSTANT0]], %[[CONSTANT1]],
// CHECK-SAME:         %[[BINDING0_SPAN]], %[[BINDING1_SPAN]], %[[BINDING2_SPAN]],
// CHECK-SAME:         %[[X]], %[[Y]], %[[Z]],
// CHECK-SAME:         %[[SIZE_XYZ]], %[[SIZE_XYZ]], %[[SIZE_XYZ]],
// CHECK-SAME:         %[[COUNT_X]], %[[COUNT_Y]], %[[COUNT_Z]])
// CHECK:   return

// CHECK-LABEL: @dispatch0
// CHECK-SAME: (%[[RESOURCE0:.+]]: !stream.resource<constant>,
// CHECK-SAME:  %[[RESOURCE1:.+]]: !stream.resource<transient>,
// CHECK-SAME:  %[[RESOURCE2:.+]]: !stream.resource<external>)
func.func private @dispatch0(%resource0: !stream.resource<constant>, %resource1: !stream.resource<transient>, %resource2: !stream.resource<external>) {
  %workload_x = arith.constant 1000 : index
  %workload_y = arith.constant 1001 : index
  %constant0 = arith.constant 4 : i32
  %constant1 = arith.constant 5 : i32
  %binding0_offset = arith.constant 200 : index
  %binding0_length = arith.constant 128 : index
  %binding1_offset = arith.constant 300 : index
  %binding1_length = arith.constant 256 : index
  %binding2_offset = arith.constant 400 : index
  %binding2_length = arith.constant 512 : index
  %resource_size = arith.constant 1024 : index
  %0 = stream.cmd.execute with(%resource0 as %resource0_inner: !stream.resource<constant>{%resource_size},
                               %resource1 as %resource1_inner: !stream.resource<transient>{%resource_size},
                               %resource2 as %resource2_inner: !stream.resource<external>{%resource_size}) {
    // CHECK: stream.cmd.dispatch
    // CHECK: hal_inline.target = @__dispatch_ex_dispatch_0
    stream.cmd.dispatch @ex::@vmvx_ir::@dispatch_0[%workload_x, %workload_y](%constant0, %constant1 : i32, i32) {
      ro %resource0_inner[%binding0_offset for %binding0_length] : !stream.resource<constant>{%resource_size},
      ro %resource1_inner[%binding1_offset for %binding1_length] : !stream.resource<transient>{%resource_size},
      wo %resource2_inner[%binding2_offset for %binding2_length] : !stream.resource<external>{%resource_size}
    } attributes {
      hal.interface.bindings = [
        #hal.interface.binding<0, 0>,
        #hal.interface.binding<0, 1>,
        #hal.interface.binding<0, 2>
      ]
    }
  } => !stream.timepoint
  return
}
