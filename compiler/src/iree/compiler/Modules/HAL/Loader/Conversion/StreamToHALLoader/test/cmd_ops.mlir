// RUN: iree-opt --split-input-file --iree-hal-loader-conversion --canonicalize %s | FileCheck %s

// NOTE: all other stream.cmd.* ops are handled by the hal_inline conversions.

// Executables are required to translate the dispatch calls.
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<4, storage_buffer>
  ]>,
  #hal.descriptor_set.layout<1, bindings = [
    #hal.descriptor_set.binding<5, storage_buffer>
  ]>
]>
hal.executable private @ex {
  hal.executable.variant public @variant target(#hal.executable.target<"llvm", "embedded-elf-x86_64">) {
    hal.executable.export public @dispatch ordinal(16) layout(#pipeline_layout) {
    ^bb0(%device: !hal.device, %workload_x: index, %workload_y: index):
      %count_x = affine.apply affine_map<()[s0] -> (s0 ceildiv 4)>()[%workload_x]
      %count_y = affine.apply affine_map<()[s0] -> (s0 ceildiv 4)>()[%workload_y]
      %count_z = arith.constant 1 : index
      hal.return %count_x, %count_y, %count_z : index, index, index
    }
    builtin.module {
      // Opaque at this point (in some target-specific dialects).
    }
  }
}

// NOTE: %buffer0 is transient and will map to a !util.buffer, while
//       %buffer1 is external and will map to a !hal.buffer.

// CHECK-LABEL: @cmdDispatch
// CHECK-SAME: (%[[BUFFER0:.+]]: !util.buffer, %[[BUFFER0_SIZE:.+]]: index,
// CHECK-SAME:  %[[BUFFER1:.+]]: !hal.buffer, %[[BUFFER1_SIZE:.+]]: index)
util.func public @cmdDispatch(%buffer0: !stream.resource<transient>, %buffer0_size: index,
                       %buffer1: !stream.resource<external>, %buffer1_size: index) -> !stream.timepoint {
  // (ends up by the dispatch below)
  %workload_x = arith.constant 1000 : index
  %workload_y = arith.constant 1001 : index

  // CHECK-DAG: %[[CONSTANT0:.+]] = arith.constant 4
  %constant0 = arith.constant 4 : i32
  // CHECK-DAG: %[[CONSTANT1:.+]] = arith.constant 5
  %constant1 = arith.constant 5 : i32

  // CHECK-DAG: %[[BUFFER0_REL_OFFSET:.+]] = arith.constant 200
  %buffer0_offset = arith.constant 200 : index
  // CHECK-DAG: %[[BUFFER0_REL_LENGTH:.+]] = arith.constant 128
  %buffer0_length = arith.constant 128 : index
  // CHECK-DAG: %[[BUFFER1_REL_OFFSET:.+]] = arith.constant 300
  %buffer1_offset = arith.constant 300 : index
  // CHECK-DAG: %[[BUFFER1_REL_LENGTH:.+]] = arith.constant 256
  %buffer1_length = arith.constant 256 : index

  %fence = stream.cmd.execute with(%buffer0 as %buffer0_inner: !stream.resource<transient>{%buffer0_size},
                                   %buffer1 as %buffer1_inner: !stream.resource<external>{%buffer1_size}) {
    // Lookup the loaded executable (resolved during iree-hal-loader-materialize-executables):
    // CHECK-DAG: %[[EXECUTABLE:.+]] = hal_loader.executable.lookup executable(@ex) : !hal.executable

    // %buffer1 is external and a subspan is needed to resolve the absolute
    // storage range. This will (mostly) eventually fold/canonicalize away.
    // CHECK-DAG: %[[BUFFER1_STORAGE:.+]] = hal_inline.buffer.storage<%[[BUFFER1]]

    // Workload calculation gets inlined and folds during conversion; this is
    // the original worload ceildiv 4 on x/y:
    // CHECK-DAG: %[[COUNT_X:.+]] = arith.constant 250
    // CHECK-DAG: %[[COUNT_Y:.+]] = arith.constant 251
    // CHECK-DAG: %[[COUNT_Z:.+]] = arith.constant 1

    // CHECK-DAG: %[[ORDINAL:.+]] = hal_loader.executable.export.ordinal target(@ex::@variant::@dispatch) : index

    //      CHECK: hal_loader.executable.dispatch
    // CHECK-SAME:   executable(%[[EXECUTABLE]] : !hal.executable)[%[[ORDINAL]]]
    // CHECK-SAME:   workgroups([%[[COUNT_X]], %[[COUNT_Y]], %[[COUNT_Z]]])
    // CHECK-SAME:   constants([%[[CONSTANT0]], %[[CONSTANT1]]])
    // CHECK-SAME:   bindings([
    // CHECK-NEXT:     (%[[BUFFER0]] : !util.buffer)[%[[BUFFER0_REL_OFFSET]], %[[BUFFER0_REL_LENGTH]]],
    // CHECK-NEXT:     (%[[BUFFER1_STORAGE]] : !util.buffer)[%[[BUFFER1_REL_OFFSET]], %[[BUFFER1_REL_LENGTH]]]
    // CHECK-NEXT:   ])
    stream.cmd.dispatch @ex::@dispatch[%workload_x, %workload_y](%constant0, %constant1 : i32, i32) {
      ro %buffer0_inner[%buffer0_offset for %buffer0_length] : !stream.resource<transient>{%buffer0_size},
      wo %buffer1_inner[%buffer1_offset for %buffer1_length] : !stream.resource<external>{%buffer1_size}
    } attributes {
      hal.interface.bindings = [
        #hal.interface.binding<0, 4>,
        #hal.interface.binding<1, 5>
      ]
    }
  } => !stream.timepoint
  // CHECK: return %c0
  util.return %fence : !stream.timepoint
}
