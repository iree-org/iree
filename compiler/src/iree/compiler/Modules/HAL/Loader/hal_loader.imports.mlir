// IREE Inline Hardware Abstraction Layer (HAL) loader module imports.
// This is only used to dynamically load and dispatch executable libraries.
//
// This is embedded in the compiler binary and inserted into any module
// containing inline HAL loader dialect ops (hal_loader.*) that is lowered to
// the VM dialect.
vm.module @hal_loader {

//===----------------------------------------------------------------------===//
// iree_hal_executable_t
//===----------------------------------------------------------------------===//

// Queries whether the given executable format is supported.
vm.import private @executable.query_support(
  %executable_format : !vm.buffer
) -> i32
attributes {nosideeffects}

// Creates and dynamically links an executable library.
vm.import private @executable.load(
  %executable_format : !vm.buffer,
  %executable_data : !vm.buffer,
  %constants : !vm.buffer
) -> !vm.ref<!hal.executable>
attributes {nosideeffects}

// Dispatches a grid with the given densely-packed and 0-aligned push constants
// and bindings.
vm.import private @executable.dispatch(
  %executable : !vm.ref<!hal.executable>,
  %entry_point : i32,
  %workgroup_x : i32,
  %workgroup_y : i32,
  %workgroup_z : i32,
  %push_constants : i32 ...,
  // <buffer, offset, length>
  %bindings : tuple<!vm.buffer, i64, i64>...
)

}  // module
