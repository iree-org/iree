// IREE Inline Hardware Abstraction Layer (HAL) runtime module imports.
// This is only used to provide ABI compatibility with the full HAL module and
// user programs that use !hal.buffer/!hal.buffer_view as IO.
//
// This is embedded in the compiler binary and inserted into any module
// containing inline HAL dialect ops (hal_inline.*) that is lowered to the VM
// dialect.
vm.module @hal_inline {

//===----------------------------------------------------------------------===//
// iree_hal_buffer_t
//===----------------------------------------------------------------------===//

// Allocates an empty buffer.
vm.import private @buffer.allocate(
  %minimum_alignment : i32,
  %allocation_size : i64
) -> (!vm.ref<!hal.buffer>, !vm.buffer)
attributes {nosideeffects}

// Allocates a buffer with an initial value provided by a VM byte buffer.
vm.import private @buffer.allocate.initialized(
  %minimum_alignment : i32,
  %source : !vm.buffer,
  %offset : i64,
  %length : i64
) -> (!vm.ref<!hal.buffer>, !vm.buffer)
attributes {nosideeffects}

// Wraps a VM byte buffer in a HAL buffer.
vm.import private @buffer.wrap(
  %source : !vm.buffer,
  %offset : i64,
  %length : i64
) -> !vm.ref<!hal.buffer>
attributes {nosideeffects}

// Returns a reference to a subspan of the buffer.
vm.import private @buffer.subspan(
  %source_buffer : !vm.ref<!hal.buffer>,
  %source_offset : i64,
  %length : i64
) -> !vm.ref<!hal.buffer>
attributes {nosideeffects}

// TODO(benvanik): make storage return length and remove dedicated length?

// Returns the byte length of the buffer (may be less than total allocation).
vm.import private @buffer.length(
  %buffer : !vm.ref<!hal.buffer>
) -> i64
attributes {nosideeffects}

// Returns a mapping to the underlying storage of the buffer sliced to the
// logical subspan of the HAL buffer.
vm.import private @buffer.storage(
  %buffer : !vm.ref<!hal.buffer>
) -> !vm.buffer
attributes {nosideeffects}

//===----------------------------------------------------------------------===//
// iree_hal_buffer_view_t
//===----------------------------------------------------------------------===//

// Creates a reference to a buffer with a particular shape and element type.
vm.import private @buffer_view.create(
  %source_buffer : !vm.ref<!hal.buffer>,
  %source_offset : i64,
  %source_length : i64,
  %element_type : i32,
  %encoding_type : i32,
  %shape : i64 ...
) -> !vm.ref<!hal.buffer_view>
attributes {nosideeffects}

// Asserts a buffer view matches the given tensor encoding and shape.
vm.import private @buffer_view.assert(
  %buffer_view : !vm.ref<!hal.buffer_view>,
  %message : !vm.buffer,
  %element_type : i32,
  %encoding_type : i32,
  %shape : i64 ...
)

// Returns the backing buffer of the buffer view.
vm.import private @buffer_view.buffer(
  %buffer_view : !vm.ref<!hal.buffer_view>
) -> !vm.ref<!hal.buffer>
attributes {nosideeffects}

// Returns the element type of the buffer view.
vm.import private @buffer_view.element_type(
  %buffer_view : !vm.ref<!hal.buffer_view>,
) -> i32
attributes {nosideeffects}

// Returns the encoding type of the buffer view.
vm.import private @buffer_view.encoding_type(
  %buffer_view : !vm.ref<!hal.buffer_view>,
) -> i32
attributes {nosideeffects}

// Returns the rank of the buffer view.
vm.import private @buffer_view.rank(
  %buffer_view : !vm.ref<!hal.buffer_view>,
) -> i32
attributes {nosideeffects}

// Returns the value of the given dimension.
vm.import private @buffer_view.dim(
  %buffer_view : !vm.ref<!hal.buffer_view>,
  %index : i32
) -> i64
attributes {nosideeffects}

// Prints out the content of buffer views.
vm.import private @buffer_view.trace(
  %key : !vm.buffer,
  %operands : !vm.ref<!hal.buffer_view> ...
)

//===----------------------------------------------------------------------===//
// iree_hal_device_t
//===----------------------------------------------------------------------===//

// Returns a tuple of (ok, value) for the given configuration key.
vm.import private @device.query.i64(
  %category : !vm.buffer,
  %key : !vm.buffer
) -> (i32, i64)
attributes {nosideeffects}

}  // module
