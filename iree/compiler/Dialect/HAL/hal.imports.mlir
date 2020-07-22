// IREE Hardware Abstraction Layer (HAL) runtime module imports.
//
// This is embedded in the compiler binary and inserted into any module
// containing HAL dialect ops (hal.*) that is lowered to the VM dialect.
vm.module @hal {

//===----------------------------------------------------------------------===//
// Experimental/temporary ops
//===----------------------------------------------------------------------===//

vm.import @ex.shared_device() -> !vm.ref<!hal.device>
attributes {nosideeffects}

vm.import @ex.defer_release(
  %operand : !vm.ref<?>
)

vm.import @ex.submit_and_wait(
  %device : !vm.ref<!hal.device>,
  %command_buffer : !vm.ref<!hal.command_buffer>
)

//===----------------------------------------------------------------------===//
// iree::hal::Allocator
//===----------------------------------------------------------------------===//

// Computes the byte size required for a buffer of the given shape and type.
vm.import @allocator.compute_size(
  %allocator : !vm.ref<!hal.allocator>,
  %shape : i32 ...,
  %element_type : i32
) -> i32
attributes {nosideeffects}

// Computes an element byte offset within a buffer.
vm.import @allocator.compute_offset(
  %allocator : !vm.ref<!hal.allocator>,
  %shape : i32 ...,
  %element_type : i32,
  %indices : i32 ...
) -> i32
attributes {nosideeffects}

// Computes a byte range within a buffer for one or more elements.
vm.import @allocator.compute_range(
  %allocator : !vm.ref<!hal.allocator>,
  %shape : i32 ...,
  %element_type : i32,
  %indices : i32 ...,
  %lengths : i32 ...
) -> (i32, i32)
attributes {nosideeffects}

// Allocates a buffer from the allocator.
vm.import @allocator.allocate(
  %allocator : !vm.ref<!hal.allocator>,
  %memory_types : i32,
  %buffer_usage : i32,
  %allocation_size : i32
) -> !vm.ref<!hal.buffer>

// Allocates a buffer from the allocator with the given constant contents.
vm.import @allocator.allocate.const(
  %allocator : !vm.ref<!hal.allocator>,
  %memory_types : i32,
  %buffer_usage : i32,
  %shape : i32 ...,
  %element_type : i32,
  %value : !vm.ref<!iree.byte_buffer>
) -> !vm.ref<!hal.buffer>

//===----------------------------------------------------------------------===//
// iree::hal::Buffer
//===----------------------------------------------------------------------===//

// Returns the allocator the buffer was allocated with.
vm.import @buffer.allocator(
  %buffer : !vm.ref<!hal.buffer>
) -> !vm.ref<!hal.allocator>

// Returns a reference to a subspan of the buffer.
vm.import @buffer.subspan(
  %source_buffer : !vm.ref<!hal.buffer>,
  %source_offset : i32,
  %length : i32
) -> !vm.ref<!hal.buffer>

// Fills the target buffer with the given repeating value.
vm.import @buffer.fill(
  %target_buffer : !vm.ref<!hal.buffer>,
  %target_offset : i32,
  %length : i32,
  %pattern : i32
)

// Reads a block of byte data from the resource at the given offset.
vm.import @buffer.read_data(
  %source_buffer : !vm.ref<!hal.buffer>,
  %source_offset : i32,
  %target_buffer : !vm.ref<!iree.mutable_byte_buffer>,
  %target_offset : i32,
  %length : i32
)

// Writes a block of byte data into the resource at the given offset.
vm.import @buffer.write_data(
  %target_buffer : !vm.ref<!hal.buffer>,
  %target_offset : i32,
  %source_buffer : !vm.ref<!iree.byte_buffer>,
  %source_offset : i32,
  %length : i32
)

// Copies data from the provided source_buffer into the buffer.
vm.import @buffer.copy_data(
  %source_buffer : !vm.ref<!hal.buffer>,
  %source_offset : i32,
  %target_buffer : !vm.ref<!hal.buffer>,
  %target_offset : i32,
  %length : i32
)

// Loads a value from a buffer by mapping it.
vm.import @buffer.load(
  %source_buffer : !vm.ref<!hal.buffer>,
  %source_offset : i32,
  %length : i32
) -> i32

// Stores a value into a buffer by mapping it.
vm.import @buffer.store(
  %value : i32,
  %target_buffer : !vm.ref<!hal.buffer>,
  %target_offset : i32,
  %length : i32
)

//===----------------------------------------------------------------------===//
// iree::hal::BufferView
//===----------------------------------------------------------------------===//

// Creates a reference to a buffer with a particular shape and element type.
vm.import @buffer_view.create(
  %buffer : !vm.ref<!hal.buffer>,
  %shape : i32 ...,
  %element_type : i32
) -> !vm.ref<!hal.buffer_view>
attributes {nosideeffects}

// Returns a view into a buffer. The buffer is not copied and both the original
// and sliced references must be synchronized.
vm.import @buffer_view.subview(
  %buffer_view : !vm.ref<!hal.buffer_view>,
  %indices : i32 ...,
  %lengths : i32 ...
) -> !vm.ref<!hal.buffer_view>
attributes {nosideeffects}

// Returns the backing buffer of the buffer view.
vm.import @buffer_view.buffer(
  %buffer_view : !vm.ref<!hal.buffer_view>
) -> !vm.ref<!hal.buffer>
attributes {nosideeffects}

// Returns the allocated size of a shaped buffer view in bytes.
vm.import @buffer_view.byte_length(
  %buffer_view : !vm.ref<!hal.buffer_view>
) -> i32
attributes {nosideeffects}

// Computes an element byte offset within a buffer.
vm.import @buffer_view.compute_offset(
  %buffer_view : !vm.ref<!hal.buffer_view>,
  %indices : i32 ...
) -> i32
attributes {nosideeffects}

// Computes a byte range within a buffer for one or more elements.
vm.import @buffer_view.compute_range(
  %buffer_view : !vm.ref<!hal.buffer_view>,
  %indices : i32 ...,
  %lengths : i32 ...
) -> (i32, i32)
attributes {nosideeffects}

// Returns the rank of the buffer view.
vm.import @buffer_view.rank(
  %buffer_view : !vm.ref<!hal.buffer_view>,
) -> i32
attributes {nosideeffects}

// Returns the value of the given dimension.
vm.import @buffer_view.dim(
  %buffer_view : !vm.ref<!hal.buffer_view>,
  %index : i32
) -> i32
attributes {nosideeffects}

// Returns N dimension values.
vm.import @buffer_view.dims.1(
  %buffer_view : !vm.ref<!hal.buffer_view>
) -> (i32)
attributes {nosideeffects}
vm.import @buffer_view.dims.2(
  %buffer_view : !vm.ref<!hal.buffer_view>
) -> (i32, i32)
attributes {nosideeffects}
vm.import @buffer_view.dims.3(
  %buffer_view : !vm.ref<!hal.buffer_view>
) -> (i32, i32, i32)
attributes {nosideeffects}
vm.import @buffer_view.dims.4(
  %buffer_view : !vm.ref<!hal.buffer_view>
) -> (i32, i32, i32, i32)
attributes {nosideeffects}

// Prints out the content of buffers.
vm.import @buffer_view.trace(
  %operands : !vm.ref<!hal.buffer_view> ...
)

//===----------------------------------------------------------------------===//
// iree::hal::CommandBuffer
//===----------------------------------------------------------------------===//

// Returns a command buffer from the device pool ready to begin recording.
vm.import @command_buffer.create(
  %device : !vm.ref<!hal.device>,
  %modes : i32,
  %command_categories : i32
) -> !vm.ref<!hal.command_buffer>

// Resets and begins recording into the command buffer, clearing all previously
// recorded contents.
vm.import @command_buffer.begin(
  %command_buffer : !vm.ref<!hal.command_buffer>
)

// Ends recording into the command buffer.
vm.import @command_buffer.end(
  %command_buffer : !vm.ref<!hal.command_buffer>
)

// Defines a memory dependency between commands recorded before and after the
// barrier.
vm.import @command_buffer.execution_barrier(
  %command_buffer : !vm.ref<!hal.command_buffer>,
  %source_stage_mask : i32,
  %target_stage_mask : i32,
  // TODO(benvanik): tuple types.
  %memory_barriers : i32 ...,
  %buffer_barriers : i32 ...
)

// Fills the target buffer with the given repeating value.
vm.import @command_buffer.fill_buffer(
  %command_buffer : !vm.ref<!hal.command_buffer>,
  %target_buffer : !vm.ref<!hal.buffer>,
  %target_offset : i32,
  %length : i32,
  %pattern : i32
)

// Copies a range of one buffer to another.
vm.import @command_buffer.copy_buffer(
  %command_buffer : !vm.ref<!hal.command_buffer>,
  %source_buffer : !vm.ref<!hal.buffer>,
  %source_offset : i32,
  %target_buffer : !vm.ref<!hal.buffer>,
  %target_offset : i32,
  %length : i32
)

// Pushes constants for consumption by dispatches.
vm.import @command_buffer.push_constants(
  %command_buffer : !vm.ref<!hal.command_buffer>,
  %executable_layout : !vm.ref<!hal.executable_layout>,
  %offset : i32,
  %values : i32 ...
)

// Pushes a descriptor set to the given set number.
vm.import @command_buffer.push_descriptor_set(
  %command_buffer : !vm.ref<!hal.command_buffer>,
  %executable_layout : !vm.ref<!hal.executable_layout>,
  %set : i32,
  %bindings : i32 ...,
  %binding_buffers : !vm.ref<!hal.buffer>...,
  %binding_offsets : i32 ...,
  %binding_lengths : i32 ...
)

// Binds a descriptor set to the given set number.
vm.import @command_buffer.bind_descriptor_set(
  %command_buffer : !vm.ref<!hal.command_buffer>,
  %executable_layout : !vm.ref<!hal.executable_layout>,
  %set : i32,
  %descriptor_set : !vm.ref<!hal.descriptor_set>,
  %dynamic_offsets : i32 ...
)

// Dispatches an execution request.
vm.import @command_buffer.dispatch(
  %command_buffer : !vm.ref<!hal.command_buffer>,
  %executable : !vm.ref<!hal.executable>,
  %entry_point : i32,
  %workgroup_x : i32,
  %workgroup_y : i32,
  %workgroup_z : i32
)

// Dispatches an execution request with the dispatch parameters loaded from the
// given buffer.
vm.import @command_buffer.dispatch.indirect(
  %command_buffer : !vm.ref<!hal.command_buffer>,
  %executable : !vm.ref<!hal.executable>,
  %entry_point : i32,
  %workgroups_buffer : !vm.ref<!hal.buffer>,
  %workgroups_offset : i32
)

//===----------------------------------------------------------------------===//
// iree::hal::DescriptorSet
//===----------------------------------------------------------------------===//

// Creates a new immutable descriptor set based on the given layout.
vm.import @descriptor_set.create(
  %device : !vm.ref<!hal.device>,
  %set_layout : !vm.ref<!hal.descriptor_set_layout>,
  %bindings : i32 ...,
  %binding_buffers : !vm.ref<!hal.buffer>...,
  %binding_offsets : i32 ...,
  %binding_lengths : i32 ...
) -> !vm.ref<!hal.descriptor_set>

//===----------------------------------------------------------------------===//
// iree::hal::DescriptorSetLayout
//===----------------------------------------------------------------------===//

// Creates a descriptor set layout that defines the bindings used within a set.
vm.import @descriptor_set_layout.create(
  %device : !vm.ref<!hal.device>,
  %usage_type : i32,
  // <binding, type, access>
  %bindings : tuple<i32, i32, i32>...
) -> !vm.ref<!hal.descriptor_set_layout>
attributes {nosideeffects}

//===----------------------------------------------------------------------===//
// iree::hal::Device
//===----------------------------------------------------------------------===//

// Returns the allocator that can be used to allocate buffers compatible with
// the device.
vm.import @device.allocator(
  %device : !vm.ref<!hal.device>
) -> !vm.ref<!hal.allocator>
attributes {nosideeffects}

// Returns true if the device ID matches the pattern.
vm.import @device.match.id(
  %device : !vm.ref<!hal.device>,
  %pattern : !vm.ref<!iree.byte_buffer>
) -> i32
attributes {nosideeffects}

//===----------------------------------------------------------------------===//
// iree::hal::ExecutableCache
//===----------------------------------------------------------------------===//

// Creates an executable cache with the given identifier.
vm.import @executable_cache.create(
  %device : !vm.ref<!hal.device>,
  %identifier : !vm.ref<!iree.byte_buffer>
) -> !vm.ref<!hal.executable_cache>
attributes {nosideeffects}

// Returns the index of the preferred format of the cache from the given set
// or -1 if none can be used. Preparation may still fail if the particular
// version or features required by the executable are not supported.
vm.import @executable_cache.select_format(
  %executable_cache : !vm.ref<!hal.executable_cache>,
  %available_formats : i32 ...
) -> i32
attributes {nosideeffects}

// Caches an executable for use with the specified device.
// The executable may be shared with other contexts but as it is immutable
// this does not matter.
vm.import @executable_cache.prepare(
  %executable_cache : !vm.ref<!hal.executable_cache>,
  %executable_layout : !vm.ref<!hal.executable_layout>,
  %caching_mode : i32,
  %executable_data : !vm.ref<!iree.byte_buffer>
) -> !vm.ref<!hal.executable>
attributes {nosideeffects}

//===----------------------------------------------------------------------===//
// iree::hal::ExecutableLayout
//===----------------------------------------------------------------------===//

// Creates an executable layout from the given descriptor sets and push constant
// required size.
vm.import @executable_layout.create(
  %device : !vm.ref<!hal.device>,
  %set_layouts : !vm.ref<!hal.descriptor_set_layout>...,
  %push_constants : i32
) -> !vm.ref<!hal.executable_layout>
attributes {nosideeffects}

//===----------------------------------------------------------------------===//
// iree::hal::Semaphore
//===----------------------------------------------------------------------===//

// Returns a semaphore from the device pool with the given initial value.
vm.import @semaphore.create(
  %device : !vm.ref<!hal.device>,
  %initial_value : i32
) -> !vm.ref<!hal.semaphore>
attributes {nosideeffects}

// Queries the current payload and returns a tuple of `(status, value)`.
// As the payload is monotonically increasing it is guaranteed that
// the value is at least equal to the previous result of a
// `hal.semaphore.signal` call and coherent with any waits for a
// specified value via `hal.semaphore.await`.
vm.import @semaphore.query(
  %semaphore : !vm.ref<!hal.semaphore>
) -> (i32, i32)

// Signals the semaphore to the given payload value.
// The call is ignored if the current payload value exceeds |new_value|.
vm.import @semaphore.signal(
  %semaphore : !vm.ref<!hal.semaphore>,
  %new_value : i32
)

// Signals the semaphore with a failure. The |status| will be returned from
// `hal.semaphore.query` and `hal.semaphore.signal` for the lifetime
// of the semaphore.
vm.import @semaphore.fail(
  %semaphore : !vm.ref<!hal.semaphore>,
  %status : i32
)

// Yields the caller until the semaphore reaches or exceeds the specified
// payload |value|.
//
// Returns the status of the semaphore after the wait, with a non-zero value
// indicating failure.
vm.import @semaphore.await(
  %semaphore : !vm.ref<!hal.semaphore>,
  %min_value : i32
) -> i32
// TODO(benvanik): yield point trait.

}  // module
