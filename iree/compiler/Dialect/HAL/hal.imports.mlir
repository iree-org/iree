// IREE Hardware Abstraction Layer (HAL) runtime module imports.
//
// This is embedded in the compiler binary and inserted into any module
// containing HAL dialect ops (hal.*) that is lowered to the VM dialect.
vm.module @hal {

//===----------------------------------------------------------------------===//
// Experimental/temporary ops
//===----------------------------------------------------------------------===//

vm.import @ex.shared_device() -> !iree.ref<!hal.device>
attributes {nosideeffects}

// Returns one of the provided executable formats that can be used by the
// device or 0 if none are supported.
vm.import @ex.match_supported_executable_format(
  %device : !iree.ref<!hal.device>,
  %available_formats : i32 ...
) -> i32
attributes {nosideeffects}

// Caches an executable for use with the specified device.
// The executable may be shared with other contexts but as it is immutable
// this does not matter.
vm.import @ex.cache_executable(
  %device : !iree.ref<!hal.device>,
  %executable_format : i32,
  %executable_data : !iree.byte_buffer_ref
) -> !iree.ref<!hal.executable>
attributes {nosideeffects}

vm.import @ex.push_binding(
  %command_buffer : !iree.ref<!hal.command_buffer>,
  %ordinal : i32,
  %buffer : !iree.ref<!hal.buffer>,
  %shape : i32 ...,
  %element_type : i32
)

vm.import @ex.defer_release(
  %operand : !iree.opaque_ref
)

vm.import @ex.submit_and_wait(
  %device : !iree.ref<!hal.device>,
  %command_buffer : !iree.ref<!hal.command_buffer>
)

//===----------------------------------------------------------------------===//
// iree::hal::Allocator
//===----------------------------------------------------------------------===//

// Computes the byte size required for a buffer of the given shape and type.
vm.import @allocator.compute_size(
  %allocator : !iree.ref<!hal.allocator>,
  %shape : i32 ...,
  %element_type : i32
) -> i32
attributes {nosideeffects}

// Computes an element byte offset within a buffer.
vm.import @allocator.compute_offset(
  %allocator : !iree.ref<!hal.allocator>,
  %shape : i32 ...,
  %element_type : i32,
  %indices : i32 ...
) -> i32
attributes {nosideeffects}

// Computes a byte range within a buffer for one or more elements.
vm.import @allocator.compute_range(
  %allocator : !iree.ref<!hal.allocator>,
  %shape : i32 ...,
  %element_type : i32,
  %indices : i32 ...,
  %lengths : i32 ...
) -> (i32, i32)
attributes {nosideeffects}

// Allocates a buffer from the allocator.
vm.import @allocator.allocate(
  %allocator : !iree.ref<!hal.allocator>,
  %memory_types : i32,
  %buffer_usage : i32,
  %allocation_size : i32
) -> !iree.ref<!hal.buffer>

// Allocates a buffer from the allocator with the given constant contents.
vm.import @allocator.allocate.const(
  %allocator : !iree.ref<!hal.allocator>,
  %memory_types : i32,
  %buffer_usage : i32,
  %shape : i32 ...,
  %element_type : i32,
  %value : !iree.byte_buffer_ref
) -> !iree.ref<!hal.buffer>

//===----------------------------------------------------------------------===//
// iree::hal::Buffer
//===----------------------------------------------------------------------===//

// Returns the allocator the buffer was allocated with.
vm.import @buffer.allocator(
  %buffer : !iree.ref<!hal.buffer>
) -> !iree.ref<!hal.allocator>

// Returns a reference to a subspan of the buffer.
vm.import @buffer.subspan(
  %source_buffer : !iree.ref<!hal.buffer>,
  %source_offset : i32,
  %length : i32
) -> !iree.ref<!hal.buffer>

// Fills the target buffer with the given repeating value.
vm.import @buffer.fill(
  %target_buffer : !iree.ref<!hal.buffer>,
  %target_offset : i32,
  %length : i32,
  %pattern : i32
)

// Reads a block of byte data from the resource at the given offset.
vm.import @buffer.read_data(
  %source_buffer : !iree.ref<!hal.buffer>,
  %source_offset : i32,
  %target_buffer : !iree.mutable_byte_buffer_ref,
  %target_offset : i32,
  %length : i32
)

// Writes a block of byte data into the resource at the given offset.
vm.import @buffer.write_data(
  %target_buffer : !iree.ref<!hal.buffer>,
  %target_offset : i32,
  %source_buffer : !iree.byte_buffer_ref,
  %source_offset : i32,
  %length : i32
)

// Copies data from the provided source_buffer into the buffer.
vm.import @buffer.copy_data(
  %source_buffer : !iree.ref<!hal.buffer>,
  %source_offset : i32,
  %target_buffer : !iree.ref<!hal.buffer>,
  %target_offset : i32,
  %length : i32
)

// Loads a value from a buffer by mapping it.
vm.import @buffer.load(
  %source_buffer : !iree.ref<!hal.buffer>,
  %source_offset : i32,
  %length : i32
) -> i32

// Stores a value into a buffer by mapping it.
vm.import @buffer.store(
  %value : i32,
  %target_buffer : !iree.ref<!hal.buffer>,
  %target_offset : i32,
  %length : i32
)

//===----------------------------------------------------------------------===//
// iree::hal::BufferView
//===----------------------------------------------------------------------===//

// Creates a reference to a buffer with a particular shape and element type.
vm.import @buffer_view.create(
  %buffer : !iree.ref<!hal.buffer>,
  %shape : i32 ...,
  %element_type : i32
) -> !iree.ref<!hal.buffer_view>
attributes {nosideeffects}

// Returns a view into a buffer. The buffer is not copied and both the original
// and sliced references must be synchronized.
vm.import @buffer_view.subview(
  %buffer_view : !iree.ref<!hal.buffer_view>,
  %indices : i32 ...,
  %lengths : i32 ...
) -> !iree.ref<!hal.buffer_view>
attributes {nosideeffects}

// Returns the backing buffer of the buffer view.
vm.import @buffer_view.buffer(
  %buffer_view : !iree.ref<!hal.buffer_view>
) -> !iree.ref<!hal.buffer>
attributes {nosideeffects}

// Returns the allocated size of a shaped buffer view in bytes.
vm.import @buffer_view.byte_length(
  %buffer_view : !iree.ref<!hal.buffer_view>
) -> i32
attributes {nosideeffects}

// Computes an element byte offset within a buffer.
vm.import @buffer_view.compute_offset(
  %buffer_view : !iree.ref<!hal.buffer_view>,
  %indices : i32 ...
) -> i32
attributes {nosideeffects}

// Computes a byte range within a buffer for one or more elements.
vm.import @buffer_view.compute_range(
  %buffer_view : !iree.ref<!hal.buffer_view>,
  %indices : i32 ...,
  %lengths : i32 ...
) -> (i32, i32)
attributes {nosideeffects}

//===----------------------------------------------------------------------===//
// iree::hal::CommandBuffer
//===----------------------------------------------------------------------===//

// Returns a command buffer from the device pool ready to begin recording.
vm.import @command_buffer.create(
  %device : !iree.ref<!hal.device>,
  %modes : i32,
  %command_categories : i32
) -> !iree.ref<!hal.command_buffer>

// Resets and begins recording into the command buffer, clearing all previously
// recorded contents.
vm.import @command_buffer.begin(
  %command_buffer : !iree.ref<!hal.command_buffer>
)

// Ends recording into the command buffer.
vm.import @command_buffer.end(
  %command_buffer : !iree.ref<!hal.command_buffer>
)

// Defines a memory dependency between commands recorded before and after the
// barrier.
vm.import @command_buffer.execution_barrier(
  %command_buffer : !iree.ref<!hal.command_buffer>,
  %source_stage_mask : i32,
  %target_stage_mask : i32,
  // TODO(benvanik): tuple types.
  %memory_barriers : i32 ...,
  %buffer_barriers : i32 ...
)

// Fills the target buffer with the given repeating value.
vm.import @command_buffer.fill_buffer(
  %command_buffer : !iree.ref<!hal.command_buffer>,
  %target_buffer : !iree.ref<!hal.buffer>,
  %target_offset : i32,
  %length : i32,
  %pattern : i32
)

// Copies a range of one buffer to another.
vm.import @command_buffer.copy_buffer(
  %command_buffer : !iree.ref<!hal.command_buffer>,
  %source_buffer : !iree.ref<!hal.buffer>,
  %source_offset : i32,
  %target_buffer : !iree.ref<!hal.buffer>,
  %target_offset : i32,
  %length : i32
)

// Binds a descriptor set to the given set number.
vm.import @command_buffer.bind_descriptor_set(
  %command_buffer : !iree.ref<!hal.command_buffer>,
  %executable_layout : !iree.ref<!hal.executable_layout>,
  %set : i32,
  %descriptor_set : !iree.ref<!hal.descriptor_set>,
  %dynamic_offsets : i32 ...
)

// Dispatches an execution request.
vm.import @command_buffer.dispatch(
  %command_buffer : !iree.ref<!hal.command_buffer>,
  %executable : !iree.ref<!hal.executable>,
  %entry_point : i32,
  %workgroup_x : i32,
  %workgroup_y : i32,
  %workgroup_z : i32
)

// Dispatches an execution request with the dispatch parameters loaded from the
// given buffer.
vm.import @command_buffer.dispatch.indirect(
  %command_buffer : !iree.ref<!hal.command_buffer>,
  %executable : !iree.ref<!hal.executable>,
  %entry_point : i32,
  %workgroups_buffer : !iree.ref<!hal.buffer>,
  %workgroups_offset : i32
)

//===----------------------------------------------------------------------===//
// iree::hal::DescriptorSet
//===----------------------------------------------------------------------===//

// Creates a new immutable descriptor set based on the given layout.
vm.import @descriptor_set.create(
  %device : !iree.ref<!hal.device>,
  %set_layout : !iree.ref<!hal.descriptor_set_layout>,
  // <binding, buffer, offset, length>
  %bindings : tuple<i32, !iree.ref<!hal.buffer>, i32, i32>...
) -> !iree.ref<!hal.descriptor_set>

//===----------------------------------------------------------------------===//
// iree::hal::DescriptorSetLayout
//===----------------------------------------------------------------------===//

// Creates a descriptor set layout that defines the bindings used within a set.
vm.import @descriptor_set_layout.create(
  %device : !iree.ref<!hal.device>,
  // <binding, type, access>
  %bindings : tuple<i32, i32, i32>...
) -> !iree.ref<!hal.descriptor_set_layout>
attributes {nosideeffects}

//===----------------------------------------------------------------------===//
// iree::hal::Device
//===----------------------------------------------------------------------===//

// Returns the allocator that can be used to allocate buffers compatible with
// the device.
vm.import @device.allocator(
  %device : !iree.ref<!hal.device>
) -> !iree.ref<!hal.allocator>
attributes {nosideeffects}

//===----------------------------------------------------------------------===//
// iree::hal::ExecutableLayout
//===----------------------------------------------------------------------===//

// Creates an executable layout from the given descriptor sets and push constant
// required size.
vm.import @executable_layout.create(
  %device : !iree.ref<!hal.device>,
  %set_layouts : !iree.ref<!hal.descriptor_set_layout>...,
  %push_constants : i32
) -> !iree.ref<!hal.executable_layout>
attributes {nosideeffects}

}  // module
