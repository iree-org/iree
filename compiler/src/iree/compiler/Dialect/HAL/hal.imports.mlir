// IREE Hardware Abstraction Layer (HAL) runtime module imports.
//
// This is embedded in the compiler binary and inserted into any module
// containing HAL dialect ops (hal.*) that is lowered to the VM dialect.
vm.module @hal {

//===----------------------------------------------------------------------===//
// Experimental/temporary ops
//===----------------------------------------------------------------------===//

// Creates a file mapped into a byte range of a host buffer.
// EXPERIMENTAL: may be removed in future versions.
vm.import private @ex.file.from_memory(
  %device : !vm.ref<!hal.device>,
  %queue_affinity : i64,
  %access : i32,
  %buffer : !vm.buffer,
  %offset : i64,
  %length : i64,
  %flags : i32
) -> !vm.ref<!hal.file>

//===----------------------------------------------------------------------===//
// iree_hal_allocator_t
//===----------------------------------------------------------------------===//

// Allocates a buffer from the allocator. The resulting buffer will have a
// length of at least that requested.
vm.import private @allocator.allocate(
  %allocator : !vm.ref<!hal.allocator>,
  %queue_affinity : i64,
  %memory_types : i32,
  %buffer_usage : i32,
  %allocation_size : i64
) -> !vm.ref<!hal.buffer>

// Imports a host byte buffer into a device visible buffer.
// If try!=0 then returns null if the given memory type cannot be mapped.
// Host-local+constant requests will always succeed.
vm.import private @allocator.import(
  %allocator : !vm.ref<!hal.allocator>,
  %try : i32,
  %queue_affinity : i64,
  %memory_types : i32,
  %buffer_usage : i32,
  %source : !vm.buffer,
  %offset : i64,
  %length : i64
) -> !vm.ref<!hal.buffer>

//===----------------------------------------------------------------------===//
// iree_hal_buffer_t
//===----------------------------------------------------------------------===//

// Returns the allocator the buffer was allocated with.
vm.import private @buffer.assert(
  %buffer : !vm.ref<!hal.buffer>,
  %message : !vm.buffer,
  %allocator : !vm.ref<!hal.allocator>,
  %minimum_length : i64,
  %memory_types : i32,
  %buffer_usage : i32
)

// Preserves the underlying buffer allocation for the caller.
vm.import private @buffer.allocation.preserve(
  %buffer : !vm.ref<!hal.buffer>
)

// Discards ownership of the underlying buffer allocation by the caller and
// returns true if the caller was the last owner.
vm.import private @buffer.allocation.discard(
  %buffer : !vm.ref<!hal.buffer>
) -> i32

// Returns true if the underlying buffer allocation has a single owner.
vm.import private @buffer.allocation.is_terminal(
  %buffer : !vm.ref<!hal.buffer>
) -> i32

// Returns a reference to a subspan of the buffer.
vm.import private @buffer.subspan(
  %source_buffer : !vm.ref<!hal.buffer>,
  %source_offset : i64,
  %length : i64
) -> !vm.ref<!hal.buffer>
attributes {nosideeffects}

// Returns the byte length of the buffer (may be less than total allocation).
vm.import private @buffer.length(
  %buffer : !vm.ref<!hal.buffer>
) -> i64
attributes {nosideeffects}

// TODO(benvanik): remove load/store and instead return a mapped !vm.buffer.
// This will let us perform generic VM buffer operations directly on the HAL
// buffer memory. The tricky part is ensuring the mapping lifetime or adding
// an invalidation mechanism.

// Loads a value from a buffer by mapping it.
vm.import private @buffer.load(
  %source_buffer : !vm.ref<!hal.buffer>,
  %source_offset : i64,
  %length : i32
) -> i32

// Stores a value into a buffer by mapping it.
vm.import private @buffer.store(
  %value : i32,
  %target_buffer : !vm.ref<!hal.buffer>,
  %target_offset : i64,
  %length : i32
)

//===----------------------------------------------------------------------===//
// iree_hal_buffer_view_t
//===----------------------------------------------------------------------===//

// Creates a reference to a buffer with a particular shape and element type.
vm.import private @buffer_view.create(
  %buffer : !vm.ref<!hal.buffer>,
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
// iree_hal_channel_t
//===----------------------------------------------------------------------===//

// Creates a new channel for collective communication.
vm.import private @channel.create(
  %device : !vm.ref<!hal.device>,
  %queue_affinity : i64,
  %flags : i64,
  %id : !vm.buffer,
  %group : !vm.buffer,
  %rank : i32,
  %count : i32
) -> !vm.ref<!hal.channel>
attributes {nosideeffects}

// Splits a collective communication channel.
vm.import private @channel.split(
  %channel : !vm.ref<!hal.channel>,
  %color : i32,
  %key : i32,
  %flags : i64
) -> !vm.ref<!hal.channel>
attributes {nosideeffects}

// Returns the rank of the local participant in the group and the group count.
vm.import private @channel.rank_and_count(
  %channel : !vm.ref<!hal.channel>
) -> (i32, i32)
attributes {nosideeffects}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_t
//===----------------------------------------------------------------------===//

// Returns a command buffer from the device pool ready to begin recording.
vm.import private @command_buffer.create(
  %device : !vm.ref<!hal.device>,
  %modes : i32,
  %command_categories : i32,
  %queue_affinity : i64,
  %binding_capacity : i32
) -> !vm.ref<!hal.command_buffer>
attributes {
  minimum_version = 6 : i32  // command buffer API version
}

// Finalizes recording into the command buffer and prepares it for submission.
// No more commands can be recorded afterward.
vm.import private @command_buffer.finalize(
  %command_buffer : !vm.ref<!hal.command_buffer>
)

// Pushes a new debug group with the given |label|.
vm.import private @command_buffer.begin_debug_group(
  %command_buffer : !vm.ref<!hal.command_buffer>,
  %label : !vm.buffer
)

// Pops a debug group from the stack.
vm.import private @command_buffer.end_debug_group(
  %command_buffer : !vm.ref<!hal.command_buffer>
)

// Defines an execution dependency between all commands recorded before the
// barrier and all commands recorded after the barrier. Only the stages provided
// will be affected.
vm.import private @command_buffer.execution_barrier(
  %command_buffer : !vm.ref<!hal.command_buffer>,
  %source_stage_mask : i32,
  %target_stage_mask : i32,
  %flags : i64
)

// Advises the device about the usage of the given buffer.
// The device may use this information to perform cache management or ignore it
// entirely.
vm.import private @command_buffer.advise_buffer(
  %command_buffer : !vm.ref<!hal.command_buffer>,
  %buffer : !vm.ref<!hal.buffer>,
  %flags : i64,
  %arg0 : i64,
  %arg1 : i64,
  %buffer_slot : i32
)

// Fills the target buffer with the given repeating value.
// NOTE: order slightly differs from op in order to get better arg alignment.
vm.import private @command_buffer.fill_buffer(
  %command_buffer : !vm.ref<!hal.command_buffer>,
  %target_buffer : !vm.ref<!hal.buffer>,
  %target_offset : i64,
  %length : i64,
  %target_buffer_slot : i32,
  %pattern : i64,
  %pattern_length : i32,
  %flags : i64
)

// Updates a device buffer with the captured contents of a host buffer.
// NOTE: order slightly differs from op in order to get better arg alignment.
vm.import private @command_buffer.update_buffer(
  %command_buffer : !vm.ref<!hal.command_buffer>,
  %source_buffer : !vm.buffer,
  %source_offset : i64,
  %target_buffer : !vm.ref<!hal.buffer>,
  %target_offset : i64,
  %length : i64,
  %target_buffer_slot : i32,
  %flags : i64
)

// Copies a range of one buffer to another.
// NOTE: order slightly differs from op in order to get better arg alignment.
vm.import private @command_buffer.copy_buffer(
  %command_buffer : !vm.ref<!hal.command_buffer>,
  %source_buffer_slot : i32,
  %target_buffer_slot : i32,
  %source_buffer : !vm.ref<!hal.buffer>,
  %source_offset : i64,
  %target_buffer : !vm.ref<!hal.buffer>,
  %target_offset : i64,
  %length : i64,
  %flags : i64
)

// Dispatches a collective operation defined by |op| using the given buffers.
// NOTE: order slightly differs from op in order to get better arg alignment.
vm.import private @command_buffer.collective(
  %command_buffer : !vm.ref<!hal.command_buffer>,
  %channel : !vm.ref<!hal.channel>,
  %op : i32,
  %param : i32,
  %send_buffer_slot : i32,
  %recv_buffer_slot : i32,
  %send_buffer : !vm.ref<!hal.buffer>,
  %recv_buffer : !vm.ref<!hal.buffer>,
  %send_offset : i64,
  %send_length : i64,
  %recv_offset : i64,
  %recv_length : i64,
  %element_count : i64
)

// Dispatches an execution request.
vm.import private @command_buffer.dispatch(
  %command_buffer : !vm.ref<!hal.command_buffer>,
  %executable : !vm.ref<!hal.executable>,
  %entry_point : i32,
  %workgroup_x : i32,
  %workgroup_y : i32,
  %workgroup_z : i32,
  %flags : i64,
  %constants : i32 ...,
  // <reserved, slot, buffer, offset, length>
  %bindings : tuple<i32, i32, !vm.ref<!hal.buffer>, i64, i64>...
)

// Dispatches an execution request with the dispatch parameters loaded from the
// given buffer.
vm.import private @command_buffer.dispatch.indirect(
  %command_buffer : !vm.ref<!hal.command_buffer>,
  %executable : !vm.ref<!hal.executable>,
  %entry_point : i32,
  %workgroups_buffer_slot : i32,
  %workgroups_buffer : !vm.ref<!hal.buffer>,
  %workgroups_offset : i64,
  %flags : i64,
  %constants : i32 ...,
  // <reserved, slot, buffer, offset, length>
  %bindings : tuple<i32, i32, !vm.ref<!hal.buffer>, i64, i64>...
)

//===----------------------------------------------------------------------===//
// iree_hal_device_t
//===----------------------------------------------------------------------===//

// Returns the allocator that can be used to allocate buffers compatible with
// the device.
vm.import private @device.allocator(
  %device : !vm.ref<!hal.device>
) -> !vm.ref<!hal.allocator>
attributes {nosideeffects}

// Returns a tuple of (ok, value) for the given configuration key.
vm.import private @device.query.i64(
  %device : !vm.ref<!hal.device>,
  %category : !vm.buffer,
  %key : !vm.buffer
) -> (i32, i64)
attributes {nosideeffects}

// Returns a queue-ordered transient buffer that will be available for use when
// the signal fence is reached. The allocation will not be made until the
// wait fence has been reached.
vm.import private @device.queue.alloca(
  %device : !vm.ref<!hal.device>,
  %queue_affinity : i64,
  %wait_fence : !vm.ref<!hal.fence>,
  %signal_fence : !vm.ref<!hal.fence>,
  %pool : i64,
  %memory_types : i32,
  %buffer_usage : i32,
  %allocation_size : i64,
  %flags : i64
) -> !vm.ref<!hal.buffer>

// Deallocates a queue-ordered transient buffer.
// The deallocation will not be made until the wait fence has been reached and
// once the storage is available for reuse the signal fence will be signaled.
vm.import private @device.queue.dealloca(
  %device : !vm.ref<!hal.device>,
  %queue_affinity : i64,
  %wait_fence : !vm.ref<!hal.fence>,
  %signal_fence : !vm.ref<!hal.fence>,
  %buffer : !vm.ref<!hal.buffer>,
  %flags : i64
)

// Enqueues a single queue-ordered fill operation.
vm.import private @device.queue.fill(
  %device : !vm.ref<!hal.device>,
  %queue_affinity : i64,
  %wait_fence : !vm.ref<!hal.fence>,
  %signal_fence : !vm.ref<!hal.fence>,
  %target_buffer : !vm.ref<!hal.buffer>,
  %target_offset : i64,
  %length : i64,
  %pattern : i64,
  %pattern_length: i32,
  %flags : i64
)

// Enqueues a single queue-ordered buffer update operation.
vm.import private @device.queue.update(
  %device : !vm.ref<!hal.device>,
  %queue_affinity : i64,
  %wait_fence : !vm.ref<!hal.fence>,
  %signal_fence : !vm.ref<!hal.fence>,
  %source_buffer : !vm.buffer,
  %source_offset : i64,
  %target_buffer : !vm.ref<!hal.buffer>,
  %target_offset : i64,
  %length : i64,
  %flags : i64
)

// Enqueues a single queue-ordered buffer copy operation.
vm.import private @device.queue.copy(
  %device : !vm.ref<!hal.device>,
  %queue_affinity : i64,
  %wait_fence : !vm.ref<!hal.fence>,
  %signal_fence : !vm.ref<!hal.fence>,
  %source_buffer : !vm.ref<!hal.buffer>,
  %source_offset : i64,
  %target_buffer : !vm.ref<!hal.buffer>,
  %target_offset : i64,
  %length : i64,
  %flags : i64
)

// Reads a segment from a file into a device buffer.
vm.import private @device.queue.read(
  %device : !vm.ref<!hal.device>,
  %queue_affinity : i64,
  %wait_fence : !vm.ref<!hal.fence>,
  %signal_fence : !vm.ref<!hal.fence>,
  %source_file : !vm.ref<!hal.file>,
  %source_offset : i64,
  %target_buffer : !vm.ref<!hal.buffer>,
  %target_offset : i64,
  %length : i64,
  %flags : i64
)

// Writes a segment from device buffer into a file.
vm.import private @device.queue.write(
  %device : !vm.ref<!hal.device>,
  %queue_affinity : i64,
  %wait_fence : !vm.ref<!hal.fence>,
  %signal_fence : !vm.ref<!hal.fence>,
  %source_buffer : !vm.ref<!hal.buffer>,
  %source_offset : i64,
  %target_file : !vm.ref<!hal.file>,
  %target_offset : i64,
  %length : i64,
  %flags : i64
)

// Signals the provided fence once the wait fence is reached.
vm.import private @device.queue.barrier(
  %device : !vm.ref<!hal.device>,
  %queue_affinity : i64,
  %wait_fence : !vm.ref<!hal.fence>,
  %signal_fence : !vm.ref<!hal.fence>,
  %flags : i64
)

// Executes a command buffer on a device queue.
// No commands will execute until the wait fence has been reached and the signal
// fence will be signaled when all commands have completed.
vm.import private @device.queue.execute(
  %device : !vm.ref<!hal.device>,
  %queue_affinity : i64,
  %wait_fence : !vm.ref<!hal.fence>,
  %signal_fence : !vm.ref<!hal.fence>,
  %command_buffer : !vm.ref<!hal.command_buffer>,
  %flags : i64
)

// Executes a command buffer on a device queue with the given binding table.
// No commands will execute until the wait fence has been reached and the signal
// fence will be signaled when all commands have completed.
vm.import private @device.queue.execute.indirect(
  %device : !vm.ref<!hal.device>,
  %queue_affinity : i64,
  %wait_fence : !vm.ref<!hal.fence>,
  %signal_fence : !vm.ref<!hal.fence>,
  %command_buffer : !vm.ref<!hal.command_buffer>,
  %flags : i64,
  // <buffer, offset, length>
  %binding_table : tuple<!vm.ref<!hal.buffer>, i64, i64>...
)

// Flushes any locally-pending submissions in the queue.
// When submitting many queue operations this can be used to eagerly flush
// earlier submissions while later ones are still being constructed.
vm.import private @device.queue.flush(
  %device : !vm.ref<!hal.device>,
  %queue_affinity : i64
)

//===----------------------------------------------------------------------===//
// iree_hal_device_t management
//===----------------------------------------------------------------------===//

vm.import private @devices.count() -> i32
attributes {nosideeffects}

vm.import private @devices.get(%index : i32) -> !vm.ref<!hal.device>
attributes {nosideeffects}

//===----------------------------------------------------------------------===//
// iree_hal_executable_t
//===----------------------------------------------------------------------===//

// Creates an executable for use with the specified device.
vm.import private @executable.create(
  %device : !vm.ref<!hal.device>,
  %queue_affinity : i64,
  %executable_format : !vm.buffer,
  %executable_data : !vm.buffer,
  %constants : !vm.buffer
) -> !vm.ref<!hal.executable>
attributes {nosideeffects}

//===----------------------------------------------------------------------===//
// iree_hal_fence_t
//===----------------------------------------------------------------------===//

// Returns an unsignaled fence that defines a point in time.
vm.import private @fence.create(
  %device : !vm.ref<!hal.device>,
  %flags : i64
) -> !vm.ref<!hal.fence>

// Returns a fence that joins the input fences as a wait-all operation.
vm.import private @fence.join(
  %flags : i64,
  %fences : !vm.ref<!hal.fence> ...
) -> !vm.ref<!hal.fence>
attributes {nosideeffects}

// Queries whether the fence has been reached and returns its status.
// Returns OK if the fence has been signaled successfully, DEFERRED if it is
// unsignaled, and otherwise an error indicating the failure.
vm.import private @fence.query(
  %fence : !vm.ref<!hal.fence>
) -> i32

// Signals the fence.
vm.import private @fence.signal(
  %fence : !vm.ref<!hal.fence>
)

// Signals the fence with a failure. The |status| will be returned from
// `hal.fence.query` and `hal.fence.await`.
vm.import private @fence.fail(
  %fence : !vm.ref<!hal.fence>,
  %status : i32
)

// Yields the caller until all fences are reached.
vm.import private @fence.await(
  %timeout_millis : i32,
  %flags : i64,
  %fences : !vm.ref<!hal.fence> ...
) -> i32
attributes {vm.yield}

}  // module
