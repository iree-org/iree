vm.module @io_parameters {

vm.import private @load(
  %device : !vm.ref<!hal.device>,
  %queue_affinity : i64,
  %wait_fence : !vm.ref<!hal.fence>,
  %signal_fence : !vm.ref<!hal.fence>,
  %source_scope : !vm.buffer,
  %source_key : !vm.buffer,
  %source_offset : i64,
  %target_queue_affinity : i64,
  %target_memory_type : i32,
  %target_buffer_usage : i32,
  %length : i64
) -> !vm.ref<!hal.buffer>

vm.import private @read(
  %device : !vm.ref<!hal.device>,
  %queue_affinity : i64,
  %wait_fence : !vm.ref<!hal.fence>,
  %signal_fence : !vm.ref<!hal.fence>,
  %source_scope : !vm.buffer,
  %source_key : !vm.buffer,
  %source_offset : i64,
  %target_buffer : !vm.ref<!hal.buffer>,
  %target_offset : i64,
  %length : i64
)

vm.import private @write(
  %device : !vm.ref<!hal.device>,
  %queue_affinity : i64,
  %wait_fence : !vm.ref<!hal.fence>,
  %signal_fence : !vm.ref<!hal.fence>,
  %target_scope : !vm.buffer,
  %target_key : !vm.buffer,
  %target_offset : i64,
  %source_buffer : !vm.ref<!hal.buffer>,
  %source_offset : i64,
  %length : i64
)

vm.import private @gather(
  %device : !vm.ref<!hal.device>,
  %queue_affinity : i64,
  %wait_fence : !vm.ref<!hal.fence>,
  %signal_fence : !vm.ref<!hal.fence>,
  %source_scope : !vm.buffer,
  %target_buffer : !vm.ref<!hal.buffer>,
  %key_table : !vm.buffer,
  %key_data : !vm.buffer,
  %spans : !vm.buffer
)

vm.import private @scatter(
  %device : !vm.ref<!hal.device>,
  %queue_affinity : i64,
  %wait_fence : !vm.ref<!hal.fence>,
  %signal_fence : !vm.ref<!hal.fence>,
  %source_buffer : !vm.ref<!hal.buffer>,
  %target_scope : !vm.buffer,
  %key_table : !vm.buffer,
  %key_data : !vm.buffer,
  %spans : !vm.buffer
)

}  // module
