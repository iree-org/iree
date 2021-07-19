[[block]] struct Params {
  offset : u32;
  length : u32;
  pattern : u32;
};
[[group(0), binding(0)]] var<uniform> params: Params;

[[block]] struct OpaqueBuffer {
  data : array<u32>;
};
[[group(1), binding(0)]] var<storage, read_write> buffer: OpaqueBuffer;

[[stage(compute), workgroup_size(64, 1, 1)]]
fn main([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {
  var element_index : u32 = GlobalInvocationID.x;
  var element_range_start : u32 = element_index * 64u;
  var element_range_end : u32 = min(element_range_start + 64u, params.length / 4u);
  for (var i : u32 = element_range_start; i < element_range_end; i = i + 1u) {
    buffer.data[i] = params.pattern;
  }
}
