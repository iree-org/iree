// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// JS companion for the iree_webgpu wasm import module.
//
// Provides the ~27 functions imported by the C WebGPU HAL driver under
// __attribute__((import_module("iree_webgpu"))). Used by the wasm binary
// bundler: the bundler inlines this file, wraps it in an IIFE, and calls
// createImports(context) with shared state from the host.
//
// The handle table maps uint32_t handles to JS WebGPU objects. Handle 0 is
// reserved (null). Lookup is array index. Allocation is pop from free stack
// or bump high water. Deallocation pushes to the free stack for reuse.
//
// Async operations (request_adapter, adapter_request_device, buffer_map_async,
// queue_on_submitted_work_done) deliver completions through the proactor's
// completion ring. Each takes a uint32_t token that the proactor allocated.
// When the Promise resolves, JS writes {token, status_code} to the ring.
//
// Required context fields:
//   memory    — WebAssembly.Memory (set after instantiation).
//   complete  — function(token, statusCode) to write a completion entry to the
//               proactor ring. Typically context.proactorImports.complete().
//   gpu       — The navigator.gpu object (browser) or dawn.create() (Node.js).
//               Used only for request_adapter.

// Handle table. Mirrors the C-side iree_hal_webgpu_handle_table_t.
class HandleTable {
  constructor() {
    this._entries = [null];  // Index 0 reserved (null handle).
    this._freeStack = [];
  }

  insert(object) {
    if (this._freeStack.length > 0) {
      const index = this._freeStack.pop();
      this._entries[index] = object;
      return index;
    }
    this._entries.push(object);
    return this._entries.length - 1;
  }

  get(handle) {
    return this._entries[handle] || null;
  }

  remove(handle) {
    if (!handle) return null;  // Handle 0 is reserved (null handle).
    const object = this._entries[handle];
    this._entries[handle] = null;
    this._freeStack.push(handle);
    return object;
  }
}

// GPUMapMode constants.
const MAP_READ = 1;
const MAP_WRITE = 2;

// iree_status_code_t values used in completions.
const STATUS_OK = 0;
const STATUS_CANCELLED = 1;
const STATUS_INTERNAL = 13;
const STATUS_UNAVAILABLE = 14;

// ISA opcode constants — must match webgpu_isa.h.
const OP_UPDATE_BUFFER = 0x01;
const OP_ENCODER_BEGIN = 0x10;
const OP_ENCODER_END = 0x11;
const OP_COPY_BUFFER = 0x12;
const OP_FILL_BUFFER = 0x13;
const OP_DISPATCH = 0x15;
const OP_BARRIER = 0x20;
const OP_RETURN = 0xFF;

// Workgroup size used by builtin fill/copy shaders — must match WGSL source.
const BUILTIN_WORKGROUP_SIZE = 256;

// Creates the iree_webgpu import implementations.
export function createImports(context) {
  const handles = new HandleTable();

  // Expose handle insertion so the JS host can pre-register WebGPU objects
  // (GPUDevice, GPUAdapter) before wasm starts. This is the production API
  // for iree_hal_webgpu_device_wrap() — the host inserts a GPUDevice and
  // passes the handle ID to the C wrap function.
  context.insertWebGPUHandle = (object) => handles.insert(object);

  // Cached typed array views over wasm linear memory. Rebuilt lazily when
  // memory.grow() detaches the underlying ArrayBuffer.
  let _cachedBuffer = null;
  let _u8 = null;
  let _u32 = null;
  function refreshViews() {
    if (_cachedBuffer !== context.memory.buffer) {
      _cachedBuffer = context.memory.buffer;
      _u8 = new Uint8Array(_cachedBuffer);
      _u32 = new Uint32Array(_cachedBuffer);
    }
  }
  function u8() {
    refreshViews();
    return _u8;
  }
  function u32() {
    refreshViews();
    return _u32;
  }

  // Helper: deliver a proactor completion.
  function complete(token, statusCode) {
    context.complete(token, statusCode);
  }

  // Helper: read a UTF-8 string from wasm linear memory.
  function readString(ptr, length) {
    return new TextDecoder().decode(u8().subarray(ptr, ptr + length));
  }

  //============================================================================
  // Instruction stream processor
  //============================================================================

  // Processes an instruction stream against a binding table (buffers +
  // offsets).
  //
  // |device|: GPUDevice.
  // |queue|: GPUQueue.
  // |text|: Uint32Array of instruction words.
  // |buffers|: Array of GPUBuffer objects, indexed by slot.
  // |offsets|: Uint32Array of base offsets per slot.
  // |builtins|: {fillPipeline, fillBGL, copyPipeline, copyBGL}.
  function processInstructions(
      device, queue, text, buffers, offsets, builtins) {
    let cursor = 0;
    let encoder = null;
    const pendingCommandBuffers = [];

    while (cursor < text.length) {
      const header = text[cursor];
      const opcode = header & 0xFF;
      const sizeWords = header >>> 16;

      switch (opcode) {
        case OP_ENCODER_BEGIN: {
          encoder = device.createCommandEncoder();
          break;
        }

        case OP_ENCODER_END: {
          pendingCommandBuffers.push(encoder.finish());
          encoder = null;
          break;
        }

        case OP_UPDATE_BUFFER: {
          const dstSlot = text[cursor + 1];
          const dstOffset = text[cursor + 2];
          const length = text[cursor + 3];
          const effectiveOffset = offsets[dstSlot] + dstOffset;

          // Flush any pending encoder work before the queue operation.
          if (pendingCommandBuffers.length > 0) {
            queue.submit(pendingCommandBuffers);
            pendingCommandBuffers.length = 0;
          }

          // The inline data starts at cursor + 4. Copy it to a temporary
          // Uint8Array for writeBuffer. The data may not be uint32-aligned in
          // the instruction stream when the original length isn't a multiple
          // of 4 (it's zero-padded to a uint32 boundary).
          const dataWords =
              text.subarray(cursor + 4, cursor + 4 + Math.ceil(length / 4));
          const dataBytes =
              new Uint8Array(dataWords.buffer, dataWords.byteOffset, length);

          // queue.writeBuffer requires a 4-byte aligned buffer offset. When
          // unaligned, use a staging buffer + copy compute shader.
          if ((effectiveOffset & 3) === 0) {
            queue.writeBuffer(buffers[dstSlot], effectiveOffset, dataBytes);
          } else {
            // Staging buffer with the inline data.
            const stagingSize = Math.ceil(length / 4) * 4;
            const staging = device.createBuffer({
              usage: 0x0080 | 0x0004,  // STORAGE | COPY_SRC
              size: stagingSize,
              mappedAtCreation: true,
            });
            new Uint8Array(staging.getMappedRange()).set(dataBytes);
            staging.unmap();

            // Params: {src_offset, dst_offset, length, pad}.
            const params = device.createBuffer({
              usage: 0x0040 | 0x0008,  // UNIFORM | COPY_DST
              size: 16,
              mappedAtCreation: true,
            });
            new Uint32Array(params.getMappedRange()).set([
              0, effectiveOffset, length, 0
            ]);
            params.unmap();

            const bindGroup = device.createBindGroup({
              layout: builtins.copyBGL,
              entries: [
                {binding: 0, resource: {buffer: staging, size: stagingSize}},
                {binding: 1, resource: {buffer: buffers[dstSlot]}},
                {binding: 2, resource: {buffer: params, size: 16}},
              ],
            });

            const alignedEnd = (effectiveOffset + length + 3) & ~3;
            const wordCount = (alignedEnd - (effectiveOffset & ~3)) / 4;
            const workgroups = Math.ceil(wordCount / BUILTIN_WORKGROUP_SIZE);

            const enc = device.createCommandEncoder();
            const pass = enc.beginComputePass();
            pass.setPipeline(builtins.copyPipeline);
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(workgroups);
            pass.end();
            queue.submit([enc.finish()]);
            staging.destroy();
            params.destroy();
          }
          break;
        }

        case OP_COPY_BUFFER: {
          const srcSlot = text[cursor + 1];
          const srcOffset = text[cursor + 2];
          const dstSlot = text[cursor + 3];
          const dstOffset = text[cursor + 4];
          const length = text[cursor + 5];
          const effectiveSrc = offsets[srcSlot] + srcOffset;
          const effectiveDst = offsets[dstSlot] + dstOffset;

          if ((effectiveSrc & 3) === 0 && (effectiveDst & 3) === 0 &&
              (length & 3) === 0) {
            // Aligned: use native copyBufferToBuffer.
            encoder.copyBufferToBuffer(
                buffers[srcSlot], effectiveSrc, buffers[dstSlot], effectiveDst,
                length);
          } else {
            // Unaligned: dispatch copy builtin shader.
            const params = device.createBuffer({
              usage: 0x0040 | 0x0008,  // UNIFORM | COPY_DST
              size: 16,
              mappedAtCreation: true,
            });
            new Uint32Array(params.getMappedRange()).set([
              effectiveSrc, effectiveDst, length, 0
            ]);
            params.unmap();

            const bindGroup = device.createBindGroup({
              layout: builtins.copyBGL,
              entries: [
                {binding: 0, resource: {buffer: buffers[srcSlot]}},
                {binding: 1, resource: {buffer: buffers[dstSlot]}},
                {binding: 2, resource: {buffer: params, size: 16}},
              ],
            });

            const alignedEnd = (effectiveDst + length + 3) & ~3;
            const wordCount = (alignedEnd - (effectiveDst & ~3)) / 4;
            const workgroups = Math.ceil(wordCount / BUILTIN_WORKGROUP_SIZE);

            const pass = encoder.beginComputePass();
            pass.setPipeline(builtins.copyPipeline);
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(workgroups);
            pass.end();
            params.destroy();
          }
          break;
        }

        case OP_FILL_BUFFER: {
          const dstSlot = text[cursor + 1];
          const dstOffset = text[cursor + 2];
          const length = text[cursor + 3];
          const pattern = text[cursor + 4];
          // text[cursor + 5] is pattern_length — used by the WGSL shader
          // internally for edge words, but the params buffer receives the
          // pre-replicated uint32 pattern.
          const effectiveOffset = offsets[dstSlot] + dstOffset;

          const params = device.createBuffer({
            usage: 0x0040 | 0x0008,  // UNIFORM | COPY_DST
            size: 16,
            mappedAtCreation: true,
          });
          new Uint32Array(params.getMappedRange()).set([
            effectiveOffset, length, pattern, 0
          ]);
          params.unmap();

          const bindGroup = device.createBindGroup({
            layout: builtins.fillBGL,
            entries: [
              {binding: 0, resource: {buffer: buffers[dstSlot]}},
              {binding: 1, resource: {buffer: params, size: 16}},
            ],
          });

          const alignedStart = effectiveOffset & ~3;
          const alignedEnd = (effectiveOffset + length + 3) & ~3;
          const wordCount = (alignedEnd - alignedStart) / 4;
          const workgroups = Math.ceil(wordCount / BUILTIN_WORKGROUP_SIZE);

          const pass = encoder.beginComputePass();
          pass.setPipeline(builtins.fillPipeline);
          pass.setBindGroup(0, bindGroup);
          pass.dispatchWorkgroups(workgroups);
          pass.end();
          params.destroy();
          break;
        }

        case OP_DISPATCH: {
          const pipelineHandle = text[cursor + 1];
          const bglHandle = text[cursor + 2];
          const workgroupX = text[cursor + 3];
          const workgroupY = text[cursor + 4];
          const workgroupZ = text[cursor + 5];

          const pipeline = handles.get(pipelineHandle);
          const bgl = handles.get(bglHandle);
          const bindingCount = (sizeWords - 6) / 3;

          const entries = [];
          for (let i = 0; i < bindingCount; i++) {
            const base = cursor + 6 + i * 3;
            const slot = text[base];
            const offset = text[base + 1];
            const size = text[base + 2];
            const effectiveOffset = offsets[slot] + offset;
            entries.push({
              binding: i,
              resource: {buffer: buffers[slot], offset: effectiveOffset, size},
            });
          }

          const bindGroup = device.createBindGroup({layout: bgl, entries});

          const pass = encoder.beginComputePass();
          pass.setPipeline(pipeline);
          pass.setBindGroup(0, bindGroup);
          pass.dispatchWorkgroups(workgroupX, workgroupY, workgroupZ);
          pass.end();
          break;
        }

        case OP_BARRIER: {
          // No-op in WebGPU — ordering is implicit within a single queue.
          break;
        }

        case OP_RETURN: {
          // Flush any remaining pending command buffers.
          if (pendingCommandBuffers.length > 0) {
            queue.submit(pendingCommandBuffers);
            pendingCommandBuffers.length = 0;
          }
          return;
        }

        default: {
          throw new Error(`Unknown WebGPU ISA opcode 0x${
              opcode.toString(16)} at word ${cursor}`);
        }
      }

      cursor += sizeWords;
    }
  }

  // Reads a builtins descriptor from wasm memory (4 uint32 handles).
  function readBuiltins(builtinsPtr) {
    const view = u32();
    const base = builtinsPtr >> 2;
    return {
      fillPipeline: handles.get(view[base]),
      fillBGL: handles.get(view[base + 1]),
      copyPipeline: handles.get(view[base + 2]),
      copyBGL: handles.get(view[base + 3]),
    };
  }

  // Reads instruction blocks zero-copy from wasm memory. The block table is an
  // array of uint32_t* pointers (stored as uint32 wasm pointers). For a single
  // block (the common case), returns a direct subarray view. For multiple
  // blocks, concatenates into a temporary Uint32Array.
  function readBlocksZeroCopy(
      view, blockTablePtr, blockCount, blockWordCapacity, lastBlockWordCount) {
    const tableBase = blockTablePtr >> 2;
    if (blockCount === 1) {
      const blockPtr = view[tableBase];
      return view.subarray(blockPtr >> 2, (blockPtr >> 2) + lastBlockWordCount);
    }
    // Multi-block: concatenate into a flat array.
    return readBlocksCopy(
        view, blockTablePtr, blockCount, blockWordCapacity, lastBlockWordCount);
  }

  // Copies instruction blocks from wasm memory into a single contiguous
  // Uint32Array. Used by create_recording (the recording outlives the blocks)
  // and as the multi-block fallback for execute_instructions.
  function readBlocksCopy(
      view, blockTablePtr, blockCount, blockWordCapacity, lastBlockWordCount) {
    const tableBase = blockTablePtr >> 2;
    const totalWords =
        (blockCount - 1) * blockWordCapacity + lastBlockWordCount;
    const result = new Uint32Array(totalWords);
    let destOffset = 0;
    for (let i = 0; i < blockCount; i++) {
      const blockPtr = view[tableBase + i];
      const blockBase = blockPtr >> 2;
      const wordCount =
          (i < blockCount - 1) ? blockWordCapacity : lastBlockWordCount;
      result.set(view.subarray(blockBase, blockBase + wordCount), destOffset);
      destOffset += wordCount;
    }
    return result;
  }

  // Reads a binding table from wasm memory. Each entry is {handle, base_offset}
  // (2 uint32 words). Returns {buffers: Array[GPUBuffer], offsets:
  // Uint32Array}.
  function readBindingTable(ptr, count) {
    const view = u32();
    const base = ptr >> 2;
    const result_buffers = new Array(count);
    const result_offsets = new Uint32Array(count);
    for (let i = 0; i < count; i++) {
      result_buffers[i] = handles.get(view[base + i * 2]);
      result_offsets[i] = view[base + i * 2 + 1];
    }
    return {buffers: result_buffers, offsets: result_offsets};
  }

  //============================================================================
  // Recording (cached reusable command buffer)
  //============================================================================

  // A recording caches the instruction stream, static bindings, and builtin
  // references for a reusable command buffer. Only dynamic bindings are
  // resolved per issue.
  class Recording {
    constructor(device, text, staticBindings, dynamicCount, builtins) {
      this.device = device;
      this.text = text;  // Uint32Array (copied from wasm).
      this.dynamicCount = dynamicCount;
      this.totalSlots = dynamicCount + staticBindings.buffers.length;
      this.builtins = builtins;

      // SoA binding cache.
      this.buffers = new Array(this.totalSlots).fill(null);
      this.offsets = new Uint32Array(this.totalSlots);

      // Resolve static bindings once.
      for (let i = 0; i < staticBindings.buffers.length; i++) {
        const slot = dynamicCount + i;
        this.buffers[slot] = staticBindings.buffers[i];
        this.offsets[slot] = staticBindings.offsets[i];
      }
    }

    execute(queue, dynamicBindingTablePtr) {
      // Resolve dynamic bindings from wasm memory.
      if (this.dynamicCount > 0) {
        const view = u32();
        const base = dynamicBindingTablePtr >> 2;
        for (let i = 0; i < this.dynamicCount; i++) {
          this.buffers[i] = handles.get(view[base + i * 2]);
          this.offsets[i] = view[base + i * 2 + 1];
        }
      }

      // Execute the cached instruction stream.
      processInstructions(
          this.device, queue, this.text, this.buffers, this.offsets,
          this.builtins);

      // Release dynamic buffer references immediately after submission.
      for (let i = 0; i < this.dynamicCount; i++) {
        this.buffers[i] = null;
        this.offsets[i] = 0;
      }
    }
  }

  return {
    //==========================================================================
    // Device / Adapter
    //==========================================================================

    request_adapter(optionsFlags, outAdapterHandlePtr, token) {
      const gpu = context.gpu;
      if (!gpu) {
        complete(token, STATUS_UNAVAILABLE);
        return;
      }
      gpu.requestAdapter().then(
          (adapter) => {
            if (!adapter) {
              complete(token, STATUS_UNAVAILABLE);
              return;
            }
            const handle = handles.insert(adapter);
            u32()[outAdapterHandlePtr >> 2] = handle;
            complete(token, STATUS_OK);
          },
          () => complete(token, STATUS_UNAVAILABLE),
      );
    },

    adapter_request_device(adapterHandle, outDeviceHandlePtr, token) {
      const adapter = handles.get(adapterHandle);
      if (!adapter) {
        complete(token, STATUS_INTERNAL);
        return;
      }
      adapter.requestDevice().then(
          (device) => {
            if (!device) {
              complete(token, STATUS_INTERNAL);
              return;
            }
            const handle = handles.insert(device);
            u32()[outDeviceHandlePtr >> 2] = handle;
            complete(token, STATUS_OK);
          },
          () => complete(token, STATUS_INTERNAL),
      );
    },

    device_destroy(deviceHandle) {
      const device = handles.remove(deviceHandle);
      if (device) device.destroy();
    },

    device_get_queue(deviceHandle) {
      const device = handles.get(deviceHandle);
      if (!device) return 0;
      // Check if we already stored the queue handle for this device.
      // If not, insert the queue into the handle table.
      if (!device._ireeQueueHandle) {
        device._ireeQueueHandle = handles.insert(device.queue);
      }
      return device._ireeQueueHandle;
    },

    //==========================================================================
    // Buffers
    //==========================================================================

    device_create_buffer(deviceHandle, usage, size, mappedAtCreation) {
      const device = handles.get(deviceHandle);
      if (!device) return 0;
      const buffer = device.createBuffer({
        usage,
        size: Number(size),
        mappedAtCreation: mappedAtCreation !== 0,
      });
      return handles.insert(buffer);
    },

    buffer_destroy(bufferHandle) {
      const buffer = handles.remove(bufferHandle);
      if (buffer) buffer.destroy();
    },

    buffer_map_async(bufferHandle, mode, offset, size, token) {
      const buffer = handles.get(bufferHandle);
      if (!buffer) {
        complete(token, STATUS_INTERNAL);
        return;
      }
      const mapMode = mode === MAP_READ ? GPUMapMode.READ : GPUMapMode.WRITE;
      buffer.mapAsync(mapMode, Number(offset), Number(size))
          .then(
              () => complete(token, STATUS_OK),
              () => complete(token, STATUS_INTERNAL),
          );
    },

    buffer_get_mapped_range(bufferHandle, offset, size, destPtr) {
      const buffer = handles.get(bufferHandle);
      if (!buffer) return;
      const mapped = new Uint8Array(
          buffer.getMappedRange(Number(offset), Number(size)),
      );
      u8().set(mapped, destPtr);
    },

    buffer_set_mapped_range(bufferHandle, offset, size, srcPtr) {
      const buffer = handles.get(bufferHandle);
      if (!buffer) return;
      const mapped = new Uint8Array(
          buffer.getMappedRange(Number(offset), Number(size)),
      );
      mapped.set(u8().subarray(srcPtr, srcPtr + Number(size)));
    },

    buffer_unmap(bufferHandle) {
      const buffer = handles.get(bufferHandle);
      if (buffer) buffer.unmap();
    },

    //==========================================================================
    // Command Encoding
    //==========================================================================

    device_create_command_encoder(deviceHandle) {
      const device = handles.get(deviceHandle);
      if (!device) return 0;
      return handles.insert(device.createCommandEncoder());
    },

    encoder_begin_compute_pass(encoderHandle) {
      const encoder = handles.get(encoderHandle);
      if (!encoder) return 0;
      return handles.insert(encoder.beginComputePass());
    },

    pass_set_pipeline(passHandle, pipelineHandle) {
      const pass = handles.get(passHandle);
      const pipeline = handles.get(pipelineHandle);
      if (pass && pipeline) pass.setPipeline(pipeline);
    },

    pass_set_bind_group(
        passHandle, index, bindGroupHandle, dynamicOffsetsPtr,
        dynamicOffsetsCount) {
      const pass = handles.get(passHandle);
      const bindGroup = handles.get(bindGroupHandle);
      if (!pass || !bindGroup) return;
      if (dynamicOffsetsCount > 0) {
        const base = dynamicOffsetsPtr >> 2;
        pass.setBindGroup(
            index, bindGroup, u32().subarray(base, base + dynamicOffsetsCount));
      } else {
        pass.setBindGroup(index, bindGroup);
      }
    },

    pass_dispatch_workgroups(passHandle, x, y, z) {
      const pass = handles.get(passHandle);
      if (pass) pass.dispatchWorkgroups(x, y, z);
    },

    pass_end(passHandle) {
      const pass = handles.remove(passHandle);
      if (pass) pass.end();
    },

    encoder_copy_buffer_to_buffer(
        encoderHandle, srcHandle, srcOffset, dstHandle, dstOffset, size) {
      const encoder = handles.get(encoderHandle);
      const src = handles.get(srcHandle);
      const dst = handles.get(dstHandle);
      if (encoder && src && dst) {
        encoder.copyBufferToBuffer(
            src,
            Number(srcOffset),
            dst,
            Number(dstOffset),
            Number(size),
        );
      }
    },

    encoder_finish(encoderHandle) {
      const encoder = handles.remove(encoderHandle);
      if (!encoder) return 0;
      return handles.insert(encoder.finish());
    },

    //==========================================================================
    // Pipeline / Bind Group
    //==========================================================================

    device_create_compute_pipeline(
        deviceHandle, layoutHandle, wgslPtr, wgslLength, entryPointPtr,
        entryPointLength) {
      const device = handles.get(deviceHandle);
      if (!device) return 0;
      const code = readString(wgslPtr, wgslLength);
      const entryPoint = readString(entryPointPtr, entryPointLength);
      const module = device.createShaderModule({code});
      const layout = layoutHandle === 0 ? 'auto' : handles.get(layoutHandle);
      const pipeline = device.createComputePipeline({
        layout,
        compute: {module, entryPoint},
      });
      return handles.insert(pipeline);
    },

    device_create_pipeline_layout(deviceHandle, layoutsPtr, layoutCount) {
      const device = handles.get(deviceHandle);
      if (!device) return 0;
      const view = u32();
      const base = layoutsPtr >> 2;
      const layouts = [];
      for (let i = 0; i < layoutCount; i++) {
        layouts.push(handles.get(view[base + i]));
      }
      const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: layouts,
      });
      return handles.insert(pipelineLayout);
    },

    device_create_bind_group_layout(deviceHandle, entriesPtr, entryCount) {
      const device = handles.get(deviceHandle);
      if (!device) return 0;

      // Each entry in wasm memory:
      //   uint32_t binding
      //   uint32_t visibility (GPUShaderStage flags)
      //   uint32_t buffer_type (0=uniform, 1=storage, 2=read-only-storage)
      //   uint32_t has_dynamic_offset
      //   uint64_t min_binding_size (as two uint32s: lo, hi)
      const ENTRY_UINT32S = 6;
      const view = u32();
      const viewBase = entriesPtr >> 2;

      const bufferTypes = ['uniform', 'storage', 'read-only-storage'];
      const entries = [];
      for (let i = 0; i < entryCount; i++) {
        const base = viewBase + i * ENTRY_UINT32S;
        const binding = view[base];
        const visibility = view[base + 1];
        const bufferType = bufferTypes[view[base + 2]] || 'storage';
        const hasDynamicOffset = view[base + 3] !== 0;
        const minBindingSize = view[base + 4] + view[base + 5] * 0x100000000;
        entries.push({
          binding,
          visibility,
          buffer: {type: bufferType, hasDynamicOffset, minBindingSize},
        });
      }

      const layout = device.createBindGroupLayout({entries});
      return handles.insert(layout);
    },

    device_create_bind_group(
        deviceHandle, layoutHandle, entriesPtr, entryCount) {
      const device = handles.get(deviceHandle);
      const layout = handles.get(layoutHandle);
      if (!device || !layout) return 0;

      // Each entry in wasm memory:
      //   uint32_t binding
      //   uint32_t buffer_handle
      //   uint64_t offset (as two uint32s: lo, hi)
      //   uint64_t size (as two uint32s: lo, hi)
      const ENTRY_UINT32S = 6;
      const view = u32();
      const viewBase = entriesPtr >> 2;

      const entries = [];
      for (let i = 0; i < entryCount; i++) {
        const base = viewBase + i * ENTRY_UINT32S;
        const binding = view[base];
        const buffer = handles.get(view[base + 1]);
        const offset = view[base + 2] + view[base + 3] * 0x100000000;
        const size = view[base + 4] + view[base + 5] * 0x100000000;
        entries.push({binding, resource: {buffer, offset, size}});
      }

      const bindGroup = device.createBindGroup({layout, entries});
      return handles.insert(bindGroup);
    },

    //==========================================================================
    // Instruction stream execution
    //==========================================================================

    execute_instructions(
        deviceHandle, queueHandle, blockTablePtr, blockCount, blockWordCapacity,
        lastBlockWordCount, bindingTablePtr, bindingCount, builtinsPtr) {
      const device = handles.get(deviceHandle);
      const queue = handles.get(queueHandle);
      if (!device || !queue) return 1;

      // Read instruction stream zero-copy from wasm memory blocks.
      // processInstructions runs synchronously — C can't mutate wasm memory.
      const view = u32();
      const text = readBlocksZeroCopy(
          view, blockTablePtr, blockCount, blockWordCapacity,
          lastBlockWordCount);

      // Read binding table.
      const {buffers, offsets} =
          readBindingTable(bindingTablePtr, bindingCount);

      // Read builtins.
      const builtins = readBuiltins(builtinsPtr);

      processInstructions(device, queue, text, buffers, offsets, builtins);
      return 0;
    },

    create_recording(
        deviceHandle, blockTablePtr, blockCount, blockWordCapacity,
        lastBlockWordCount, staticBindingTablePtr, staticBindingCount,
        dynamicBindingCount, builtinsPtr) {
      const device = handles.get(deviceHandle);
      if (!device) return 0;

      // Copy instruction stream blocks into a single JS-owned Uint32Array.
      // The recording outlives the builder blocks.
      const view = u32();
      const text = readBlocksCopy(
          view, blockTablePtr, blockCount, blockWordCapacity,
          lastBlockWordCount);

      // Read static bindings.
      const staticBindings =
          readBindingTable(staticBindingTablePtr, staticBindingCount);

      // Read builtins.
      const builtins = readBuiltins(builtinsPtr);

      const recording = new Recording(
          device, text, staticBindings, dynamicBindingCount, builtins);
      return handles.insert(recording);
    },

    execute_recording(recordingHandle, queueHandle, dynamicBindingTablePtr) {
      const recording = handles.get(recordingHandle);
      const queue = handles.get(queueHandle);
      if (!recording || !queue) return 1;
      recording.execute(queue, dynamicBindingTablePtr);
      return 0;
    },

    //==========================================================================
    // Queue
    //==========================================================================

    queue_submit(queueHandle, commandBufferHandle) {
      const queue = handles.get(queueHandle);
      const commandBuffer = handles.remove(commandBufferHandle);
      if (queue && commandBuffer) {
        queue.submit([commandBuffer]);
      }
    },

    queue_on_submitted_work_done(queueHandle, token) {
      const queue = handles.get(queueHandle);
      if (!queue) {
        complete(token, STATUS_INTERNAL);
        return;
      }
      queue.onSubmittedWorkDone().then(
          () => complete(token, STATUS_OK),
          () => complete(token, STATUS_INTERNAL),
      );
    },

    queue_write_buffer(
        queueHandle, bufferHandle, bufferOffset, dataPtr, dataSize) {
      const queue = handles.get(queueHandle);
      const buffer = handles.get(bufferHandle);
      if (!queue || !buffer) return;
      const size = Number(dataSize);
      queue.writeBuffer(
          buffer, Number(bufferOffset), u8().subarray(dataPtr, dataPtr + size));
    },

    //==========================================================================
    // File ↔ GPU transfer
    //==========================================================================

    // Writes data from a file object directly to a GPU buffer, bypassing
    // wasm linear memory. The file object is accessed via context.fileObjects
    // (set up by the iree_file import module's file_imports.mjs).
    queue_write_buffer_from_file(
        queueHandle, bufferHandle, bufferOffset, fd, fileOffset, dataSize) {
      const queue = handles.get(queueHandle);
      const buffer = handles.get(bufferHandle);
      const fileObject = context.fileObjects?.get(fd);
      if (!queue || !buffer || !fileObject) return;
      const data = new Uint8Array(
          fileObject,
          Number(fileOffset),
          Number(dataSize),
      );
      queue.writeBuffer(buffer, Number(bufferOffset), data);
    },

    // Returns the byte length of a file object. Returns 0n (BigInt) if the
    // fd is invalid or the file object table is unavailable.
    file_get_length(fd) {
      const fileObject = context.fileObjects?.get(fd);
      if (!fileObject) return 0n;
      return BigInt(fileObject.byteLength);
    },

    // Copies data from a mapped GPU buffer into an existing file object.
    // The buffer must be in mapped state. Reads from the mapped buffer range
    // and writes into the file object at the given offset.
    file_write_from_mapped(bufferHandle, bufferOffset, size, fd, fileOffset) {
      const buffer = handles.get(bufferHandle);
      const fileObject = context.fileObjects?.get(fd);
      if (!buffer || !fileObject) return;
      const mapped = buffer.getMappedRange(Number(bufferOffset), Number(size));
      const source = new Uint8Array(mapped);
      const target =
          new Uint8Array(fileObject, Number(fileOffset), Number(size));
      target.set(source);
    },

    // Exports the mapped range of a GPU buffer as a file object. The buffer
    // must be in mapped state. Copies the data into a new ArrayBuffer (the
    // mapped range is neutered when the buffer is unmapped). Returns the fd
    // (file object table index), or 0 if the file object table is unavailable.
    buffer_export_mapped_to_file(bufferHandle, offset, size) {
      const buffer = handles.get(bufferHandle);
      if (!buffer || !context.fileObjects) return 0;
      const mapped = buffer.getMappedRange(Number(offset), Number(size));
      // Copy — the mapped ArrayBuffer is neutered on unmap.
      const copy = new ArrayBuffer(Number(size));
      new Uint8Array(copy).set(new Uint8Array(mapped));
      return context.fileObjects.register(copy);
    },

    //==========================================================================
    // Queries
    //==========================================================================

    adapter_get_info(adapterHandle, destPtr) {
      const adapter = handles.get(adapterHandle);
      if (!adapter) return;
      // Write adapter info to wasm memory. Layout defined by C struct.
      const view = u32();
      const base = destPtr >> 2;
      const info = adapter.info || {};
      view[base] = info.vendor ? parseInt(info.vendor, 16) || 0 : 0;
      view[base + 1] = info.device ? parseInt(info.device, 16) || 0 : 0;
      view[base + 2] = 0;  // Reserved.
      view[base + 3] = 0;  // Reserved.
    },

    device_get_limits(deviceHandle, destPtr) {
      const device = handles.get(deviceHandle);
      if (!device) return;
      const limits = device.limits;
      const view = u32();
      const base = destPtr >> 2;
      view[base] = limits.maxBufferSize & 0xFFFFFFFF;
      view[base + 1] = Math.floor(limits.maxBufferSize / 0x100000000);
      view[base + 2] = limits.maxStorageBufferBindingSize & 0xFFFFFFFF;
      view[base + 3] =
          Math.floor(limits.maxStorageBufferBindingSize / 0x100000000);
      view[base + 4] = limits.maxComputeWorkgroupSizeX || 256;
      view[base + 5] = limits.maxComputeWorkgroupSizeY || 256;
      view[base + 6] = limits.maxComputeWorkgroupSizeZ || 64;
      view[base + 7] = limits.maxComputeWorkgroupsPerDimension || 65535;
    },

    //==========================================================================
    // Execution context queries
    //==========================================================================

    // Returns 1 if the current context supports blocking waits (Atomics.wait),
    // 0 otherwise. Blocking is available on Web Workers with cross-origin
    // isolation but never on the main thread.
    can_block() {
      // Atomics.wait throws TypeError on the main thread. The most reliable
      // detection: try a zero-timeout wait on a dummy buffer. If it throws,
      // we can't block.
      try {
        Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 0);
        return 1;
      } catch {
        return 0;
      }
    },

    // Returns a non-zero handle for a GPUDevice pre-configured by the JS host,
    // or 0 if none was provided. The host sets context.preConfiguredDevice
    // before wasm starts to support inline-mode device creation (browser main
    // thread, node.js with dawn, CTS).
    get_preconfigured_device() {
      return context.preConfiguredDevice || 0;
    },

    //==========================================================================
    // Resource cleanup
    //==========================================================================

    handle_release(handle) {
      handles.remove(handle);
    },
  };
}
