// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import assert from 'node:assert/strict';
import {test} from 'node:test';

import {createImports} from './webgpu_imports.mjs';

for (const [arch, base] of [['wasm32', 0x80000100], ['wasm64', 0x100000100]]) {
  test(`${arch} host pointers`, () => {
    const pointer = offset => arch === 'wasm64' ? BigInt(offset) : offset | 0;
    const context = {
      memory: {buffer: new ArrayBuffer(base + 256)},
      complete() {},
    };
    const imports = createImports(context);
    const view = new Uint32Array(context.memory.buffer);
    const buffer = {};
    const writes = [];
    const device = context.insertWebGPUHandle({
      limits: {
        maxBufferSize: 0x100000000,
        maxStorageBufferBindingSize: 0x20000000,
      },
    });
    const queue = context.insertWebGPUHandle({
      writeBuffer(target, offset, bytes) {
        assert.equal(target, buffer);
        writes.push([offset, [...bytes]]);
      },
    });
    const table = base;
    const firstBlock = base + 32;
    const secondBlock = base + 64;
    const bindings = base + 96;
    const builtins = base + 128;
    const output = base + 160;
    const pointerBytes = arch === 'wasm64' ? 8 : 4;
    for (const [index, block] of [firstBlock, secondBlock].entries()) {
      const word = (table + index * pointerBytes) / 4;
      view[word] = block >>> 0;
      if (arch === 'wasm64') view[word + 1] = Math.floor(block / 0x100000000);
    }
    view.set([context.insertWebGPUHandle(buffer), 4], bindings / 4);
    view.set([0x00050101, 0, 0, 4, 0x12345678], firstBlock / 4);
    view[secondBlock / 4] = 0x000100FF;

    imports.device_get_limits(device, pointer(output));
    assert.deepEqual(
        [...view.subarray(output / 4, output / 4 + 4)], [0, 1, 0x20000000, 0]);
    for (const blockCount of [1, 2]) {
      writes.length = 0;
      assert.equal(
          imports.execute_instructions(
              device, queue, pointer(table), blockCount, 5,
              blockCount === 1 ? 5 : 1, pointer(bindings), 1,
              pointer(builtins)),
          0);
      assert.deepEqual(writes, [[4, [0x78, 0x56, 0x34, 0x12]]]);
    }

    for (const misalignment of [1, 2, 3]) {
      assert.throws(
          () =>
              imports.device_get_limits(device, pointer(output + misalignment)),
          RangeError);
      view[table / 4] = (firstBlock + misalignment) >>> 0;
      assert.throws(
          () => imports.execute_instructions(
              device, queue, pointer(table), 1, 5, 5, pointer(bindings), 1,
              pointer(builtins)),
          RangeError);
      view[table / 4] = firstBlock >>> 0;
    }
  });
}

for (const arch of ['wasm32', 'wasm64']) {
  for (const [count, failSubmission] of [
           [0, false], [1, false], [7, false], [17, false], [7, true]]) {
    test(
        `${arch} uniform dispatch ${count} constants${
            failSubmission ? ' submission failure' : ''}`,
        () => {
          const context = {
            memory: {buffer: new ArrayBuffer(4096)},
            complete() {}
          };
          const imports = createImports(context);
          const view = new Uint32Array(context.memory.buffer);
          const pointer = n => arch === 'wasm64' ? BigInt(n) : n;
          const pipeline = {
            getBindGroupLayout(i) {
              assert.equal(i, 1);
              return constantsLayout;
            }
          };
          const bufferLayout = {};
          const constantsLayout = {};
          const buffer = {};
          const captured = [];
          const allocated = [];
          let submitted = false;
          let dispatches = 0;
          let destroyed = 0;
          const pass = {
            setPipeline(value) {
              assert.equal(value, pipeline);
            },
            setBindGroup(index, group) {
              if (index === 1)
                captured.push(
                    [...new Uint32Array(group.buffer.data, 0, count)]);
              else
                assert.equal(index, 0);
            },
            dispatchWorkgroups(x, y, z) {
              assert.deepEqual([x, y, z], [2, 3, 1]);
              dispatches++;
            },
            end() {},
          };
          const device = {
            createBuffer(descriptor) {
              assert.deepEqual(descriptor, {
                usage: 0x40,
                size: count * 4,
                mappedAtCreation: true
              });
              const value = {
                data: new ArrayBuffer(descriptor.size),
                getMappedRange() {
                  return this.data;
                },
                unmap() {},
                destroy() {
                  assert(submitted, 'Destroyed before submission');
                  destroyed++;
                },
              };
              allocated.push(value);
              return value;
            },
            createBindGroup(descriptor) {
              if (descriptor.layout === constantsLayout) {
                assert.equal(descriptor.entries.length, 1);
                assert.equal(descriptor.entries[0].binding, 0);
                return {buffer: descriptor.entries[0].resource.buffer};
              }
              assert.equal(descriptor.layout, bufferLayout);
              assert.deepEqual(
                  descriptor.entries,
                  [{binding: 0, resource: {buffer, offset: 0, size: 4}}]);
              return {};
            },
            createCommandEncoder() {
              return {
                beginComputePass() {
                  return pass;
                },
                finish() {
                  return {};
                }
              };
            },
          };
          const queue = {
            submit() {
              submitted = true;
              if (failSubmission) throw new Error('Submission failed');
            }
          };
          const dh = context.insertWebGPUHandle(device),
                qh = context.insertWebGPUHandle(queue);
          const ph = context.insertWebGPUHandle(pipeline),
                lh = context.insertWebGPUHandle(bufferLayout);
          view.set([context.insertWebGPUHandle(buffer), 0], 80);
          view[16] = 512;
          if (arch === 'wasm64') view[17] = 0;
          const patterns = [0, 0xffffffff, 0x80000000, 0x12345678];
          const values = Array.from(
              {length: count}, (_, i) => patterns[i % patterns.length]);
          const dispatch = count ?
              [
                ((10 + count) << 16) | 0x115, ph, lh, 2, 3, 1, 1, 0, 0, 4,
                ...values
              ] :
              [(9 << 16) | 0x15, ph, lh, 2, 3, 1, 0, 0, 4];
          const words = [0x10010, ...dispatch, ...dispatch, 0x10011, 0x100ff];
          view.set(words, 128);
          assert.equal(
              imports.execute_instructions(
                  dh, qh, pointer(64), 1, words.length, words.length,
                  pointer(320), 1, pointer(448)),
              failSubmission ? 13 : 0);
          assert.equal(dispatches, 2);
          assert.equal(allocated.length, count ? 2 : 0);
          assert.equal(destroyed, allocated.length);
          assert.deepEqual(captured, count ? [values, values] : []);
          view.fill(0, 128, 128 + words.length);
          assert.deepEqual(captured, count ? [values, values] : []);
        });
  }
}
