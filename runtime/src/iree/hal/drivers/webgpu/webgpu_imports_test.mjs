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
