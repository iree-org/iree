// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import assert from 'node:assert/strict';
import {test} from 'node:test';

import {createImports} from './proactor_imports.mjs';

test('inline completions continue after a partial drain', async () => {
  const context = {
    inline: true,
    memory: new WebAssembly.Memory({initial: 1}),
  };
  const imports = createImports(context);
  const received = [];
  context.exports = {
    proactor_drain() {
      const count = imports.ring_drain(0, 2);
      received.push(...new Int32Array(context.memory.buffer, 0, count * 2));
    },
  };

  // One burst exceeds the two-entry drain buffer; no later completion wakes it.
  context.complete(1, 0);
  context.complete(2, 1);
  context.complete(3, 2);
  await new Promise(resolve => setTimeout(resolve, 0));

  assert.deepEqual(received, [1, 0, 2, 1, 3, 2]);
});
