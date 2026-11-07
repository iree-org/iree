// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';

const bundle = process.argv[2];
assert.ok(bundle, 'Expected the bundled WASI fixture path');

// Each child starts a fresh WASI instance: proc_exit must not terminate the
// controller before it can check the host process's observable exit status.
const outcomes = [['success', 0], ['failure', 7], ['exit', 7], ['trap', 1]];
for (const [argument, expected] of outcomes) {
  const result = spawnSync(process.execPath, [bundle, argument], {
    encoding: 'utf8',
    timeout: 30000,
  });
  assert.ifError(result.error);
  assert.equal(result.signal, null, result.stderr);
  assert.equal(result.status, expected, `${argument}: ${result.stderr}`);
  if (argument === 'trap') {
    assert.match(result.stderr, /RuntimeError: unreachable/);
  }
}
