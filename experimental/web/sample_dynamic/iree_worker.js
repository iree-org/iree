// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// TODO(scotttodd): configure this through the build system / scripts?
// const MAIN_SCRIPT_URL = 'web-sample-dynamic-multithreaded.js';
const MAIN_SCRIPT_URL = 'web-sample-dynamic-sync.js';

let wasmLoadProgramFn;
var Module = {
  print: function(text) {
    console.log('(C)', text);
  },
  printErr: function(text) {
    console.error('(C)', text);
  },
  onRuntimeInitialized: function() {
    console.log('WebAssembly module onRuntimeInitialized()');

    wasmLoadProgramFn =
        Module.cwrap('load_program', 'number', ['number', 'number']);

    postMessage({
      'messageType': 'initialized',
    });
  },
  noInitialRun: true,
};

function loadProgram(id, vmfbPath) {
  console.log('fetching program at \'%s\'', vmfbPath);

  const fetchRequest = new XMLHttpRequest();

  fetchRequest.onload = function(progressEvent) {
    console.log('XMLHttpRequest completed, passing to Wasm module');

    const programDataBuffer = progressEvent.target.response;
    const programDataView = new Int8Array(programDataBuffer);

    const programDataWasmBuffer = Module._malloc(
        programDataView.length * programDataView.BYTES_PER_ELEMENT);
    Module.HEAP8.set(programDataView, programDataWasmBuffer);

    const result =
        wasmLoadProgramFn(programDataWasmBuffer, programDataBuffer.byteLength);
    console.log('Result from loadProgramFn():', result);
    Module._free(programDataWasmBuffer);

    if (result !== 0) {
      postMessage({
        'messageType': 'loadProgramResult',
        'id': id,
        'error': 'Wasm module error, check console for details',
      });
    } else {
      postMessage({
        'messageType': 'loadProgramResult',
        'id': id,
        'payload': 'success',
      });
    }
  };

  fetchRequest.open('GET', vmfbPath);
  fetchRequest.responseType = 'arraybuffer';
  fetchRequest.send();
}

self.onmessage = function(messageEvent) {
  const {messageType, id, payload} = messageEvent.data;

  console.log('worker received message:', messageEvent.data);

  if (messageType == 'loadProgram') {
    loadProgram(id, payload);
  }
};

importScripts(MAIN_SCRIPT_URL);
