// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// TODO(scotttodd): configure this through the build system / scripts?
// const MAIN_SCRIPT_URL = 'web-sample-dynamic-multithreaded.js';
const MAIN_SCRIPT_URL = 'web-sample-dynamic-sync.js';

let wasmSetupSampleFn;
let wasmCleanupSampleFn;
let wasmLoadProgramFn;
let wasmInspectProgramFn;
let wasmUnloadProgramFn;
let wasmCallFunctionFn;

let sampleState;

var Module = {
  print: function(text) {
    console.log('(C)', text);
  },
  printErr: function(text) {
    console.error('(C)', text);
  },
  onRuntimeInitialized: function() {
    wasmSetupSampleFn = Module.cwrap('setup_sample', 'number', []);
    wasmCleanupSampleFn = Module.cwrap('cleanup_sample', null, ['number']);
    wasmLoadProgramFn =
        Module.cwrap('load_program', 'number', ['number', 'number', 'number']);
    wasmInspectProgramFn = Module.cwrap('inspect_program', null, ['number']);
    wasmUnloadProgramFn = Module.cwrap('unload_program', null, ['number']);
    wasmCallFunctionFn = Module.cwrap(
        'call_function', 'number', ['number', 'string', 'string', 'number']);

    sampleState = wasmSetupSampleFn();

    postMessage({
      'messageType': 'initialized',
    });
  },
  noInitialRun: true,
};

function loadProgramBuffer(id, programDataBuffer) {
  const programDataView = new Int8Array(programDataBuffer);

  const programDataWasmBuffer = Module._malloc(
      programDataView.length * programDataView.BYTES_PER_ELEMENT);
  Module.HEAP8.set(programDataView, programDataWasmBuffer);

  // Note: we transfer ownership of the FlatBuffer data here, so there is
  // no need to call `Module._free(programDataWasmBuffer)` later.
  const programState = wasmLoadProgramFn(
      sampleState, programDataWasmBuffer, programDataBuffer.byteLength);

  if (programState !== 0) {
    postMessage({
      'messageType': 'callResult',
      'id': id,
      'payload': programState,
    });
  } else {
    postMessage({
      'messageType': 'callResult',
      'id': id,
      'error': 'Wasm module error, check console for details',
    });
  }
}

function loadProgram(id, vmfbPathOrBuffer) {
  if (vmfbPathOrBuffer instanceof ArrayBuffer) {
    loadProgramBuffer(id, vmfbPathOrBuffer);
    return;
  }

  const fetchRequest = new XMLHttpRequest();
  fetchRequest.onload = function(progressEvent) {
    loadProgramBuffer(id, progressEvent.target.response);
  };
  fetchRequest.open('GET', vmfbPathOrBuffer);
  fetchRequest.responseType = 'arraybuffer';
  fetchRequest.send();
}

function inspectProgram(id, programState) {
  wasmInspectProgramFn(programState);

  postMessage({
    'messageType': 'callResult',
    'id': id,
  });
}

function unloadProgram(id, programState) {
  wasmUnloadProgramFn(programState);

  postMessage({
    'messageType': 'callResult',
    'id': id,
  });
}

function callFunction(id, functionParams) {
  const {programState, functionName, inputs, iterations} = functionParams;

  let inputsJoined;
  if (Array.isArray(inputs)) {
    inputsJoined = inputs.join(';');
  } else if (typeof (inputs) === 'string') {
    inputsJoined = inputs;
  } else {
    postMessage({
      'messageType': 'callResult',
      'id': id,
      'error': 'Expected \'inputs\' to be a String or an array of Strings',
    });
    return;
  }

  // Receive as a pointer, convert, then free. This avoids a memory leak, see
  // https://github.com/emscripten-core/emscripten/issues/6484
  const returnValuePtr =
      wasmCallFunctionFn(programState, functionName, inputsJoined, iterations);
  const returnValue = Module.UTF8ToString(returnValuePtr);

  if (returnValue === '') {
    postMessage({
      'messageType': 'callResult',
      'id': id,
      'error': 'Wasm module error, check console for details',
    });
  } else {
    Module._free(returnValuePtr);
    postMessage({
      'messageType': 'callResult',
      'id': id,
      'payload': JSON.parse(returnValue),
    });
  }
}

self.onmessage = function(messageEvent) {
  const {messageType, id, payload} = messageEvent.data;

  if (messageType == 'loadProgram') {
    loadProgram(id, payload);
  } else if (messageType == 'inspectProgram') {
    inspectProgram(id, payload);
  } else if (messageType == 'unloadProgram') {
    unloadProgram(id, payload);
  } else if (messageType == 'callFunction') {
    callFunction(id, payload);
  }
};

importScripts(MAIN_SCRIPT_URL);
