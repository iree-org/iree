// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Promise-based API for interacting with the IREE runtime.

const EMSCRIPTEN_SCRIPT_URL = 'web-sample-webgpu.js';

// ------------------------------------------------------------------------- //
// - API                                                                   - //
// ------------------------------------------------------------------------- //

// Initializes IREE's runtime.
async function ireeInitialize() {
  return _ireeInitialize();
}

// Loads an IREE program stored in a .vmfb file.
//
// Accepts either a string path to a file (XMLHttpRequest compatible) or an
// ArrayBuffer containing an already loaded file.
//
// In order to call functions on the program it must be compiled in a supported
// configuration, such as with these flags:
//     --iree-hal-target-backends=webgpu
//
// Resolves with an opaque pointer to the program state on success.
async function ireeLoadProgram(vmfbPathOrBuffer) {
  return _ireeLoadProgram(vmfbPathOrBuffer);
}

// Inspects a program.
async function ireeInspectProgram(programState) {
  return _ireeInspectProgram(programState);
}

// Unloads a program.
async function ireeUnloadProgram(programState) {
  return _ireeUnloadProgram(programState);
}

// Calls a function on a loaded program.
//
// Resolves with a parsed JSON object on success:
// {
//   "total_invoke_time_ms": [number],
//   "outputs": [semicolon delimited list of formatted outputs]
// }
async function ireeCallFunction(
    programState, functionName, inputs, iterations) {
  return _ireeCallFunction(programState, functionName, inputs, iterations);
}

// ------------------------------------------------------------------------- //
// - Implementation                                                        - //
// ------------------------------------------------------------------------- //

// TODO(scotttodd): namespace / scope these (don't pollute window object)
let wasmSetupSampleFn;
let wasmCleanupSampleFn;
let wasmLoadProgramFn;
let wasmInspectProgramFn;
let wasmUnloadProgramFn;
let wasmCallFunctionFn;

let initializedPromise, initializePromiseResolve, initializePromiseReject;
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
    wasmLoadProgramFn = Module.cwrap(
        'load_program',
        'number',
        ['number', 'number', 'number'],
    );
    wasmInspectProgramFn = Module.cwrap('inspect_program', null, ['number']);
    wasmUnloadProgramFn = Module.cwrap('unload_program', null, ['number']);
    wasmCallFunctionFn = Module.cwrap(
        'call_function',
        'number',
        ['number', 'string', 'string', 'number'],
    );

    sampleState = wasmSetupSampleFn();
    if (!sampleState) {
      initializePromiseReject('Runtime initialization failed');
      return;
    }
    initializePromiseResolve();
  },
  noInitialRun: true,
};

async function _ireeInitialize() {
  if (initializedPromise) return initializedPromise;

  initializedPromise = new Promise((resolve, reject) => {
    initializePromiseResolve = resolve;
    initializePromiseReject = reject;
  });

  // Preinitialize a WebGPU device here. We could let the C program request the
  // adapter and device itself, but that would jump through layers of Emscripten
  // binding code and C/JS callbacks. This is much more concise.
  // const instance = -1; // No wgpuCreateInstance function in JS (yet?).
  if (navigator['gpu'] === undefined) {
    throw 'No \'gpu\' property on navigator, can\'t initialize WebGPU (missing #enable-unsafe-webgpu or an origin trial?)';
  }
  const adapter = await navigator['gpu']['requestAdapter']();
  const deviceDescriptor = {
    'label': 'IREE WebGPU device',
    'requiredFeatures': [],
    'requiredLimits': {
      'maxBindGroups': 4,
      'maxStorageBuffersPerShaderStage': 8,
    },
    'defaultQueue': {},
  };
  const device = await adapter['requestDevice'](deviceDescriptor);
  // Emscripten makes this available via emscripten_webgpu_get_device() in C.
  Module['preinitializedWebGPUDevice'] = device;

  const mainScript = document.createElement('script');
  mainScript.setAttribute('src', EMSCRIPTEN_SCRIPT_URL);
  document.body.appendChild(mainScript);

  return initializedPromise;
}

function _ireeLoadProgramBuffer(programDataBuffer) {
  const programDataView = new Int8Array(programDataBuffer);

  const programDataWasmBuffer = Module._malloc(
      programDataView.length * programDataView.BYTES_PER_ELEMENT);
  Module.HEAP8.set(programDataView, programDataWasmBuffer);

  // Note: we transfer ownership of the FlatBuffer data here, so there is
  // no need to call `Module._free(programDataWasmBuffer)` later.
  const programState = wasmLoadProgramFn(
      sampleState, programDataWasmBuffer, programDataBuffer.byteLength);
  return programState;
}

function _ireeLoadProgram(vmfbPathOrBuffer) {
  if (vmfbPathOrBuffer instanceof ArrayBuffer) {
    const programState = _ireeLoadProgramBuffer(vmfbPathOrBuffer);
    if (programState !== 0) {
      return Promise.resolve(programState);
    } else {
      return Promise.reject('Wasm module error loading program');
    }
  }

  return new Promise((resolve, reject) => {
    const fetchRequest = new XMLHttpRequest();
    fetchRequest.onload = function(progressEvent) {
      const programState =
          _ireeLoadProgramBuffer(progressEvent.target.response);
      if (programState !== 0) {
        resolve(programState);
      } else {
        reject('Wasm module error loading program');
      }
    };
    fetchRequest.onerror = function(progressEvent) {
      reject(progressEvent.error);
    };
    fetchRequest.open('GET', vmfbPathOrBuffer);
    fetchRequest.responseType = 'arraybuffer';
    fetchRequest.send();
  });
}

function _ireeInspectProgram(programState) {
  wasmInspectProgramFn(programState);
  return Promise.resolve();
}

function _ireeUnloadProgram(programState) {
  wasmUnloadProgramFn(programState);
  return Promise.resolve();
}

function _ireeCallFunction(programState, functionName, inputs, iterations) {
  iterations = iterations !== undefined ? iterations : 1;

  let inputsJoined;
  if (Array.isArray(inputs)) {
    inputsJoined = inputs.join(';');
  } else if (typeof (inputs) === 'string') {
    inputsJoined = inputs;
  } else {
    return Promise.reject(
        'Expected \'inputs\' to be a String or an array of Strings');
  }

  // Receive as a pointer, convert, then free. This avoids a memory leak, see
  // https://github.com/emscripten-core/emscripten/issues/6484
  const returnValuePtr =
      wasmCallFunctionFn(programState, functionName, inputsJoined, iterations);
  const returnValue = Module.UTF8ToString(returnValuePtr);

  if (returnValue === '') {
    return Promise.reject('Wasm module error calling function');
  } else {
    Module._free(returnValuePtr);
    return Promise.resolve(JSON.parse(returnValue));
  }
}
