// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Helpers that prefix logs with the 'name' from DedicatedWorkerGlobalScope.
// This helps with debugging thread creation.
function ireeNamePrefix() {
  return self.name ? (self.name + ':') : '(unnamed scope)';
}
function ireeLog(...args) {
  console.log(ireeNamePrefix(), ...args);
}
function ireeError(...args) {
  console.error(ireeNamePrefix(), ...args);
}

// TODO(scotttodd): configure this through the build system / scripts?
// TODO(scotttodd): fix multithreading (startup silently fails on some emsdk
//                  versions, memory growth or high initial memory also req.)
// const MAIN_SCRIPT_URL = 'web-sample-static-multithreaded.js';
const MAIN_SCRIPT_URL = 'web-sample-static-sync.js';

let wasmSetupSampleFn;
let wasmCleanupSampleFn;
let wasmRunSampleFn;
let wasmState;
let initialized = false;

const IMAGE_PIXEL_COUNT = 28 * 28;
const imageTypedArray = new Float32Array(IMAGE_PIXEL_COUNT);
let imageBuffer;

var Module = {
  print: function(text) {
    ireeLog('(C/Wasm) ' + text);
  },
  printErr: function(text) {
    ireeError('(C/Wasm) ' + text);
  },
  onRuntimeInitialized: function() {
    ireeLog('WebAssembly module onRuntimeInitialized()');

    wasmSetupSampleFn = Module.cwrap('setup_sample', 'number', []);
    wasmCleanupSampleFn = Module.cwrap('cleanup_sample', null, ['number']);
    wasmRunSampleFn =
        Module.cwrap('run_sample', 'number', ['number', 'number']);

    initializeSample();
  },
  mainScriptUrlOrBlob: MAIN_SCRIPT_URL,
  noInitialRun: true,
};

function initializeSample() {
  wasmState = wasmSetupSampleFn();
  imageBuffer =
      Module._malloc(IMAGE_PIXEL_COUNT * Float32Array.BYTES_PER_ELEMENT);
  initialized = true;

  postMessage({
    'messageType': 'initialized',
  });
}

// TODO(scotttodd): call this on page suspend?
function cleanupSample() {
  initialized = false;
  Module._free(imageDataBuffer);
  wasmCleanupSampleFn();
  wasmState = null;
}

// https://becominghuman.ai/passing-and-returning-webassembly-array-parameters-a0f572c65d97
// https://developers.google.com/web/updates/2018/03/emscripting-a-c-library#get_an_image_from_javascript_into_wasm
function preprocessImageDataIntoHeap(rawImageData) {
  // rawImageData is a Uint8ClampedArray with RGBA image data
  // * this MNIST model takes tensor<1x28x28x1xf32> with grayscale pixels
  //   in [0.0, 1.0]

  // This conversion is terrible, but this is a toy demo with a small image
  // Hopefully there aren't any logic / iteration order issues...
  for (let y = 0; y < 28; ++y) {
    for (let x = 0; x < 28; ++x) {
      const typedIndex = y * 28 + x;
      const rawIndex = 4 * (y * 28 + x) + 3;  // Assume colorSpace srgb
      imageTypedArray[typedIndex] = rawImageData.data[rawIndex] / 255.0;
    }
  }

  // Copy into Wasm heap.
  // Note: we could have done the conversion in-place, but this is demo code
  Module.HEAPF32.set(imageTypedArray, imageBuffer >> 2);
}

function handlePredict(id, canvasData) {
  if (!initialized) return;

  preprocessImageDataIntoHeap(canvasData);
  result = wasmRunSampleFn(wasmState, imageBuffer);

  if (result == -1) {
    postMessage({
      'messageType': 'predictResult',
      'id': id,
      'error': 'Wasm module error, check console for details',
    });
  } else {
    postMessage({
      'messageType': 'predictResult',
      'id': id,
      'payload': result,
    });
  }
}

self.onmessage = function(messageEvent) {
  const {messageType, id, payload} = messageEvent.data;

  if (messageType == 'predict') {
    handlePredict(id, payload);
  }
};

ireeLog('iree_worker importing \'' + MAIN_SCRIPT_URL + '\'');

importScripts(MAIN_SCRIPT_URL);
