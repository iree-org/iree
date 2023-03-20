// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Promise-based API for interacting with the IREE runtime.

let ireeWorker = null;
let nextMessageId = 0;
const pendingPromises = {};

// Communication protocol to and from the worker:
// {
//     'messageType': string
//         * the type of message (initialized, callResult, etc.)
//     'id': number?
//         * optional id to disambiguate messages of the same type
//     'payload': Object?
//         * optional message data, format defined by message type
//     'error': string?
//         * optional error message
// }

function _handleMessageFromWorker(messageEvent) {
  const {messageType, id, payload, error} = messageEvent.data;

  if (messageType == 'initialized') {
    pendingPromises['initialize']['resolve']();
    delete pendingPromises['initialize'];
  } else if (messageType == 'callResult') {
    if (error !== undefined) {
      pendingPromises[id]['reject'](error);
    } else {
      pendingPromises[id]['resolve'](payload);
    }
    delete pendingPromises[id];
  }
}

function _callIntoWorker(baseMessage) {
  return new Promise((resolve, reject) => {
    const message = baseMessage;
    const messageId = nextMessageId++;
    message['id'] = messageId;

    pendingPromises[messageId] = {
      'resolve': resolve,
      'reject': reject,
    };

    ireeWorker.postMessage(message);
  });
}

// Initializes IREE's web worker asynchronously.
//
// Resolves with no return value when the worker is fully initialized.
function ireeInitializeWorker() {
  return new Promise((resolve, reject) => {
    pendingPromises['initialize'] = {
      'resolve': resolve,
      'reject': reject,
    };

    ireeWorker = new Worker('iree_worker.js', {name: 'IREE-main'});
    ireeWorker.onmessage = _handleMessageFromWorker;
  });
}

// Loads an IREE program stored in a .vmfb file, asynchronously.
//
// Accepts either a string path to a file (XMLHttpRequest compatible) or an
// ArrayBuffer containing an already loaded file.
//
// In order to call functions on the program it must be compiled in a supported
// configuration, such as with these flags:
//     --iree-hal-target-backends=llvm
//     --iree-llvmcpu-target-triple=wasm32-unknown-emscripten
//
// Resolves with an opaque pointer to the program state on success.
function ireeLoadProgram(vmfbPathOrBuffer) {
  return _callIntoWorker({
    'messageType': 'loadProgram',
    'payload': vmfbPathOrBuffer,
  });
}

// Inspects a program, asynchronously.
function ireeInspectProgram(programState) {
  return _callIntoWorker({
    'messageType': 'inspectProgram',
    'payload': programState,
  });
}

// Unloads a program, asynchronously.
function ireeUnloadProgram(programState) {
  return _callIntoWorker({
    'messageType': 'unloadProgram',
    'payload': programState,
  });
}

// Calls a function on a loaded program, asynchronously.
//
// Returns a parsed JSON object on success:
// {
//   "total_invoke_time_ms": [number],
//   "outputs": [semicolon delimited list of formatted outputs]
// }
function ireeCallFunction(programState, functionName, inputs, iterations) {
  return _callIntoWorker({
    'messageType': 'callFunction',
    'payload': {
      'programState': programState,
      'functionName': functionName,
      'inputs': inputs,
      'iterations': iterations !== undefined ? iterations : 1,
    },
  });
}
