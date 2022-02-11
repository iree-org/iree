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
//         * the type of message (initialized, predict, etc.)
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
  } else if (messageType == 'predictResult') {
    if (payload !== undefined) {
      pendingPromises[id]['resolve'](payload);
    } else {
      pendingPromises[id]['reject'](error);
    }
    delete pendingPromises[id];
  }
}

// Initializes IREE's web worker asynchronously.
// Resolves when the worker is fully initialized.
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

// Predicts the handwritten digit in a provided image asynchronously.
// Input: 28x28 pixel data from CanvasRenderingContext2D.getImageData()
// Resolves with a Number in [0, 9] (inclusive) on success
function ireePredictDigit(imageData) {
  return new Promise((resolve, reject) => {
    const messageId = nextMessageId++;
    const message = {
      'messageType': 'predict',
      'id': messageId,
      'payload': imageData,
    };

    pendingPromises[messageId] = {
      'resolve': resolve,
      'reject': reject,
    };

    ireeWorker.postMessage(message);
  });
}
