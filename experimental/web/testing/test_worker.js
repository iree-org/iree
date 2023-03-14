// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

var Module = {
  // We can't interact with the DOM from a worker, so each line of output has
  // to be routed back to the main/UI thread. We could buffer these to avoid
  // all the interrupts, but any performance critical code shouldn't be logging
  // anyways, and unbuffered output is easier to debug with.
  print: function(text) {
    postMessage({
      'messageType': 'print',
      'payload': text,
    });
  },
  printErr: function(text) {
    postMessage({
      'messageType': 'printErr',
      'payload': text,
    });
  },
  onRuntimeInitialized: function() {
    // TODO(scotttodd): record test time start?
  },
  quit: function(status, error) {
    // Report errors from Emscripten (not our test/application code) directly.
    // Useful for debugging the test runner itself.
    if (!(error instanceof ExitStatus)) console.error(error);

    postMessage({
      'messageType': 'testResult',
      'payload': status,
    });
  },
};

function runTest(parameters) {
  const {name, sourceFile, workingDirectory, requiredFiles, args} = parameters;

  Module['arguments'] = args;

  // Set up the 'locateFile' and 'preInit' functions to handle loading the
  // different types of files needed by our tests.
  //
  // 'locateFile' is used by Emscripten when loading other JS and Wasm files,
  // so we should search relative to the test source file. On the other hand,
  // C/C++ code attempting to read files from disk (e.g. with `fopen()`) does
  // _not_ use locateFile, but rather interfaces with Emscripten's File System
  // API.
  //
  // Consider this example:
  //     workingDirectory = tests/e2e/
  //     sourceFile       = tools/iree-check-module.js
  //     requiredFiles    = [check_sqrt.mlir_module.vmfb]
  // * Emscripten will need to load `iree-check-module.wasm`, so locateFile
  //   should point to /tools/.
  // * The test wants to read `check_sqrt.mlir_module.vmfb` from the working
  //   directory, so we should load `tests/e2e/check_sqrt.mlir_module.vmfb`
  //   into Emscripten's file system. There are multiple ways to do that, but
  //   calling createLazyFile in preInit seems simplest.
  Module['locateFile'] = function(path, prefix) {
    // https://emscripten.org/docs/api_reference/module.html#Module.locateFile
    // `path/to/source_file.js` -> `path/to/`
    const sourcePath = sourceFile.substring(0, sourceFile.lastIndexOf('/'));
    return '/' + sourcePath + '/' + path;
  };
  Module['preInit'] = function() {
    requiredFiles.forEach((file) => {
      // https://emscripten.org/docs/api_reference/Filesystem-API.html
      FS.createLazyFile(
          '/', file, '/' + workingDirectory + '/' + file, /*canRead=*/ true,
          /*canWrite=*/ false);
    });
  };

  // TODO(scotttodd): allow-list of script URLs? Restrict to same domain?
  importScripts('/' + sourceFile);
}

self.onmessage = function(messageEvent) {
  const {messageType, payload} = messageEvent.data;

  if (messageType == 'runTest') {
    runTest(payload);
  }
};
