// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Simple unit test that demonstrates compiling using the CAPI.
// There is room for improvement on the high level APIs, and if some of what
// is here is extracted into new APIs, please simplify this test accordingly.
//
// Originally contributed due to the work of edubart who figured out how to
// be the first user of the combined MLIR+IREE CAPI:
// https://github.com/openxla/iree/pull/8582

#include <stdio.h>
#include <string.h>

#include "iree/compiler/API/MLIRInterop.h"
#include "iree/compiler/embedding_api.h"

struct compiler_state_t {
  iree_compiler_session_t* session;
  MlirContext context;
};

struct invocation_state_t {
  iree_compiler_invocation_t* inv;
};

static void initializeCompiler(struct compiler_state_t* state) {
  ireeCompilerGlobalInitialize();
  state->session = ireeCompilerSessionCreate();
  state->context = ireeCompilerSessionGetContext(state->session);
}

static void shutdownCompiler(struct compiler_state_t* state) {
  ireeCompilerSessionDestroy(state->session);
  ireeCompilerGlobalShutdown();
}

int main(int argc, char** argv) {
  struct compiler_state_t state;
  initializeCompiler(&state);

  // Important: The compiler expects a top-level 'module' and in order to
  // parse that, it must be explicitly wrapped as such.
  MlirOperation module = mlirOperationCreateParse(
      state.context,
      mlirStringRefCreateFromCString(
          "module {"
          "func.func @simple_mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) "
          "-> "
          "tensor<4xf32> {\n"
          "  %0 = arith.mulf %arg0, %arg1 : tensor<4xf32>\n"
          "  return %0 : tensor<4xf32>\n"
          "}\n"
          "}\n"),
      mlirStringRefCreateFromCString("source.mlir"));
  if (mlirOperationIsNull(module)) {
    // TODO: Shutdown.
    return false;
  }

  // Set flags.
  iree_compiler_error_t* err;
  const char* flags[] = {
      "--iree-hal-target-backends=vmvx",
  };
  err = ireeCompilerSessionSetFlags(state.session, 1, flags);
  if (err) {
    fprintf(stderr, "ERROR: %s\n", ireeCompilerErrorGetMessage(err));
    mlirOperationDestroy(module);
    shutdownCompiler(&state);
    return 1;
  }

  // Import module.
  iree_compiler_invocation_t* inv = ireeCompilerInvocationCreate(state.session);
  if (!ireeCompilerInvocationImportModule(inv, module)) {
    // ireeCompilerInvocationCreate takes ownership of the module regardless
    // of success or error, so we let it destroy it.
    ireeCompilerInvocationDestroy(inv);
    shutdownCompiler(&state);
    return 1;
  }

  // Compile.
  if (!ireeCompilerInvocationPipeline(inv, IREE_COMPILER_PIPELINE_STD)) {
    ireeCompilerInvocationDestroy(inv);
    shutdownCompiler(&state);
    return 1;
  }

  // Output.
  iree_compiler_output_t* output;
  err = ireeCompilerOutputOpenMembuffer(&output);
  if (err) {
    fprintf(stderr, "ERROR: %s\n", ireeCompilerErrorGetMessage(err));
    ireeCompilerInvocationDestroy(inv);
    shutdownCompiler(&state);
    return 1;
  }
  err = ireeCompilerInvocationOutputVMBytecode(inv, output);
  if (err) {
    fprintf(stderr, "ERROR: %s\n", ireeCompilerErrorGetMessage(err));
    ireeCompilerOutputDestroy(output);
    ireeCompilerInvocationDestroy(inv);
    shutdownCompiler(&state);
    return 1;
  }

  // Map memory and print size.
  void* bytecode;
  uint64_t bytecodeSize;
  err = ireeCompilerOutputMapMemory(output, &bytecode, &bytecodeSize);
  if (err) {
    fprintf(stderr, "ERROR: %s\n", ireeCompilerErrorGetMessage(err));
    ireeCompilerOutputDestroy(output);
    ireeCompilerInvocationDestroy(inv);
    shutdownCompiler(&state);
    return 1;
  }

  printf("Success! Generated vmfb size: %d\n", (int)bytecodeSize);
  ireeCompilerOutputDestroy(output);
  ireeCompilerInvocationDestroy(inv);
  shutdownCompiler(&state);
  return 0;
}
