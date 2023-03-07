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

#include "iree/base/api.h"
#include "iree/compiler/API2/MLIRInterop.h"

static void bytecode_builder_callback(MlirStringRef str, void* userdata) {
  iree_string_builder_t* builder = (iree_string_builder_t*)userdata;
  iree_string_builder_append_string(
      builder, iree_make_string_view(str.data, str.length));
}

// Compiles MLIR code into VM bytecode for the given target backend.
static bool iree_compile_mlir_to_bytecode(iree_string_view_t mlir_source,
                                          iree_string_view_t target_backend,
                                          iree_string_builder_t* out_builder) {
  // TODO: support customizing compiling flags?
  // TODO: return IREE status with error information instead of a boolean?
  // TODO: only call registers once to speedup second calls?
  // TODO: cache MLIR context, pass manager to speedup second calls?

  // Expects string builder to be initialized.
  if (out_builder == NULL) {
    return false;
  }

  // The IREE source code states that this function should be called before
  // creating any MLIRContext if one expects all the possible target backends
  // to be available.
  ireeCompilerRegisterTargetBackends();

  // Register passes that may be required in the lowering pipeline.
  ireeCompilerRegisterAllPasses();

  // Create MLIR context.
  MlirContext context = mlirContextCreate();

  // Register all IREE dialects and dialects it depends on.
  ireeCompilerRegisterAllDialects(context);

  // Create MLIR module from a chunk of text.
  MlirModule module = mlirModuleCreateParse(
      context, mlirStringRefCreate(mlir_source.data, mlir_source.size));
  if (mlirModuleIsNull(module)) {
    return false;
  }

  // Prepare target backend flag.
  char target_buf[128];
  iree_string_builder_t target_builder;
  iree_string_builder_initialize_with_storage(target_buf, sizeof(target_buf),
                                              &target_builder);
  iree_string_builder_append_cstring(&target_builder,
                                     "--iree-hal-target-backends=");
  iree_string_builder_append_string(&target_builder, target_backend);

  // Create compiler options.
  IreeCompilerOptions options = ireeCompilerOptionsCreate();
  const char* compiler_flags[] = {iree_string_builder_buffer(&target_builder)};
  MlirLogicalResult status =
      ireeCompilerOptionsSetFlags(options, 1, compiler_flags, /*onError=*/NULL,
                                  /*userData=*/NULL);
  if (mlirLogicalResultIsFailure(status)) {
    ireeCompilerOptionsDestroy(options);
    mlirModuleDestroy(module);
    mlirContextDestroy(context);
    return false;
  }

  // Run MLIR pass pipeline to lower the input IR down to the IREE VM dialect.
  MlirPassManager pass = mlirPassManagerCreate(context);
  MlirOpPassManager op_pass = mlirPassManagerGetAsOpPassManager(pass);
  ireeCompilerBuildIREEVMPassPipeline(options, op_pass);
  status = mlirPassManagerRunOnOp(pass, mlirModuleGetOperation(module));
  if (mlirLogicalResultIsFailure(status)) {
    mlirPassManagerDestroy(pass);
    ireeCompilerOptionsDestroy(options);
    mlirModuleDestroy(module);
    mlirContextDestroy(context);
    return false;
  }

  // Compile MLIR VM code to VM bytecode.
  status = ireeCompilerTranslateModuletoVMBytecode(
      options, mlirModuleGetOperation(module), bytecode_builder_callback,
      out_builder);

  // Cleanups.
  mlirPassManagerDestroy(pass);
  ireeCompilerOptionsDestroy(options);
  mlirModuleDestroy(module);
  mlirContextDestroy(context);
  return mlirLogicalResultIsSuccess(status);
}

int main(int argc, char** argv) {
  // MLIR code that we will compile
  iree_string_view_t mlir_code = iree_make_cstring_view(
      "func.func @simple_mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> "
      "tensor<4xf32> {\n"
      "  %0 = arith.mulf %arg0, %arg1 : tensor<4xf32>\n"
      "  return %0 : tensor<4xf32>\n"
      "}\n");

  // Initializes string builder that will contains the output bytecode.
  iree_string_builder_t bytecode_builder;
  iree_string_builder_initialize(iree_allocator_system(), &bytecode_builder);

  // Compiles MLIR to VM bytecode.
  bool status = iree_compile_mlir_to_bytecode(
      mlir_code, iree_make_cstring_view("vmvx"), &bytecode_builder);
  if (!status) {
    iree_string_builder_deinitialize(&bytecode_builder);
    fprintf(stderr, "failed to compile MLIR code\n");
    return -1;
  }

  // For testing purposes, just print the length vs the full contents.
  iree_string_view_t bytecode = iree_string_builder_view(&bytecode_builder);
  printf("Success! Generated vmfb size: %d\n", (int)bytecode.size);

  // Cleanups.
  iree_string_builder_deinitialize(&bytecode_builder);
  return 0;
}
