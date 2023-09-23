// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "llvm/Support/CommandLine.h"
#include "openxla/partitioner/embedding_api.h"

extern "C" {
OPENXLA_PARTITIONER_EMBED_EXPORTED int openxlaPartitionerMain(int argc,
                                                              char **argv);
}

namespace {
enum class OutputFormat {
  none,
  ir_text,
};
}  // namespace

int openxlaPartitionerMain(int argc, char **argv) {
  openxlaPartitionerGlobalInitialize();
  static llvm::cl::OptionCategory mainOptions("OpenXLA Partitioner Options");

  llvm::cl::opt<std::string> inputPath(
      llvm::cl::Positional, llvm::cl::desc("<input file>"), llvm::cl::Required,
      llvm::cl::cat(mainOptions));
  llvm::cl::opt<std::string> outputFilename(
      "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
      llvm::cl::init("-"), llvm::cl::cat(mainOptions));
  llvm::cl::opt<std::string> pipelineName(
      "pipeline-name", llvm::cl::desc("Partitioner pipeline to run"),
      llvm::cl::init("gspmd"), llvm::cl::cat(mainOptions));
  llvm::cl::opt<OutputFormat> outputFormat(
      "output-format", llvm::cl::desc("Format of output"),
      llvm::cl::values(clEnumValN(OutputFormat::ir_text, "ir-text",
                                  "StableHLO IR Text (default)")),
      llvm::cl::init(OutputFormat::ir_text), llvm::cl::cat(mainOptions));

  openxlaPartitionerSetupGlobalCL(argc, const_cast<const char **>(argv),
                                  "OpenXLA Partitioner CLI",
                                  /*installSignalHandlers=*/true);
  struct MainState {
    ~MainState() {
      if (inv) {
        openxlaPartitionerInvocationDestroy(inv);
      }
      if (source) {
        openxlaPartitionerSourceDestroy(source);
      }
      if (output) {
        openxlaPartitionerOutputDestroy(output);
      }
      openxlaPartitionerSessionDestroy(session);
      openxlaPartitionerGlobalShutdown();
    }
    void handleError(openxla_partitioner_error_t *err) {
      fprintf(stderr, "ERROR: %s\n", openxlaPartitionerErrorGetMessage(err));
      openxlaPartitionerErrorDestroy(err);
    }
    openxla_partitioner_session_t *session = openxlaPartitionerSessionCreate();
    openxla_partitioner_invocation_t *inv =
        openxlaPartitionerInvocationCreate(session);
    openxla_partitioner_source_t *source = nullptr;
    openxla_partitioner_output_t *output = nullptr;
  };
  MainState mainState;

  // Parse source.
  if (auto err = openxlaPartitionerSourceOpenFile(
          mainState.session, inputPath.c_str(), &mainState.source)) {
    mainState.handleError(err);
    return 1;
  }
  if (!openxlaPartitionerInvocationParseSource(mainState.inv,
                                               mainState.source)) {
    return 1;
  }

  // Open output.
  if (auto err = openxlaPartitionerOutputOpenFile(outputFilename.c_str(),
                                                  &mainState.output)) {
    mainState.handleError(err);
    return 1;
  }

  // Run pipeline.
  if (!openxlaPartitionerInvocationPipeline(mainState.inv,
                                            pipelineName.c_str())) {
    return 1;
  }

  // Write output.
  openxla_partitioner_error_t *outputError = nullptr;
  switch (outputFormat) {
    case OutputFormat::ir_text:
      outputError =
          openxlaPartitionerInvocationOutputIR(mainState.inv, mainState.output);
      break;
    default:
      fprintf(stderr, "INTERNAL ERROR: Unknown output format\n");
      return 1;
  }
  if (outputError) {
    mainState.handleError(outputError);
    return 1;
  }

  // Everything is successful, mark the output as kept.
  openxlaPartitionerOutputKeep(mainState.output);
  return 0;
}
