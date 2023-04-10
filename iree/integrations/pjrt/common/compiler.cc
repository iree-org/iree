// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/integrations/pjrt/common/compiler.h"

#include <functional>
#include <iostream>  // TODO: Remove
#include <vector>

namespace iree::pjrt {

//===----------------------------------------------------------------------===//
// In-process stub compiler
//===----------------------------------------------------------------------===//

namespace {

class MMapCompilerOutput : public CompilerOutput {
 public:
  MMapCompilerOutput(iree_compiler_output_t* output, void* data, size_t length)
      : output_(output), data_(data), length_(length) {}
  ~MMapCompilerOutput() { ireeCompilerOutputDestroy(output_); }
  void* GetData() { return data_; }
  size_t GetDataSize() { return length_; }

 private:
  iree_compiler_output_t* output_;
  void* data_;
  size_t length_;
};

using SessionRecycler = std::function<void(iree_compiler_session_t*)>;
class InprocessCompilerJob : public CompilerJob {
 public:
  // Takes ownership of both |session| and |inv|. On destruction, destroys
  // |inv| and passes |session| to the recycler (this can be used to implement
  // session pooling).
  InprocessCompilerJob(iree_compiler_session_t* session,
                       iree_compiler_invocation_t* inv,
                       SessionRecycler session_recycler)
      : session_(session), inv_(inv), session_recycler_(session_recycler) {}
  ~InprocessCompilerJob() {
    if (error_) {
      ireeCompilerErrorDestroy(error_);
    }
    for (auto* source : retained_sources_) {
      ireeCompilerSourceDestroy(source);
    }
    ireeCompilerInvocationDestroy(inv_);
    session_recycler_(session_);

    if (output_) {
      ireeCompilerOutputDestroy(output_);
    }
  }

  std::string GetErrorMessage() override {
    if (!error_) return std::string();
    const char* cstr = ireeCompilerErrorGetMessage(error_);
    return std::string(cstr);
  }

  void EnableCrashDumps(
      ArtifactDumper::Transaction* artifact_transaction) override {
    if (crash_dump_transaction_) return;
    crash_dump_transaction_ = artifact_transaction;
    ireeCompilerInvocationSetCrashHandler(
        inv_, /*genLocalReproducer=*/false,
        [](iree_compiler_output_t** outOutput,
           void* userData) -> iree_compiler_error_t* {
          auto* self = static_cast<InprocessCompilerJob*>(userData);
          auto maybePath = self->crash_dump_transaction_->AllocateArtifactPath(
              /*label=*/"crash_reproducer", /*extension=*/"mlir",
              /*index=*/self->crash_dump_count_++);
          if (!maybePath) {
            *outOutput = nullptr;
            return nullptr;
          }

          return ireeCompilerOutputOpenFile(maybePath->c_str(), outOutput);
        },
        this);
  }

  bool SetFlag(const char* flag) override {
    auto* error = ireeCompilerSessionSetFlags(session_, 1, &flag);
    if (error) {
      SetError(error);
      return false;
    }
    return true;
  }

  std::string GetFlags() override {
    std::string flags;
    ireeCompilerSessionGetFlags(
        session_, /*nonDefaultOnly=*/false,
        [](const char* flag, size_t length, void* userData) {
          std::string* capture_flags = static_cast<std::string*>(userData);
          if (!capture_flags->empty()) {
            capture_flags->append(" ");
          }
          capture_flags->append(flag, length);
        },
        &flags);
    return flags;
  }

  bool ParseSourceBuffer(const void* buffer, size_t length) override {
    iree_compiler_source_t* source;
    auto* error = ireeCompilerSourceWrapBuffer(
        session_, "<jit>", static_cast<const char*>(buffer), length,
        /*isNullTerminated=*/false, &source);
    if (error) {
      SetError(error);
      return false;
    }
    retained_sources_.push_back(source);

    return ireeCompilerInvocationParseSource(inv_, source);
  }

  std::unique_ptr<CompilerOutput> CompileStandardPipeline() override {
    iree_compiler_error_t* error;
    if (!ireeCompilerInvocationPipeline(inv_, IREE_COMPILER_PIPELINE_STD)) {
      return nullptr;
    }

    error = ireeCompilerOutputOpenMembuffer(&output_);
    if (error) {
      SetError(error);
      return nullptr;
    }

    // Output.
    error = ireeCompilerInvocationOutputVMBytecode(inv_, output_);
    if (error) {
      SetError(error);
      return nullptr;
    }

    // Map the data.
    void* output_data = nullptr;
    uint64_t size = -1;
    error = ireeCompilerOutputMapMemory(output_, &output_data, &size);
    if (error) {
      SetError(error);
      return nullptr;
    }

    // Transfer the output_ to MMapCompilerOutput since the mapping is only
    // valid for the life of the output.
    iree_compiler_output_t* local_output = output_;
    output_ = nullptr;
    return std::make_unique<MMapCompilerOutput>(local_output, output_data,
                                                size);
  }

 private:
  void SetError(iree_compiler_error_t* error) {
    if (error_) {
      ireeCompilerErrorDestroy(error_);
    }
    error_ = error;
  }

  iree_compiler_session_t* session_;
  iree_compiler_invocation_t* inv_;
  SessionRecycler session_recycler_;

  std::vector<iree_compiler_source_t*> retained_sources_;
  iree_compiler_error_t* error_ = nullptr;
  ArtifactDumper::Transaction* crash_dump_transaction_ = nullptr;
  int crash_dump_count_ = 0;

  // Output.
  iree_compiler_output_t* output_ = nullptr;
};

}  // namespace

std::unique_ptr<CompilerJob> InprocessCompiler::StartJob() {
  auto* session = ireeCompilerSessionCreate();
  auto* inv = ireeCompilerInvocationCreate(session);

  // TODO: Capture diagnostics, etc vs spewing to stderr.
  ireeCompilerInvocationEnableConsoleDiagnostics(inv);

  return std::make_unique<InprocessCompilerJob>(
      session, inv, [](iree_compiler_session_t* session) {
        ireeCompilerSessionDestroy(session);
      });
}

}  // namespace iree::pjrt
