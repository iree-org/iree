// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <functional>
#include <iostream>  // TODO: Remove
#include <vector>

#include "iree_pjrt/common/compiler.h"
#include "iree_pjrt/partitioner_api/embedding_api.h"

namespace iree::pjrt {

//===----------------------------------------------------------------------===//
// IREE compiler.
//===----------------------------------------------------------------------===//

namespace {

class MMapCompilerOutput : public CompilerOutput {
 public:
  MMapCompilerOutput(openxla_partitioner_output_t* output, void* data,
                     size_t length)
      : output_(output), data_(data), length_(length) {}
  ~MMapCompilerOutput() { openxlaPartitionerOutputDestroy(output_); }
  void* GetData() { return data_; }
  size_t GetDataSize() { return length_; }

 private:
  openxla_partitioner_output_t* output_;
  void* data_;
  size_t length_;
};

using SessionRecycler = std::function<void(openxla_partitioner_session_t*)>;
class OpenXLAPartitionerJob : public CompilerJob {
 public:
  // Takes ownership of both |session| and |inv|. On destruction, destroys
  // |inv| and passes |session| to the recycler (this can be used to implement
  // session pooling).
  OpenXLAPartitionerJob(openxla_partitioner_session_t* session,
                        openxla_partitioner_invocation_t* inv,
                        SessionRecycler session_recycler)
      : session_(session), inv_(inv), session_recycler_(session_recycler) {}
  ~OpenXLAPartitionerJob() {
    if (error_) {
      openxlaPartitionerErrorDestroy(error_);
    }
    for (auto* source : retained_sources_) {
      openxlaPartitionerSourceDestroy(source);
    }
    openxlaPartitionerInvocationDestroy(inv_);
    session_recycler_(session_);

    if (output_) {
      openxlaPartitionerOutputDestroy(output_);
    }
  }

  std::string GetErrorMessage() override {
    if (!error_) return std::string();
    const char* cstr = openxlaPartitionerErrorGetMessage(error_);
    return std::string(cstr);
  }

  void EnableCrashDumps(
      ArtifactDumper::Transaction* artifact_transaction) override {
    if (crash_dump_transaction_) return;
    crash_dump_transaction_ = artifact_transaction;
    openxlaPartitionerInvocationSetCrashHandler(
        inv_, /*genLocalReproducer=*/false,
        [](openxla_partitioner_output_t** outOutput,
           void* userData) -> openxla_partitioner_error_t* {
          auto* self = static_cast<OpenXLAPartitionerJob*>(userData);
          auto maybePath = self->crash_dump_transaction_->AllocateArtifactPath(
              /*label=*/"crash_reproducer", /*extension=*/"mlir",
              /*index=*/self->crash_dump_count_++);
          if (!maybePath) {
            *outOutput = nullptr;
            return nullptr;
          }

          return openxlaPartitionerOutputOpenFile(maybePath->c_str(),
                                                  outOutput);
        },
        this);
  }

  bool SetFlag(const char* flag) override {
    auto* error = openxlaPartitionerSessionSetFlags(session_, 1, &flag);
    if (error) {
      SetError(error);
      return false;
    }
    return true;
  }

  // TODO: Find another way to deal with this.
  // bool SetFlags(xla::CompileOptions options) override {
  //   int num_partitions = options.executable_build_options.num_partitions();
  //   int num_replicas = options.executable_build_options.num_replicas();
  //   bool use_spmd_partitioning =
  //       options.executable_build_options.use_spmd_partitioning();
  //   auto allow_spmd_sharding_propagation_to_output =
  //       options.executable_build_options
  //           .allow_spmd_sharding_propagation_to_output();
  //   if (!SetFlag(absl::StrCat("--openxla-partitioner-gspmd-num-partitions=",
  //                             num_partitions)
  //                    .c_str())) {
  //     return false;
  //   }
  //   if (!SetFlag(absl::StrCat("--openxla-partitioner-gspmd-replica-count=",
  //                             num_replicas)
  //                    .c_str())) {
  //     return false;
  //   }
  //   if (!SetFlag(
  //           absl::StrCat("--openxla-partitioner-gspmd-use-spmd-partitioning=",
  //                        use_spmd_partitioning)
  //               .c_str())) {
  //     return false;
  //   }
  //   if (!SetFlag(
  //           absl::StrCat(
  //               "--openxla-partitioner-gspmd-allow-spmd-"
  //               "sharding-propagation-to-output=",
  //               absl::StrJoin(allow_spmd_sharding_propagation_to_output, ","))
  //               .c_str())) {
  //     return false;
  //   }
  //   return true;
  // }

  std::string GetFlags() override {
    std::string flags;
    openxlaPartitionerSessionGetFlags(
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
    openxla_partitioner_source_t* source;
    auto* error = openxlaPartitionerSourceWrapBuffer(
        session_, "<jit>", static_cast<const char*>(buffer), length,
        /*isNullTerminated=*/false, &source);
    if (error) {
      SetError(error);
      return false;
    }
    retained_sources_.push_back(source);

    return openxlaPartitionerInvocationParseSource(inv_, source);
  }

  std::unique_ptr<CompilerOutput> CompileStandardPipeline() override {
    if (!openxlaPartitionerInvocationPipeline(inv_, "gspmd")) {
      return nullptr;
    }

    openxla_partitioner_error_t* error =
        openxlaPartitionerOutputOpenMembuffer(&output_);
    if (error) {
      SetError(error);
      return nullptr;
    }

    // Output.
    error = openxlaPartitionerInvocationOutputIR(inv_, output_);
    if (error) {
      SetError(error);
      return nullptr;
    }

    // Map the data.
    void* output_data = nullptr;
    uint64_t size = -1;
    error = openxlaPartitionerOutputMapMemory(output_, &output_data, &size);
    if (error) {
      SetError(error);
      return nullptr;
    }

    // Transfer the output_ to MMapCompilerOutput since the mapping is only
    // valid for the life of the output.
    openxla_partitioner_output_t* local_output = output_;
    output_ = nullptr;
    return std::make_unique<MMapCompilerOutput>(local_output, output_data,
                                                size);
  }

 private:
  void SetError(openxla_partitioner_error_t* error) {
    if (error_) {
      openxlaPartitionerErrorDestroy(error_);
    }
    error_ = error;
  }

  openxla_partitioner_session_t* session_;
  openxla_partitioner_invocation_t* inv_;
  SessionRecycler session_recycler_;

  std::vector<openxla_partitioner_source_t*> retained_sources_;
  openxla_partitioner_error_t* error_ = nullptr;
  ArtifactDumper::Transaction* crash_dump_transaction_ = nullptr;
  int crash_dump_count_ = 0;

  // Output.
  openxla_partitioner_output_t* output_ = nullptr;
};

}  // namespace

std::unique_ptr<CompilerJob> OpenXLAPartitioner::StartJob() {
  auto* session = openxlaPartitionerSessionCreate();
  auto* inv = openxlaPartitionerInvocationCreate(session);

  // TODO: Capture diagnostics, etc vs spewing to stderr.
  openxlaPartitionerInvocationEnableConsoleDiagnostics(inv);

  return std::make_unique<OpenXLAPartitionerJob>(
      session, inv, [](openxla_partitioner_session_t* session) {
        openxlaPartitionerSessionDestroy(session);
      });
}

std::string OpenXLAPartitioner::GetRevision() {
  std::string result;
  const char* revision = openxlaPartitionerGetRevision();
  result.append(revision[0] ? revision : "<unknown>");
  result.append(" (API version ");
  int packed_api_version = openxlaPartitionerGetAPIVersion();
  result.append(std::to_string(packed_api_version >> 16));
  result.append(".");
  result.append(std::to_string(packed_api_version & 0xffff));
  result.append(")");
  return result;
}

}  // namespace iree::pjrt
