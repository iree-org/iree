// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef IREE_BINDINGS_PYTHON_PYIREE_COMPILER_H_
#define IREE_BINDINGS_PYTHON_PYIREE_COMPILER_H_

#include <mutex>  // NOLINT
#include <string>

#include "bindings/python/pyiree/common/binding.h"
#include "iree/compiler/Dialect/VM/Target/Bytecode/BytecodeModuleTarget.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"

namespace iree {
namespace python {

// Wrapper around a blob of memory.
// Used to transport blobs back and forth between C++ and Python.
class OpaqueBlob {
 public:
  OpaqueBlob() : data_(nullptr), size_(0) {}
  OpaqueBlob(void* data, size_t size) : data_(data), size_(size) {}
  virtual ~OpaqueBlob() = default;

  void* data() { return data_; }
  const void* data() const { return data_; }
  size_t size() const { return size_; }

  // Create a free function from the OpaqueBlob shared pointer.
  using BufferFreeFn = void (*)(void* self, iree_byte_span_t);
  static std::pair<BufferFreeFn, void*> CreateFreeFn(
      std::shared_ptr<OpaqueBlob> blob) {
    // Note that there are more efficient ways to write this which
    // don't bounce through an extra heap alloc, but this is not
    // intended to be a high impact code path.
    struct Holder {
      std::shared_ptr<OpaqueBlob> blob;
    };
    Holder* holder = new Holder{std::move(blob)};
    auto free_fn = +([](void* self, iree_byte_span_t) {
      Holder* self_holder = static_cast<Holder*>(self);
      delete self_holder;
    });
    return {free_fn, holder};
  }

  static iree_allocator_t CreateDeallocator(std::shared_ptr<OpaqueBlob> blob) {
    // Note that there are more efficient ways to write this which
    // don't bounce through an extra heap alloc, but this is not
    // intended to be a high impact code path.
    struct Holder {
      std::shared_ptr<OpaqueBlob> blob;
    };
    Holder* holder = new Holder{std::move(blob)};
    auto free_fn = +([](void* self, void*) -> iree_status_t {
      Holder* self_holder = static_cast<Holder*>(self);
      delete self_holder;
      return IREE_STATUS_OK;
    });
    return {holder /* self */, nullptr /* alloc */, free_fn /* free */};
  }

 protected:
  void* data_;
  size_t size_;
};

// Opaque blob that owns a vector.
class OpaqueByteVectorBlob : public OpaqueBlob {
 public:
  OpaqueByteVectorBlob(std::vector<uint8_t> v)
      : OpaqueBlob(), v_(std::move(v)) {
    data_ = v_.data();
    size_ = v_.size();
  }

 private:
  std::vector<uint8_t> v_;
};

class OpaqueStringBlob : public OpaqueBlob {
 public:
  OpaqueStringBlob(std::string s) : OpaqueBlob(), s_(std::move(s)) {
    data_ = &s_[0];
    size_ = s_.size();
  }

 private:
  std::string s_;
};

class CompilerContextBundle;
class CompilerModuleBundle;

// Wraps an MLIR module and its producing context.
class CompilerModuleBundle {
 public:
  CompilerModuleBundle(std::shared_ptr<CompilerContextBundle> context,
                       mlir::ModuleOp module_op)
      : context_(std::move(context)), module_op_(std::move(module_op)) {}

  mlir::ModuleOp& module_op() { return module_op_; }
  std::string ToAsm(bool enableDebugInfo, bool prettyForm,
                    int64_t largeElementLimit);

  // Runs one or more pass pipelines (as is mlir::parsePassPipeline).
  void RunPassPipeline(const std::vector<std::string>& pipelines);

  // Compile to a VM module.
  std::shared_ptr<OpaqueBlob> Compile(
      mlir::iree_compiler::IREE::VM::BytecodeTargetOptions options,
      std::vector<std::string> target_backends);

 private:
  std::shared_ptr<CompilerContextBundle> context_;
  mlir::ModuleOp module_op_;
};

// Registers to receive diagnostics for a scope.
// When this goes out of scope, any remaining diagnostics will be added to
// the parent.
class DiagnosticCapture {
 public:
  DiagnosticCapture(mlir::MLIRContext* mlir_context, DiagnosticCapture* parent);
  ~DiagnosticCapture();
  DiagnosticCapture(DiagnosticCapture&& other);

  std::vector<mlir::Diagnostic>& diagnostics() { return diagnostics_; }

  // Consumes/clears diagnostics.
  std::string ConsumeDiagnosticsAsString(const char* error_message);
  void ClearDiagnostics();

 private:
  mlir::MLIRContext* mlir_context_;
  DiagnosticCapture* parent_;
  std::vector<mlir::Diagnostic> diagnostics_;
  mlir::DiagnosticEngine::HandlerID handler_id_;
};

// Bundle of MLIRContext related things that facilitates interop with
// Python.
class CompilerContextBundle
    : public std::enable_shared_from_this<CompilerContextBundle> {
 public:
  CompilerContextBundle();
  ~CompilerContextBundle();

  mlir::MLIRContext* mlir_context() { return &mlir_context_; }

  CompilerModuleBundle ParseAsm(const std::string& asm_text);

  // Gets the default diagnostic capture.
  DiagnosticCapture& DefaultDiagnosticCapture() { return default_capture_; }

  // Creates a new diagnostic region.
  // Note that this only supports one deep at present.
  DiagnosticCapture CaptureDiagnostics() {
    return DiagnosticCapture(&mlir_context_, &default_capture_);
  }

  // Consumes/clears diagnostics.
  std::string ConsumeDiagnosticsAsString() {
    return default_capture_.ConsumeDiagnosticsAsString(nullptr);
  }
  void ClearDiagnostics() { default_capture_.ClearDiagnostics(); }

  // Default crash reproducer path.
  static absl::optional<std::string> default_crash_reproducer_path() {
    std::lock_guard<std::mutex> lock(static_config_lock_);
    return default_crash_reproducer_path_;
  }
  static void set_default_crash_reproducer_path(
      absl::optional<std::string> default_crash_reproducer_path) {
    std::lock_guard<std::mutex> lock(static_config_lock_);
    default_crash_reproducer_path_ = std::move(default_crash_reproducer_path);
  }

  // Crash reproducer (if not set, uses the static default).
  // If neither are set or are the empty string, then the crash reproducer
  // will not be used.
  absl::optional<std::string> crash_reproducer_path() const {
    if (crash_reproducer_path_) {
      return crash_reproducer_path_;
    }
    return default_crash_reproducer_path();
  }
  void set_crash_reproducer_path(
      absl::optional<std::string> crash_reproducer_path) {
    crash_reproducer_path_ = std::move(crash_reproducer_path);
  }

 private:
  static std::mutex static_config_lock_;
  static absl::optional<std::string> default_crash_reproducer_path_;

  mlir::MLIRContext mlir_context_;
  DiagnosticCapture default_capture_;
  absl::optional<std::string> crash_reproducer_path_;
};

void SetupCommonCompilerBindings(pybind11::module m);

}  // namespace python
}  // namespace iree

#endif  // IREE_BINDINGS_PYTHON_PYIREE_COMPILER_H_
