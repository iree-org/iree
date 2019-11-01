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

#include <mutex>  // NOLINT

#include "bindings/python/pyiree/binding.h"
#include "bindings/python/pyiree/compiler.h"
#include "bindings/python/pyiree/hal.h"
#include "bindings/python/pyiree/rt.h"
#include "bindings/python/pyiree/status_utils.h"
#include "bindings/python/pyiree/tf_interop/register_tensorflow.h"
#include "bindings/python/pyiree/vm.h"
#include "iree/base/initializer.h"
#include "iree/base/tracing.h"
#include "wtf/event.h"
#include "wtf/macros.h"

namespace iree {
namespace python {

namespace {

// Wrapper around wtf::ScopedEvent to make it usable as a python context
// object.
class PyScopedEvent {
 public:
  PyScopedEvent(std::string name_spec)
      : scoped_event_(InternEvent(std::move(name_spec))) {}

  bool Enter() {
    if (scoped_event_) {
      scoped_event_->Enter();
      return true;
    }
    return false;
  }

  void Exit(py::args args) {
    if (scoped_event_) scoped_event_->Leave();
  }

 private:
  static ::wtf::ScopedEvent<>* InternEvent(std::string name_spec) {
    if (!::wtf::kMasterEnable) return nullptr;
    std::lock_guard<std::mutex> lock(mu_);
    auto it = scoped_event_intern_.find(name_spec);
    if (it == scoped_event_intern_.end()) {
      // Name spec must live forever.
      std::string* dup_name_spec = new std::string(std::move(name_spec));
      // So must the event.
      auto scoped_event = new ::wtf::ScopedEvent<>(dup_name_spec->c_str());
      scoped_event_intern_.insert(std::make_pair(*dup_name_spec, scoped_event));
      return scoped_event;
    } else {
      return it->second;
    }
  }

  static std::mutex mu_;
  static std::unordered_map<std::string, ::wtf::ScopedEvent<>*>
      scoped_event_intern_;
  ::wtf::ScopedEvent<>* scoped_event_;
};

std::mutex PyScopedEvent::mu_;
std::unordered_map<std::string, ::wtf::ScopedEvent<>*>
    PyScopedEvent::scoped_event_intern_;

void SetupTracingBindings(pybind11::module m) {
  m.def("enable_thread", []() { WTF_AUTO_THREAD_ENABLE(); });
  m.def("is_available", []() { return IsTracingAvailable(); });
  m.def(
      "flush",
      [](absl::optional<std::string> explicit_trace_path) {
        absl::optional<absl::string_view> sv_path;
        if (explicit_trace_path) sv_path = explicit_trace_path;
        FlushTrace(explicit_trace_path);
      },
      py::arg("explicit_trace_path") = absl::optional<absl::string_view>());
  m.def(
      "autoflush",
      [](float period) { StartTracingAutoFlush(absl::Seconds(period)); },
      py::arg("period") = 5.0f);
  m.def("stop", []() { StopTracing(); });

  py::class_<PyScopedEvent>(m, "ScopedEvent")
      .def(py::init<std::string>())
      .def("__enter__", &PyScopedEvent::Enter)
      .def("__exit__", &PyScopedEvent::Exit);
}

}  // namespace

PYBIND11_MODULE(binding, m) {
  IREE_RUN_MODULE_INITIALIZERS();

  m.doc() = "IREE Binding Backend Helpers";
  py::class_<OpaqueBlob, std::shared_ptr<OpaqueBlob>>(m, "OpaqueBlob");

  auto compiler_m = m.def_submodule("compiler", "IREE compiler support");
  SetupCompilerBindings(compiler_m);

  auto hal_m = m.def_submodule("hal", "IREE HAL support");
  SetupHalBindings(hal_m);

  auto rt_m = m.def_submodule("rt", "IREE RT api");
  SetupRtBindings(rt_m);

  auto vm_m = m.def_submodule("vm", "IREE VM api");
  SetupVmBindings(vm_m);

  auto tracing_m = m.def_submodule("tracing", "IREE tracing api");
  SetupTracingBindings(tracing_m);

// TensorFlow.
#if defined(IREE_TENSORFLOW_ENABLED)
  auto tf_m = m.def_submodule("tf_interop", "IREE TensorFlow interop");
  SetupTensorFlowBindings(tf_m);
#endif
}

}  // namespace python
}  // namespace iree
