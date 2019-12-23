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

#ifndef IREE_BINDINGS_PYTHON_PYIREE_FUNCTION_ABI_H_
#define IREE_BINDINGS_PYTHON_PYIREE_FUNCTION_ABI_H_

#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "bindings/python/pyiree/binding.h"
#include "bindings/python/pyiree/hal.h"
#include "bindings/python/pyiree/host_types.h"
#include "bindings/python/pyiree/vm.h"
#include "iree/base/signature_mangle.h"

namespace iree {
namespace python {

// Forward declarations.
class HalDevice;

// Instantiated with function attributes in order to process inputs/outputs.
class FunctionAbi {
 public:
  using AttributeLookup =
      std::function<absl::optional<absl::string_view>(absl::string_view)>;
  FunctionAbi(HalDevice& device,
              std::shared_ptr<HostTypeFactory> host_type_factory)
      : device_(HalDevice::RetainAndCreate(device.raw_ptr())),
        host_type_factory_(std::move(host_type_factory)) {}
  virtual ~FunctionAbi() = default;

  using Description = RawSignatureParser::Description;
  using InputDescriptionVector = absl::InlinedVector<Description, 4>;
  using ResultDescriptionVector = absl::InlinedVector<Description, 1>;

  struct RawConfig {
    InputDescriptionVector inputs;
    ResultDescriptionVector results;

    // The following are retained to aid debugging but may be empty if
    // disabled.
    std::string signature;
  };

  // Creates an instance based on the function attributes.
  static std::unique_ptr<FunctionAbi> Create(
      HalDevice& device, std::shared_ptr<HostTypeFactory> host_type_factory,
      AttributeLookup lookup);

  RawConfig& raw_config() { return raw_config_; }
  int raw_input_arity() const { return raw_config_.inputs.size(); }
  int raw_result_arity() const { return raw_config_.results.size(); }

  // Raw packing. These always operate on the linear span of raw inputs and
  // results. Some ABIs perform a higher level of mapping on top of this,
  // which can be accessed via the non-prefixed Pack/Unpack methods.
  // Given a span of descriptions, packs the given py_args into the span
  // of function args. All spans must be of the same size.
  void RawPack(absl::Span<const Description> descs,
               absl::Span<py::handle> py_args, VmVariantList& args,
               bool writable);

  // Raw unpacks f_results into py_results.
  // Note that this consumes entries in f_results as needed, leaving them
  // as nullptr.
  // Ordinarily, this will be invoked along with AllocateResults() but it
  // is broken out for testing.
  void RawUnpack(absl::Span<const Description> descs, VmVariantList& f_results,
                 absl::Span<py::object> py_results);

  // Given bound function arguments (from RawPack or equiv) and signature
  // descriptors, allocates results for the function invocation. For fully
  // specified result types, this can be done purely by matching up
  // reflection metadata and an oracle for determining layout. For dynamically
  // shaped or data-dependent shaped results, the metadata about the function
  // arguments may be required to generate additional allocation function calls.
  // Finally, in truly data-dependent cases, some results may not be resolvable
  // ahead of time, resulting in a nullptr in f_results. In such cases, the
  // invocation must ensure proper barriers are in place to fully execute the
  // function prior to delivering results to the user layer.
  void AllocateResults(absl::Span<const Description> descs,
                       VmVariantList& f_args, VmVariantList& f_results);

  // Gets the string representation.
  std::string DebugString() const;

 private:
  void PackBuffer(const RawSignatureParser::Description& desc,
                  py::handle py_arg, VmVariantList& f_args, bool writable);

  HalDevice device_;
  std::shared_ptr<HostTypeFactory> host_type_factory_;
  RawConfig raw_config_;
};

void SetupFunctionAbiBindings(pybind11::module m);

}  // namespace python
}  // namespace iree

#endif  // IREE_BINDINGS_PYTHON_PYIREE_FUNCTION_ABI_H_
