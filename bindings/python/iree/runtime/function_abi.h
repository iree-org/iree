// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BINDINGS_PYTHON_IREE_RT_FUNCTION_ABI_H_
#define IREE_BINDINGS_PYTHON_IREE_RT_FUNCTION_ABI_H_

#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "binding.h"
#include "hal.h"
#include "host_types.h"
#include "iree/base/signature_parser.h"
#include "vm.h"

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

  // Structured packing. Linearizes structures according to the ABI and
  // delegates to RawPack.
  void Pack(pybind11::tuple& py_args, pybind11::dict& kwargs,
            absl::Span<const Description> descs, VmVariantList& args,
            bool writable);

  // Structured unpacking. Delegates to RawUnpack and delinearizes according to
  // the ABI.
  pybind11::object Unpack(absl::Span<const Description> descs,
                          VmVariantList& f_results);

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
  // If present, the SIP signature maps a "structured signature" to linearized
  // input and result lists. In layman's terms, this maps the normal python
  // *args and **kwargs calling convention with nested dicts and sequences.
  // It is used by TensorFlow, which lacks higher level types for such things.
  absl::optional<std::string> sip_signature_;
};

void SetupFunctionAbiBindings(pybind11::module m);

}  // namespace python
}  // namespace iree

#endif  // IREE_BINDINGS_PYTHON_IREE_RT_FUNCTION_ABI_H_
