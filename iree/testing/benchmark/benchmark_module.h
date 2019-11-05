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

#ifndef IREE_BENCHMARK_BENCHMARK_MODULE_H_
#define IREE_BENCHMARK_BENCHMARK_MODULE_H_

#include <memory>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "benchmark/benchmark.h"
#include "iree/base/shaped_buffer.h"
#include "iree/base/status.h"
#include "iree/vm/bytecode_module.h"

namespace iree {

Status RunModuleBenchmark(benchmark::State& state,
                          ref_ptr<vm::ModuleFile> main_module_file,
                          absl::string_view main_function_name,
                          absl::string_view driver_name,
                          absl::Span<const ShapedBuffer> arguments);

}  // namespace iree

#endif  // IREE_BENCHMARK_BENCHMARK_MODULE_H_
