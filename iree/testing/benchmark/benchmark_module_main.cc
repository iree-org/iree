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

#include <iostream>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "benchmark/benchmark.h"
#include "iree/base/file_io.h"
#include "iree/base/init.h"
#include "iree/base/shaped_buffer_string_util.h"
#include "iree/base/status.h"
#include "iree/testing/benchmark/benchmark_module.h"
#include "iree/vm/bytecode_module.h"

ABSL_FLAG(std::string, main_module, "", "Main module with entry point.");
ABSL_FLAG(std::string, main_function, "",
          "Function within the main module to execute.");

ABSL_FLAG(std::string, input_values, "", "Input shapes and optional values.");
ABSL_FLAG(std::string, input_file, "",
          "Input shapes and optional values serialized in a file.");

namespace iree {
namespace {

// Parses a list of input shapes and values from a string of newline-separated
// inputs. Expects the contents to have one value per line with each value
// listed as
//   [shape]xtype=[value]
// Example:
//   4x4xi8=0,1,2,3
StatusOr<std::vector<ShapedBuffer>> ParseInputsFromString(
    absl::string_view contents) {
  std::vector<ShapedBuffer> inputs;
  for (const auto& line :
       absl::StrSplit(contents, '\n', absl::SkipWhitespace())) {
    ASSIGN_OR_RETURN(auto input, ParseShapedBufferFromString(line));
    inputs.push_back(std::move(input));
  }
  return inputs;
}

Status Run(benchmark::State& state) {
  ASSIGN_OR_RETURN(
      auto main_module_file,
      vm::ModuleFile::LoadFile(ModuleDefIdentifier(),
                               absl::GetFlag(FLAGS_main_module)),
      _ << "while loading module file " << absl::GetFlag(FLAGS_main_module));

  std::string arguments_file_contents;
  if (!absl::GetFlag(FLAGS_input_values).empty()) {
    arguments_file_contents =
        absl::StrReplaceAll(absl::GetFlag(FLAGS_input_values), {{"\\n", "\n"}});
  } else if (!absl::GetFlag(FLAGS_input_file).empty()) {
    ASSIGN_OR_RETURN(arguments_file_contents,
                     file_io::GetFileContents(absl::GetFlag(FLAGS_input_file)));
  }

  ASSIGN_OR_RETURN(auto arguments,
                   ParseInputsFromString(arguments_file_contents));

  return RunModuleBenchmark(state, std::move(main_module_file),
                            absl::GetFlag(FLAGS_main_function),
                            /*driver_name=*/"interpreter", arguments);
}

void BM_RunModule(benchmark::State& state) {
  // Delegate to a status-returning function so we can use the status macros.
  CHECK_OK(Run(state));
}

// By default only the main thread is included in CPU time. Include all the
// threads instead. To make single and multi-threaded benchmarks more
// comparable, use the wall time to determine how many iterations to run.
// See https://github.com/google/benchmark#cpu-timers,
BENCHMARK(BM_RunModule)->MeasureProcessCPUTime()->UseRealTime();

}  // namespace

extern "C" int main(int argc, char** argv) {
  // The benchmark library uses a different mechanism for its flags. This
  // consumes any arguments it understands from argv. It must come before
  // InitializeEnvironment to avoid failures on unknown flags.
  ::benchmark::Initialize(&argc, argv);
  InitializeEnvironment(&argc, &argv);
  size_t run_benchmark_count = ::benchmark::RunSpecifiedBenchmarks();
  CHECK_GT(run_benchmark_count, 0) << "No benchmarks were run";
  return 0;
}

}  // namespace iree
