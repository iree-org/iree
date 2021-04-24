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

#include "iree/base/internal/flags.h"
#include "iree/testing/benchmark.h"

int main(int argc, char** argv) {
  // Pass through flags to benchmark.
  // Note that we handle --help so we have to include benchmark's flags here.
  iree_flags_set_usage(
      NULL,
      "\n\n"
      "  Optional flags from third_party/benchmark/src/benchmark.cc:\n"
      "    [--benchmark_list_tests={true|false}]\n"
      "    [--benchmark_filter=<regex>]\n"
      "    [--benchmark_min_time=<min_time>]\n"
      "    [--benchmark_repetitions=<num_repetitions>]\n"
      "    [--benchmark_report_aggregates_only={true|false}]\n"
      "    [--benchmark_display_aggregates_only={true|false}]\n"
      "    [--benchmark_format=<console|json|csv>]\n"
      "    [--benchmark_out=<filename>]\n"
      "    [--benchmark_out_format=<json|console|csv>]\n"
      "    [--benchmark_color={auto|true|false}]\n"
      "    [--benchmark_counters_tabular={true|false}]\n"
      "    [--v=<verbosity>]\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_UNDEFINED_OK, &argc, &argv);
  iree_benchmark_initialize(&argc, argv);
  iree_benchmark_run_specified();
  return 0;
}
