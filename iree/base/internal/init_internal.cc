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

#include "iree/base/internal/init_internal.h"

#include <string.h>

#include <set>

#include "absl/flags/parse.h"

namespace iree {

static Initializer::NameMap* static_name_map = nullptr;

struct Initializer::InitializerData {
  Initializer* initializer_obj;
  std::set<std::string> dependency_names;

  InitializerData() : initializer_obj(nullptr) {}
  explicit InitializerData(Initializer* i) : initializer_obj(i) {}
};

Initializer::DependencyRegisterer::DependencyRegisterer(
    const char* name, Initializer* initializer, const Dependency& dependency) {
  NameMap* name_map = InitializerNameMap();

  // Insert 'dependency' into the 'dependency_names' set for 'initializer'.
  InitializerData* initializer_data = &(*name_map)[name];
  initializer_data->dependency_names.insert(dependency.name);

  // Ensure that 'dependency' exists in the map.
  InitializerData* dependency_data = &(*name_map)[dependency.name];
  dependency_data->initializer_obj = dependency.initializer;
}

Initializer::Initializer(const char* name, InitializerFunc function)
    : name_(name), function_(function), done_(false) {
  // Register this Initializer instance (wrapped by an InitializerData) within
  // the static name map.
  NameMap* name_map = InitializerNameMap();
  InitializerData* initializer_data = &(*name_map)[name];
  initializer_data->initializer_obj = this;
}

void Initializer::RunInitializers() {
  // Run each registered Initializer, in lexicographic order of their names.
  // Initializer dependencies will be run first as needed.
  NameMap* name_map = InitializerNameMap();
  for (auto& p : *name_map) {
    RunInitializer(&p.second);
  }
}

void Initializer::Require() {
  NameMap* name_map = InitializerNameMap();
  InitializerData* initializer_data = &(name_map->find(name_)->second);
  RunInitializer(initializer_data);
}

Initializer::NameMap* Initializer::InitializerNameMap() {
  if (static_name_map == nullptr) {
    static_name_map = new Initializer::NameMap;
  }
  return static_name_map;
}

void Initializer::RunInitializer(InitializerData* initializer_data) {
  if (initializer_data->initializer_obj->done_) {
    return;
  }

  // Run Initializer dependencies first.
  NameMap* name_map = InitializerNameMap();
  for (const auto& dependency_name : initializer_data->dependency_names) {
    auto dep_init = name_map->find(dependency_name);
    RunInitializer(&dep_init->second);
  }

  // Finally run the Initializer itself.
  initializer_data->initializer_obj->function_();
  initializer_data->initializer_obj->done_ = true;
}

void InitializeEnvironment(int* argc, char*** argv) {
  auto positional_args = absl::ParseCommandLine(*argc, *argv);
  if (positional_args.size() < *argc) {
    // Edit the passed argument refs to only include positional args.
    *argc = positional_args.size();
    for (int i = 0; i < *argc; ++i) {
      (*argv)[i] = positional_args[i];
    }
    (*argv)[*argc + 1] = nullptr;
  }

  IREE_RUN_MODULE_INITIALIZERS();
}

}  // namespace iree
