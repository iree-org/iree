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

#ifndef IREE_BASE_INTERNAL_INIT_INTERNAL_H_
#define IREE_BASE_INTERNAL_INIT_INTERNAL_H_

#include <map>
#include <string>

#include "iree/base/target_platform.h"

namespace iree {

// A static instance of this class is declared for each piece of initialization
// code using the initializer macros.
class Initializer {
 public:
  typedef void (*InitializerFunc)();

  Initializer(const char* name, InitializerFunc function);

  // Runs all registered initializers that have not yet run.
  // The initializers are invoked in lexicographically increasing order by name,
  // except as necessary to satisfy dependencies.
  //
  // This is normally called by InitializeEnvironment(), so application code
  // typically should not call it directly.
  static void RunInitializers();

  // Runs this initializer if it has not yet run, including any dependencies.
  void Require();

  struct Dependency {
    Dependency(const char* n, Initializer* i) : name(n), initializer(i) {}
    const char* const name;
    Initializer* const initializer;
  };

  // A static instance of this class is declared for each piece of
  // initializer ordering definition.
  struct DependencyRegisterer {
    DependencyRegisterer(const char* name, Initializer* initializer,
                         const Dependency& dependency);
  };

  struct InitializerData;
  typedef std::map<std::string, InitializerData> NameMap;

 private:
  static NameMap* InitializerNameMap();
  static void RunInitializer(InitializerData* initializer_data);

  const std::string name_;
  InitializerFunc function_;
  bool done_;
};

// In iree/base/init.h:
void InitializeEnvironment(int* argc, char*** argv);

}  // namespace iree

#define IREE_DECLARE_MODULE_INITIALIZER(name) \
  extern ::iree::Initializer iree_initializer_##name

#define IREE_REGISTER_MODULE_INITIALIZER(name, body) \
  static void iree_init_##name() { body; }           \
  ::iree::Initializer iree_initializer_##name(#name, iree_init_##name)

#define IREE_REGISTER_MODULE_INITIALIZER_SEQUENCE(name1, name2)                \
  namespace {                                                                  \
  static ::iree::Initializer::DependencyRegisterer                             \
      iree_initializer_dependency_##name1##_##name2(                           \
          #name2, &iree_initializer_##name2,                                   \
          ::iree::Initializer::Dependency(#name1, &iree_initializer_##name1)); \
  }

#define IREE_REQUIRE_MODULE_INITIALIZED(name) \
  do {                                        \
    IREE_DECLARE_MODULE_INITIALIZER(name);    \
    iree_initializer_##name.Require();        \
  } while (0)

#define IREE_RUN_MODULE_INITIALIZERS()      \
  do {                                      \
    ::iree::Initializer::RunInitializers(); \
  } while (0)

#if !defined(IREE_COMPILER_MSVC)
#define IREE_ATTRIBUTE_USED __attribute__((used))
#else
#define IREE_ATTRIBUTE_USED
#endif  // IREE_COMPILER_MSVC

#define IREE_REQUIRE_MODULE_LINKED(name)                                   \
  IREE_ATTRIBUTE_USED static ::iree::Initializer* iree_module_ref_##name = \
      &iree_initializer_##name

#endif  // IREE_BASE_INTERNAL_INIT_INTERNAL_H_
