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

#ifndef IREE_BINDINGS_PYTHON_PYIREE_INITIALIZE_H_
#define IREE_BINDINGS_PYTHON_PYIREE_INITIALIZE_H_

#include <vector>

namespace iree {
namespace python {

// Performs once-only initialization of the extension, which is required
// prior to any use of the runtime. Optionally, arguments can be provided.
// If automatic initialization has already taken place, then does nothing.
// In the future, it would be nice to have more of the process level init
// happen automatically and rely less on this kind of init the world
// function.
void InitializeExtension(const std::vector<std::string>& arguments);

}  // namespace python
}  // namespace iree

#endif  // IREE_BINDINGS_PYTHON_PYIREE_INITIALIZE_H_
