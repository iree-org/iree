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

// Vulkan "Hello Triangle" + IREE API Integration Sample.

#include <SDL.h>

#include <string>

#include "iree/base/api.h"
#include "iree/base/logging.h"
#include "iree/hal/api.h"
#include "iree/rt/api.h"
#include "iree/vm/api.h"

extern "C" int main(int argc, char** argv) {
  iree_api_version_t actual_version;
  iree_status_t status =
      iree_api_version_check(IREE_API_VERSION_LATEST, &actual_version);

  if (status != IREE_STATUS_OK) {
    LOG(FATAL) << "Unsupported runtime API version " << actual_version;
  } else {
    LOG(INFO) << "IREE runtime API version " << actual_version;
  }

  // --------------------------------------------------------------------------
  // Create a window.
  if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0) {
    LOG(FATAL) << "Failed to initialize SDL";
    return 1;
  }

  // Setup window
  SDL_WindowFlags window_flags = (SDL_WindowFlags)(
      SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
  SDL_Window* window =
      SDL_CreateWindow("IREE Samples - Vulkan Triangle", SDL_WINDOWPOS_CENTERED,
                       SDL_WINDOWPOS_CENTERED, 1280, 720, window_flags);

  // --------------------------------------------------------------------------

  // Main loop
  bool done = false;
  while (!done) {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_QUIT) {
        done = true;
      }
    }
  }

  // TODO(scotttodd): Vulkan Hello Triangle

  // TODO(scotttodd): ImGui
  // https://github.com/ocornut/imgui/blob/master/examples/example_sdl_vulkan/main.cpp

  // Cleanup
  SDL_DestroyWindow(window);
  SDL_Quit();

  return 0;
}
