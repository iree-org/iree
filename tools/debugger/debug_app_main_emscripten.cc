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

// Emscripten debug_app entry point.
// Though we are using SDL here we need to do some emscripten-specific magic to
// handle the different main looping mode (as we can't block in main() like on
// other platforms) as well as support some emscripten-specific features for
// file upload/download/etc.

#include <SDL.h>
#include <emscripten.h>

#include "base/init.h"
#include "tools/debugger/debug_app.h"

namespace iree {
namespace rt {
namespace debug {

extern "C" int main(int argc, char** argv) {
  InitializeEnvironment(&argc, &argv);

  if (SDL_Init(SDL_INIT_VIDEO) != 0) {
    printf("Error: %s\n", SDL_GetError());
    return -1;
  }

  const char* glsl_version = "#version 100";
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);

  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
  SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
  SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
  SDL_DisplayMode current;
  SDL_GetCurrentDisplayMode(0, &current);
  SDL_WindowFlags window_flags = (SDL_WindowFlags)(
      SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
  SDL_Window* window =
      SDL_CreateWindow("IREE Debugger", SDL_WINDOWPOS_CENTERED,
                       SDL_WINDOWPOS_CENTERED, 1280, 720, window_flags);
  SDL_GLContext gl_context = SDL_GL_CreateContext(window);
  if (!gl_context) {
    printf("Failed to initialize WebGL context!\n");
    return 1;
  }

  auto app = absl::make_unique<DebugApp>(window, gl_context, glsl_version);
  ::emscripten_set_main_loop_arg(DebugApp::PumpMainLoopThunk, app.release(), 0,
                                 false);
  return 0;
}

}  // namespace debug
}  // namespace rt
}  // namespace iree
