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

#include "iree/tools/debugger/debug_app_embedded.h"

#include <SDL.h>

#include <thread>  // NOLINT

#include "absl/base/thread_annotations.h"
#include "absl/memory/memory.h"
#include "absl/synchronization/mutex.h"
#include "iree/base/memory.h"
#include "iree/base/status.h"
#include "iree/tools/debugger/debug_app.h"
#include "third_party/SDL2/include/SDL_thread.h"

namespace iree {
namespace rt {
namespace debug {

class InProcessEmbeddedDebugger : public EmbeddedDebugger {
 public:
  explicit InProcessEmbeddedDebugger(std::unique_ptr<DebugApp> app)
      : app_(std::move(app)) {
    thread_ =
        SDL_CreateThread(&ThreadMainThunk, "InProcessEmbeddedDebugger", this);
  }

  ~InProcessEmbeddedDebugger() override {
    VLOG(1) << "Setting shutdown flag and waiting on thread...";
    shutdown_flag_ = true;
    int status = 0;
    SDL_WaitThread(thread_, &status);
    VLOG(1) << "Thread shutdown, killing app...";
    app_.reset();
  }

  Status AwaitClose() override {
    await_mutex_.LockWhen(absl::Condition(
        +[](bool* is_shutdown) { return *is_shutdown; }, &is_shutdown_));
    auto status = std::move(shutdown_status_);
    await_mutex_.Unlock();
    return status;
  }

 private:
  static int ThreadMainThunk(void* arg) {
    return reinterpret_cast<InProcessEmbeddedDebugger*>(arg)->ThreadMain();
  }

  int ThreadMain() {
    VLOG(1) << "Thread entry";
    while (!shutdown_flag_) {
      auto status = app_->PumpMainLoop();
      if (IsCancelled(status)) {
        shutdown_flag_ = true;
        break;
      } else if (!shutdown_flag_ && !status.ok()) {
        absl::MutexLock lock(&await_mutex_);
        shutdown_status_ = std::move(status);
        // TODO(benvanik): don't check unless no one is watching.
        CHECK_OK(shutdown_status_);
      }
    }
    app_.reset();
    {
      absl::MutexLock lock(&await_mutex_);
      is_shutdown_ = true;
    }
    VLOG(1) << "Thread exit";
    return 0;
  }

  std::unique_ptr<DebugApp> app_;
  SDL_Thread* thread_;
  std::atomic<bool> shutdown_flag_ = {false};
  absl::Mutex await_mutex_;
  bool is_shutdown_ ABSL_GUARDED_BY(await_mutex_) = false;
  Status shutdown_status_ ABSL_GUARDED_BY(await_mutex_);
};

StatusOr<std::unique_ptr<EmbeddedDebugger>> LaunchDebugger() {
  return AttachDebugger("");
}

StatusOr<std::unique_ptr<EmbeddedDebugger>> AttachDebugger(
    absl::string_view service_address) {
  LOG(INFO) << "Launching embedded debugger; service=" << service_address;
  // Workaround for terrible bad SDL/graphics driver leaks.
  IREE_DISABLE_LEAK_CHECKS();

  if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0) {
    return InternalErrorBuilder(IREE_LOC)
           << "Unable to init SDL: " << SDL_GetError();
  }

#if __APPLE__
  // GL 3.2 Core + GLSL 150
  const char* glsl_version = "#version 150";
  SDL_GL_SetAttribute(
      SDL_GL_CONTEXT_FLAGS,
      SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG);  // Always required on Mac
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
#else
  // GL 3.0 + GLSL 130
  const char* glsl_version = "#version 130";
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
#endif

  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
  SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
  SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
  SDL_DisplayMode current;
  SDL_GetCurrentDisplayMode(0, &current);
  SDL_WindowFlags window_flags = (SDL_WindowFlags)(
      SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
  SDL_Window* window =
      SDL_CreateWindow("IREE Debugger (embedded)", SDL_WINDOWPOS_CENTERED,
                       SDL_WINDOWPOS_CENTERED, 1280, 720, window_flags);
  SDL_GLContext gl_context = SDL_GL_CreateContext(window);
  SDL_GL_MakeCurrent(nullptr, nullptr);

  IREE_ENABLE_LEAK_CHECKS();

  auto app = absl::make_unique<DebugApp>(window, gl_context, glsl_version);
  if (!service_address.empty()) {
    RETURN_IF_ERROR(app->Connect(service_address));
  }

  auto handle = absl::make_unique<InProcessEmbeddedDebugger>(std::move(app));
  return handle;
}

}  // namespace debug
}  // namespace rt
}  // namespace iree
