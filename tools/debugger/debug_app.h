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

#ifndef IREE_TOOLS_DEBUGGER_DEBUG_APP_H_
#define IREE_TOOLS_DEBUGGER_DEBUG_APP_H_

#include <SDL.h>

#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "base/status.h"
#include "rt/debug/debug_client.h"

// NOTE: order matters here, imgui must come first:
#include "third_party/dear_imgui/imgui.h"
// NOTE: must follow imgui.h:
#include "third_party/dear_imgui/examples/imgui_impl_opengl3.h"
#include "third_party/dear_imgui/examples/imgui_impl_sdl.h"

namespace iree {
namespace rt {
namespace debug {

// Debug client app UI.
// Uses a DebugClient to communicate with a remote DebugServer and ImGui to
// display a nifty UI.
//
// See the ImGui site for more info: https://github.com/ocornut/imgui
// The most useful thing is the imgui_demo.cpp file that contains example usage
// of most features.
class DebugApp : private DebugClient::Listener {
 public:
  struct UserBreakpoint {
    RemoteBreakpoint::Type type = RemoteBreakpoint::Type::kBytecodeFunction;
    const RemoteBreakpoint* active_breakpoint = nullptr;
    bool wants_enabled = true;
    bool is_enabling = false;
    // TODO(benvanik): reuse BreakpointDef here?
    std::string module_name;
    std::string function_name;
    int function_ordinal = -1;
    int bytecode_offset = 0;
    std::string native_function;
  };

  static void PumpMainLoopThunk(void* arg);

  DebugApp(SDL_Window* window, SDL_GLContext gl_context,
           const char* glsl_version);
  ~DebugApp();

  // Connects to the service at the specified address.
  Status Connect(absl::string_view service_address);
  // Disconnects from the currently connected service, if any.
  Status Disconnect();

  // Returns true if the remote service is paused at our request.
  bool is_paused() const;

  // Pumps the main UI loop once.
  // This polls the DebugClient, SDL input, and renders the UI.
  // It should be called as frequently as possible to ensure snappy UI updates.
  // Returns CancelledError if the app is being closed by the user.
  Status PumpMainLoop();

  // Defines how NavigationToCodeView methods behave.
  enum class NavigationMode {
    // The target will be opened in a new document tab.
    kNewDocument,
    // The target will be opened in the current document tab, replacing the
    // current contents.
    kCurrentDocument,
    // The target will be opened in a document tab that mostly matches (like
    // the same function in a module at a different offset), otherwise a new
    // document will be opened.
    kMatchDocument,
  };

  // Navigates to a particular function offset based on resolution of the given
  // arguments. Navigation may happen asynchronously if targets need to be
  // resolved or contents fetched.
  Status NavigateToCodeView(absl::string_view module_name, int function_ordinal,
                            int offset, NavigationMode navigation_mode);
  Status NavigateToCodeView(absl::string_view module_name,
                            absl::string_view function_name, int offset,
                            NavigationMode navigation_mode);
  Status NavigateToCodeView(const RemoteInvocation& invocation,
                            int stack_frame_index,
                            NavigationMode navigation_mode);
  Status NavigateToCodeView(const UserBreakpoint& user_breakpoint,
                            NavigationMode navigation_mode);

 private:
  struct CodeViewDocument {
    // Document display title (and ID).
    std::string title;
    // Function (and offset within the function) being displayed.
    RemoteFunction* function = nullptr;
    int bytecode_offset = 0;
    // Set to a bytecode offset to have the document focus there.
    absl::optional<int> focus_offset;
    // Cached info for bytecode display.
    struct {
      std::vector<std::string> lines;
    } bytecode_info;
  };

  CodeViewDocument* FindMatchingDocument(absl::string_view module_name,
                                         int function_ordinal);
  RemoteInvocation* GetSelectedInvocation() const;

  Status RefreshActiveBreakpoints();
  bool IsStoppedAtBreakpoint(const UserBreakpoint& user_breakpoint) const;
  int FindMatchingUserBreakpointIndex(absl::string_view module_name,
                                      int function_ordinal, int offset);
  int FindMatchingUserBreakpointIndex(absl::string_view module_name,
                                      absl::string_view function_name,
                                      int offset);
  Status ResumeFromBreakpoint(UserBreakpoint* user_breakpoint);

  Status OnContextRegistered(const RemoteContext& context) override;
  Status OnContextUnregistered(const RemoteContext& context) override;
  Status OnModuleLoaded(const RemoteContext& context,
                        const RemoteModule& module) override;
  Status OnInvocationRegistered(const RemoteInvocation& invocation) override;
  Status OnInvocationUnregistered(const RemoteInvocation& invocation) override;
  Status OnBreakpointHit(const RemoteBreakpoint& breakpoint,
                         const RemoteInvocation& invocation) override;

  Status LayoutInitialDockSpace();

  Status DrawUI();
  Status DrawMainMenu();
  Status DrawToolbar();

  Status DrawBreakpointListPanel();
  StatusOr<bool> DrawBreakpoint(UserBreakpoint* user_breakpoint);
  Status DrawAddBreakpointDialogs(
      absl::optional<RemoteBreakpoint::Type> add_breakpoint_type);
  Status DrawAddBytecodeFunctionBreakpointDialog();
  Status DrawAddNativeFunctionBreakpointDialog();

  Status DrawModuleListPanel();
  Status DrawContext(const RemoteContext& context,
                     const ImGuiTextFilter& filter);
  Status DrawModule(RemoteModule* module, const ImGuiTextFilter& filter);

  Status DrawLocalListPanel();
  Status DrawLocal(RemoteInvocation* invocation, int stack_frame_index,
                   int local_index, const rpc::BufferViewDefT& local);

  Status DrawInvocationListPanel();
  Status DrawInvocation(const RemoteInvocation& invocation);

  Status DrawCodeViewPanels();
  StatusOr<bool> DrawCodeViewDocument(CodeViewDocument* document);
  Status PrepareBytecodeCodeView(CodeViewDocument* document);
  Status DrawBytecodeCodeView(CodeViewDocument* document);

  SDL_Window* window_ = nullptr;
  SDL_GLContext gl_context_ = nullptr;

  ImGuiID dockspace_id_;
  ImGuiID dock_top_id_;
  ImGuiID dock_left_id_;
  ImGuiID dock_bottom_id_;
  ImGuiID dock_bottom_left_id_;
  ImGuiID dock_bottom_right_id_;
  ImGuiID dock_right_id_;
  ImGuiID dock_content_id_;

  std::unique_ptr<DebugClient> debug_client_;
  std::vector<UserBreakpoint> user_breakpoint_list_;

  bool is_paused_ = false;
  std::vector<const RemoteBreakpoint*> hit_breakpoints_;
  bool is_stepping_ = false;

  absl::optional<int> selected_invocation_id_;
  absl::optional<int> selected_stack_frame_index_;

  std::vector<std::unique_ptr<CodeViewDocument>> documents_;
};

}  // namespace debug
}  // namespace rt
}  // namespace iree

#endif  // IREE_TOOLS_DEBUGGER_DEBUG_APP_H_
