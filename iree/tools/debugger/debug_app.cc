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

#include "iree/tools/debugger/debug_app.h"

#include <GLES2/gl2.h>

#include <algorithm>
#include <cstdio>

#include "absl/flags/flag.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/types/optional.h"
#include "third_party/dear_imgui/imgui.h"
#include "third_party/dear_imgui/imgui_internal.h"
#include "iree/base/memory.h"
#include "iree/base/source_location.h"
#include "iree/base/status.h"
#include "iree/rt/debug/debug_client.h"
#include "iree/schemas/debug_service_generated.h"
#include "iree/vm/bytecode_printer.h"
#include "iree/vm/bytecode_tables_sequencer.h"
#include "iree/vm/module.h"
#include "iree/vm/source_map.h"

namespace iree {
namespace rt {
namespace debug {
namespace {

void PushButtonHue(float hue) {
  ImGui::PushStyleColor(ImGuiCol_Button,
                        (ImVec4)ImColor::HSV(hue / 7.0f, 0.6f, 0.6f));
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                        (ImVec4)ImColor::HSV(hue / 7.0f, 0.7f, 0.7f));
  ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                        (ImVec4)ImColor::HSV(hue / 7.0f, 0.8f, 0.8f));
}

void PushButtonColor(const ImVec4& color) {
  ImGui::PushStyleColor(ImGuiCol_Button, color);
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, color);
  ImGui::PushStyleColor(ImGuiCol_ButtonActive, color);
}

void PopButtonStyle() { ImGui::PopStyleColor(3); }

bool AreBreakpointsEqual(const RemoteBreakpoint& breakpoint,
                         const DebugApp::UserBreakpoint& user_breakpoint) {
  if (user_breakpoint.active_breakpoint == &breakpoint) {
    return true;
  } else if (user_breakpoint.type != breakpoint.type()) {
    return false;
  }
  switch (breakpoint.type()) {
    case RemoteBreakpoint::Type::kBytecodeFunction:
      if (user_breakpoint.function_ordinal != -1 &&
          user_breakpoint.function_ordinal != breakpoint.function_ordinal()) {
        return false;
      }
      return breakpoint.module_name() == user_breakpoint.module_name &&
             breakpoint.function_name() == user_breakpoint.function_name &&
             breakpoint.bytecode_offset() == user_breakpoint.bytecode_offset;
    case RemoteBreakpoint::Type::kNativeFunction:
      return breakpoint.function_name() == user_breakpoint.native_function;
    default:
      return false;
  }
}

}  // namespace

// static
void DebugApp::PumpMainLoopThunk(void* arg) {
  auto status = reinterpret_cast<DebugApp*>(arg)->PumpMainLoop();
  if (IsCancelled(status)) {
    return;
  } else if (!status.ok()) {
    CHECK_OK(status);
  }
}

DebugApp::DebugApp(SDL_Window* window, SDL_GLContext gl_context,
                   const char* glsl_version)
    : window_(window), gl_context_(gl_context) {
  VLOG(1) << "DebugApp initializing...";
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

  // TODO(benvanik): ini file for settings.
  io.IniFilename = nullptr;
  // ImGui::LoadIniSettingsFromMemory()
  // ImGui::SaveIniSettingsToMemory()

  // TODO(benvanik): theming.
  ImGui::StyleColorsDark();

  // Setup Platform/Renderer bindings
  ImGui_ImplSDL2_InitForOpenGL(window_, gl_context_);
  ImGui_ImplOpenGL3_Init(glsl_version);
  SDL_GL_MakeCurrent(nullptr, nullptr);
  VLOG(1) << "DebugApp initialized";
}

DebugApp::~DebugApp() {
  VLOG(1) << "DebugApp shutting down...";
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplSDL2_Shutdown();
  ImGui::DestroyContext();

  SDL_GL_DeleteContext(gl_context_);
  SDL_GL_MakeCurrent(nullptr, nullptr);
  SDL_DestroyWindow(window_);
  SDL_Quit();
  VLOG(1) << "DebugApp shut down (SDL_Quit)";
}

Status DebugApp::Connect(absl::string_view service_address) {
  VLOG(1) << "Connecting to debug service at " << service_address << "...";
  ASSIGN_OR_RETURN(debug_client_, DebugClient::Connect(service_address, this));

  // TODO(benvanik): load breakpoints from file.
  UserBreakpoint user_breakpoint;
  user_breakpoint.module_name = "module";
  user_breakpoint.function_name = "main";
  user_breakpoint.bytecode_offset = 0;
  user_breakpoint.wants_enabled = true;
  user_breakpoint_list_.push_back(std::move(user_breakpoint));
  RETURN_IF_ERROR(RefreshActiveBreakpoints());

  // Set paused so that we need to resume to continue execution.
  is_paused_ = true;
  return OkStatus();
}

Status DebugApp::Disconnect() {
  VLOG(1) << "Disconnecting from debug service";
  debug_client_.reset();
  return OkStatus();
}

bool DebugApp::is_paused() const {
  if (!debug_client_) {
    return false;
  }
  if (!hit_breakpoints_.empty()) {
    return true;  // One or more breakpoints hit.
  }
  return is_paused_ || !is_stepping_;
}

RemoteInvocation* DebugApp::GetSelectedInvocation() const {
  if (!debug_client_ || !selected_invocation_id_.has_value()) {
    return nullptr;
  }
  for (auto* invocation : debug_client_->invocations()) {
    if (invocation->id() == selected_invocation_id_.value()) {
      return invocation;
    }
  }
  return nullptr;
}

Status DebugApp::RefreshActiveBreakpoints() {
  // Set all breakpoints to disabled. We'll re-enable them as we find them
  // below.
  for (auto& user_breakpoint : user_breakpoint_list_) {
    user_breakpoint.active_breakpoint = nullptr;
  }

  // If not connected then no breakpoints are active.
  if (!debug_client_) {
    return OkStatus();
  }

  // Reconcile the user breakpoint list with the breakpoints available on the
  // server.
  for (auto* breakpoint : debug_client_->breakpoints()) {
    auto it =
        std::find_if(user_breakpoint_list_.begin(), user_breakpoint_list_.end(),
                     [breakpoint](const UserBreakpoint& user_breakpoint) {
                       return AreBreakpointsEqual(*breakpoint, user_breakpoint);
                     });
    if (it == user_breakpoint_list_.end()) {
      // Breakpoint not found - add to user list.
      UserBreakpoint user_breakpoint;
      user_breakpoint.type = breakpoint->type();
      user_breakpoint.active_breakpoint = breakpoint;
      user_breakpoint.module_name = breakpoint->module_name();
      user_breakpoint.function_name = breakpoint->function_name();
      user_breakpoint.function_ordinal = breakpoint->function_ordinal();
      user_breakpoint.bytecode_offset = breakpoint->bytecode_offset();
      user_breakpoint_list_.push_back(std::move(user_breakpoint));
    } else {
      // Breakpoint found - set the active pointer.
      UserBreakpoint& user_breakpoint = *it;
      user_breakpoint.active_breakpoint = breakpoint;
      user_breakpoint.is_enabling = false;
      user_breakpoint.module_name = breakpoint->module_name();
      user_breakpoint.function_name = breakpoint->function_name();
      user_breakpoint.function_ordinal = breakpoint->function_ordinal();
      user_breakpoint.bytecode_offset = breakpoint->bytecode_offset();
    }
  }

  // Ensure any breakpoint the user wants enabled is active/otherwise.
  for (auto& user_breakpoint : user_breakpoint_list_) {
    if (user_breakpoint.wants_enabled && !user_breakpoint.is_enabling &&
        !user_breakpoint.active_breakpoint) {
      // Add breakpoint on server.
      switch (user_breakpoint.type) {
        case RemoteBreakpoint::Type::kBytecodeFunction:
          RETURN_IF_ERROR(debug_client_->AddFunctionBreakpoint(
              user_breakpoint.module_name, user_breakpoint.function_name,
              user_breakpoint.bytecode_offset,
              [&user_breakpoint](const RemoteBreakpoint& breakpoint) {
                user_breakpoint.function_ordinal =
                    breakpoint.function_ordinal();
              }));
          break;
        case RemoteBreakpoint::Type::kNativeFunction:
          // TODO(benvanik): native breakpoint support.
          return UnimplementedErrorBuilder(IREE_LOC)
                 << "Native function breakpoints are TODO";
        default:
          return UnimplementedErrorBuilder(IREE_LOC)
                 << "Unimplemented breakpoint type";
      }
      user_breakpoint.is_enabling = true;
    } else if (!user_breakpoint.wants_enabled &&
               user_breakpoint.active_breakpoint) {
      // Remove breakpoint from server.
      RETURN_IF_ERROR(
          debug_client_->RemoveBreakpoint(*user_breakpoint.active_breakpoint));

      user_breakpoint.active_breakpoint = nullptr;
    }
  }

  return OkStatus();
}

bool DebugApp::IsStoppedAtBreakpoint(
    const UserBreakpoint& user_breakpoint) const {
  return std::find(hit_breakpoints_.begin(), hit_breakpoints_.end(),
                   user_breakpoint.active_breakpoint) != hit_breakpoints_.end();
}

int DebugApp::FindMatchingUserBreakpointIndex(absl::string_view module_name,
                                              int function_ordinal,
                                              int offset) {
  for (int i = 0; i < user_breakpoint_list_.size(); ++i) {
    auto& user_breakpoint = user_breakpoint_list_[i];
    if (user_breakpoint.module_name == module_name &&
        user_breakpoint.function_ordinal == function_ordinal &&
        user_breakpoint.bytecode_offset == offset) {
      return i;
    }
  }
  return -1;
}

int DebugApp::FindMatchingUserBreakpointIndex(absl::string_view module_name,
                                              absl::string_view function_name,
                                              int offset) {
  for (int i = 0; i < user_breakpoint_list_.size(); ++i) {
    auto& user_breakpoint = user_breakpoint_list_[i];
    if (user_breakpoint.module_name == module_name &&
        user_breakpoint.function_name == function_name &&
        user_breakpoint.bytecode_offset == offset) {
      return i;
    }
  }
  return -1;
}

Status DebugApp::ResumeFromBreakpoint(UserBreakpoint* user_breakpoint) {
  if (!user_breakpoint->active_breakpoint) {
    return FailedPreconditionErrorBuilder(IREE_LOC) << "Breakpoint not active";
  }
  VLOG(1) << "Resuming from breakpoint "
          << user_breakpoint->active_breakpoint->id() << "...";
  auto it = std::find(hit_breakpoints_.begin(), hit_breakpoints_.end(),
                      user_breakpoint->active_breakpoint);
  if (it == hit_breakpoints_.end()) {
    return NotFoundErrorBuilder(IREE_LOC) << "Breakpoint not found";
  }
  hit_breakpoints_.erase(it);
  return debug_client_->MakeReady();
}

Status DebugApp::OnContextRegistered(const RemoteContext& context) {
  // Ack event.
  return debug_client_->MakeReady();
}

Status DebugApp::OnContextUnregistered(const RemoteContext& context) {
  // Close documents that may reference modules in the context.
  std::vector<CodeViewDocument*> closing_documents;
  for (auto& document : documents_) {
    auto* module = document->function->module();
    if (module->context_id() != context.id()) {
      // Document is not from this context so it's fine.
      continue;
    }

    // See if any other live context still has the module loaded. We can change
    // the document over to that.
    RemoteModule* replacement_module = nullptr;
    for (auto* context : debug_client_->contexts()) {
      for (auto* other_module : context->modules()) {
        if (other_module->name() == module->name()) {
          replacement_module = other_module;
          break;
        }
      }
      if (replacement_module) break;
    }
    if (replacement_module && replacement_module->is_loaded()) {
      // Replace document module reference.
      int function_ordinal = document->function->ordinal();
      auto functions = replacement_module->functions();
      if (function_ordinal < functions.size()) {
        document->function = functions[function_ordinal];
      } else {
        document->function = nullptr;
      }
    } else {
      document->function = nullptr;
    }

    if (!document->function) {
      // Close the document if we don't have a valid function for it.
      VLOG(1)
          << "Closing document " << document->title
          << " because the last context using the module is being unregistered";
      closing_documents.push_back(document.get());
    }
  }
  for (auto* document : closing_documents) {
    auto it = std::find_if(
        documents_.begin(), documents_.end(),
        [document](const std::unique_ptr<CodeViewDocument>& open_document) {
          return document == open_document.get();
        });
    documents_.erase(it);
  }

  // Ack event.
  return debug_client_->MakeReady();
}

Status DebugApp::OnModuleLoaded(const RemoteContext& context,
                                const RemoteModule& module) {
  // Ack event.
  return debug_client_->MakeReady();
}

Status DebugApp::OnInvocationRegistered(const RemoteInvocation& invocation) {
  if (!selected_invocation_id_.has_value()) {
    selected_invocation_id_ = invocation.id();
    selected_stack_frame_index_ = {};
  }

  // Ack event.
  return debug_client_->MakeReady();
}

Status DebugApp::OnInvocationUnregistered(const RemoteInvocation& invocation) {
  if (selected_invocation_id_.has_value() &&
      selected_invocation_id_.value() == invocation.id()) {
    selected_invocation_id_ = {};
    selected_stack_frame_index_ = {};
  }

  // Ack event.
  return debug_client_->MakeReady();
}

Status DebugApp::OnBreakpointHit(const RemoteBreakpoint& breakpoint,
                                 const RemoteInvocation& invocation) {
  // Keep track of where we are stopped.
  hit_breakpoints_.push_back(&breakpoint);
  return NavigateToCodeView(invocation, -1, NavigationMode::kMatchDocument);
}

Status DebugApp::PumpMainLoop() {
  ImGuiIO& io = ImGui::GetIO();

  if (debug_client_) {
    RETURN_IF_ERROR(debug_client_->Poll());
  }
  RETURN_IF_ERROR(RefreshActiveBreakpoints());

  SDL_GL_MakeCurrent(window_, gl_context_);

  SDL_Event event;
  while (SDL_PollEvent(&event)) {
    ImGui_ImplSDL2_ProcessEvent(&event);
    if (event.type == SDL_QUIT) {
      return CancelledErrorBuilder(IREE_LOC) << "Quit hotkey";
    } else if (event.type == SDL_WINDOWEVENT &&
               event.window.event == SDL_WINDOWEVENT_CLOSE &&
               event.window.windowID == SDL_GetWindowID(window_)) {
      return CancelledErrorBuilder(IREE_LOC) << "Window closed";
    }
  }
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplSDL2_NewFrame(window_);
  ImGui::NewFrame();

  auto draw_status = DrawUI();
  if (!draw_status.ok()) {
    // TODO(benvanik): show on screen? Probably all messed up.
    LOG(ERROR) << draw_status;
  }

  // Blit the entire ImGui UI.
  ImGui::Render();
  SDL_GL_MakeCurrent(window_, gl_context_);
  glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);
  glClearColor(0.45f, 0.55f, 0.60f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT);
  // Workaround for terrible bad SDL/graphics driver leaks.
  IREE_DISABLE_LEAK_CHECKS();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
  IREE_ENABLE_LEAK_CHECKS();

  // Render additional viewport windows (desktop only).
  if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
    SDL_Window* backup_current_window = SDL_GL_GetCurrentWindow();
    SDL_GLContext backup_current_context = SDL_GL_GetCurrentContext();
    ImGui::UpdatePlatformWindows();
    ImGui::RenderPlatformWindowsDefault();
    SDL_GL_MakeCurrent(backup_current_window, backup_current_context);
  }

  SDL_GL_SwapWindow(window_);
  return OkStatus();
}

Status DebugApp::LayoutInitialDockSpace() {
  dockspace_id_ = ImGui::GetID("MainDockSpace");
  if (ImGui::DockBuilderGetNode(dockspace_id_)) {
    // Already configured.
    return OkStatus();
  }
  ImGui::DockBuilderAddNode(dockspace_id_, ImGuiDockNodeFlags_DockSpace);

  dock_content_id_ = dockspace_id_;
  dock_top_id_ = ImGui::DockBuilderSplitNode(dock_content_id_, ImGuiDir_Up,
                                             0.05f, nullptr, &dock_content_id_);
  dock_left_id_ = ImGui::DockBuilderSplitNode(
      dock_content_id_, ImGuiDir_Left, 0.20f, nullptr, &dock_content_id_);
  dock_bottom_id_ = ImGui::DockBuilderSplitNode(
      dock_content_id_, ImGuiDir_Down, 0.20f, nullptr, &dock_content_id_);
  dock_right_id_ = ImGui::DockBuilderSplitNode(
      dock_content_id_, ImGuiDir_Right, 0.20f, nullptr, &dock_content_id_);
  dock_bottom_left_id_ = ImGui::DockBuilderSplitNode(
      dock_bottom_id_, ImGuiDir_Left, 0.50f, nullptr, &dock_bottom_right_id_);

  ImGui::DockBuilderDockWindow("Toolbar", dock_top_id_);
  auto* dock_top_node = ImGui::DockBuilderGetNode(dock_top_id_);
  dock_top_node->LocalFlags = ImGuiDockNodeFlags_NoSplit |
                              ImGuiDockNodeFlags_NoResize |
                              ImGuiDockNodeFlags_AutoHideTabBar;

  ImGui::DockBuilderDockWindow("Modules", dock_left_id_);
  ImGui::DockBuilderDockWindow("Locals", dock_bottom_left_id_);
  ImGui::DockBuilderDockWindow("Invocations", dock_bottom_right_id_);
  ImGui::DockBuilderDockWindow("Breakpoints", dock_bottom_right_id_);

  ImGui::DockBuilderFinish(dockspace_id_);
  return OkStatus();
}

Status DebugApp::DrawUI() {
  ImGuiWindowFlags window_flags =
      ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;
  window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
                  ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                  ImGuiWindowFlags_NoNavFocus;

  ImGuiViewport* viewport = ImGui::GetMainViewport();
  ImGui::SetNextWindowPos(viewport->Pos);
  ImGui::SetNextWindowSize(viewport->Size);
  ImGui::SetNextWindowViewport(viewport->ID);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
  ImGui::Begin("IREEDebugRoot", nullptr, window_flags);
  ImGui::PopStyleVar(3);

  RETURN_IF_ERROR(LayoutInitialDockSpace());
  ImGui::DockSpace(dockspace_id_, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_None);

  RETURN_IF_ERROR(DrawMainMenu());
  RETURN_IF_ERROR(DrawToolbar());

  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(2, 2));
  RETURN_IF_ERROR(DrawBreakpointListPanel());
  RETURN_IF_ERROR(DrawModuleListPanel());
  RETURN_IF_ERROR(DrawLocalListPanel());
  RETURN_IF_ERROR(DrawInvocationListPanel());
  ImGui::PopStyleVar();

  RETURN_IF_ERROR(DrawCodeViewPanels());

  ImGui::End();
  return OkStatus();
}

Status DebugApp::DrawMainMenu() {
  if (!ImGui::BeginMenuBar()) return OkStatus();

  // TODO(benvanik): main menu.
  if (ImGui::BeginMenu("File")) {
    ImGui::EndMenu();
  }

  ImGui::EndMenuBar();
  return OkStatus();
}

Status DebugApp::DrawToolbar() {
  // TODO(benvanik): figure out how to make this not grow.
  ImGui::Begin("Toolbar", nullptr,
               ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar |
                   ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                   ImGuiWindowFlags_NoScrollbar);
  ImGui::BeginGroup();

#if !defined(IMGUI_DISABLE_DEMO_WINDOWS)
  static bool show_demo_window = false;
  if (ImGui::Button("Demo")) {
    show_demo_window = !show_demo_window;
  }
  if (show_demo_window) {
    ImGui::SetNextWindowDockID(dock_content_id_);
    ImGui::ShowDemoWindow(&show_demo_window);
  }
#endif  // !IMGUI_DISABLE_DEMO_WINDOWS

  ImGui::SameLine();
  if (!debug_client_) {
    if (ImGui::Button("Connect")) {
      // TODO(benvanik): connection dialog and/or autoconnect.
    }
  } else {
    if (ImGui::Button("Disconnect")) {
      debug_client_.reset();
    }
  }

  ImGui::SameLine();
  if (debug_client_) {
    ImGui::Text("<status>");
  } else {
    ImGui::TextDisabled("disconnected");
  }

  ImGui::SameLine();
  ImGui::Spacing();
  ImGui::SameLine();
  ImGui::Spacing();

  ImGui::SameLine();
  ImGui::BeginGroup();
  ImGui::Text("Invocation: ");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(300);
  auto* selected_invocation = GetSelectedInvocation();
  const std::string& active_invocation_name =
      selected_invocation ? selected_invocation->name() : "";
  if (ImGui::BeginCombo("##active_invocation", active_invocation_name.c_str(),
                        ImGuiComboFlags_PopupAlignLeft)) {
    if (debug_client_) {
      for (auto* invocation : debug_client_->invocations()) {
        ImGui::PushID(invocation->id());
        bool is_selected = invocation == selected_invocation;
        if (ImGui::Selectable(invocation->name().c_str(), is_selected)) {
          RETURN_IF_ERROR(NavigateToCodeView(*invocation, -1,
                                             NavigationMode::kMatchDocument));
        }
        if (is_selected) {
          ImGui::SetItemDefaultFocus();
        }
        ImGui::PopID();
      }
    }
    ImGui::EndCombo();
  }
  ImGui::EndGroup();

  ImGui::SameLine();
  ImGui::BeginGroup();
  static const float kPauseButtonHue = 0.0f;
  static const float kResumeButtonHue = 2.0f;
  static const float kStepButtonHue = 1.0f;
  if (debug_client_ && !is_paused()) {
    PushButtonHue(kPauseButtonHue);
    if (ImGui::Button("Pause")) {
      RETURN_IF_ERROR(debug_client_->SuspendAllInvocations());
    }
    PopButtonStyle();
  } else if (debug_client_ && is_paused()) {
    ImGui::PushStyleColor(ImGuiCol_Button, 0xFF666666);
    ImGui::PushStyleColor(ImGuiCol_Text, 0xFFAAAAAA);
    ImGui::ButtonEx("Pause", {}, ImGuiButtonFlags_Disabled);
    ImGui::PopStyleColor(2);
  }
  if (debug_client_ && is_paused()) {
    ImGui::SameLine();
    PushButtonHue(kResumeButtonHue);
    if (ImGui::Button("Resume")) {
      if (is_paused_) {
        is_paused_ = false;
        RETURN_IF_ERROR(debug_client_->MakeReady());
      }
      while (!hit_breakpoints_.empty()) {
        hit_breakpoints_.pop_back();
        RETURN_IF_ERROR(debug_client_->MakeReady());
      }
    }
    PopButtonStyle();
  } else {
    ImGui::PushStyleColor(ImGuiCol_Button, 0xFF666666);
    ImGui::PushStyleColor(ImGuiCol_Text, 0xFFAAAAAA);
    ImGui::SameLine();
    ImGui::ButtonEx("Resume", {}, ImGuiButtonFlags_Disabled);
    ImGui::PopStyleColor(2);
  }

  if (debug_client_ && is_paused() && selected_invocation) {
    ImGui::SameLine();
    PushButtonHue(kStepButtonHue);
    if (ImGui::Button("Step Into")) {
      RETURN_IF_ERROR(
          debug_client_->StepInvocation(*selected_invocation, [this]() {
            is_paused_ = true;
            is_stepping_ = false;
          }));
      is_stepping_ = true;
    }
    PopButtonStyle();
    ImGui::SameLine();
    if (ImGui::Button("Step Over")) {
      RETURN_IF_ERROR(
          debug_client_->StepInvocationOver(*selected_invocation, [this]() {
            is_paused_ = true;
            is_stepping_ = false;
          }));
      is_stepping_ = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Step Out")) {
      RETURN_IF_ERROR(
          debug_client_->StepInvocationOut(*selected_invocation, [this]() {
            is_paused_ = true;
            is_stepping_ = false;
          }));
      is_stepping_ = true;
    }
    if (ImGui::BeginPopup("Step to...")) {
      // TODO(benvanik): step to Invoke exit, next FFI call, etc
      ImGui::MenuItem("(stuff)");
      ImGui::EndPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("Step to...")) {
      ImGui::OpenPopup("Step to...");
    }
  } else {
    ImGui::PushStyleColor(ImGuiCol_Button, 0xFF666666);
    ImGui::PushStyleColor(ImGuiCol_Text, 0xFFAAAAAA);
    ImGui::SameLine();
    ImGui::ButtonEx("Step Into", {}, ImGuiButtonFlags_Disabled);
    ImGui::SameLine();
    ImGui::ButtonEx("Step Over", {}, ImGuiButtonFlags_Disabled);
    ImGui::SameLine();
    ImGui::ButtonEx("Step Out", {}, ImGuiButtonFlags_Disabled);
    ImGui::SameLine();
    ImGui::ButtonEx("Step to...", {}, ImGuiButtonFlags_Disabled);
    ImGui::PopStyleColor(2);
  }
  ImGui::EndGroup();

  ImGui::EndGroup();
  ImGui::End();
  return OkStatus();
}

Status DebugApp::DrawBreakpointListPanel() {
  static bool is_panel_visible = true;
  if (!ImGui::Begin("Breakpoints", &is_panel_visible, ImGuiWindowFlags_None)) {
    ImGui::End();
    return OkStatus();
  }

  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 8));
  absl::optional<RemoteBreakpoint::Type> add_breakpoint_type;
  if (ImGui::BeginPopup("+ Function")) {
    if (ImGui::MenuItem("Bytecode Function")) {
      add_breakpoint_type = RemoteBreakpoint::Type::kBytecodeFunction;
    }
    if (ImGui::MenuItem("Native Function")) {
      add_breakpoint_type = RemoteBreakpoint::Type::kNativeFunction;
    }
    ImGui::EndPopup();
  }
  ImGui::PopStyleVar();
  if (ImGui::Button("+ Function")) {
    ImGui::OpenPopup("+ Function");
  }
  RETURN_IF_ERROR(DrawAddBreakpointDialogs(add_breakpoint_type));

  ImGui::SameLine();
  if (ImGui::Button("Remove All")) {
    // TODO(benvanik): removal all is broken - need removebreakpoints or a
    // 'want_removal' flag so that RefreshActiveBreakpoints handles things.
    // Right now if you have 2 breakpoints and hit remove all the second will
    // come back during the next refresh (as the server hasn't removed it yet).
    for (auto& user_breakpoint : user_breakpoint_list_) {
      if (user_breakpoint.active_breakpoint) {
        RETURN_IF_ERROR(debug_client_->RemoveBreakpoint(
            *user_breakpoint.active_breakpoint));
        user_breakpoint.active_breakpoint = nullptr;
      }
    }
    user_breakpoint_list_.clear();
  }
  ImGui::Separator();

  ImGui::BeginChild("BreakpointList", ImVec2(-1, -1), false,
                    ImGuiWindowFlags_AlwaysVerticalScrollbar);
  std::vector<UserBreakpoint*> dead_breakpoints;
  for (auto& user_breakpoint : user_breakpoint_list_) {
    ASSIGN_OR_RETURN(bool should_keep, DrawBreakpoint(&user_breakpoint));
    if (!should_keep) {
      dead_breakpoints.push_back(&user_breakpoint);
    }
  }
  for (auto* user_breakpoint : dead_breakpoints) {
    for (auto it = user_breakpoint_list_.begin();
         it != user_breakpoint_list_.end(); ++it) {
      if (&*it == user_breakpoint) {
        if (user_breakpoint->active_breakpoint) {
          RETURN_IF_ERROR(debug_client_->RemoveBreakpoint(
              *user_breakpoint->active_breakpoint));
        }
        user_breakpoint_list_.erase(it);
        break;
      }
    }
  }
  ImGui::EndChild();

  ImGui::End();
  return OkStatus();
}

StatusOr<bool> DebugApp::DrawBreakpoint(UserBreakpoint* user_breakpoint) {
  std::string breakpoint_name;
  switch (user_breakpoint->type) {
    case RemoteBreakpoint::Type::kBytecodeFunction:
      breakpoint_name =
          absl::StrCat("[bytecode] ", user_breakpoint->module_name, ":",
                       user_breakpoint->function_name, ":",
                       user_breakpoint->bytecode_offset);
      if (user_breakpoint->function_ordinal != -1) {
        absl::StrAppend(&breakpoint_name, "  @",
                        user_breakpoint->function_ordinal);
      }
      break;
    case RemoteBreakpoint::Type::kNativeFunction:
      breakpoint_name =
          absl::StrCat("[native  ] ", user_breakpoint->native_function);
      break;
  }
  ImGui::BeginGroup();
  bool is_closing = true;
  bool is_expanded = ImGui::CollapsingHeader(
      ("##" + breakpoint_name).c_str(), &is_closing,
      ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_NoTreePushOnOpen |
          ImGuiTreeNodeFlags_NoAutoOpenOnLog | ImGuiTreeNodeFlags_OpenOnArrow |
          ImGuiTreeNodeFlags_OpenOnDoubleClick);
  ImGui::SameLine();
  ImGui::Checkbox(breakpoint_name.c_str(), &user_breakpoint->wants_enabled);
  ImGui::EndGroup();
  if (!is_expanded) {
    return is_closing;
  }
  ImGui::PushID(breakpoint_name.c_str());

  ImGui::Text("(breakpoint stats/etc)");

  ImGui::PopID();
  return is_closing;
}

Status DebugApp::DrawAddBreakpointDialogs(
    absl::optional<RemoteBreakpoint::Type> add_breakpoint_type) {
  if (add_breakpoint_type.has_value()) {
    switch (add_breakpoint_type.value()) {
      case RemoteBreakpoint::Type::kBytecodeFunction:
        ImGui::OpenPopup("Add Bytecode Function Breakpoint");
        break;
      case RemoteBreakpoint::Type::kNativeFunction:
        ImGui::OpenPopup("Add Native Function Breakpoint");
        break;
    }
  }
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 8));
  RETURN_IF_ERROR(DrawAddBytecodeFunctionBreakpointDialog());
  RETURN_IF_ERROR(DrawAddNativeFunctionBreakpointDialog());
  ImGui::PopStyleVar();
  return OkStatus();
}

Status DebugApp::DrawAddBytecodeFunctionBreakpointDialog() {
  ImGui::SetNextWindowSize(ImVec2(400, 400), ImGuiCond_FirstUseEver);
  bool close_popup = true;
  if (!ImGui::BeginPopupModal("Add Bytecode Function Breakpoint", &close_popup,
                              ImGuiWindowFlags_None)) {
    return OkStatus();
  }
  ImGui::BeginGroup();
  ImGui::BeginChild("##data_entry",
                    ImVec2(0, -ImGui::GetFrameHeightWithSpacing()));

  ImGui::TextWrapped(
      "Adds a breakpoint set on the entry of the function (offset=0).");
  ImGui::Separator();

  // TODO(benvanik): fancy list, filtering, etc.

  static char module_name[256] = {0};
  ImGui::InputText("Module", module_name, sizeof(module_name));
  ImGui::SetItemDefaultFocus();

  static char function_name[256] = {0};
  ImGui::InputText("Function", function_name, sizeof(function_name));

  ImGui::EndChild();
  ImGui::Separator();

  if (ImGui::Button("Add")) {
    int offset = 0;
    if (FindMatchingUserBreakpointIndex(module_name, function_name, offset) ==
        -1) {
      UserBreakpoint user_breakpoint;
      user_breakpoint.type = RemoteBreakpoint::Type::kBytecodeFunction;
      user_breakpoint.module_name = module_name;
      user_breakpoint.function_name = function_name;
      user_breakpoint.bytecode_offset = offset;
      user_breakpoint.wants_enabled = true;
      user_breakpoint_list_.push_back(std::move(user_breakpoint));
    }
    ImGui::CloseCurrentPopup();
  }
  ImGui::SameLine();
  if (ImGui::Button("Cancel")) {
    ImGui::CloseCurrentPopup();
  }

  ImGui::EndGroup();
  ImGui::EndPopup();
  return OkStatus();
}

Status DebugApp::DrawAddNativeFunctionBreakpointDialog() {
  ImGui::SetNextWindowSize(ImVec2(400, 400), ImGuiCond_FirstUseEver);
  bool close_popup = true;
  if (!ImGui::BeginPopupModal("Add Native Function Breakpoint", &close_popup,
                              ImGuiWindowFlags_None)) {
    return OkStatus();
  }
  ImGui::BeginGroup();
  ImGui::BeginChild("##data_entry",
                    ImVec2(0, -ImGui::GetFrameHeightWithSpacing()));

  ImGui::TextWrapped(
      "Adds a breakpoint set on any call to the given FFI imported "
      "function.");
  ImGui::Separator();

  static char function_name[256] = {0};
  ImGui::InputText("Function", function_name, sizeof(function_name));
  ImGui::SetItemDefaultFocus();

  ImGui::EndChild();
  ImGui::Separator();

  if (ImGui::Button("Add")) {
    UserBreakpoint user_breakpoint;
    user_breakpoint.type = RemoteBreakpoint::Type::kNativeFunction;
    user_breakpoint.native_function = function_name;
    user_breakpoint.wants_enabled = true;
    user_breakpoint_list_.push_back(std::move(user_breakpoint));
    ImGui::CloseCurrentPopup();
  }
  ImGui::SameLine();
  if (ImGui::Button("Cancel")) {
    ImGui::CloseCurrentPopup();
  }

  ImGui::EndGroup();
  ImGui::EndPopup();
  return OkStatus();
}

Status DebugApp::DrawModuleListPanel() {
  static bool is_panel_visible = true;
  if (!ImGui::Begin("Modules", &is_panel_visible, ImGuiWindowFlags_None)) {
    ImGui::End();
    return OkStatus();
  } else if (!debug_client_) {
    ImGui::TextDisabled("disconnected");
    ImGui::End();
    return OkStatus();
  }
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(4, 4));

  ImGui::BeginGroup();
  ImGui::SetNextItemWidth(ImGui::GetContentRegionAvailWidth());
  static char function_name_filter_text[256] = {0};
  ImGui::InputTextWithHint(
      "##function_name_filter", "Filter functions", function_name_filter_text,
      sizeof(function_name_filter_text), ImGuiInputTextFlags_AutoSelectAll);
  ImGuiTextFilter function_name_filter(function_name_filter_text);
  ImGui::EndGroup();

  ImGui::Separator();

  ImGui::BeginGroup();
  ImGui::BeginChild("##context_list", ImVec2(0, -ImGui::GetFrameHeight()));
  for (auto* context : debug_client_->contexts()) {
    RETURN_IF_ERROR(DrawContext(*context, function_name_filter));
  }
  ImGui::EndChild();
  ImGui::EndGroup();

  ImGui::PopStyleVar();
  ImGui::End();
  return OkStatus();
}

Status DebugApp::DrawContext(const RemoteContext& context,
                             const ImGuiTextFilter& filter) {
  std::string context_name = absl::StrCat("Context ", context.id());
  if (!ImGui::CollapsingHeader(context_name.c_str(), nullptr,
                               ImGuiTreeNodeFlags_DefaultOpen |
                                   ImGuiTreeNodeFlags_Framed |
                                   ImGuiTreeNodeFlags_NoTreePushOnOpen |
                                   ImGuiTreeNodeFlags_NoAutoOpenOnLog |
                                   ImGuiTreeNodeFlags_OpenOnArrow |
                                   ImGuiTreeNodeFlags_OpenOnDoubleClick)) {
    return OkStatus();
  }
  ImGui::PushID(context.id());
  for (auto* module : context.modules()) {
    RETURN_IF_ERROR(DrawModule(module, filter));
  }
  ImGui::PopID();
  return OkStatus();
}

Status DebugApp::DrawModule(RemoteModule* module,
                            const ImGuiTextFilter& filter) {
  ImGui::PushID(module->name().c_str());
  if (ImGui::TreeNodeEx(module->name().c_str(),
                        ImGuiTreeNodeFlags_Framed |
                            ImGuiTreeNodeFlags_DefaultOpen |
                            ImGuiTreeNodeFlags_OpenOnDoubleClick |
                            ImGuiTreeNodeFlags_OpenOnArrow)) {
    if (module->CheckLoadedOrRequest()) {
      for (auto* function : module->functions()) {
        char function_name[128];
        if (function->name().empty()) {
          std::snprintf(function_name, sizeof(function_name), "@%d",
                        function->ordinal());
        } else {
          std::snprintf(function_name, sizeof(function_name), "@%d %s",
                        function->ordinal(), function->name().c_str());
        }
        if (filter.IsActive() && !filter.PassFilter(function_name)) {
          continue;
        }
        ImGui::PushID(function->ordinal());
        bool is_selected = false;
        if (ImGui::Selectable("##selectable", &is_selected,
                              ImGuiSelectableFlags_AllowDoubleClick |
                                  ImGuiSelectableFlags_DrawFillAvailWidth)) {
          if (is_selected) {
            RETURN_IF_ERROR(NavigateToCodeView(module->name(),
                                               function->ordinal(), 0,
                                               NavigationMode::kMatchDocument));
          }
        }
        ImGui::SameLine();
        // TODO(benvanik): detect if breakpoint active at offset 0.
        ImGui::BulletText("%s", function_name);
        ImGui::PopID();
      }
    } else {
      ImGui::TextDisabled("Loading...");
    }
    ImGui::TreePop();
  }
  ImGui::PopID();
  return OkStatus();
}

Status DebugApp::DrawLocalListPanel() {
  static bool is_panel_visible = true;
  if (!ImGui::Begin("Locals", &is_panel_visible, ImGuiWindowFlags_None)) {
    ImGui::End();
    return OkStatus();
  } else if (!debug_client_) {
    ImGui::TextDisabled("disconnected");
    ImGui::End();
    return OkStatus();
  }
  auto* invocation = GetSelectedInvocation();
  if (!invocation) {
    ImGui::TextDisabled("select a invocation to view locals");
    ImGui::End();
    return OkStatus();
  } else if (invocation->def().frames.empty()) {
    ImGui::TextDisabled("(invocation has no frames)");
    ImGui::End();
    return OkStatus();
  }
  int stack_frame_index = selected_stack_frame_index_.value_or(-1);
  if (stack_frame_index == -1) {
    stack_frame_index = invocation->def().frames.size() - 1;
  }
  auto& stack_frame = invocation->def().frames[stack_frame_index];

  // TODO(benvanik): toggle for IREE VM locals vs. source locals.
  for (int i = 0; i < stack_frame->locals.size(); ++i) {
    auto& local = stack_frame->locals[i];
    RETURN_IF_ERROR(DrawLocal(invocation, stack_frame_index, i, *local));
  }

  ImGui::End();
  return OkStatus();
}

Status DebugApp::DrawLocal(RemoteInvocation* invocation, int stack_frame_index,
                           int local_index, const rpc::BufferViewDefT& local) {
  // TODO(benvanik): columns and such in fancy table.
  ImGui::Text("l%d", local_index);
  ImGui::SameLine(50);
  if (local.is_valid) {
    auto shape_str =
        absl::StrCat(absl::StrJoin(local.shape, "x"), "x", local.element_size);
    ImGui::Text("%s", shape_str.c_str());
  } else {
    ImGui::TextDisabled("∅");
  }
  // TODO(benvanik): editing options (change shape, change contents, upload).
  // TODO(benvanik): save/download/log options.
  return OkStatus();
}

Status DebugApp::DrawInvocationListPanel() {
  static bool is_panel_visible = true;
  if (!ImGui::Begin("Invocations", &is_panel_visible, ImGuiWindowFlags_None)) {
    ImGui::End();
    return OkStatus();
  } else if (!debug_client_) {
    ImGui::TextDisabled("disconnected");
    ImGui::End();
    return OkStatus();
  }
  for (auto* invocation : debug_client_->invocations()) {
    RETURN_IF_ERROR(DrawInvocation(*invocation));
  }
  ImGui::End();
  return OkStatus();
}

Status DebugApp::DrawInvocation(const RemoteInvocation& invocation) {
  // TODO(benvanik): expand if any breakpoints are stopped in invocation.
  if (selected_invocation_id_.has_value() &&
      selected_invocation_id_.value() == invocation.id()) {
    ImGui::SetNextTreeNodeOpen(true);
  }
  if (!ImGui::CollapsingHeader(invocation.name().c_str())) {
    return OkStatus();
  }
  ImGui::PushID(invocation.id());

  for (int i = 0; i < invocation.def().frames.size(); ++i) {
    const auto& stack_frame = invocation.def().frames[i];
    ImGui::PushID(i);
    // TODO(benvanik): highlight frames with breakpoints in them.
    bool is_selected = selected_invocation_id_.has_value() &&
                       selected_invocation_id_.value() == invocation.id() &&
                       selected_stack_frame_index_.has_value() &&
                       selected_stack_frame_index_.value() == i;
    if (ImGui::Selectable("##selectable", &is_selected,
                          ImGuiSelectableFlags_AllowDoubleClick |
                              ImGuiSelectableFlags_DrawFillAvailWidth)) {
      // TODO(benvanik): detect when clicking but already selected.
      if (is_selected) {
        RETURN_IF_ERROR(
            NavigateToCodeView(invocation, i, NavigationMode::kMatchDocument));
      }
    }
    ImGui::SameLine();
    ImGui::Bullet();
    ImGui::SameLine();
    // TODO(benvanik): better naming/etc (resolve function).
    ImGui::Text("%s:%d:%d", stack_frame->module_name.c_str(),
                stack_frame->function_ordinal, stack_frame->offset);

    ImGui::PopID();
  }

  ImGui::PopID();
  return OkStatus();
}

DebugApp::CodeViewDocument* DebugApp::FindMatchingDocument(
    absl::string_view module_name, int function_ordinal) {
  for (auto& document : documents_) {
    if (document->function->module()->name() == module_name &&
        document->function->ordinal() == function_ordinal) {
      return document.get();
    }
  }
  return nullptr;
}

Status DebugApp::NavigateToCodeView(absl::string_view module_name,
                                    int function_ordinal, int offset,
                                    NavigationMode navigation_mode) {
  if (!debug_client_) {
    return UnavailableErrorBuilder(IREE_LOC) << "No connection established";
  }
  VLOG(1) << "NavigateToCodeView(" << module_name << ", " << function_ordinal
          << ", " << offset << ")";
  CodeViewDocument* existing_document = nullptr;
  switch (navigation_mode) {
    case NavigationMode::kNewDocument:
      // Fall through and create below.
      break;
    case NavigationMode::kCurrentDocument:
      // Not yet done - treat as a new document.
      break;
    case NavigationMode::kMatchDocument:
      existing_document = FindMatchingDocument(module_name, function_ordinal);
      break;
  }
  if (existing_document) {
    ImGui::SetWindowFocus(existing_document->title.c_str());
    return OkStatus();
  }

  // TODO(benvanik): make this common code.
  RETURN_IF_ERROR(debug_client_->GetFunction(
      std::string(module_name), function_ordinal,
      [this, offset](StatusOr<RemoteFunction*> function_or) {
        if (!function_or.ok()) {
          // TODO(benvanik): error dialog.
          CHECK_OK(function_or.status());
        }
        auto* function = function_or.ValueOrDie();
        auto document = absl::make_unique<CodeViewDocument>();
        document->title =
            absl::StrCat(function->module()->name(), ":", function->name());
        document->function = function;
        document->focus_offset = offset;
        ImGui::SetWindowFocus(document->title.c_str());
        documents_.push_back(std::move(document));
      }));
  return OkStatus();
}

Status DebugApp::NavigateToCodeView(absl::string_view module_name,
                                    absl::string_view function_name, int offset,
                                    NavigationMode navigation_mode) {
  if (!debug_client_) {
    return UnavailableErrorBuilder(IREE_LOC) << "No connection established";
  }
  return debug_client_->ResolveFunction(
      std::string(module_name), std::string(function_name),
      [this, navigation_mode, module_name,
       offset](StatusOr<int> function_ordinal) {
        CHECK_OK(function_ordinal.status());
        CHECK_OK(NavigateToCodeView(module_name, function_ordinal.ValueOrDie(),
                                    offset, navigation_mode));
      });
}

Status DebugApp::NavigateToCodeView(const RemoteInvocation& invocation,
                                    int stack_frame_index,
                                    NavigationMode navigation_mode) {
  if (!debug_client_) {
    return UnavailableErrorBuilder(IREE_LOC) << "No connection established";
  }
  const auto& stack_frame = stack_frame_index == -1
                                ? *invocation.def().frames.back()
                                : *invocation.def().frames[stack_frame_index];
  selected_invocation_id_ = invocation.id();
  selected_stack_frame_index_ = stack_frame_index;
  return NavigateToCodeView(stack_frame.module_name,
                            stack_frame.function_ordinal, stack_frame.offset,
                            NavigationMode::kMatchDocument);
}

Status DebugApp::NavigateToCodeView(const UserBreakpoint& user_breakpoint,
                                    NavigationMode navigation_mode) {
  if (!debug_client_) {
    return UnavailableErrorBuilder(IREE_LOC) << "No connection established";
  }
  switch (user_breakpoint.type) {
    case RemoteBreakpoint::Type::kBytecodeFunction:
      if (user_breakpoint.function_ordinal != -1) {
        return NavigateToCodeView(
            user_breakpoint.module_name, user_breakpoint.function_ordinal,
            user_breakpoint.bytecode_offset, navigation_mode);
      } else {
        return NavigateToCodeView(
            user_breakpoint.module_name, user_breakpoint.function_name,
            user_breakpoint.bytecode_offset, navigation_mode);
      }
    case RemoteBreakpoint::Type::kNativeFunction:
      return UnimplementedErrorBuilder(IREE_LOC)
             << "Navigation to non-bytecode functions unimplemented";
  }
}

Status DebugApp::DrawCodeViewPanels() {
  // If we've disconnected then we need to clear bodies.
  // TODO(benvanik): allow documents to persist by caching all required info.
  if (!debug_client_) {
    documents_.clear();
    return OkStatus();
  }

  std::vector<CodeViewDocument*> closing_documents;
  for (auto& document : documents_) {
    ASSIGN_OR_RETURN(bool is_open, DrawCodeViewDocument(document.get()));
    if (!is_open) {
      closing_documents.push_back(document.get());
    }
  }
  for (auto* closing_document : closing_documents) {
    auto it = std::find_if(
        documents_.begin(), documents_.end(),
        [closing_document](const std::unique_ptr<CodeViewDocument>& document) {
          return document.get() == closing_document;
        });
    documents_.erase(it);
  }
  return OkStatus();
}

StatusOr<bool> DebugApp::DrawCodeViewDocument(CodeViewDocument* document) {
  ImGui::SetNextWindowDockID(dockspace_id_, ImGuiCond_FirstUseEver);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
  bool is_open = true;
  bool is_visible =
      ImGui::Begin(document->title.c_str(), &is_open, ImGuiWindowFlags_None);
  if (!is_open || !is_visible) {
    ImGui::End();
    ImGui::PopStyleVar();
    return is_open;
  }
  ImGui::PopStyleVar();

  auto* remote_module = document->function->module();
  auto* remote_function = document->function;
  if (remote_module->CheckLoadedOrRequest() &&
      remote_function->CheckLoadedOrRequest()) {
    // TODO(benvanik): draw function signature.
    if (remote_function->bytecode()) {
      RETURN_IF_ERROR(DrawBytecodeCodeView(document));
    } else {
      // TODO(benvanik): display native registration info.
      ImGui::TextDisabled("(native)");
    }
  } else {
    ImGui::TextDisabled("loading...");
  }

  ImGui::End();
  return true;
}

Status DebugApp::PrepareBytecodeCodeView(CodeViewDocument* document) {
  auto* remote_module = document->function->module();
  auto* remote_function = document->function;

  ASSIGN_OR_RETURN(auto module, vm::Module::FromDef(remote_module->def()));

  // TODO(benvanik): source map support.
  // Want line count including source lines, IR lines, and bytecode lines.
  // May want lower level (JIT/etc) lines too.

  // TODO(benvanik): bytecode iterator for richer display.
  auto source_map_resolver = vm::SourceMapResolver::FromFunction(
      module->def(), remote_function->ordinal());
  vm::BytecodePrinter printer(vm::sequencer_opcode_table(),
                              module->function_table(),
                              module->executable_table(), source_map_resolver);
  ASSIGN_OR_RETURN(std::string full_string,
                   printer.Print(*remote_function->bytecode()));
  document->bytecode_info.lines = absl::StrSplit(full_string, '\n');

  return OkStatus();
}

Status DebugApp::DrawBytecodeCodeView(CodeViewDocument* document) {
  // Ensure we have cached our line information.
  RETURN_IF_ERROR(PrepareBytecodeCodeView(document));

  auto* remote_module = document->function->module();
  auto* remote_function = document->function;

  ImGui::BeginGroup();
  ImGui::BeginChild("##bytecode_view", ImVec2(0, 0), false,
                    ImGuiWindowFlags_AlwaysVerticalScrollbar);
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));

  // TODO(benvanik): cache breakpoints for this function for faster lookup.

  auto& bytecode_info = document->bytecode_info;
  ImGuiListClipper clipper(bytecode_info.lines.size(),
                           ImGui::GetTextLineHeightWithSpacing());
  while (clipper.Step()) {
    for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
      ImGui::PushID(i);

      // TODO(benvanik): lookup line info.
      int bytecode_offset = 0;
      int breakpoint_index = FindMatchingUserBreakpointIndex(
          remote_module->name(), remote_function->ordinal(), bytecode_offset);
      bool has_breakpoint = breakpoint_index != -1;
      bool active_on_any_invocation = false;
      bool active_on_selected_invocation = false;

      ImGui::Dummy(ImVec2(4, 0));

      // Gutter breakpoint button.
      ImGui::SameLine();
      if (has_breakpoint) {
        PushButtonHue(0.0f);  // Red
        if (ImGui::Button(" ##toggle_breakpoint")) {
          CHECK_GE(breakpoint_index, 0);
          auto& user_breakpoint = user_breakpoint_list_[breakpoint_index];
          if (user_breakpoint.active_breakpoint) {
            RETURN_IF_ERROR(debug_client_->RemoveBreakpoint(
                *user_breakpoint.active_breakpoint));
          }
          user_breakpoint_list_.erase(user_breakpoint_list_.begin() +
                                      breakpoint_index);
        }
        PopButtonStyle();
        if (ImGui::IsItemHovered()) {
          ImGui::SetTooltip("Remove the breakpoint at this offset.");
        }
      } else {
        PushButtonColor(ImGui::GetStyleColorVec4(ImGuiCol_ChildBg));
        if (ImGui::Button(" ##toggle_breakpoint")) {
          UserBreakpoint user_breakpoint;
          user_breakpoint.type = RemoteBreakpoint::Type::kBytecodeFunction;
          user_breakpoint.module_name = remote_module->name();
          user_breakpoint.function_name = remote_function->name();
          user_breakpoint.bytecode_offset = bytecode_offset;
          user_breakpoint.wants_enabled = true;
          user_breakpoint_list_.push_back(std::move(user_breakpoint));
        }
        PopButtonStyle();
        if (ImGui::IsItemHovered()) {
          ImGui::SetTooltip("Add a breakpoint at this offset.");
        }
      }

      // Active execution chevron (shows when active or any invocation is
      // executing this region).
      ImGui::SameLine();
      if (active_on_selected_invocation) {
        // The selected invocation is active here.
        ImGui::TextColored(ImGui::GetStyleColorVec4(ImGuiCol_SeparatorActive),
                           " > ");
      } else if (active_on_any_invocation) {
        // At least one other invocation is active here.
        ImGui::TextColored(ImGui::GetStyleColorVec4(ImGuiCol_Separator), " > ");
      } else {
        // Not active.
        ImGui::Text("   ");
      }

      // Line contents.
      ImGui::SameLine();
      ImGui::Text("%s", bytecode_info.lines[i].c_str());

      if (document->focus_offset.has_value() &&
          bytecode_offset == document->focus_offset.value()) {
        document->bytecode_offset = document->focus_offset.value();
        document->focus_offset = {};
        ImGui::SetScrollHereY();
      }

      ImGui::PopID();
    }
  }

  ImGui::PopStyleVar();
  ImGui::EndChild();
  ImGui::EndGroup();

  return OkStatus();
}

}  // namespace debug
}  // namespace rt
}  // namespace iree
