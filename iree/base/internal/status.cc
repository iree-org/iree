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

#include "iree/base/internal/status.h"

#include <memory>

#include "absl/base/attributes.h"
#include "absl/strings/str_cat.h"

namespace iree {

std::ostream& operator<<(std::ostream& os, const StatusCode& x) {
  os << StatusCodeToString(x);
  return os;
}

Status::Status(iree_status_t status) {
  // TODO(#265): just store status.
  if (!iree_status_is_ok(status)) {
    state_ = std::make_unique<State>();
    state_->code = static_cast<StatusCode>(iree_status_code(status));
    state_->message = std::string("TODO");
    iree_status_ignore(status);
  }
}

Status::Status(StatusCode code, absl::string_view message) {
  if (code != StatusCode::kOk) {
    state_ = std::make_unique<State>();
    state_->code = code;
    state_->message = std::string(message);
  }
}

Status::Status(const Status& x) {
  if (x.ok()) return;

  state_ = std::make_unique<State>();
  state_->code = x.state_->code;
  state_->message = x.state_->message;
}

Status& Status::operator=(const Status& x) {
  if (x.ok()) {
    state_ = nullptr;
  } else {
    state_ = std::make_unique<State>();
    state_->code = x.state_->code;
    state_->message = x.state_->message;
  }
  return *this;
}

Status::~Status() {}

bool Status::ok() const { return state_ == nullptr; }

StatusCode Status::code() const {
  return ok() ? StatusCode::kOk : state_->code;
}

absl::string_view Status::message() const {
  return ok() ? absl::string_view() : absl::string_view(state_->message);
}

std::string Status::ToString() const {
  if (ok()) {
    return "OK";
  }

  std::string text;
  absl::StrAppend(&text, StatusCodeToString(state_->code), ": ",
                  state_->message);
  // TODO(scotttodd): Payloads (stack traces)
  return text;
}

std::ostream& operator<<(std::ostream& os, const Status& x) {
  os << x.ToString();
  return os;
}

Status OkStatus() { return Status(); }

Status Annotate(const Status& s, absl::string_view msg) {
  if (s.ok() || msg.empty()) return s;

  absl::string_view new_msg = msg;
  std::string annotated;
  if (!s.message().empty()) {
    absl::StrAppend(&annotated, s.message(), "; ", msg);
    new_msg = annotated;
  }
  Status result(s.code(), new_msg);
  // TODO(scotttodd): Copy payload(s) into the new Status
  return result;
}

}  // namespace iree
