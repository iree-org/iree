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

#ifndef IREE_BASE_INTERNAL_STATUS_BUILDER_H_
#define IREE_BASE_INTERNAL_STATUS_BUILDER_H_

#include "iree/base/internal/ostringstream.h"
#include "iree/base/internal/status.h"

namespace iree {

class IREE_MUST_USE_RESULT StatusBuilder;

// Creates a status based on an original_status, but enriched with additional
// information. The builder implicitly converts to Status and StatusOr<T>
// allowing for it to be returned directly.
class StatusBuilder {
 public:
  // Creates a `StatusBuilder` based on an original status.
  explicit StatusBuilder(Status&& original_status, SourceLocation location);
  explicit StatusBuilder(Status&& original_status, SourceLocation location,
                         const char* format, ...);

  // Creates a `StatusBuilder` from a status code.
  explicit StatusBuilder(StatusCode code, SourceLocation location);
  explicit StatusBuilder(StatusCode code, SourceLocation location,
                         const char* format, ...);

  StatusBuilder(const StatusBuilder& sb) = delete;
  StatusBuilder& operator=(const StatusBuilder& sb) = delete;
  StatusBuilder(StatusBuilder&&) noexcept;
  StatusBuilder& operator=(StatusBuilder&&) noexcept;

  // Appends to the extra message that will be added to the original status.
  template <typename T>
  StatusBuilder& operator<<(const T& value) &;
  template <typename T>
  StatusBuilder&& operator<<(const T& value) &&;

  // Returns true if the Status created by this builder will be ok().
  bool ok() const;

  StatusCode code() const { return status_.code(); }

  IREE_MUST_USE_RESULT Status ToStatus() {
    if (!status_.ok()) Flush();
    return exchange(status_, status_.code());
  }

  // Implicit conversion to Status. Eats the status object but preserves the
  // status code so the builder remains !ok().
  operator Status() && {
    if (!status_.ok()) Flush();
    return exchange(status_, status_.code());
  }

  // Implicit conversion to iree_status_t.
  operator iree_status_t() && {
    if (!status_.ok()) Flush();
    Status status = exchange(status_, status_.code());
    return static_cast<iree_status_t>(std::move(status));
  }

  friend bool operator==(const StatusBuilder& lhs, const StatusCode& rhs) {
    return lhs.code() == rhs;
  }
  friend bool operator!=(const StatusBuilder& lhs, const StatusCode& rhs) {
    return !(lhs == rhs);
  }

  friend bool operator==(const StatusCode& lhs, const StatusBuilder& rhs) {
    return lhs == rhs.code();
  }
  friend bool operator!=(const StatusCode& lhs, const StatusBuilder& rhs) {
    return !(lhs == rhs);
  }

 private:
  void Flush();

  // The status that is being built.
  // This may be an existing status that we are appending an annotation to or
  // just a code that we are building from scratch.
  Status status_;

  // Lazy construction of the expensive stream.
  struct Rep {
    explicit Rep() = default;
    Rep(const Rep& r);

    // Gathers additional messages added with `<<` for use in the final status.
    std::string stream_message;
    iree::OStringStream stream{&stream_message};
  };
  std::unique_ptr<Rep> rep_;
};

inline StatusBuilder::StatusBuilder(StatusBuilder&& sb) noexcept
    : status_(exchange(sb.status_, sb.code())), rep_(std::move(sb.rep_)) {}

inline StatusBuilder& StatusBuilder::operator=(StatusBuilder&& sb) noexcept {
  status_ = exchange(sb.status_, sb.code());
  rep_ = std::move(sb.rep_);
  return *this;
}

// Disable << streaming when status messages are disabled.
#if (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_ANNOTATIONS) == 0
template <typename T>
StatusBuilder& StatusBuilder::operator<<(const T& value) & {
  return *this;
}
#else
template <typename T>
StatusBuilder& StatusBuilder::operator<<(const T& value) & {
  if (status_.ok()) return *this;
  if (rep_ == nullptr) rep_ = std::make_unique<Rep>();
  rep_->stream << value;
  return *this;
}
#endif  // (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_ANNOTATIONS) == 0

template <typename T>
StatusBuilder&& StatusBuilder::operator<<(const T& value) && {
  return std::move(operator<<(value));
}

}  // namespace iree

#endif  // IREE_BASE_INTERNAL_STATUS_BUILDER_H_
