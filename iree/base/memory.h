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

#ifndef IREE_BASE_MEMORY_H_
#define IREE_BASE_MEMORY_H_

#include <memory>
#include <type_traits>
#include <utility>

#include "absl/types/span.h"

namespace iree {

// reinterpret_cast for Spans, preserving byte size.
template <typename T, typename U>
constexpr absl::Span<const T> ReinterpretSpan(absl::Span<const U> value) {
  return absl::MakeSpan(reinterpret_cast<const T*>(value.data()),
                        (value.size() * sizeof(U)) / sizeof(T));
}
template <typename T, typename U>
constexpr absl::Span<T> ReinterpretSpan(absl::Span<U> value) {
  return absl::MakeSpan(reinterpret_cast<T*>(value.data()),
                        (value.size() * sizeof(U)) / sizeof(T));
}

// Cast a span of std::unique_ptr to a span of raw pointers.
template <typename T>
inline absl::Span<T*> RawPtrSpan(absl::Span<std::unique_ptr<T>> value) {
  return absl::MakeSpan(reinterpret_cast<T**>(value.data()), value.size());
}

// TODO(benvanik): replace with an absl version when it exists.
// A move-only RAII object that calls a stored cleanup functor when
// destroyed. Cleanup<F> is the return type of iree::MakeCleanup(F).
template <typename F>
class Cleanup {
 public:
  Cleanup() : released_(true), f_() {}
  template <typename G>
  explicit Cleanup(G&& f)          // NOLINT
      : f_(std::forward<G>(f)) {}  // NOLINT(build/c++11)
  Cleanup(Cleanup&& src)           // NOLINT
      : released_(src.is_released()), f_(src.release()) {}

  // Implicitly move-constructible from any compatible Cleanup<G>.
  // The source will be released as if src.release() were called.
  // A moved-from Cleanup can be safely destroyed or reassigned.
  template <typename G>
  Cleanup(Cleanup<G>&& src)  // NOLINT
      : released_(src.is_released()), f_(src.release()) {}

  // Assignment to a Cleanup object behaves like destroying it
  // and making a new one in its place, analogous to unique_ptr
  // semantics.
  Cleanup& operator=(Cleanup&& src) {  // NOLINT
    if (!released_) std::move(f_)();
    released_ = src.released_;
    f_ = src.release();
    return *this;
  }

  ~Cleanup() {
    if (!released_) std::move(f_)();
  }

  // Releases the cleanup function instead of running it.
  // Hint: use c.release()() to run early.
  F release() {
    released_ = true;
    return std::move(f_);
  }

  bool is_released() const { return released_; }

 private:
  static_assert(!std::is_reference<F>::value, "F must not be a reference");

  bool released_ = false;
  F f_;
};

// MakeCleanup(f) returns an RAII cleanup object that calls 'f' in its
// destructor. The easiest way to use MakeCleanup is with a lambda argument,
// capturing the return value in an 'auto' local variable. Most users will not
// need more sophisticated syntax than that.
//
// Example:
//   void func() {
//     FILE* fp = fopen("data.txt", "r");
//     if (fp == nullptr) return;
//     auto fp_cleaner = MakeCleanup([fp] { fclose(fp); });
//     // No matter what, fclose(fp) will happen.
//   }
//
// You can call 'release()' on a Cleanup object to cancel the cleanup.
template <int&... ExplicitParameterBarrier, typename F,
          typename DecayF = typename std::decay<F>::type>
ABSL_MUST_USE_RESULT Cleanup<DecayF> MakeCleanup(F&& f) {
  return Cleanup<DecayF>(std::forward<F>(f));
}

}  // namespace iree

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define IREE_CONFIG_ASAN 1
#endif  // __has_feature(address_sanitizer)
#endif  // __has_feature

// If you see these macros being used it means that the code between is not
// really under our control and not a leak we would be able to prevent.
#if defined(IREE_CONFIG_ASAN)
#include <sanitizer/lsan_interface.h>
#define IREE_DISABLE_LEAK_CHECKS() __lsan_disable()
#define IREE_ENABLE_LEAK_CHECKS() __lsan_enable()
#else
#define IREE_DISABLE_LEAK_CHECKS()
#define IREE_ENABLE_LEAK_CHECKS()
#endif  // IREE_CONFIG_ASAN

#endif  // IREE_BASE_MEMORY_H_
