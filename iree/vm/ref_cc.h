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

#ifndef IREE_VM_REF_CC_H_
#define IREE_VM_REF_CC_H_

#include <memory>
#include <utility>

#include "absl/base/attributes.h"
#include "iree/base/api.h"
#include "iree/vm/ref.h"

#ifndef __cplusplus
#error "This header is meant for use with C++ implementations."
#endif  // __cplusplus

namespace iree {
namespace vm {

// TODO(benvanik): make this automatic for most types, or use type lookup.
// This could be done with SFINAE to detect iree_vm_ref_object_t or RefObject
// types. We may still need the iree_vm_ref_type_t exposed but that's relatively
// simple compared to getting the typed retain/release functions.

// Users may override this with their custom types to allow the packing code to
// access their registered type ID at runtime.
template <typename T>
ABSL_ATTRIBUTE_ALWAYS_INLINE void ref_type_retain(T* p) {
  iree_vm_ref_object_retain(p, ref_type_descriptor<T>::get());
}

template <typename T>
ABSL_ATTRIBUTE_ALWAYS_INLINE void ref_type_release(T* p) {
  iree_vm_ref_object_release(p, ref_type_descriptor<T>::get());
}

// Reference counted pointer container wrapping iree_vm_ref_t.
// This is modeled on boost::instrusive_ptr in that it requires no
// extra storage over the pointer type and should compile to almost
// no additional code. It also allows us to round-trip object pointers
// through regular pointers, which is critical when having to round-trip
// them through JNI/etc where we can't use things like unique_ptr/shared_ptr.
//
// The ref wrapper calls the iree_vm_ref_* functions and uses the
// iree_vm_ref_type_descriptor_t registered for the type T to manipulate the
// reference counter and, when needed, destroy the object using
// iree_vm_ref_destroy_t. Any iree_vm_ref_t can be used interchangably with
// ref<T> when RAII is needed.
//
// Example:
//   ref<Foo> p1(new Foo());    // ref count 1
//   ref<Foo> p2(p1);           // ref count 2
//   p1.reset();                // ref count 1
//   p2.reset();                // ref count 0, deleted
//
// When round-tripping the pointer through external APIs, use release():
//   ref<Foo> p1(new Foo());    // ref count 1
//   Foo* raw_p = p1.release(); // ref count 1
//   // pass to API
//   ref<Foo> p2(raw_p);        // ref count 1 (don't add ref)
//   p2.reset();                // ref count 0, deleted
//
// See the boost intrusive_ptr docs for details of behavior:
// http://www.boost.org/doc/libs/1_55_0/libs/smart_ptr/intrusive_ptr.html
//
// The retain_ref and assign_ref helpers can be used to make it easier to
// declare and use ref types:
//   ref<Foo> p = assign_ref(new Foo());  // ref count 1
//   PassRefWithRetain(retain_ref(p));
//   PassRefWithMove(std::move(p));       // ala unique_ptr/shared_ptr
//
// ref manages the target objects in a thread-safe way, though you'll want
// to take care with objects that may have pinned threads for deallocation. If
// you release the last reference to an object on a thread other than what it
// was expecting you're gonna have a bad time.
//
// Compatible only with types that implement the following methods:
//   ref_type_retain(T*)
//   ref_type_release(T*)
//   ref_type_descriptor<T>::get()
//
// If you get link errors pertaining to ref_type_descriptor then ensure that you
// have included the header file containing the IREE_VM_DECLARE_TYPE_ADAPTERS
// for the given type.
//
// TODO(benvanik): reconcile RefObject, iree_vm_ref_t, and this.
template <typename T>
class ref {
 private:
  typedef ref this_type;
  typedef T* this_type::*unspecified_bool_type;

 public:
  ABSL_ATTRIBUTE_ALWAYS_INLINE iree_vm_ref_type_t type() const noexcept {
    return ref_type_descriptor<T>::get()->type;
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE ref() noexcept
      : ref_({
            0,
            ref_type_descriptor<T>::get()->offsetof_counter,
            ref_type_descriptor<T>::get()->type,
        }) {}
  ABSL_ATTRIBUTE_ALWAYS_INLINE ref(std::nullptr_t) noexcept  // NOLINT
      : ref_({
            0,
            ref_type_descriptor<T>::get()->offsetof_counter,
            ref_type_descriptor<T>::get()->type,
        }) {}
  ABSL_ATTRIBUTE_ALWAYS_INLINE ref(T* p) noexcept  // NOLINT
      : ref_({
            p,
            ref_type_descriptor<T>::get()->offsetof_counter,
            ref_type_descriptor<T>::get()->type,
        }) {}
  ABSL_ATTRIBUTE_ALWAYS_INLINE ~ref() noexcept { ref_type_release<T>(get()); }

  // Don't use implicit ref copying; use retain_ref instead to make things more
  // readable. We can't delete the ctor (or, I couldn't find a way not to)
  // because the templated parameter packing magic needs it.
  ref(const ref& rhs) noexcept : ref_(rhs.ref_) { ref_type_retain<T>(get()); }
  ref& operator=(const ref&) noexcept = delete;

  // Move support to transfer ownership from one ref to another.
  ref(ref&& rhs) noexcept : ref_(rhs.ref_) { rhs.release(); }
  ref& operator=(ref&& rhs) noexcept {
    if (get() != rhs.get()) {
      ref_type_release<T>(get());
      ref_ = rhs.ref_;
      rhs.release();
    }
    return *this;
  }

  // Move support from another compatible type.
  template <typename U>
  ref(ref<U>&& rhs) noexcept {  // NOLINT
    ref_.ptr = static_cast<T*>(rhs.release());
    ref_.offsetof_counter = rhs.ref_.offsetof_counter;
    ref_.type = rhs.ref_.type;
  }
  template <typename U>
  ref& operator=(ref<U>&& rhs) noexcept {
    if (get() != rhs.get()) {
      ref_type_release<T>(get());
      ref_.ptr = static_cast<T*>(rhs.release());
    }
    return *this;
  }

  // Resets the object to nullptr and decrements the reference count, possibly
  // deleting it.
  void reset() noexcept {
    ref_type_release<T>(get());
    ref_.ptr = nullptr;
  }

  // Releases a pointer.
  // Returns the current pointer held by this object without having
  // its reference count decremented and resets the ref to empty.
  // Returns nullptr if the ref holds no value.
  // To re-wrap in a ref use either ref<T>(value) or assign().
  ABSL_ATTRIBUTE_ALWAYS_INLINE T* release() noexcept {
    T* p = get();
    ref_.ptr = nullptr;
    return p;
  }

  // Assigns a pointer.
  // The pointer will be accepted by the ref and its reference count will
  // not be incremented.
  ABSL_ATTRIBUTE_ALWAYS_INLINE void assign(T* value) noexcept {
    reset();
    ref_.ptr = value;
  }

  // Gets the pointer referenced by this instance.
  // operator* and operator-> will assert() if there is no current object.
  constexpr T* get() const noexcept { return reinterpret_cast<T*>(ref_.ptr); }
  constexpr T& operator*() const noexcept { return *get(); }
  constexpr T* operator->() const noexcept { return get(); }

  // Returns a pointer to the inner pointer storage.
  // This allows passing a pointer to the ref as an output argument to C-style
  // creation functions.
  constexpr T** operator&() noexcept {  // NOLINT
    return reinterpret_cast<T**>(&ref_.ptr);
  }

  // Support boolean expression evaluation ala unique_ptr/shared_ptr:
  // https://en.cppreference.com/w/cpp/memory/shared_ptr/operator_bool
  constexpr operator unspecified_bool_type() const noexcept {  // NOLINT
    return get() ? reinterpret_cast<unspecified_bool_type>(&this_type::ref_.ptr)
                 : nullptr;
  }
  // Supports unary expression evaluation.
  constexpr bool operator!() const noexcept { return !get(); }

  // Swap support.
  void swap(ref& rhs) { std::swap(ref_.ptr, rhs.ref_.ptr); }

  // Allows directly passing the ref to a C-API function for creation.
  // Example:
  //    iree::vm::ref<my_type_t> value;
  //    my_type_create(..., &value);
  constexpr operator iree_vm_ref_t*() const noexcept {  // NOLINT
    return &ref_;
  }

 private:
  mutable iree_vm_ref_t ref_;
};

// Adds a reference to the given ref and returns the same ref.
//
// Usage:
//  ref<MyType> a = AcquireRefFromSomewhere();
//  ref<MyType> b = retain_ref(a);  // ref count + 1
//  retain_ref(b);  // ref count + 1
template <typename T>
inline ref<T> retain_ref(const ref<T>& value) {
  ref_type_retain<T>(value.get());
  return ref<T>(value.get());
}

// Adds a reference to the given raw pointer and returns it wrapped in a ref.
//
// Usage:
//  MyType* raw_ptr = AcquirePointerFromSomewhere();
//  ref<MyType> p = retain_ref(raw_ptr);  // ref count + 1
template <typename T>
inline ref<T> retain_ref(T* value) {
  ref_type_retain<T>(value);
  return ref<T>(value);
}

// Assigns a raw pointer to a ref without adding a reference.
//
// Usage:
//  ref<MyType> p = assign_ref(new MyType());  // ref count untouched
template <typename T>
inline ref<T> assign_ref(T* value) {
  return ref<T>(value);
}

template <class T, class U>
inline bool operator==(ref<T> const& a, ref<U> const& b) {
  return a.get() == b.get();
}

template <class T, class U>
inline bool operator!=(ref<T> const& a, ref<U> const& b) {
  return a.get() != b.get();
}

template <class T, class U>
inline bool operator==(ref<T> const& a, U* b) {
  return a.get() == b;
}

template <class T, class U>
inline bool operator!=(ref<T> const& a, U* b) {
  return a.get() != b;
}

template <class T, class U>
inline bool operator==(T* a, ref<U> const& b) {
  return a == b.get();
}

template <class T, class U>
inline bool operator!=(T* a, ref<U> const& b) {
  return a != b.get();
}

template <class T>
inline bool operator<(ref<T> const& a, ref<T> const& b) {
  return a.get() < b.get();
}

// Swaps the pointers of two refs.
template <class T>
void swap(ref<T>& lhs, ref<T>& rhs) {
  lhs.swap(rhs);
}

// An opaque reference that does not make any assertions about the type of the
// ref contained. This can be used to accept arbitrary ref objects that are then
// dynamically handled based on type.
class opaque_ref {
 public:
  opaque_ref() = default;
  opaque_ref(const opaque_ref&) = delete;
  opaque_ref& operator=(const opaque_ref&) = delete;
  opaque_ref(opaque_ref&& rhs) noexcept {
    iree_vm_ref_move(&rhs.value_, &value_);
  }
  opaque_ref& operator=(opaque_ref&& rhs) noexcept {
    iree_vm_ref_move(&rhs.value_, &value_);
    return *this;
  }
  ~opaque_ref() { iree_vm_ref_release(&value_); }

  constexpr iree_vm_ref_t* get() const noexcept { return &value_; }
  constexpr operator iree_vm_ref_t*() const noexcept {  // NOLINT
    return &value_;
  }
  constexpr bool operator!() const noexcept { return !value_.ptr; }

  // Returns a pointer to the inner pointer storage.
  // This allows passing a pointer to the ref as an output argument to C-style
  // creation functions.
  constexpr iree_vm_ref_t* operator&() noexcept { return &value_; }  // NOLINT

 private:
  mutable iree_vm_ref_t value_ = {0};
};

}  // namespace vm
}  // namespace iree

#endif  // IREE_VM_REF_CC_H_
