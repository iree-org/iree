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

#ifndef IREE_HAL_VULKAN_UTIL_REF_PTR_H_
#define IREE_HAL_VULKAN_UTIL_REF_PTR_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "iree/base/attributes.h"
#include "iree/base/logging.h"

namespace iree {

// Use this to get really verbose refptr logging:
// #define IREE_VERBOSE_REF_PTR

template <class T>
class ref_ptr;

// Allocates a new ref_ptr type.
// Like make_unique, but for ref_ptr.
//
// Usage:
//  ref_ptr<MyType> p = make_ref<MyType>(1, 2, 3);
template <typename T, typename... Args>
ref_ptr<T> make_ref(Args&&... args) {
  return ref_ptr<T>(new T(std::forward<Args>(args)...));
}

// Assigns a raw pointer to a ref_ptr without adding a reference.
//
// Usage:
//  ref_ptr<MyType> p = assign_ref(new MyType());
template <typename T>
inline ref_ptr<T> assign_ref(T* value) {
  return ref_ptr<T>(value);
}

// Adds a reference to the given raw pointer.
//
// Usage:
//  MyType* raw_ptr = AcquirePointerFromSomewhere();
//  ref_ptr<MyType> p = add_ref(raw_ptr);
template <typename T>
inline ref_ptr<T> add_ref(T* value) {
  if (value) ref_ptr_add_ref(value);
  return ref_ptr<T>(value);
}

// Adds a reference to the given ref_ptr.
//
// Usage:
//  ref_ptr<MyType> a = make_ref<MyType>();
//  ref_ptr<MyType> p = add_ref(a);
template <typename T>
inline ref_ptr<T> add_ref(const ref_ptr<T>& value) {
  if (value.get()) ref_ptr_add_ref(value.get());
  return ref_ptr<T>(value.get());
}

// Reference counted pointer container.
// This is modeled on boost::instrusive_ptr in that it requires no
// extra storage over the pointer type and should compile to almost
// no additional code. It also allows us to round-trip object pointers
// through regular pointers, which is critical when having to round-trip
// them through JNI/etc where we can't use things like unique_ptr/shared_ptr.
//
//   ref_ptr<Foo> p1(new Foo());    // ref count 1
//   ref_ptr<Foo> p2(p1);           // ref count 2
//   p1.reset();                    // ref count 1
//   p2.reset();                    // ref count 0, deleted
//
// When round-tripping the pointer through external APIs, use release():
//   ref_ptr<Foo> p1(new Foo());    // ref count 1
//   Foo* raw_p = p1.release();     // ref count 1
//   // pass to API
//   ref_ptr<Foo> p2(raw_p);        // ref count 1 (don't add ref)
//   p2.reset();                    // ref count 0, deleted
//
// See the boost intrusive_ptr docs for details of behavior:
// http://www.boost.org/doc/libs/1_55_0/libs/smart_ptr/intrusive_ptr.html
//
// ref_ptr manages the target objects in a thread-safe way, though you'll want
// to take care with objects that may have pinned threads for deallocation. If
// you release the last reference to an object on a thread other than what it
// was expecting you're gonna have a bad time.
//
// Compatible only with types that subclass RefObject or implement the following
// methods:
//   ref_ptr_add_ref
//   ref_ptr_release_ref
template <class T>
class ref_ptr {
 private:
  typedef ref_ptr this_type;
  typedef T* this_type::*unspecified_bool_type;

 public:
  // Initializes with nullptr.
  IREE_ATTRIBUTE_ALWAYS_INLINE ref_ptr() noexcept = default;

  // Initializes with nullptr so that there is no way to create an
  // uninitialized ref_ptr.
  IREE_ATTRIBUTE_ALWAYS_INLINE ref_ptr(std::nullptr_t) noexcept {}  // NOLINT

  // Initializes the pointer to the given value.
  // The value will not have its reference count incremented (as it is with
  // unique_ptr). Use Retain to add to the reference count.
  IREE_ATTRIBUTE_ALWAYS_INLINE explicit ref_ptr(T* p) noexcept : px_(p) {}

  // Decrements the reference count of the owned pointer.
  IREE_ATTRIBUTE_ALWAYS_INLINE ~ref_ptr() noexcept {
    if (px_) ref_ptr_release_ref(px_);
  }

  // No implicit ref_ptr copying allowed; use add_ref instead.
  ref_ptr(const ref_ptr&) noexcept = delete;
  ref_ptr& operator=(const ref_ptr&) noexcept = delete;

  // Move support to transfer ownership from one ref_ptr to another.
  ref_ptr(ref_ptr&& rhs) noexcept : px_(rhs.release()) {}
  ref_ptr& operator=(ref_ptr&& rhs) noexcept {
    if (px_ != rhs.px_) {
      if (px_) ref_ptr_release_ref(px_);
      px_ = rhs.release();
    }
    return *this;
  }

  // Move support from another compatible type.
  template <typename U>
  ref_ptr(ref_ptr<U>&& rhs) noexcept : px_(rhs.release()) {}  // NOLINT
  template <typename U>
  ref_ptr& operator=(ref_ptr<U>&& rhs) noexcept {
    if (px_ != rhs.get()) {
      if (px_) ref_ptr_release_ref(px_);
      px_ = rhs.release();
    }
    return *this;
  }

  // Resets the object to nullptr and decrements the reference count, possibly
  // deleting it.
  void reset() noexcept {
    if (px_) {
      ref_ptr_release_ref(px_);
      px_ = nullptr;
    }
  }

  // Releases a pointer.
  // Returns the current pointer held by this object without having
  // its reference count decremented and resets the ref_ptr to empty.
  // Returns nullptr if the ref_ptr holds no value.
  // To re-wrap in a ref_ptr use either ref_ptr<T>(value) or assign().
  IREE_ATTRIBUTE_ALWAYS_INLINE T* release() noexcept {
    T* p = px_;
    px_ = nullptr;
    return p;
  }

  // Assigns a pointer.
  // The pointer will be accepted by the ref_ptr and its reference count will
  // not be incremented.
  IREE_ATTRIBUTE_ALWAYS_INLINE void assign(T* value) noexcept {
    reset();
    px_ = value;
  }

  // Gets the pointer referenced by this instance.
  // operator* and operator-> will assert() if there is no current object.
  constexpr T* get() const noexcept { return px_; }
  constexpr T& operator*() const noexcept { return *px_; }
  constexpr T* operator->() const noexcept { return px_; }

  // Support boolean expression evaluation ala unique_ptr/shared_ptr:
  // https://en.cppreference.com/w/cpp/memory/shared_ptr/operator_bool
  constexpr operator unspecified_bool_type() const noexcept {
    return px_ ? &this_type::px_ : nullptr;
  }
  // Supports unary expression evaluation.
  constexpr bool operator!() const noexcept { return !px_; }

  // Swap support.
  void swap(ref_ptr& rhs) { std::swap(px_, rhs.px_); }

 private:
  T* px_ = nullptr;
};

// Base class for reference counted objects.
// Reference counted objects should be used with the ref_ptr pointer type.
// As reference counting can be tricky always prefer to use unique_ptr and
// avoid this type. Only use this when unique_ptr is not possible, such as
// when round-tripping objects through marshaling boundaries (v8/Java) or
// any objects that may have their lifetime tied to a garbage collected
// object.
//
// Subclasses should protect their dtor so that reference counting must
// be used.
//
// This is designed to avoid the need for extra vtable space or for adding
// methods to the vtable of subclasses. This differs from the boost Pointable
// version of this object.
// Inspiration for this comes from Peter Weinert's Dr. Dobb's article:
// http://www.drdobbs.com/cpp/a-base-class-for-intrusively-reference-c/229218807
//
// RefObjects are thread safe and may be used with ref_ptrs from multiple
// threads.
//
// Subclasses may implement a custom Delete operator to handle their
// deallocation. It should be thread safe as it may be called from any thread.
//
// Usage:
//   class MyRefObject : public RefObject<MyRefObject> {
//    public:
//     MyRefObject() = default;
//     // Optional; can be used to return to pool/etc - must be public:
//     static void Delete(MyRefObject* ptr) {
//       ::operator delete(ptr);
//     }
//   };
template <class T>
class RefObject {
  static_assert(!std::is_array<T>::value, "T must not be an array");

  // value is true if a static Delete(T*) function is present.
  struct has_custom_deleter {
    template <typename C>
    static auto Test(C* p) -> decltype(C::Delete(nullptr), std::true_type());
    template <typename>
    static std::false_type Test(...);
    static constexpr bool value =
        std::is_same<std::true_type, decltype(Test<T>(nullptr))>::value;
  };

  template <typename V, bool has_custom_deleter>
  struct delete_thunk {
    static void Delete(V* p) {
      auto ref_obj = static_cast<RefObject<V>*>(p);
      int previous_count = ref_obj->counter_.fetch_sub(1);
#ifdef IREE_VERBOSE_REF_PTR
      IREE_LOG(INFO) << "ro-- " << typeid(V).name() << " " << p << " now "
                     << previous_count - 1
                     << (previous_count == 1 ? " DEAD (CUSTOM)" : "");
#endif  // IREE_VERBOSE_REF_PTR
      if (previous_count == 1) {
        // We delete type T pointer here to avoid the need for a virtual dtor.
        V::Delete(p);
      }
    }
    static void Destroy(V* p) { V::Delete(p); }
  };

  template <typename V>
  struct delete_thunk<V, false> {
    static void Delete(V* p) {
      auto ref_obj = static_cast<RefObject<V>*>(p);
      int previous_count = ref_obj->counter_.fetch_sub(1);
#ifdef IREE_VERBOSE_REF_PTR
      IREE_LOG(INFO) << "ro-- " << typeid(V).name() << " " << p << " now "
                     << previous_count - 1
                     << (previous_count == 1 ? " DEAD" : "");
#endif  // IREE_VERBOSE_REF_PTR
      if (previous_count == 1) {
        // We delete type T pointer here to avoid the need for a virtual dtor.
        delete p;
      }
    }
    static void Destroy(V* p) { delete p; }
  };

 public:
  // Adds a reference; used by ref_ptr.
  friend void ref_ptr_add_ref(T* p) {
    auto ref_obj = static_cast<RefObject*>(p);
    ++ref_obj->counter_;

#ifdef IREE_VERBOSE_REF_PTR
    IREE_LOG(INFO) << "ro++ " << typeid(T).name() << " " << p << " now "
                   << ref_obj->counter_;
#endif  // IREE_VERBOSE_REF_PTR
  }

  // Releases a reference, potentially deleting the object; used by ref_ptr.
  friend void ref_ptr_release_ref(T* p) {
    delete_thunk<T, has_custom_deleter::value>::Delete(p);
  }

  // Deletes the object (precondition: ref count is zero).
  friend void ref_ptr_destroy_ref(T* p) {
    delete_thunk<T, has_custom_deleter::value>::Destroy(p);
  }

  // Deletes the object (precondition: ref count is zero).
  static void DirectDestroy(void* p) {
    ref_ptr_destroy_ref(reinterpret_cast<T*>(p));
  }

  // Adds a reference.
  // ref_ptr should be used instead of this in most cases. This is required
  // for when interoperating with marshaling APIs.
  void AddReference() { ref_ptr_add_ref(static_cast<T*>(this)); }

  // Releases a reference, potentially deleting the object.
  // ref_ptr should be used instead of this in most cases. This is required
  // for when interoperating with marshaling APIs.
  void ReleaseReference() { ref_ptr_release_ref(static_cast<T*>(this)); }

  // Returns the offset of the reference counter field from the start of the
  // type T.
  //
  // This is generally unsafe to use and is here for support of the
  // iree_vm_ref_t glue that allows RefObject-derived types to be round-tripped
  // through the VM.
  //
  // For simple POD types or non-virtual classes we expect this to return 0.
  // If the type has virtual methods (dtors/etc) then it should be 4 or 8
  // (depending on pointer width). It may be other things, and instead of too
  // much crazy magic we just rely on offsetof doing the right thing here.
  static constexpr size_t offsetof_counter() { return offsetof(T, counter_); }

 protected:
  RefObject() { ref_ptr_add_ref(static_cast<T*>(this)); }
  RefObject(const RefObject&) = default;
  RefObject& operator=(const RefObject&) { return *this; }

  std::atomic<int32_t> counter_{0};
};

// Various comparison operator overloads.

template <class T, class U>
inline bool operator==(ref_ptr<T> const& a, ref_ptr<U> const& b) {
  return a.get() == b.get();
}

template <class T, class U>
inline bool operator!=(ref_ptr<T> const& a, ref_ptr<U> const& b) {
  return a.get() != b.get();
}

template <class T, class U>
inline bool operator==(ref_ptr<T> const& a, U* b) {
  return a.get() == b;
}

template <class T, class U>
inline bool operator!=(ref_ptr<T> const& a, U* b) {
  return a.get() != b;
}

template <class T, class U>
inline bool operator==(T* a, ref_ptr<U> const& b) {
  return a == b.get();
}

template <class T, class U>
inline bool operator!=(T* a, ref_ptr<U> const& b) {
  return a != b.get();
}

template <class T>
inline bool operator<(ref_ptr<T> const& a, ref_ptr<T> const& b) {
  return a.get() < b.get();
}

// Swaps the pointers of two ref_ptrs.
template <class T>
void swap(ref_ptr<T>& lhs, ref_ptr<T>& rhs) {
  lhs.swap(rhs);
}

}  // namespace iree

#endif  // IREE_HAL_VULKAN_UTIL_REF_PTR_H_
