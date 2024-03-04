/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2019) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef _LIBCUDACXX___MDSPAN_COMPRESSED_PAIR_HPP
#define _LIBCUDACXX___MDSPAN_COMPRESSED_PAIR_HPP

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__mdspan/macros.h"
#if !defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
#include "../__mdspan/no_unique_address.h"
#endif
#include "../__type_traits/enable_if.h"
#include "../__type_traits/is_empty.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 11

namespace __detail {

// For no unique address emulation, this is the case taken when neither are empty.
// For real `[[no_unique_address]]`, this case is always taken.
template <class _Tp, class _Up, class _Enable = void> struct __compressed_pair {
  __MDSPAN_NO_UNIQUE_ADDRESS _Tp __t_val;
  __MDSPAN_NO_UNIQUE_ADDRESS _Up __u_val;
  __MDSPAN_FORCE_INLINE_FUNCTION constexpr _Tp &__first() noexcept { return __t_val; }
  __MDSPAN_FORCE_INLINE_FUNCTION constexpr _Tp const &__first() const noexcept {
    return __t_val;
  }
  __MDSPAN_FORCE_INLINE_FUNCTION constexpr _Up &__second() noexcept { return __u_val; }
  __MDSPAN_FORCE_INLINE_FUNCTION constexpr _Up const &__second() const noexcept {
    return __u_val;
  }

  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  constexpr __compressed_pair() noexcept = default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  constexpr __compressed_pair(__compressed_pair const &) noexcept = default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  constexpr __compressed_pair(__compressed_pair &&) noexcept = default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  __MDSPAN_CONSTEXPR_14_DEFAULTED __compressed_pair &
  operator=(__compressed_pair const &) noexcept = default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  __MDSPAN_CONSTEXPR_14_DEFAULTED __compressed_pair &
  operator=(__compressed_pair &&) noexcept = default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  ~__compressed_pair() noexcept = default;
  template <class _TLike, class _ULike>
  __MDSPAN_INLINE_FUNCTION constexpr __compressed_pair(_TLike &&__t, _ULike &&__u)
      : __t_val((_TLike &&) __t), __u_val((_ULike &&) __u) {}
};

#if !defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)

// First empty.
template <class _Tp, class _Up>
struct __compressed_pair<
    _Tp, _Up,
    _CUDA_VSTD::enable_if_t<_LIBCUDACXX_TRAIT(_CUDA_VSTD::is_empty, _Tp) && !_LIBCUDACXX_TRAIT(_CUDA_VSTD::is_empty, _Up)>>
    : private _Tp {
  _Up __u_val;
  __MDSPAN_FORCE_INLINE_FUNCTION constexpr _Tp &__first() noexcept {
    return *static_cast<_Tp *>(this);
  }
  __MDSPAN_FORCE_INLINE_FUNCTION constexpr _Tp const &__first() const noexcept {
    return *static_cast<_Tp const *>(this);
  }
  __MDSPAN_FORCE_INLINE_FUNCTION constexpr _Up &__second() noexcept { return __u_val; }
  __MDSPAN_FORCE_INLINE_FUNCTION constexpr _Up const &__second() const noexcept {
    return __u_val;
  }

  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  constexpr __compressed_pair() noexcept = default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  constexpr __compressed_pair(__compressed_pair const &) noexcept = default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  constexpr __compressed_pair(__compressed_pair &&) noexcept = default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  __MDSPAN_CONSTEXPR_14_DEFAULTED __compressed_pair &
  operator=(__compressed_pair const &) noexcept = default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  __MDSPAN_CONSTEXPR_14_DEFAULTED __compressed_pair &
  operator=(__compressed_pair &&) noexcept = default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  ~__compressed_pair() noexcept = default;
  template <class _TLike, class _ULike>
  __MDSPAN_INLINE_FUNCTION constexpr __compressed_pair(_TLike &&__t, _ULike &&__u)
      : _Tp((_TLike &&) __t), __u_val((_ULike &&) __u) {}
};

// Second empty.
template <class _Tp, class _Up>
struct __compressed_pair<
    _Tp, _Up,
    _CUDA_VSTD::enable_if_t<!_LIBCUDACXX_TRAIT(_CUDA_VSTD::is_empty, _Tp) && _LIBCUDACXX_TRAIT(_CUDA_VSTD::is_empty, _Up)>>
    : private _Up {
  _Tp __t_val;
  __MDSPAN_FORCE_INLINE_FUNCTION constexpr _Tp &__first() noexcept { return __t_val; }
  __MDSPAN_FORCE_INLINE_FUNCTION constexpr _Tp const &__first() const noexcept {
    return __t_val;
  }
  __MDSPAN_FORCE_INLINE_FUNCTION constexpr _Up &__second() noexcept {
    return *static_cast<_Up *>(this);
  }
  __MDSPAN_FORCE_INLINE_FUNCTION constexpr _Up const &__second() const noexcept {
    return *static_cast<_Up const *>(this);
  }

  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  constexpr __compressed_pair() noexcept = default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  constexpr __compressed_pair(__compressed_pair const &) noexcept = default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  constexpr __compressed_pair(__compressed_pair &&) noexcept = default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  __MDSPAN_CONSTEXPR_14_DEFAULTED __compressed_pair &
  operator=(__compressed_pair const &) noexcept = default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  __MDSPAN_CONSTEXPR_14_DEFAULTED __compressed_pair &
  operator=(__compressed_pair &&) noexcept = default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  ~__compressed_pair() noexcept = default;

  template <class _TLike, class _ULike>
  __MDSPAN_INLINE_FUNCTION constexpr __compressed_pair(_TLike &&__t, _ULike &&__u)
      : _Up((_ULike &&) __u), __t_val((_TLike &&) __t) {}
};

// Both empty.
template <class _Tp, class _Up>
struct __compressed_pair<
    _Tp, _Up,
    _CUDA_VSTD::enable_if_t<_LIBCUDACXX_TRAIT(_CUDA_VSTD::is_empty, _Tp) && _LIBCUDACXX_TRAIT(_CUDA_VSTD::is_empty, _Up)>>
    // We need to use the __no_unique_address_emulation wrapper here to avoid
    // base class ambiguities.
#ifdef __MDSPAN_COMPILER_MSVC
// MSVC doesn't allow you to access public static member functions of a type
// when you *happen* to privately inherit from that type.
    : protected __no_unique_address_emulation<_Tp, 0>,
      protected __no_unique_address_emulation<_Up, 1>
#else
    : private __no_unique_address_emulation<_Tp, 0>,
      private __no_unique_address_emulation<_Up, 1>
#endif
{
  using __first_base_t = __no_unique_address_emulation<_Tp, 0>;
  using __second_base_t = __no_unique_address_emulation<_Up, 1>;

  __MDSPAN_FORCE_INLINE_FUNCTION constexpr _Tp &__first() noexcept {
    return this->__first_base_t::__ref();
  }
  __MDSPAN_FORCE_INLINE_FUNCTION constexpr _Tp const &__first() const noexcept {
    return this->__first_base_t::__ref();
  }
  __MDSPAN_FORCE_INLINE_FUNCTION constexpr _Up &__second() noexcept {
    return this->__second_base_t::__ref();
  }
  __MDSPAN_FORCE_INLINE_FUNCTION constexpr _Up const &__second() const noexcept {
    return this->__second_base_t::__ref();
  }

  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  constexpr __compressed_pair() noexcept = default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  constexpr __compressed_pair(__compressed_pair const &) noexcept = default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  constexpr __compressed_pair(__compressed_pair &&) noexcept = default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  __MDSPAN_CONSTEXPR_14_DEFAULTED __compressed_pair &
  operator=(__compressed_pair const &) noexcept = default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  __MDSPAN_CONSTEXPR_14_DEFAULTED __compressed_pair &
  operator=(__compressed_pair &&) noexcept = default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  ~__compressed_pair() noexcept = default;
  template <class _TLike, class _ULike>
  __MDSPAN_INLINE_FUNCTION constexpr __compressed_pair(_TLike &&__t, _ULike &&__u) noexcept
    : __first_base_t(_Tp((_TLike &&) __t)),
      __second_base_t(_Up((_ULike &&) __u))
  { }
};

#endif // !defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)

} // end namespace __detail

#endif // _LIBCUDACXX_STD_VER > 11

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___MDSPAN_COMPRESSED_PAIR_HPP
