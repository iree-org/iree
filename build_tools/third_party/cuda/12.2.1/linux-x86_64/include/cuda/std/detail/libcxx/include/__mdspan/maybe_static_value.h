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

#ifndef _LIBCUDACXX___MDSPAN_MAYBE_STATIC_VALUE_HPP
#define _LIBCUDACXX___MDSPAN_MAYBE_STATIC_VALUE_HPP

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__mdspan/dynamic_extent.h"
#include "../__mdspan/macros.h"
#if !defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
#include "../__mdspan/no_unique_address.h"
#endif

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

// This is only needed for the non-standard-layout version of partially
// static array.
// Needs to be after the includes above to work with the single header generator
#if !__MDSPAN_PRESERVE_STANDARD_LAYOUT

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 11

//==============================================================================

namespace __detail {

// static case
template <class _dynamic_t, class _static_t, _static_t __v,
          _static_t __is_dynamic_sentinal = dynamic_extent,
          size_t __array_entry_index = 0>
struct __maybe_static_value {
  static constexpr _static_t __static_value = __v;
  __MDSPAN_FORCE_INLINE_FUNCTION constexpr _dynamic_t __value() const noexcept {
    return static_cast<_dynamic_t>(__v);
  }
  template <class _Up>
  __MDSPAN_FORCE_INLINE_FUNCTION constexpr
  __mdspan_enable_fold_comma
  __set_value(_Up&& /*__rhs*/) noexcept {
    // Should we assert that the value matches the static value here?
    return {};
  }

  //--------------------------------------------------------------------------

  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  constexpr __maybe_static_value() noexcept = default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  constexpr __maybe_static_value(__maybe_static_value const&) noexcept = default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  constexpr __maybe_static_value(__maybe_static_value&&) noexcept = default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  __MDSPAN_CONSTEXPR_14_DEFAULTED __maybe_static_value& operator=(__maybe_static_value const&) noexcept = default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  __MDSPAN_CONSTEXPR_14_DEFAULTED __maybe_static_value& operator=(__maybe_static_value&&) noexcept = default;
  __MDSPAN_INLINE_FUNCTION_DEFAULTED
  ~__maybe_static_value() noexcept = default;

  __MDSPAN_INLINE_FUNCTION
  constexpr explicit __maybe_static_value(_dynamic_t const&) noexcept {
    // Should we assert that the value matches the static value here?
  }

  //--------------------------------------------------------------------------

};

// dynamic case
template <class _dynamic_t, class _static_t, _static_t __is_dynamic_sentinal, size_t __array_entry_index>
struct __maybe_static_value<_dynamic_t, _static_t, __is_dynamic_sentinal, __is_dynamic_sentinal,
                            __array_entry_index>
#if !defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
    : __no_unique_address_emulation<_Tp>
#endif
{
  static constexpr _static_t __static_value = __is_dynamic_sentinal;
#if defined(__MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
  __MDSPAN_NO_UNIQUE_ADDRESS _dynamic_t __v = {};
  __MDSPAN_FORCE_INLINE_FUNCTION constexpr _dynamic_t __value() const noexcept {
    return __v;
  }
  __MDSPAN_FORCE_INLINE_FUNCTION constexpr _dynamic_t &__ref() noexcept {
    return __v;
  }
  template <class _Up>
  __MDSPAN_FORCE_INLINE_FUNCTION constexpr
  __mdspan_enable_fold_comma
  __set_value(_Up&& __rhs) noexcept {
    __v = (_Up &&)rhs;
    return {};
  }
#else
  __MDSPAN_FORCE_INLINE_FUNCTION constexpr _dynamic_t __value() const noexcept {
    return this->__no_unique_address_emulation<_dynamic_t>::__ref();
  }
  __MDSPAN_FORCE_INLINE_FUNCTION constexpr _dynamic_t &__ref() noexcept {
    return this->__no_unique_address_emulation<_dynamic_t>::__ref();
  }
  template <class _Up>
  __MDSPAN_FORCE_INLINE_FUNCTION constexpr
  __mdspan_enable_fold_comma
  __set_value(_Up&& __rhs) noexcept {
    this->__no_unique_address_emulation<_dynamic_t>::__ref() = (_Up &&)__rhs;
    return {};
  }
#endif
};

} // namespace __detail

//==============================================================================


#endif // _LIBCUDACXX_STD_VER > 11

_LIBCUDACXX_END_NAMESPACE_STD

#endif // !__MDSPAN_PRESERVE_STANDARD_LAYOUT

#endif // _LIBCUDACXX___MDSPAN_MAYBE_STATIC_VALUE_HPP
