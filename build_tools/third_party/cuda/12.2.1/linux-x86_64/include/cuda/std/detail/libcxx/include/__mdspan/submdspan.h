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


#ifndef _LIBCUDACXX___MDSPAN_SUBMDSPAN_HPP
#define _LIBCUDACXX___MDSPAN_SUBMDSPAN_HPP

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__mdspan/dynamic_extent.h"
#include "../__mdspan/full_extent_t.h"
#include "../__mdspan/layout_left.h"
#include "../__mdspan/layout_right.h"
#include "../__mdspan/layout_stride.h"
#include "../__mdspan/macros.h"
#include "../__mdspan/mdspan.h"
#include "../__type_traits/conditional.h"
#include "../__type_traits/integral_constant.h"
#include "../__type_traits/is_convertible.h"
#include "../__type_traits/is_same.h"
#include "../__type_traits/is_signed.h"
#include "../__type_traits/remove_const.h"
#include "../__type_traits/remove_reference.h"
#include "../__utility/move.h"
#include "../__utility/pair.h"
#include "../tuple"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 11

namespace __detail {

template <size_t _OldExtent, size_t _OldStaticStride, class _Tp>
struct __slice_wrap {
  _Tp slice;
  size_t old_extent;
  size_t old_stride;
};

//--------------------------------------------------------------------------------

template <size_t _OldExtent, size_t _OldStaticStride>
__MDSPAN_INLINE_FUNCTION constexpr
__slice_wrap<_OldExtent, _OldStaticStride, size_t>
__wrap_slice(size_t __val, size_t __ext, size_t __stride) { return { __val, __ext, __stride }; }

template <size_t _OldExtent, size_t _OldStaticStride, class _IntegerType, _IntegerType _Value0>
__MDSPAN_INLINE_FUNCTION constexpr
__slice_wrap<_OldExtent, _OldStaticStride, integral_constant<_IntegerType, _Value0>>
__wrap_slice(size_t __val, size_t __ext, integral_constant<_IntegerType, _Value0> __stride)
{
#if __MDSPAN_HAS_CXX_17
  if constexpr (_CUDA_VSTD::is_signed_v<_IntegerType>) {
    static_assert(_Value0 >= _IntegerType(0), "Invalid slice specifier");
  }
#endif // __MDSPAN_HAS_CXX_17

  return { __val, __ext, __stride };
}

template <size_t _OldExtent, size_t _OldStaticStride>
__MDSPAN_INLINE_FUNCTION constexpr
__slice_wrap<_OldExtent, _OldStaticStride, full_extent_t>
__wrap_slice(full_extent_t __val, size_t __ext, size_t __stride) { return { __val, __ext, __stride }; }

// TODO generalize this to anything that works with get<0> and get<1>
template <size_t _OldExtent, size_t _OldStaticStride>
__MDSPAN_INLINE_FUNCTION constexpr
__slice_wrap<_OldExtent, _OldStaticStride, _CUDA_VSTD::tuple<size_t, size_t>>
__wrap_slice(_CUDA_VSTD::tuple<size_t, size_t> const& __val, size_t __ext, size_t __stride)
{
  return { __val, __ext, __stride };
}

template <size_t _OldExtent, size_t _OldStaticStride,
	  class _IntegerType0, _IntegerType0 _Value0,
	  class _IntegerType1, _IntegerType1 _Value1>
__MDSPAN_INLINE_FUNCTION constexpr
  __slice_wrap<_OldExtent, _OldStaticStride,
               _CUDA_VSTD::tuple<integral_constant<_IntegerType0, _Value0>,
                          integral_constant<_IntegerType1, _Value1>>>
__wrap_slice(_CUDA_VSTD::tuple<integral_constant<_IntegerType0, _Value0>, integral_constant<_IntegerType1, _Value1>> const& __val, size_t __ext, size_t __stride)
{
  static_assert(_Value1 >= _Value0, "Invalid slice tuple");
  return { __val, __ext, __stride };
}

//--------------------------------------------------------------------------------


// a layout right remains a layout right if it is indexed by 0 or more scalars,
// then optionally a pair and finally 0 or more all
template <
  // what we encountered until now preserves the layout right
  bool _Result=true,
  // we only encountered 0 or more scalars, no pair or all
  bool _EncounteredOnlyScalar=true
>
struct preserve_layout_right_analysis : integral_constant<bool, _Result> {
  using layout_type_if_preserved = layout_right;
  using encounter_pair = preserve_layout_right_analysis<
    // if we encounter a pair, the layout remains a layout right only if it was one before
    // and that only scalars were encountered until now
    _Result && _EncounteredOnlyScalar,
    // if we encounter a pair, we didn't encounter scalars only
    false
  >;
  using encounter_all = preserve_layout_right_analysis<
    // if we encounter a all, the layout remains a layout right if it was one before
    _Result,
    // if we encounter a all, we didn't encounter scalars only
    false
  >;
  using encounter_scalar = preserve_layout_right_analysis<
    // if we encounter a scalar, the layout remains a layout right only if it was one before
    // and that only scalars were encountered until now
    _Result && _EncounteredOnlyScalar,
    // if we encounter a scalar, the fact that we encountered scalars only doesn't change
    _EncounteredOnlyScalar
  >;
};

// a layout left remains a layout left if it is indexed by 0 or more all,
// then optionally a pair and finally 0 or more scalars
template <
  bool _Result=true,
  bool _EncounteredOnlyAll=true
>
struct preserve_layout_left_analysis : integral_constant<bool, _Result> {
  using layout_type_if_preserved = layout_left;
  using encounter_pair = preserve_layout_left_analysis<
    // if we encounter a pair, the layout remains a layout left only if it was one before
    // and that only all were encountered until now
    _Result && _EncounteredOnlyAll,
    // if we encounter a pair, we didn't encounter all only
    false
  >;
  using encounter_all = preserve_layout_left_analysis<
    // if we encounter a all, the layout remains a layout left only if it was one before
    // and that only all were encountered until now
    _Result && _EncounteredOnlyAll,
    // if we encounter a all, the fact that we encountered scalars all doesn't change
    _EncounteredOnlyAll
  >;
  using encounter_scalar = preserve_layout_left_analysis<
    // if we encounter a scalar, the layout remains a layout left if it was one before
    _Result,
    // if we encounter a scalar, we didn't encounter scalars only
    false
  >;
};

struct ignore_layout_preservation : integral_constant<bool, false> {
  using layout_type_if_preserved = void;
  using encounter_pair = ignore_layout_preservation;
  using encounter_all = ignore_layout_preservation;
  using encounter_scalar = ignore_layout_preservation;
};

template <class _Layout>
struct preserve_layout_analysis
  : ignore_layout_preservation { };
template <>
struct preserve_layout_analysis<layout_right>
  : preserve_layout_right_analysis<> { };
template <>
struct preserve_layout_analysis<layout_left>
  : preserve_layout_left_analysis<> { };

//--------------------------------------------------------------------------------

template <
  class _IndexT,
  class _PreserveLayoutAnalysis,
  class _OffsetsArray=__partially_static_sizes<_IndexT, size_t>,
  class _ExtsArray=__partially_static_sizes<_IndexT, size_t>,
  class _StridesArray=__partially_static_sizes<_IndexT, size_t>,
  class = _CUDA_VSTD::make_index_sequence<_OffsetsArray::__size>,
  class = _CUDA_VSTD::make_index_sequence<_ExtsArray::__size>,
  class = _CUDA_VSTD::make_index_sequence<_StridesArray::__size>
>
struct __assign_op_slice_handler;

/* clang-format: off */
template <
  class _IndexT,
  class _PreserveLayoutAnalysis,
  size_t... _Offsets,
  size_t... _Exts,
  size_t... _Strides,
  size_t... _OffsetIdxs,
  size_t... _ExtIdxs,
  size_t... _StrideIdxs>
struct __assign_op_slice_handler<
  _IndexT,
  _PreserveLayoutAnalysis,
  __partially_static_sizes<_IndexT, size_t, _Offsets...>,
  __partially_static_sizes<_IndexT, size_t, _Exts...>,
  __partially_static_sizes<_IndexT, size_t, _Strides...>,
  _CUDA_VSTD::integer_sequence<size_t, _OffsetIdxs...>,
  _CUDA_VSTD::integer_sequence<size_t, _ExtIdxs...>,
  _CUDA_VSTD::integer_sequence<size_t, _StrideIdxs...>>
{
  // TODO remove this for better compiler performance
  static_assert(
    __MDSPAN_FOLD_AND((_Strides == dynamic_extent || _Strides > 0) /* && ... */),
    " "
  );
  static_assert(
    __MDSPAN_FOLD_AND((_Offsets == dynamic_extent || _Offsets >= 0) /* && ... */),
    " "
  );

  using __offsets_storage_t = __partially_static_sizes<_IndexT, size_t, _Offsets...>;
  using __extents_storage_t = __partially_static_sizes<_IndexT, size_t, _Exts...>;
  using __strides_storage_t = __partially_static_sizes<_IndexT, size_t, _Strides...>;
  __offsets_storage_t __offsets;
  __extents_storage_t __exts;
  __strides_storage_t __strides;

#ifdef __INTEL_COMPILER
#if __INTEL_COMPILER <= 1800
  __MDSPAN_INLINE_FUNCTION constexpr __assign_op_slice_handler(__assign_op_slice_handler&& __other) noexcept
    : __offsets(_CUDA_VSTD::move(__other.__offsets)), __exts(_CUDA_VSTD::move(__other.__exts)), __strides(_CUDA_VSTD::move(__other.__strides))
  { }
  __MDSPAN_INLINE_FUNCTION constexpr __assign_op_slice_handler(
    __offsets_storage_t&& __o,
    __extents_storage_t&& __e,
    __strides_storage_t&& __s
  ) noexcept
    : __offsets(_CUDA_VSTD::move(__o)), __exts(_CUDA_VSTD::move(__e)), __strides(_CUDA_VSTD::move(__s))
  { }
#endif
#endif

// Don't define this unless we need it; they have a cost to compile
#ifndef __MDSPAN_USE_RETURN_TYPE_DEDUCTION
  using __extents_type = _CUDA_VSTD::extents<_IndexT, _Exts...>;
#endif

  // For size_t slice, skip the extent and stride, but add an offset corresponding to the value
  template <size_t _OldStaticExtent, size_t _OldStaticStride>
  __MDSPAN_FORCE_INLINE_FUNCTION // NOLINT (misc-unconventional-assign-operator)
  constexpr auto
  operator=(__slice_wrap<_OldStaticExtent, _OldStaticStride, size_t>&& __slice) noexcept
    -> __assign_op_slice_handler<
         _IndexT,
         typename _PreserveLayoutAnalysis::encounter_scalar,
         __partially_static_sizes<_IndexT, size_t, _Offsets..., dynamic_extent>,
         __partially_static_sizes<_IndexT, size_t, _Exts...>,
         __partially_static_sizes<_IndexT, size_t, _Strides...>/* intentional space here to work around ICC bug*/> {
    return {
      __partially_static_sizes<_IndexT, size_t, _Offsets..., dynamic_extent>(
        __construct_psa_from_all_exts_values_tag,
        __offsets.template __get_n<_OffsetIdxs>()..., __slice.slice),
      _CUDA_VSTD::move(__exts),
      _CUDA_VSTD::move(__strides)
    };
  }

  // Treat integral_constant slice like size_t slice, but with a compile-time offset.
  // The result's extents_type can't take advantage of that,
  // but it might help for specialized layouts.
  template <size_t _OldStaticExtent, size_t _OldStaticStride, class _IntegerType, _IntegerType _Value0>
  __MDSPAN_FORCE_INLINE_FUNCTION // NOLINT (misc-unconventional-assign-operator)
  constexpr auto
  operator=(__slice_wrap<_OldStaticExtent, _OldStaticStride, integral_constant<_IntegerType, _Value0>>&&) noexcept
    -> __assign_op_slice_handler<
         _IndexT,
         typename _PreserveLayoutAnalysis::encounter_scalar,
         __partially_static_sizes<_IndexT, size_t, _Offsets..., _Value0>,
         __partially_static_sizes<_IndexT, size_t, _Exts...>,
         __partially_static_sizes<_IndexT, size_t, _Strides...>/* intentional space here to work around ICC bug*/> {
#if __MDSPAN_HAS_CXX_17
    if constexpr (_CUDA_VSTD::is_signed_v<_IntegerType>) {
      static_assert(_Value0 >= _IntegerType(0), "Invalid slice specifier");
    }
#endif // __MDSPAN_HAS_CXX_17
    return {
      __partially_static_sizes<_IndexT, size_t, _Offsets..., _Value0>(
        __construct_psa_from_all_exts_values_tag,
        __offsets.template __get_n<_OffsetIdxs>()..., size_t(_Value0)),
      _CUDA_VSTD::move(__exts),
      _CUDA_VSTD::move(__strides)
    };
  }

  // For a _CUDA_VSTD::full_extent, offset 0 and old extent
  template <size_t _OldStaticExtent, size_t _OldStaticStride>
  __MDSPAN_FORCE_INLINE_FUNCTION // NOLINT (misc-unconventional-assign-operator)
  constexpr auto
  operator=(__slice_wrap<_OldStaticExtent, _OldStaticStride, full_extent_t>&& __slice) noexcept
    -> __assign_op_slice_handler<
         _IndexT,
         typename _PreserveLayoutAnalysis::encounter_all,
         __partially_static_sizes<_IndexT, size_t, _Offsets..., 0>,
         __partially_static_sizes<_IndexT, size_t, _Exts..., _OldStaticExtent>,
         __partially_static_sizes<_IndexT, size_t, _Strides..., _OldStaticStride>/* intentional space here to work around ICC bug*/> {
    return {
      __partially_static_sizes<_IndexT, size_t, _Offsets..., 0>(
        __construct_psa_from_all_exts_values_tag,
        __offsets.template __get_n<_OffsetIdxs>()..., size_t(0)),
      __partially_static_sizes<_IndexT, size_t, _Exts..., _OldStaticExtent>(
        __construct_psa_from_all_exts_values_tag,
        __exts.template __get_n<_ExtIdxs>()..., __slice.old_extent),
      __partially_static_sizes<_IndexT, size_t, _Strides..., _OldStaticStride>(
        __construct_psa_from_all_exts_values_tag,
        __strides.template __get_n<_StrideIdxs>()..., __slice.old_stride)
    };
  }

  // For a _CUDA_VSTD::tuple, add an offset and add a new dynamic extent (strides still preserved)
  template <size_t _OldStaticExtent, size_t _OldStaticStride>
  __MDSPAN_FORCE_INLINE_FUNCTION // NOLINT (misc-unconventional-assign-operator)
  constexpr auto
  operator=(__slice_wrap<_OldStaticExtent, _OldStaticStride, tuple<size_t, size_t>>&& __slice) noexcept
    -> __assign_op_slice_handler<
         _IndexT,
         typename _PreserveLayoutAnalysis::encounter_pair,
         __partially_static_sizes<_IndexT, size_t, _Offsets..., dynamic_extent>,
         __partially_static_sizes<_IndexT, size_t, _Exts..., dynamic_extent>,
         __partially_static_sizes<_IndexT, size_t, _Strides..., _OldStaticStride>/* intentional space here to work around ICC bug*/> {
    return {
      __partially_static_sizes<_IndexT, size_t, _Offsets..., dynamic_extent>(
        __construct_psa_from_all_exts_values_tag,
        __offsets.template __get_n<_OffsetIdxs>()..., _CUDA_VSTD::get<0>(__slice.slice)),
      __partially_static_sizes<_IndexT, size_t, _Exts..., dynamic_extent>(
        __construct_psa_from_all_exts_values_tag,
        __exts.template __get_n<_ExtIdxs>()..., _CUDA_VSTD::get<1>(__slice.slice) - _CUDA_VSTD::get<0>(__slice.slice)),
      __partially_static_sizes<_IndexT, size_t, _Strides..., _OldStaticStride>(
        __construct_psa_from_all_exts_values_tag,
        __strides.template __get_n<_StrideIdxs>()..., __slice.old_stride)
    };
  }

  // For a _CUDA_VSTD::tuple of two integral_constant, do something like
  // we did above for a tuple of two size_t, but make sure the
  // result's extents type make the values compile-time constants.
  template <size_t _OldStaticExtent, size_t _OldStaticStride,
	    class _IntegerType0, _IntegerType0 _Value0,
	    class _IntegerType1, _IntegerType1 _Value1>
  __MDSPAN_FORCE_INLINE_FUNCTION // NOLINT (misc-unconventional-assign-operator)
  constexpr auto
  operator=(__slice_wrap<_OldStaticExtent, _OldStaticStride, tuple<integral_constant<_IntegerType0, _Value0>, integral_constant<_IntegerType1, _Value1>>>&& __slice) noexcept
    -> __assign_op_slice_handler<
         _IndexT,
         typename _PreserveLayoutAnalysis::encounter_pair,
         __partially_static_sizes<_IndexT, size_t, _Offsets..., size_t(_Value0)>,
         __partially_static_sizes<_IndexT, size_t, _Exts..., size_t(_Value1 - _Value0)>,
         __partially_static_sizes<_IndexT, size_t, _Strides..., _OldStaticStride>/* intentional space here to work around ICC bug*/> {
    static_assert(_Value1 >= _Value0, "Invalid slice specifier");
    return {
      // We're still turning the template parameters _Value0 and _Value1
      // into (constexpr) run-time values here.
      __partially_static_sizes<_IndexT, size_t, _Offsets..., size_t(_Value0) > (
        __construct_psa_from_all_exts_values_tag,
        __offsets.template __get_n<_OffsetIdxs>()..., _Value0),
      __partially_static_sizes<_IndexT, size_t, _Exts..., size_t(_Value1 - _Value0) > (
        __construct_psa_from_all_exts_values_tag,
        __exts.template __get_n<_ExtIdxs>()..., _Value1 - _Value0),
      __partially_static_sizes<_IndexT, size_t, _Strides..., _OldStaticStride>(
        __construct_psa_from_all_exts_values_tag,
        __strides.template __get_n<_StrideIdxs>()..., __slice.old_stride)
    };
  }

   // TODO defer instantiation of this?
  using layout_type = conditional_t<
    _PreserveLayoutAnalysis::value,
    typename _PreserveLayoutAnalysis::layout_type_if_preserved,
    layout_stride
  >;

  // TODO noexcept specification
  template <class NewLayout>
  __MDSPAN_INLINE_FUNCTION
  __MDSPAN_DEDUCE_RETURN_TYPE_SINGLE_LINE(
    (
      constexpr /* auto */
      _make_layout_mapping_impl(NewLayout) noexcept
    ),
    (
      /* not layout stride, so don't pass dynamic_strides */
      /* return */ typename NewLayout::template mapping<_CUDA_VSTD::extents<_IndexT, _Exts...>>(
        extents<_IndexT, _Exts...>::__make_extents_impl(_CUDA_VSTD::move(__exts))
      ) /* ; */
    )
  )

  __MDSPAN_INLINE_FUNCTION
  __MDSPAN_DEDUCE_RETURN_TYPE_SINGLE_LINE(
    (
      constexpr /* auto */
      _make_layout_mapping_impl(layout_stride) noexcept
    ),
    (
      /* return */ layout_stride::template mapping<_CUDA_VSTD::extents<_IndexT, _Exts...>>
        ::__make_mapping(_CUDA_VSTD::move(__exts), _CUDA_VSTD::move(__strides)) /* ; */
    )
  )

  template <class _OldLayoutMapping> // mostly for deferred instantiation, but maybe we'll use this in the future
  __MDSPAN_INLINE_FUNCTION
  __MDSPAN_DEDUCE_RETURN_TYPE_SINGLE_LINE(
    (
      constexpr /* auto */
      make_layout_mapping(_OldLayoutMapping const&) noexcept
    ),
    (
      /* return */ this->_make_layout_mapping_impl(layout_type{}) /* ; */
    )
  )
};

//==============================================================================

#if __MDSPAN_USE_RETURN_TYPE_DEDUCTION
// Forking this because the C++11 version will be *completely* unreadable
template <class _ET, class _ST, size_t... _Exts, class _LP, class _AP, class... _SliceSpecs, size_t... _Idxs>
__MDSPAN_INLINE_FUNCTION
constexpr auto _submdspan_impl(
  _CUDA_VSTD::integer_sequence<size_t, _Idxs...>,
  mdspan<_ET, _CUDA_VSTD::extents<_ST, _Exts...>, _LP, _AP> const& __src,
  _SliceSpecs&&... __slices
) noexcept
{
  using __index_t = _ST;
  auto __handled =
    __MDSPAN_FOLD_ASSIGN_LEFT(
      (
        __detail::__assign_op_slice_handler<
          __index_t,
          __detail::preserve_layout_analysis<_LP>
        >{
          __partially_static_sizes<__index_t, size_t>{},
          __partially_static_sizes<__index_t, size_t>{},
          __partially_static_sizes<__index_t, size_t>{}
        }
      ),
        /* = ... = */
      __detail::__wrap_slice<
        _Exts, dynamic_extent
      >(
        __slices, __src.extents().template __extent<_Idxs>(),
        __src.mapping().stride(_Idxs)
      )
    );

  size_t __offset_size = __src.mapping()(__handled.__offsets.template __get_n<_Idxs>()...);
  auto __offset_ptr = __src.accessor().offset(__src.data_handle(), __offset_size);
  auto __map = __handled.make_layout_mapping(__src.mapping());
  auto __acc_pol = typename _AP::offset_policy(__src.accessor());
  return mdspan<
    _ET, remove_const_t<_CUDA_VSTD::remove_reference_t<decltype(__map.extents())>>,
        typename decltype(__handled)::layout_type, remove_const_t<_CUDA_VSTD::remove_reference_t<decltype(__acc_pol)>>
  >(
    _CUDA_VSTD::move(__offset_ptr), _CUDA_VSTD::move(__map), _CUDA_VSTD::move(__acc_pol)
  );
}
#else

template <class _ET, class _AP, class _Src, class _Handled, size_t... _Idxs>
auto _submdspan_impl_helper(_Src&& __src, _Handled&& __h, _CUDA_VSTD::integer_sequence<size_t, _Idxs...>)
  -> mdspan<
       _ET, typename _Handled::__extents_type, typename _Handled::layout_type, typename _AP::offset_policy
     >
{
  return {
    __src.accessor().offset(__src.data_handle(), __src.mapping()(__h.__offsets.template __get_n<_Idxs>()...)),
    __h.make_layout_mapping(__src.mapping()),
    typename _AP::offset_policy(__src.accessor())
  };
}

template <class _ET, class _ST, size_t... _Exts, class _LP, class _AP, class... _SliceSpecs, size_t... _Idxs>
__MDSPAN_INLINE_FUNCTION
__MDSPAN_DEDUCE_RETURN_TYPE_SINGLE_LINE(
  (
    constexpr /* auto */ _submdspan_impl(
      _CUDA_VSTD::integer_sequence<size_t, _Idxs...> __seq,
      mdspan<_ET, _CUDA_VSTD::extents<_ST, _Exts...>, _LP, _AP> const& __src,
      _SliceSpecs&&... __slices
    ) noexcept
  ),
  (
    /* return */ _submdspan_impl_helper<_ET, _AP>(
      __src,
      __MDSPAN_FOLD_ASSIGN_LEFT(
        (
          __detail::__assign_op_slice_handler<
            size_t,
            __detail::preserve_layout_analysis<_LP>
          >{
            __partially_static_sizes<_ST, size_t>{},
            __partially_static_sizes<_ST, size_t>{},
            __partially_static_sizes<_ST, size_t>{}
          }
        ),
        /* = ... = */
        __detail::__wrap_slice<
          _Exts, dynamic_extent
        >(
          __slices, __src.extents().template __extent<_Idxs>(), __src.mapping().stride(_Idxs)
        )
      ),
      __seq
    ) /* ; */
  )
)

#endif

template <class _Tp> struct _is_layout_stride : false_type { };
template<>
struct _is_layout_stride<
  layout_stride
> : true_type
{ };

} // namespace __detail

//==============================================================================

__MDSPAN_TEMPLATE_REQUIRES(
  class _ET, class _EXT, class _LP, class _AP, class... _SliceSpecs,
  /* requires */ (
    (
      _LIBCUDACXX_TRAIT(_CUDA_VSTD::is_same, _LP, layout_left)
        || _LIBCUDACXX_TRAIT(_CUDA_VSTD::is_same, _LP, layout_right)
        || __detail::_is_layout_stride<_LP>::value
    ) &&
    __MDSPAN_FOLD_AND((
      _LIBCUDACXX_TRAIT(_CUDA_VSTD::is_convertible, _SliceSpecs, size_t)
        || _LIBCUDACXX_TRAIT(_CUDA_VSTD::is_convertible, _SliceSpecs, tuple<size_t, size_t>)
        || _LIBCUDACXX_TRAIT(_CUDA_VSTD::is_convertible, _SliceSpecs, full_extent_t)
    ) /* && ... */) &&
    sizeof...(_SliceSpecs) == _EXT::rank()
  )
)
__MDSPAN_INLINE_FUNCTION
__MDSPAN_DEDUCE_RETURN_TYPE_SINGLE_LINE(
  (
    constexpr submdspan(
      mdspan<_ET, _EXT, _LP, _AP> const& __src, _SliceSpecs... __slices
    ) noexcept
  ),
  (
    /* return */
      __detail::_submdspan_impl(_CUDA_VSTD::make_index_sequence<sizeof...(_SliceSpecs)>{}, __src, __slices...) /*;*/
  )
)
/* clang-format: on */

#endif // _LIBCUDACXX_STD_VER > 11

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___MDSPAN_SUBMDSPAN_HPP
