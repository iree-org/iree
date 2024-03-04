 /* Copyright 1993-2016 NVIDIA Corporation.  All rights reserved.
  *
  * NOTICE TO LICENSEE:
  *
  * The source code and/or documentation ("Licensed Deliverables") are
  * subject to NVIDIA intellectual property rights under U.S. and
  * international Copyright laws.
  *
  * The Licensed Deliverables contained herein are PROPRIETARY and
  * CONFIDENTIAL to NVIDIA and are being provided under the terms and
  * conditions of a form of NVIDIA software license agreement by and
  * between NVIDIA and Licensee ("License Agreement") or electronically
  * accepted by Licensee.  Notwithstanding any terms or conditions to
  * the contrary in the License Agreement, reproduction or disclosure
  * of the Licensed Deliverables to any third party without the express
  * written consent of NVIDIA is prohibited.
  *
  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
  * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
  * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  THEY ARE
  * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
  * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
  * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
  * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
  * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
  * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
  * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
  * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
  * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
  * OF THESE LICENSED DELIVERABLES.
  *
  * U.S. Government End Users.  These Licensed Deliverables are a
  * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
  * 1995), consisting of "commercial computer software" and "commercial
  * computer software documentation" as such terms are used in 48
  * C.F.R. 12.212 (SEPT 1995) and are provided to the U.S. Government
  * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
  * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
  * U.S. Government End Users acquire the Licensed Deliverables with
  * only those rights set forth herein.
  *
  * Any use of the Licensed Deliverables in individual and commercial
  * software must include, in the user documentation and internal
  * comments to the code, the above Disclaimer and U.S. Government End
  * Users Notice.
  */

#ifndef _CG_REDUCE_H_
#define _CG_REDUCE_H_

#include "info.h"
#include "helpers.h"
#include "coalesced_reduce.h"
#include "functional.h"
#include "cooperative_groups.h"

_CG_BEGIN_NAMESPACE

namespace details {

    template <class Ty>
    using _redux_is_add_supported = _CG_STL_NAMESPACE::integral_constant<
            bool,
            _CG_STL_NAMESPACE::is_integral<Ty>::value && (sizeof(Ty) <= 4)>;

    template <class Ty>
    using redux_is_add_supported = _redux_is_add_supported<Ty>;

    // A specialization for 64 bit logical operations is possible
    // but for now only accelerate 32 bit bitwise ops
    template <class Ty>
    using redux_is_logical_supported = redux_is_add_supported<Ty>;

    // Base operator support case
    template <class TyOp, class Ty> struct _redux_op_supported                 : public _CG_STL_NAMESPACE::false_type {};
#ifdef _CG_HAS_OP_REDUX
    template <class Ty> struct _redux_op_supported<cooperative_groups::plus<Ty>,    Ty> : public redux_is_add_supported<Ty> {};
    template <class Ty> struct _redux_op_supported<cooperative_groups::less<Ty>,    Ty> : public redux_is_add_supported<Ty> {};
    template <class Ty> struct _redux_op_supported<cooperative_groups::greater<Ty>, Ty> : public redux_is_add_supported<Ty> {};
    template <class Ty> struct _redux_op_supported<cooperative_groups::bit_and<Ty>, Ty> : public redux_is_logical_supported<Ty> {};
    template <class Ty> struct _redux_op_supported<cooperative_groups::bit_or<Ty>,  Ty> : public redux_is_logical_supported<Ty> {};
    template <class Ty> struct _redux_op_supported<cooperative_groups::bit_xor<Ty>, Ty> : public redux_is_logical_supported<Ty> {};
#endif

    template <class Ty, template <class> class TyOp>
    using redux_op_supported = _redux_op_supported<
            typename details::remove_qual<TyOp<Ty>>,
            Ty>;

    // Groups smaller than 16 actually have worse performance characteristics when used with redux
    // tiles of size 16 and 32 perform the same or better and have better code generation profiles
    template <class TyGroup> struct _redux_group_optimized : public _CG_STL_NAMESPACE::false_type {};

    template <unsigned int Sz, typename TyPar>
    struct _redux_group_optimized<cooperative_groups::thread_block_tile<Sz, TyPar>> : public _CG_STL_NAMESPACE::integral_constant<
                                                                                            bool,
                                                                                            (Sz >= 16)> {};
    template <unsigned int Sz, typename TyPar>
    struct _redux_group_optimized<internal_thread_block_tile<Sz, TyPar>>            : public _CG_STL_NAMESPACE::integral_constant<
                                                                                            bool,
                                                                                            (Sz >= 16)> {};
    template <>
    struct _redux_group_optimized<cooperative_groups::coalesced_group>              : public _CG_STL_NAMESPACE::true_type  {};

    template <typename TyGroup>
    using redux_group_optimized = _redux_group_optimized<details::remove_qual<TyGroup>>;

    template <template <class> class TyOp>
    _CG_STATIC_QUALIFIER int pick_redux(int mask, int val);
    template <template <class> class TyOp>
    _CG_STATIC_QUALIFIER unsigned int pick_redux(int mask, unsigned int val);

#ifdef _CG_HAS_OP_REDUX
    template <> _CG_QUALIFIER int pick_redux<cooperative_groups::plus>(int mask, int val) {
        return __reduce_add_sync(mask, val);
    }
    template <> _CG_QUALIFIER int pick_redux<cooperative_groups::less>(int mask, int val) {
        return __reduce_min_sync(mask, val);
    }
    template <> _CG_QUALIFIER int pick_redux<cooperative_groups::greater>(int mask, int val) {
        return __reduce_max_sync(mask, val);
    }
    template <> _CG_QUALIFIER int pick_redux<cooperative_groups::bit_and>(int mask, int val) {
        return __reduce_and_sync(mask, val);
    }
    template <> _CG_QUALIFIER int pick_redux<cooperative_groups::bit_xor>(int mask, int val) {
        return __reduce_xor_sync(mask, val);
    }
    template <> _CG_QUALIFIER int pick_redux<cooperative_groups::bit_or>(int mask, int val) {
        return __reduce_or_sync(mask, val);
    }

    template <> _CG_QUALIFIER unsigned int pick_redux<cooperative_groups::plus>(int mask, unsigned int val) {
        return __reduce_add_sync(mask, val);
    }
    template <> _CG_QUALIFIER unsigned int pick_redux<cooperative_groups::less>(int mask, unsigned int val) {
        return __reduce_min_sync(mask, val);
    }
    template <> _CG_QUALIFIER unsigned int pick_redux<cooperative_groups::greater>(int mask, unsigned int val) {
        return __reduce_max_sync(mask, val);
    }
    template <> _CG_QUALIFIER unsigned int pick_redux<cooperative_groups::bit_and>(int mask, unsigned int val) {
        return __reduce_and_sync(mask, val);
    }
    template <> _CG_QUALIFIER unsigned int pick_redux<cooperative_groups::bit_xor>(int mask, unsigned int val) {
        return __reduce_xor_sync(mask, val);
    }
    template <> _CG_QUALIFIER unsigned int pick_redux<cooperative_groups::bit_or>(int mask, unsigned int val) {
        return __reduce_or_sync(mask, val);
    }
#endif


    template <typename TyVal, bool = _CG_STL_NAMESPACE::is_unsigned<TyVal>::value>
    struct _accelerated_op;

    // Signed type redux intrinsic dispatch
    template <typename TyVal>
    struct _accelerated_op<TyVal, false> {
        template <template <class> class TyOp>
        _CG_STATIC_QUALIFIER TyVal redux(int mask, TyVal val) {
            return static_cast<TyVal>(pick_redux<TyOp>(mask, static_cast<int>(val)));
        }
    };

    // Unsigned type redux intrinsic dispatch
    template <typename TyVal>
    struct _accelerated_op<TyVal, true> {
        template <template <class> class TyOp>
        _CG_STATIC_QUALIFIER TyVal redux(int mask, TyVal val) {
            return static_cast<TyVal>(pick_redux<TyOp>(mask, static_cast<unsigned int>(val)));
        }
    };

    template <typename TyVal>
    using accelerated_op = _accelerated_op<TyVal>;


    template <typename TyVal, typename TyFnInput, typename TyGroup>
    class _redux_dispatch {
        template <class Ty, template <class> class TyOp>
        using _redux_is_usable = _CG_STL_NAMESPACE::integral_constant<bool,
            redux_op_supported<Ty, TyOp>::value &&
            redux_group_optimized<TyGroup>::value>;

        template <class Ty, template <class> class TyOp>
        using redux_is_usable = typename _CG_STL_NAMESPACE::enable_if<_redux_is_usable<Ty, TyOp>::value, void>::type*;

        template <class Ty, template <class> class TyOp>
        using redux_is_not_usable = typename _CG_STL_NAMESPACE::enable_if<!_redux_is_usable<Ty, TyOp>::value, void>::type*;

    public:
        // Dispatch to redux if the combination of op and args are supported
        template<
            template <class> class TyOp,
            redux_is_usable<TyFnInput, TyOp> = nullptr>
        _CG_STATIC_QUALIFIER auto reduce(const TyGroup& group, TyVal&& val, TyOp<TyFnInput>&& op) -> decltype(op(val, val)) {
            // Retrieve the mask for the group and dispatch to redux
            return accelerated_op<TyFnInput>::template redux<TyOp>(_coalesced_group_data_access::get_mask(group), _CG_STL_NAMESPACE::forward<TyVal>(val));
        }

        template<
            template <class> class TyOp,
            redux_is_usable<TyFnInput, TyOp> = nullptr>
        _CG_STATIC_QUALIFIER auto reduce(const TyGroup& group, TyVal&& val, TyOp<TyFnInput>& op) -> decltype(op(val, val)) {
            // Retrieve the mask for the group and dispatch to redux
            return accelerated_op<TyFnInput>::template redux<TyOp>(_coalesced_group_data_access::get_mask(group), _CG_STL_NAMESPACE::forward<TyVal>(val));
        }

        // Fallback shuffle sync reduction
        template <
            template <class> class TyOp,
            redux_is_not_usable<TyFnInput, TyOp> = nullptr>
        _CG_STATIC_QUALIFIER auto reduce(const TyGroup& group, TyVal&& val, TyOp<TyFnInput>&& op) -> decltype(op(val, val)) {
            //Dispatch to fallback shuffle sync accelerated reduction
            return coalesced_reduce(group, _CG_STL_NAMESPACE::forward<TyVal>(val), _CG_STL_NAMESPACE::forward<TyOp<TyFnInput>>(op));
        }

    };

    // Group support for reduce.
    template <class TyGroup> struct _reduce_group_supported : public _CG_STL_NAMESPACE::false_type {};

    template <unsigned int Sz, typename TyPar>
    struct _reduce_group_supported<cooperative_groups::thread_block_tile<Sz, TyPar>> : public _CG_STL_NAMESPACE::true_type {};
    template <unsigned int Sz, typename TyPar>
    struct _reduce_group_supported<internal_thread_block_tile<Sz, TyPar>>            : public _CG_STL_NAMESPACE::true_type {};
    template <>
    struct _reduce_group_supported<cooperative_groups::coalesced_group>              : public _CG_STL_NAMESPACE::true_type {};

    template <typename TyGroup>
    using reduce_group_supported = _reduce_group_supported<details::remove_qual<TyGroup>>;

    template <typename TyVal, typename TyFnInput, template <class> class TyOp, typename TyGroup>
    _CG_QUALIFIER auto reduce(const TyGroup& group, TyVal&& val, TyOp<TyFnInput>&& op) -> decltype(op(val, val)) {
        static_assert(details::is_op_type_same<TyFnInput, TyVal>::value, "Operator and argument types differ");

        using dispatch = details::_redux_dispatch<TyVal, TyFnInput, TyGroup>;
        return dispatch::reduce(group, _CG_STL_NAMESPACE::forward<TyVal>(val), _CG_STL_NAMESPACE::forward<TyOp<TyFnInput>>(op));
    }

    template <typename TyVal, typename TyFnInput, template <class> class TyOp, typename TyGroup>
    _CG_QUALIFIER auto reduce(const TyGroup& group, TyVal&& val, TyOp<TyFnInput>& op) -> decltype(op(val, val)) {
        static_assert(details::is_op_type_same<TyFnInput, TyVal>::value, "Operator and argument types differ");

        using dispatch = details::_redux_dispatch<TyVal, TyFnInput, TyGroup>;
        return dispatch::reduce(group, _CG_STL_NAMESPACE::forward<TyVal>(val), _CG_STL_NAMESPACE::forward<TyOp<TyFnInput>>(op));
    }


    template <typename TyVal, typename TyOp, typename TyGroup>
    _CG_QUALIFIER auto reduce(const TyGroup& group, TyVal&& val, TyOp&& op) -> decltype(op(val, val)) {
        return details::coalesced_reduce(group, _CG_STL_NAMESPACE::forward<TyVal>(val), _CG_STL_NAMESPACE::forward<TyOp>(op));
    }

    template <unsigned int GroupId>
    struct tile_reduce_dispatch;

    template <>
    struct tile_reduce_dispatch<details::coalesced_group_id> {
        template <typename TyGroup, typename TyVal, typename TyFn>
        _CG_STATIC_QUALIFIER auto reduce(const TyGroup& group, TyVal&& val, TyFn&& op) -> decltype(op(val, val)) {
            return details::reduce(group, _CG_STL_NAMESPACE::forward<TyVal>(val), _CG_STL_NAMESPACE::forward<TyFn>(op));
        }
    };

#if defined(_CG_CPP11_FEATURES)
    template <>
    struct tile_reduce_dispatch<details::multi_tile_group_id> {
        template <unsigned int Size, typename ParentT, typename TyVal, typename TyFn>
        _CG_STATIC_QUALIFIER auto reduce(const thread_block_tile<Size, ParentT>& group, TyVal&& val, TyFn&& op) -> decltype(op(val, val)) {
            using warpType = details::internal_thread_block_tile<32, __static_size_multi_warp_tile_base<Size>>;
            using TyRet = details::remove_qual<TyVal>;
            const unsigned int num_warps = Size / 32;

            auto warp_lambda = [&] (const warpType& warp, TyRet* warp_scratch_location) {
                    *warp_scratch_location =
                        details::reduce(warp, _CG_STL_NAMESPACE::forward<TyVal>(val), op);
            };
            auto inter_warp_lambda =
                [&] (const details::internal_thread_block_tile<num_warps, warpType>& subwarp, TyRet* thread_scratch_location) {
                    *thread_scratch_location =
                        details::reduce(subwarp, *thread_scratch_location, _CG_STL_NAMESPACE::forward<TyFn>(op));
            };
            return details::multi_warp_collectives_helper<TyRet>(group, warp_lambda, inter_warp_lambda);
        }
    };

    template <unsigned int GroupId>
    struct tile_async_reduce_dispatch;

    template <>
    struct tile_async_reduce_dispatch<details::coalesced_group_id> {
        template <unsigned int TySize, typename ParentT, typename TyDst, typename TyVal, typename TyFn, typename TyResHandler>
        _CG_STATIC_QUALIFIER void reduce(const thread_block_tile<TySize, ParentT>& group, TyDst& dst, TyVal&& val, TyFn&& op, TyResHandler& res_handler) {
            // Do regular, in group reduction
            auto result = details::reduce(group, _CG_STL_NAMESPACE::forward<TyVal>(val), op);

            // One thread stores/updates the destination
            if (group.thread_rank() == 0) {
                res_handler(result);
            }
        }
        template <typename TyDst, typename TyVal, typename TyFn, typename TyResHandler>
        _CG_STATIC_QUALIFIER void reduce(const coalesced_group& group, TyDst& dst, TyVal&& val, TyFn&& op, TyResHandler& res_handler) {
            // Do in group reduction to the last thread
            auto result = details::coalesced_reduce_to_one(group, _CG_STL_NAMESPACE::forward<TyVal>(val), op);

            // One thread stores/updates the destination
            if (group.thread_rank() == group.size() - 1) {
                res_handler(result);
            }
        }
    };

    template <>
    struct tile_async_reduce_dispatch<details::multi_tile_group_id> {
        template <unsigned int TySize, typename ParentT, typename TyDst, typename TyInputVal, typename TyFn, typename TyResHandler>
        _CG_STATIC_QUALIFIER void reduce(const thread_block_tile<TySize, ParentT>& group, TyDst& dst, TyInputVal&& val, TyFn&& op, TyResHandler& res_handler) {
            using TyVal = remove_qual<TyInputVal>;
            const unsigned int num_warps = TySize / 32;
            details::barrier_t* sync_location = multi_warp_sync_location_getter(group);
            auto warp_scratch_location = multi_warp_scratch_location_getter<TyVal>(group, group.thread_rank() / 32);

            // Do in warp reduce
            auto warp = details::tiled_partition_internal<32, thread_block_tile<TySize, ParentT>>();
            *warp_scratch_location = details::reduce(warp, _CG_STL_NAMESPACE::forward<TyInputVal>(val), op);

            // Tile of size num_warps from the last warp to arrive does final reduction step
            if (details::sync_warps_last_releases(sync_location, details::cta::thread_rank(), num_warps)) {
                auto subwarp = details::tiled_partition_internal<num_warps, decltype(warp)>();
                if (subwarp.meta_group_rank() == 0) {
                    auto thread_scratch_location = multi_warp_scratch_location_getter<TyVal>(group, subwarp.thread_rank());
                    auto thread_val = *thread_scratch_location;
                    // Release other warps, we read their contribution already.
                    subwarp.sync();
                    details::sync_warps_release(sync_location, subwarp.thread_rank() == 0, details::cta::thread_rank(), num_warps);
                    TyVal result = details::reduce(subwarp, thread_val, op);
                    // One thread stores the result or updates the atomic
                    if (subwarp.thread_rank() == 0) {
                        res_handler(result);
                    }
                }
                warp.sync();
            }
        }
    };
#endif

    template <typename TyGroup, typename TyInputVal, typename TyRetVal>
    _CG_QUALIFIER void check_reduce_params() {
        static_assert(details::is_op_type_same<TyInputVal, TyRetVal>::value, "Operator input and output types differ");
        static_assert(details::reduce_group_supported<TyGroup>::value, "This group does not exclusively represent a tile");
    };

    template <typename TyGroup, typename TyDstVal, typename TyInputVal, typename TyRetVal>
    _CG_QUALIFIER void check_async_reduce_params() {
        check_reduce_params<TyGroup, TyInputVal, TyRetVal>();
        static_assert(details::is_op_type_same<TyDstVal, TyInputVal>::value, "Destination and input types differ");
    }
} // details

template <typename TyGroup, typename TyVal, typename TyFn>
_CG_QUALIFIER auto reduce(const TyGroup& group, TyVal&& val, TyFn&& op) -> decltype(op(val, val)) {
    details::check_reduce_params<TyGroup, details::remove_qual<TyVal>, decltype(op(val, val))>();

    using dispatch = details::tile_reduce_dispatch<TyGroup::_group_id>;
    return dispatch::reduce(group, _CG_STL_NAMESPACE::forward<TyVal>(val), _CG_STL_NAMESPACE::forward<TyFn>(op));
}

#if defined(_CG_CPP11_FEATURES)

# if defined(_CG_HAS_STL_ATOMICS)
template<typename TyGroup, typename TyVal, cuda::thread_scope Sco, typename TyInputVal, typename TyFn>
void _CG_QUALIFIER reduce_update_async(const TyGroup& group, cuda::atomic<TyVal, Sco>& dst, TyInputVal&& val, TyFn&& op) {
    details::check_async_reduce_params<TyGroup, TyVal, details::remove_qual<TyInputVal>, decltype(op(val, val))>();
    auto update_lambda = [&] (TyVal& result) {
        details::atomic_update(dst, result, op);
    };
    using dispatch = details::tile_async_reduce_dispatch<TyGroup::_group_id>;
    dispatch::reduce(group, dst, _CG_STL_NAMESPACE::forward<TyInputVal>(val), _CG_STL_NAMESPACE::forward<TyFn>(op), update_lambda);
}

template<typename TyGroup, typename TyVal, cuda::thread_scope Sco, typename TyInputVal, typename TyFn>
void _CG_QUALIFIER reduce_update_async(const TyGroup& group, const cuda::atomic_ref<TyVal, Sco>& dst, TyInputVal&& val, TyFn&& op) {
    details::check_async_reduce_params<TyGroup, TyVal, details::remove_qual<TyInputVal>, decltype(op(val, val))>();
    auto update_lambda = [&] (TyVal& result) {
        details::atomic_update(dst, result, op);
    };
    using dispatch = details::tile_async_reduce_dispatch<TyGroup::_group_id>;
    dispatch::reduce(group, dst, _CG_STL_NAMESPACE::forward<TyInputVal>(val), _CG_STL_NAMESPACE::forward<TyFn>(op), update_lambda);
}

template<typename TyGroup, typename TyVal, cuda::thread_scope Sco, typename TyInputVal, typename TyFn>
void _CG_QUALIFIER reduce_store_async(const TyGroup& group, cuda::atomic<TyVal, Sco>& dst, TyInputVal&& val, TyFn&& op) {
    details::check_async_reduce_params<TyGroup, TyVal, details::remove_qual<TyInputVal>, decltype(op(val, val))>();
    auto store_lambda = [&] (TyVal& result) {
        details::atomic_store(dst, result);
    };
    using dispatch = details::tile_async_reduce_dispatch<TyGroup::_group_id>;
    dispatch::reduce(group, dst, _CG_STL_NAMESPACE::forward<TyInputVal>(val), _CG_STL_NAMESPACE::forward<TyFn>(op), store_lambda);
}

template<typename TyGroup, typename TyVal, cuda::thread_scope Sco, typename TyInputVal, typename TyFn>
void _CG_QUALIFIER reduce_store_async(const TyGroup& group, const cuda::atomic_ref<TyVal, Sco>& dst, TyInputVal&& val, TyFn&& op) {
    details::check_async_reduce_params<TyGroup, TyVal, details::remove_qual<TyInputVal>, decltype(op(val, val))>();
    auto store_lambda = [&] (TyVal& result) {
        details::atomic_store(dst, result);
    };
    using dispatch = details::tile_async_reduce_dispatch<TyGroup::_group_id>;
    dispatch::reduce(group, dst, _CG_STL_NAMESPACE::forward<TyInputVal>(val), _CG_STL_NAMESPACE::forward<TyFn>(op), store_lambda);
}
# endif

template<typename TyGroup, typename TyVal, typename TyInputVal, typename TyFn>
void _CG_QUALIFIER reduce_store_async(const TyGroup& group, TyVal* dst, TyInputVal&& val, TyFn&& op) {
    details::check_async_reduce_params<TyGroup, TyVal, details::remove_qual<TyInputVal>, decltype(op(val, val))>();
    auto store_lambda = [&] (TyVal& result) {
        *dst = result;
    };
    using dispatch = details::tile_async_reduce_dispatch<TyGroup::_group_id>;
    dispatch::reduce(group, dst, _CG_STL_NAMESPACE::forward<TyInputVal>(val), _CG_STL_NAMESPACE::forward<TyFn>(op), store_lambda);
}
#endif

_CG_END_NAMESPACE

#endif // _CG_REDUCE_H_
