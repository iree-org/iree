// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "./invoke.h"

#include "./hal.h"
#include "./vm.h"
#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/module.h"
#include "iree/vm/api.h"

namespace iree {
namespace python {

namespace {

class InvokeContext {
 public:
  InvokeContext(HalDevice &device) : device_(device) {}

  HalDevice &device() { return device_; }
  HalAllocator allocator() {
    // TODO: Unfortunate that we inc ref here but that is how our object model
    // is set up.
    return HalAllocator::BorrowFromRawPtr(device().allocator());
  }

 private:
  HalDevice device_;
};

using PackCallback =
    std::function<void(InvokeContext &, iree_vm_list_t *, py::handle)>;

class InvokeStatics {
 public:
  ~InvokeStatics() {
    for (auto it : py_type_to_pack_callbacks_) {
      py::handle(it.first).dec_ref();
    }
  }

  py::str kNamedTag = py::str("named");
  py::str kSlistTag = py::str("slist");
  py::str kStupleTag = py::str("stuple");
  py::str kSdictTag = py::str("sdict");

  py::int_ kZero = py::int_(0);
  py::int_ kOne = py::int_(1);
  py::int_ kTwo = py::int_(2);
  py::str kAsArray = py::str("asarray");
  py::str kMapDtypeToElementTypeAttr = py::str("map_dtype_to_element_type");
  py::str kContiguousArg = py::str("C");
  py::str kArrayProtocolAttr = py::str("__array__");
  py::str kDtypeAttr = py::str("dtype");

  // Primitive type names.
  py::str kF32 = py::str("f32");
  py::str kF64 = py::str("f64");
  py::str kI1 = py::str("i1");
  py::str kI8 = py::str("i8");
  py::str kI16 = py::str("i16");
  py::str kI32 = py::str("i32");
  py::str kI64 = py::str("i64");

  // Compound types names.
  py::str kNdarray = py::str("ndarray");

  // Attribute names.
  py::str kAttrBufferView = py::str("_buffer_view");

  // Module 'numpy'.
  py::module &numpy_module() { return numpy_module_; }

  py::object &runtime_module() {
    if (!runtime_module_) {
      runtime_module_ = py::module::import("iree.runtime");
    }
    return *runtime_module_;
  }

  py::module &array_interop_module() {
    if (!array_interop_module_) {
      array_interop_module_ = py::module::import("iree.runtime.array_interop");
    }
    return *array_interop_module_;
  }

  py::object &device_array_type() {
    if (!device_array_type_) {
      device_array_type_ = runtime_module().attr("DeviceArray");
    }
    return *device_array_type_;
  }

  py::type &hal_buffer_view_type() { return hal_buffer_view_type_; }

  py::object MapElementAbiTypeToDtype(py::object &element_abi_type) {
    try {
      return abi_type_to_dtype_[element_abi_type];
    } catch (std::exception &) {
      std::string msg("could not map abi type ");
      msg.append(py::cast<std::string>(py::repr(element_abi_type)));
      msg.append(" to numpy dtype");
      throw std::invalid_argument(std::move(msg));
    }
  }

  enum iree_hal_element_types_t MapDtypeToElementType(py::object dtype) {
    // TODO: Consider porting this from a py func to C++ as it can be on
    // the critical path.
    try {
      py::object element_type =
          array_interop_module().attr(kMapDtypeToElementTypeAttr)(dtype);
      if (element_type.is_none()) {
        throw std::invalid_argument("mapping not found");
      }
      return py::cast<enum iree_hal_element_types_t>(element_type);
    } catch (std::exception &e) {
      std::string msg("could not map dtype ");
      msg.append(py::cast<std::string>(py::repr(dtype)));
      msg.append(" to element type: ");
      msg.append(e.what());
      throw std::invalid_argument(std::move(msg));
    }
  }

  PackCallback AbiTypeToPackCallback(py::handle desc) {
    return AbiTypeToPackCallback(
        std::move(desc), /*desc_is_list=*/py::isinstance<py::list>(desc));
  }

  // Given an ABI desc, return a callback that can pack a corresponding py
  // value into a list. For efficiency, the caller must specify whether the
  // desc is a list (this check already needs to be done typically so
  // passed in).
  PackCallback AbiTypeToPackCallback(py::handle desc, bool desc_is_list) {
    // Switch based on descriptor type.
    if (desc_is_list) {
      // Compound type.
      py::object compound_type = desc[kZero];
      if (compound_type.equal(kNdarray)) {
        // Has format:
        //   ["ndarray", "f32", dim0, dim1, ...]
        // Extract static information about the target.
        std::vector<int64_t> abi_shape(py::len(desc) - 2);
        for (size_t i = 0, e = abi_shape.size(); i < e; ++i) {
          py::handle dim = desc[py::int_(i + 2)];
          abi_shape[i] = dim.is_none() ? -1 : py::cast<int64_t>(dim);
        }

        // Map abi element type to dtype.
        py::object abi_type = desc[kOne];
        py::object target_dtype = MapElementAbiTypeToDtype(abi_type);
        auto hal_element_type = MapDtypeToElementType(target_dtype);

        return [this, target_dtype = std::move(target_dtype), hal_element_type,
                abi_shape = std::move(abi_shape)](InvokeContext &c,
                                                  iree_vm_list_t *list,
                                                  py::handle py_value) {
          IREE_TRACE_SCOPE0("ArgumentPacker::ReflectionNdarray");
          HalBufferView *bv = nullptr;
          py::object retained_bv;
          if (py::isinstance(py_value, device_array_type())) {
            // Short-circuit: If a DeviceArray is provided, assume it is
            // correct.
            IREE_TRACE_SCOPE0("PackDeviceArray");
            bv = py::cast<HalBufferView *>(py_value.attr(kAttrBufferView));
          } else if (py::isinstance(py_value, hal_buffer_view_type())) {
            // Short-circuit: If a HalBufferView is provided directly.
            IREE_TRACE_SCOPE0("PackBufferView");
            bv = py::cast<HalBufferView *>(py_value);
          } else {
            // Fall back to the array protocol to generate a host side
            // array and then convert that.
            IREE_TRACE_SCOPE0("PackHostArray");
            py::object host_array;
            try {
              host_array = numpy_module().attr(kAsArray)(py_value, target_dtype,
                                                         kContiguousArg);
            } catch (std::exception &e) {
              std::string msg("could not convert value to numpy array: dtype=");
              msg.append(py::cast<std::string>(py::repr(target_dtype)));
              msg.append(", error='");
              msg.append(e.what());
              msg.append("', value=");
              msg.append(py::cast<std::string>(py::repr(py_value)));
              throw std::invalid_argument(std::move(msg));
            }

            retained_bv = c.allocator().AllocateBufferCopy(
                IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
                IREE_HAL_BUFFER_USAGE_DEFAULT | IREE_HAL_BUFFER_USAGE_MAPPING,
                host_array, hal_element_type);
            bv = py::cast<HalBufferView *>(retained_bv);
          }

          // TODO: Add some shape verification. Not strictly necessary as the VM
          // will check, but may make error reporting nicer.
          // TODO: It is theoretically possible to enqueue further conversions
          // on the device, but for now we require things to line up closely.
          // TODO: If adding further manipulation here, please make this common
          // with the generic access case.
          iree_vm_ref_t buffer_view_ref =
              iree_hal_buffer_view_retain_ref(bv->raw_ptr());
          CheckApiStatus(iree_vm_list_push_ref_move(list, &buffer_view_ref),
                         "could not push buffer view to list");
        };
      } else if (compound_type.equal(kSlistTag) ||
                 compound_type.equal(kStupleTag)) {
        // Tuple/list extraction.
        // When decoding a list or tuple, the desc object is like:
        // ['slist', [...value_type_0...], ...]
        // Where the type is either 'slist' or 'stuple'.
        std::vector<PackCallback> sub_packers(py::len(desc) - 1);
        for (size_t i = 0; i < sub_packers.size(); i++) {
          sub_packers[i] = AbiTypeToPackCallback(desc[py::int_(i + 1)]);
        }
        return [sub_packers = std::move(sub_packers)](InvokeContext &c,
                                                      iree_vm_list_t *list,
                                                      py::handle py_value) {
          if (py::len(py_value) != sub_packers.size()) {
            std::string msg("expected a sequence with ");
            msg.append(std::to_string(sub_packers.size()));
            msg.append(" values. got: ");
            msg.append(py::cast<std::string>(py::repr(py_value)));
            throw std::invalid_argument(std::move(msg));
          }
          VmVariantList item_list = VmVariantList::Create(sub_packers.size());
          for (size_t i = 0; i < sub_packers.size(); ++i) {
            py::object item_py_value;
            try {
              item_py_value = py_value[py::int_(i)];
            } catch (std::exception &e) {
              std::string msg("could not get item ");
              msg.append(std::to_string(i));
              msg.append(" from: ");
              msg.append(py::cast<std::string>(py::repr(py_value)));
              msg.append(": ");
              msg.append(e.what());
              throw std::invalid_argument(std::move(msg));
            }
            sub_packers[i](c, item_list.raw_ptr(), item_py_value);
          }

          // Push the sub list.
          iree_vm_ref_t retained =
              iree_vm_list_move_ref(item_list.steal_raw_ptr());
          iree_vm_list_push_ref_move(list, &retained);
        };
      } else if (compound_type.equal(kSdictTag)) {
        // Dict extraction.
        // The descriptor for an sdict is like:
        //   ['sdict', ['key1', value1], ...]
        std::vector<std::pair<py::object, PackCallback>> sub_packers(
            py::len(desc) - 1);
        for (size_t i = 0; i < sub_packers.size(); i++) {
          py::object sub_desc = desc[py::int_(i + 1)];
          py::object key = sub_desc[kZero];
          py::object value_desc = sub_desc[kOne];
          sub_packers[i] =
              std::make_pair(std::move(key), AbiTypeToPackCallback(value_desc));
        }
        return [sub_packers = std::move(sub_packers)](InvokeContext &c,
                                                      iree_vm_list_t *list,
                                                      py::handle py_value) {
          if (py::len(py_value) != sub_packers.size()) {
            std::string msg("expected a dict with ");
            msg.append(std::to_string(sub_packers.size()));
            msg.append(" values. got: ");
            msg.append(py::cast<std::string>(py::repr(py_value)));
            throw std::invalid_argument(std::move(msg));
          }
          VmVariantList item_list = VmVariantList::Create(sub_packers.size());
          for (size_t i = 0; i < sub_packers.size(); ++i) {
            py::object item_py_value;
            try {
              item_py_value = py_value[sub_packers[i].first];
            } catch (std::exception &e) {
              std::string msg("could not get item ");
              msg.append(py::cast<std::string>(py::repr(sub_packers[i].first)));
              msg.append(" from: ");
              msg.append(py::cast<std::string>(py::repr(py_value)));
              msg.append(": ");
              msg.append(e.what());
              throw std::invalid_argument(std::move(msg));
            }
            sub_packers[i].second(c, item_list.raw_ptr(), item_py_value);
          }

          // Push the sub list.
          iree_vm_ref_t retained =
              iree_vm_list_move_ref(item_list.steal_raw_ptr());
          iree_vm_list_push_ref_move(list, &retained);
        };
      } else {
        std::string message("Unrecognized reflection compound type: ");
        message.append(py::cast<std::string>(compound_type));
        throw std::invalid_argument(message);
      }
    } else {
      // Primtive type.
      py::str prim_type = py::cast<py::str>(desc);
      if (prim_type.equal(kF32)) {
        // f32
        return [](InvokeContext &c, iree_vm_list_t *list, py::handle py_value) {
          iree_vm_value_t vm_value =
              iree_vm_value_make_f32(py::cast<float>(py_value));
          CheckApiStatus(iree_vm_list_push_value(list, &vm_value),
                         "could not append value");
        };
      } else if (prim_type.equal(kF64)) {
        // f64
        return [](InvokeContext &c, iree_vm_list_t *list, py::handle py_value) {
          iree_vm_value_t vm_value =
              iree_vm_value_make_f64(py::cast<double>(py_value));
          CheckApiStatus(iree_vm_list_push_value(list, &vm_value),
                         "could not append value");
        };
      } else if (prim_type.equal(kI32)) {
        // i32.
        return [](InvokeContext &c, iree_vm_list_t *list, py::handle py_value) {
          iree_vm_value_t vm_value =
              iree_vm_value_make_i32(py::cast<int32_t>(py_value));
          CheckApiStatus(iree_vm_list_push_value(list, &vm_value),
                         "could not append value");
        };
      } else if (prim_type.equal(kI64)) {
        // i64.
        return [](InvokeContext &c, iree_vm_list_t *list, py::handle py_value) {
          iree_vm_value_t vm_value =
              iree_vm_value_make_i64(py::cast<int64_t>(py_value));
          CheckApiStatus(iree_vm_list_push_value(list, &vm_value),
                         "could not append value");
        };
      } else if (prim_type.equal(kI8)) {
        // i8.
        return [](InvokeContext &c, iree_vm_list_t *list, py::handle py_value) {
          iree_vm_value_t vm_value =
              iree_vm_value_make_i8(py::cast<int8_t>(py_value));
          CheckApiStatus(iree_vm_list_push_value(list, &vm_value),
                         "could not append value");
        };
      } else if (prim_type.equal(kI16)) {
        // i16.
        return [](InvokeContext &c, iree_vm_list_t *list, py::handle py_value) {
          iree_vm_value_t vm_value =
              iree_vm_value_make_i16(py::cast<int16_t>(py_value));
          CheckApiStatus(iree_vm_list_push_value(list, &vm_value),
                         "could not append value");
        };
      } else {
        std::string message("Unrecognized reflection primitive type: ");
        message.append(py::cast<std::string>(prim_type));
        throw std::invalid_argument(message);
      }
    }
  }

  PackCallback GetGenericPackCallbackFor(py::handle arg) {
    PopulatePyTypeToPackCallbacks();
    py::type clazz = py::type::of(arg);
    auto found_it = py_type_to_pack_callbacks_.find(clazz.ptr());
    if (found_it == py_type_to_pack_callbacks_.end()) {
      // Probe to see if we have a host array.
      if (py::hasattr(arg, kArrayProtocolAttr)) {
        return GetGenericPackCallbackForNdarray();
      }
      return {};
    }

    return found_it->second;
  }

 private:
  PackCallback GetGenericPackCallbackForNdarray() {
    return [this](InvokeContext &c, iree_vm_list_t *list, py::handle py_value) {
      IREE_TRACE_SCOPE0("ArgumentPacker::GenericNdarray");
      py::object host_array;
      try {
        host_array = numpy_module().attr(kAsArray)(
            py_value, /*dtype=*/py::none(), kContiguousArg);
      } catch (std::exception &e) {
        std::string msg("could not convert value to numpy array: ");
        msg.append("error='");
        msg.append(e.what());
        msg.append("', value=");
        msg.append(py::cast<std::string>(py::repr(py_value)));
        throw std::invalid_argument(std::move(msg));
      }

      auto hal_element_type =
          MapDtypeToElementType(host_array.attr(kDtypeAttr));

      // Put it on the device.
      py::object retained_bv = c.allocator().AllocateBufferCopy(
          IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
          IREE_HAL_BUFFER_USAGE_DEFAULT | IREE_HAL_BUFFER_USAGE_MAPPING,
          host_array, hal_element_type);
      HalBufferView *bv = py::cast<HalBufferView *>(retained_bv);

      // TODO: If adding further manipulation here, please make this common
      // with the reflection access case.
      iree_vm_ref_t buffer_view_ref =
          iree_hal_buffer_view_retain_ref(bv->raw_ptr());
      CheckApiStatus(iree_vm_list_push_ref_move(list, &buffer_view_ref),
                     "could not append value");
    };
  }

  void PopulatePyTypeToPackCallbacks() {
    if (!py_type_to_pack_callbacks_.empty()) return;

    // We only care about int and double in the numeric hierarchy. Since Python
    // has no further refinement of these, just treat them as vm 64 bit int and
    // floats and let the VM take care of it. There isn't much else we can do.
    AddPackCallback(
        py::type::of(py::cast(1)),
        [](InvokeContext &c, iree_vm_list_t *list, py::handle py_value) {
          iree_vm_value_t vm_value =
              iree_vm_value_make_i64(py::cast<int64_t>(py_value));
          CheckApiStatus(iree_vm_list_push_value(list, &vm_value),
                         "could not append value");
        });

    AddPackCallback(
        py::type::of(py::cast(1.0)),
        [](InvokeContext &c, iree_vm_list_t *list, py::handle py_value) {
          iree_vm_value_t vm_value =
              iree_vm_value_make_f64(py::cast<double>(py_value));
          CheckApiStatus(iree_vm_list_push_value(list, &vm_value),
                         "could not append value");
        });

    // List/tuple.
    auto sequence_callback = [this](InvokeContext &c, iree_vm_list_t *list,
                                    py::handle py_value) {
      auto py_seq = py::cast<py::sequence>(py_value);
      VmVariantList item_list = VmVariantList::Create(py::len(py_seq));
      for (py::object py_item : py_seq) {
        PackCallback sub_packer = GetGenericPackCallbackFor(py_item);
        if (!sub_packer) {
          std::string message("could not convert python value to VM: ");
          message.append(py::cast<std::string>(py::repr(py_item)));
          throw std::invalid_argument(std::move(message));
        }
        sub_packer(c, item_list.raw_ptr(), py_item);
      }
      // Push the sub list.
      iree_vm_ref_t retained = iree_vm_list_move_ref(item_list.steal_raw_ptr());
      iree_vm_list_push_ref_move(list, &retained);
    };
    AddPackCallback(py::type::of(py::list{}), sequence_callback);
    AddPackCallback(py::type::of(py::tuple{}), sequence_callback);

    // Dict.
    auto dict_callback = [this](InvokeContext &c, iree_vm_list_t *list,
                                py::handle py_value) {
      // Gets all dict items and sorts (by key).
      auto py_dict = py::cast<py::dict>(py_value);
      py::list py_keys;
      for (std::pair<py::handle, py::handle> it : py_dict) {
        py_keys.append(it.first);
      }
      py_keys.attr("sort")();

      VmVariantList item_list = VmVariantList::Create(py_keys.size());
      for (auto py_key : py_keys) {
        py::object py_item = py_dict[py_key];
        PackCallback sub_packer = GetGenericPackCallbackFor(py_item);
        if (!sub_packer) {
          std::string message("could not convert python value to VM: ");
          message.append(py::cast<std::string>(py::repr(py_item)));
          throw std::invalid_argument(std::move(message));
        }
        sub_packer(c, item_list.raw_ptr(), py_item);
      }
      // Push the sub list.
      iree_vm_ref_t retained = iree_vm_list_move_ref(item_list.steal_raw_ptr());
      iree_vm_list_push_ref_move(list, &retained);
    };
    AddPackCallback(py::type::of(py::dict{}), dict_callback);

    // HalBufferView.
    AddPackCallback(
        py::type::of<HalBufferView>(),
        [](InvokeContext &c, iree_vm_list_t *list, py::handle py_value) {
          HalBufferView *bv = py::cast<HalBufferView *>(py_value);
          iree_vm_ref_t buffer_view_ref =
              iree_hal_buffer_view_retain_ref(bv->raw_ptr());
          CheckApiStatus(iree_vm_list_push_ref_move(list, &buffer_view_ref),
                         "could not append value");
        });

    // DeviceArray.
    AddPackCallback(
        device_array_type(),
        [this](InvokeContext &c, iree_vm_list_t *list, py::handle py_value) {
          HalBufferView *bv =
              py::cast<HalBufferView *>(py_value.attr(kAttrBufferView));
          iree_vm_ref_t buffer_view_ref =
              iree_hal_buffer_view_retain_ref(bv->raw_ptr());
          CheckApiStatus(iree_vm_list_push_ref_move(list, &buffer_view_ref),
                         "could not append value");
        });
  }

  void AddPackCallback(py::handle t, PackCallback pcb) {
    assert(py_type_to_pack_callbacks_.count(t.ptr()) == 0 && "duplicate types");
    t.inc_ref();
    py_type_to_pack_callbacks_.insert(std::make_pair(t.ptr(), std::move(pcb)));
  }

  py::dict BuildAbiTypeToDtype() {
    auto d = py::dict();
    d[kF32] = numpy_module().attr("float32");
    d[kF64] = numpy_module().attr("float64");
    d[kI1] = numpy_module().attr("bool_");
    d[kI8] = numpy_module().attr("int8");
    d[kI16] = numpy_module().attr("int16");
    d[kI64] = numpy_module().attr("int64");
    d[kI32] = numpy_module().attr("int32");
    return d;
  }

  // Cached modules and types. Those that involve recursive lookup within
  // our top level module, we defer. Those outside, we cache at creation.
  py::module numpy_module_ = py::module::import("numpy");
  std::optional<py::object> runtime_module_;
  std::optional<py::module> array_interop_module_;
  std::optional<py::object> device_array_type_;
  py::type hal_buffer_view_type_ = py::type::of<HalBufferView>();

  // Maps Python type to a PackCallback that can generically code it.
  // This will have inc_ref() called on them when added.
  std::unordered_map<PyObject *, PackCallback> py_type_to_pack_callbacks_;

  // Dict of str (ABI dtype like 'f32') to numpy dtype.
  py::dict abi_type_to_dtype_ = BuildAbiTypeToDtype();
};

/// Object that can pack Python arguments into a VM List for a specific
/// function.
class ArgumentPacker {
 public:
  ArgumentPacker(InvokeStatics &statics, std::optional<py::list> arg_descs)
      : statics_(statics) {
    IREE_TRACE_SCOPE0("ArgumentPacker::Init");
    if (!arg_descs) {
      dynamic_dispatch_ = true;
    } else {
      // Reflection dispatch.
      for (py::handle desc : *arg_descs) {
        int arg_index = flat_arg_packers_.size();
        std::optional<std::string> kwarg_name;
        py::object retained_sub_desc;

        bool desc_is_list = py::isinstance<py::list>(desc);

        // Check if named.
        //   ["named", "kwarg_name", sub_desc]
        // If found, then we set kwarg_name and reset desc to the sub_desc.
        if (desc_is_list) {
          py::object maybe_named_field = desc[statics.kZero];
          if (maybe_named_field.equal(statics.kNamedTag)) {
            py::object name_field = desc[statics.kOne];
            retained_sub_desc = desc[statics.kTwo];
            kwarg_name = py::cast<std::string>(name_field);
            desc = retained_sub_desc;
            desc_is_list = py::isinstance<py::list>(desc);

            kwarg_to_index_[name_field] = arg_index;
          }
        }

        if (!kwarg_name) {
          pos_only_arg_count_ += 1;
        }

        flat_arg_packers_.push_back(
            statics.AbiTypeToPackCallback(desc, desc_is_list));
      }
    }
  }

  /// Packs positional/kw arguments into a suitable VmVariantList and returns
  /// it.
  VmVariantList Pack(InvokeContext &invoke_context, py::sequence pos_args,
                     py::dict kw_args) {
    // Dynamic dispatch.
    if (dynamic_dispatch_) {
      IREE_TRACE_SCOPE0("ArgumentPacker::PackDynamic");
      if (!kw_args.empty()) {
        throw std::invalid_argument(
            "kwargs not supported for dynamic dispatch functions");
      }

      VmVariantList arg_list = VmVariantList::Create(pos_args.size());
      for (py::handle py_arg : pos_args) {
        PackCallback packer = statics_.GetGenericPackCallbackFor(py_arg);
        if (!packer) {
          std::string message("could not convert python value to VM: ");
          message.append(py::cast<std::string>(py::repr(py_arg)));
          throw std::invalid_argument(std::move(message));
        }
        // TODO: Better error handling by catching the exception and
        // reporting which arg has a problem.
        packer(invoke_context, arg_list.raw_ptr(), py_arg);
      }
      return arg_list;
    } else {
      IREE_TRACE_SCOPE0("ArgumentPacker::PackReflection");

      // Reflection based dispatch.
      std::vector<py::handle> py_args(flat_arg_packers_.size());

      if (pos_args.size() > pos_only_arg_count_) {
        std::string message("mismatched call arity: expected ");
        message.append(std::to_string(pos_only_arg_count_));
        message.append(" got ");
        message.append(std::to_string(pos_args.size()));
        throw std::invalid_argument(std::move(message));
      }

      // Positional args.
      size_t pos_index = 0;
      for (py::handle py_arg : pos_args) {
        py_args[pos_index++] = py_arg;
      }

      // Keyword args.
      for (auto it : kw_args) {
        int found_index;
        try {
          found_index = py::cast<int>(kwarg_to_index_[it.first]);
        } catch (std::exception &) {
          std::string message("specified kwarg '");
          message.append(py::cast<py::str>(it.first));
          message.append("' is unknown");
          throw std::invalid_argument(std::move(message));
        }
        if (py_args[found_index]) {
          std::string message(
              "mismatched call arity: duplicate keyword argument '");
          message.append(py::cast<py::str>(it.first));
          message.append("'");
          throw std::invalid_argument(std::move(message));
        }
        py_args[found_index] = it.second;
      }

      // Now check to see that all args are set.
      for (size_t i = 0; i < py_args.size(); ++i) {
        if (!py_args[i]) {
          std::string message(
              "mismatched call arity: expected a value for argument ");
          message.append(std::to_string(i));
          throw std::invalid_argument(std::move(message));
        }
      }

      // Start packing into the list.
      VmVariantList arg_list = VmVariantList::Create(flat_arg_packers_.size());
      for (size_t i = 0; i < py_args.size(); ++i) {
        // TODO: Better error handling by catching the exception and
        // reporting which arg has a problem.
        flat_arg_packers_[i](invoke_context, arg_list.raw_ptr(), py_args[i]);
      }
      return arg_list;
    }
  }

 private:
  InvokeStatics &statics_;

  int pos_only_arg_count_ = 0;

  // Dictionary of py::str -> py::int_ mapping kwarg names to position in
  // the argument list. We store this as a py::dict because it is optimized
  // for py::str lookup.
  py::dict kwarg_to_index_;

  std::vector<PackCallback> flat_arg_packers_;

  // If true, then there is no dispatch metadata and we process fully
  // dynamically.
  bool dynamic_dispatch_ = false;
};

}  // namespace

void SetupInvokeBindings(pybind11::module &m) {
  py::class_<InvokeStatics>(m, "_InvokeStatics");
  py::class_<InvokeContext>(m, "InvokeContext").def(py::init<HalDevice &>());
  py::class_<ArgumentPacker>(m, "ArgumentPacker")
      .def(py::init<InvokeStatics &, std::optional<py::list>>())
      .def("pack", &ArgumentPacker::Pack);

  m.attr("_invoke_statics") = py::cast(InvokeStatics());
}

}  // namespace python
}  // namespace iree
