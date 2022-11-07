## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Helpers to serialize/deserialize objects."""

from enum import Enum
from typing import Any, Dict, Optional, OrderedDict, Sequence, Type
import collections
import dataclasses
import typing

SERIALIZE_FUNC_NAME = "__serialize__"
DESERIALIZE_FUNC_NAME = "__deserialize__"


def serialize_and_pack(obj,
                       root_obj_field_name="root_obj",
                       keyed_obj_map_field_name="keyed_obj_map"):
  if root_obj_field_name == keyed_obj_map_field_name:
    raise ValueError(
        f"root_obj and keyed_obj_map can't have the same field name")

  keyed_obj_map = collections.OrderedDict()
  root_obj = serialize(obj=obj, keyed_obj_map=keyed_obj_map)
  return {
      root_obj_field_name: root_obj,
      keyed_obj_map_field_name: keyed_obj_map
  }


def unpack_and_deserialize(data,
                           root_type: Type,
                           root_obj_field_name="root_obj",
                           keyed_obj_map_field_name="keyed_obj_map"):
  return deserialize(data=data[root_obj_field_name],
                     obj_type=root_type,
                     keyed_obj_map=data[keyed_obj_map_field_name])


def serialize(obj, keyed_obj_map: OrderedDict[str, Any]):
  """Converts the object into a serializable object.
  
  Args:
    obj: python object to be serialized.
    keyed_obj_map: mutable container to store the keyed serializable object.
  Returns
    A serializable object.
  """

  serialize_func = getattr(obj, SERIALIZE_FUNC_NAME, None)
  if serialize_func is not None:
    return serialize_func(keyed_obj_map)
  elif isinstance(obj, list):
    return [serialize(value, keyed_obj_map) for value in obj]
  elif isinstance(obj, Enum):
    return obj.name
  return obj


def deserialize(data,
                obj_type: Type,
                keyed_obj_map: Dict[str, Any],
                obj_cache: Dict[str, Any] = {}):
  """Deserializes the data back to the typed object.

  Args:
    data: serialized data.
    obj_type: type of the data.
    keyed_obj_map: container of the keyed serializable object.
  Returns:
    A deserialized object.
  """
  deserialize_func = getattr(obj_type, DESERIALIZE_FUNC_NAME, None)
  if deserialize_func is not None:
    return deserialize_func(data, keyed_obj_map, obj_cache)
  elif typing.get_origin(obj_type) == list:
    subtype, = typing.get_args(obj_type)
    return [
        deserialize(item, subtype, keyed_obj_map, obj_cache) for item in data
    ]
  elif issubclass(obj_type, Enum):
    for member in obj_type:
      if data == member.name:
        return member
    raise ValueError(f"Member {data} not found in the enum {obj_type}.")
  return data


def serializable(cls=None,
                 keyed_obj: bool = False,
                 type_key: Optional[str] = None,
                 id_field: str = "id"):
  """Decorator to make a dataclass serializable.
  
  Args:
    keyed_obj: is the class a keyed object, which is unique per id and will only
      have one copy in the serialization per id.
    type_key: string defining the object type, must be set if keyed_obj is True.
    id_field: field name of the id field of a keyed object.

  Example:
    @serializable
    @dataclass
    class A(object):
      ...

    @serialzable(keyed_obj=True, type_key="obj_b")
    @dataclass
    class B(object):
      id: str
  """

  if keyed_obj and type_key is None:
    raise ValueError("type_key must be set if keyed_by_id is true.")
  if type_key is not None and ":" in type_key:
    raise ValueError("':' is the reserved character in type_key.")

  def wrap(cls):
    if not dataclasses.is_dataclass(cls):
      raise ValueError(f"{cls} is not a dataclass.")

    fields = dataclasses.fields(cls)

    def serialize(self, keyed_obj_map: OrderedDict[str, Any]):
      if not keyed_obj:
        return _fields_to_dict(self, fields, keyed_obj_map)

      obj_id = getattr(self, id_field)
      # type_key has been checked to be not None above.
      assert type_key is not None
      obj_key = f"{type_key}:{obj_id}"
      if obj_key in keyed_obj_map:
        # If the value in the map is None, it means we have visited this object
        # before but not yet finished serializing it. This will only happen if
        # there is a circular reference.
        if keyed_obj_map[obj_key] is None:
          raise ValueError(f"Circular reference is not supported: {obj_key}.")
        return obj_id

      # Populate the keyed_obj_map with None first to detect circular reference.
      keyed_obj_map[obj_key] = None
      obj_dict = _fields_to_dict(self, fields, keyed_obj_map)
      keyed_obj_map[obj_key] = obj_dict
      return obj_id

    def deserialize(data, keyed_obj_map: Dict[str, Any], obj_cache: Dict[str,
                                                                         Any]):
      if not keyed_obj:
        field_value_map = _dict_to_fields(data, fields, keyed_obj_map,
                                          obj_cache)
        return cls(**field_value_map)

      obj_id = data
      # type_key has been checked to be not None above.
      assert type_key is not None
      obj_key = f"{type_key}:{obj_id}"
      if obj_key in obj_cache:
        return obj_cache[obj_key]

      field_value_map = _dict_to_fields(keyed_obj_map[obj_key], fields,
                                        keyed_obj_map, obj_cache)
      derialized_obj = cls(**field_value_map)
      obj_cache[obj_key] = derialized_obj
      return derialized_obj

    setattr(cls, SERIALIZE_FUNC_NAME, serialize)
    setattr(cls, DESERIALIZE_FUNC_NAME, deserialize)
    return cls

  if cls is None:
    return wrap
  return wrap(cls)


def _fields_to_dict(
    obj, fields: Sequence[dataclasses.Field],
    keyed_obj_map: OrderedDict[str, Any]) -> OrderedDict[str, Any]:
  return collections.OrderedDict(
      (field.name, serialize(getattr(obj, field.name), keyed_obj_map))
      for field in fields)


def _dict_to_fields(obj_dict, fields: Sequence[dataclasses.Field],
                    keyed_obj_map: Dict[str, Any],
                    obj_cache: Dict[str, Any]) -> Dict[str, Any]:
  return dict(
      (field.name,
       deserialize(obj_dict[field.name], field.type, keyed_obj_map, obj_cache))
      for field in fields)
