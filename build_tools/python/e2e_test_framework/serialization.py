## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Helpers to serialize/deserialize objects."""

from enum import Enum
from typing import Any, Dict, Optional, Sequence, Tuple, Type, TypeVar, Union
import dataclasses
import typing

# types.NoneType is only added after Python 3.10.
NONE_TYPE = type(None)
SERIALIZE_FUNC_NAME = "__serialize__"
DESERIALIZE_FUNC_NAME = "__deserialize__"
SUPPORTED_DICT_KEY_TYPES = {str, int, float, bool}
SUPPORTED_PRIMITIVE_TYPES = {str, int, float, bool, NONE_TYPE}


def serialize_and_pack(obj,
                       root_obj_field_name="root_obj",
                       keyed_obj_map_field_name="keyed_obj_map"):
  """Converts and packs the object into a serializable object.
  
  Args:
    obj: object to be serialized.
    root_obj_field_name: field name of the top-level object in the return dict.
    keyed_obj_map_field_name: field name of the keyed object map in the return
      dict.
  Returns
    A serializable dict.
  """

  if root_obj_field_name == keyed_obj_map_field_name:
    raise ValueError(
        f"root_obj and keyed_obj_map can't have the same field name.")

  keyed_obj_map = {}
  root_obj = _serialize(obj=obj, keyed_obj_map=keyed_obj_map)
  return {
      root_obj_field_name: root_obj,
      keyed_obj_map_field_name: keyed_obj_map
  }


T = TypeVar('T')


def unpack_and_deserialize(data,
                           root_type: Type[T],
                           root_obj_field_name="root_obj",
                           keyed_obj_map_field_name="keyed_obj_map") -> T:
  """Unpacks and deserializes the data back to the typed object.

  Args:
    data: serialized data dict.
    root_type: top-level object type of the data.
    root_obj_field_name: field name of the top-level object in the dict.
    keyed_obj_map_field_name: field name of the keyed object map in the dict.
  Returns:
    A deserialized object.
  """
  obj = _deserialize(data=data[root_obj_field_name],
                     obj_type=root_type,
                     keyed_obj_map=data[keyed_obj_map_field_name])
  return typing.cast(root_type, obj)


def _serialize(obj, keyed_obj_map: Dict[str, Any]):
  """Converts the object into a serializable object.
  
  Args:
    obj: object to be serialized.
    keyed_obj_map: mutable container to store the keyed serializable object.
  Returns
    A serializable object.
  """

  serialize_func = getattr(obj, SERIALIZE_FUNC_NAME, None)
  if serialize_func is not None:
    return serialize_func(keyed_obj_map)

  elif isinstance(obj, list):
    return [_serialize(value, keyed_obj_map) for value in obj]

  elif isinstance(obj, Enum):
    return obj.name

  elif isinstance(obj, dict):
    result_dict = {}
    for key, value in obj.items():
      if type(key) not in SUPPORTED_DICT_KEY_TYPES:
        raise ValueError(f"Unsupported key {key} in the dict {obj}.")
      result_dict[key] = _serialize(value, keyed_obj_map)
    return result_dict

  elif type(obj) in SUPPORTED_PRIMITIVE_TYPES:
    return obj

  raise ValueError(f"Unsupported object: {obj}.")


def _deserialize(data,
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
        _deserialize(item, subtype, keyed_obj_map, obj_cache) for item in data
    ]

  elif typing.get_origin(obj_type) == dict:
    _, value_type = typing.get_args(obj_type)
    return dict((key, _deserialize(value, value_type, keyed_obj_map, obj_cache))
                for key, value in data.items())

  elif typing.get_origin(obj_type) == Union:
    subtypes = typing.get_args(obj_type)
    if len(subtypes) != 2 or NONE_TYPE not in subtypes:
      raise ValueError(f"Unsupported union type: {obj_type}.")
    subtype = subtypes[0] if subtypes[1] == NONE_TYPE else subtypes[1]
    return _deserialize(data, subtype, keyed_obj_map, obj_cache)

  elif issubclass(obj_type, Enum):
    for member in obj_type:
      if data == member.name:
        return member
    raise ValueError(f"Member {data} not found in the enum {obj_type}.")

  return data


def serializable(cls=None,
                 type_key: Optional[str] = None,
                 id_field: str = "id"):
  """Decorator to make a dataclass serializable.
  
  Args:
    type_key: string defines the object type and indeicates that the class is a
      keyed object, which is unique per id and will only have one copy in the
      serialization per id.
    id_field: field name of the id field of a keyed object.

  Example:
    @serializable
    @dataclass
    class A(object):
      ...

    @serialzable(type_key="obj_b")
    @dataclass
    class B(object):
      id: str
  """

  if type_key is not None and ":" in type_key:
    raise ValueError("':' is the reserved character in type_key.")

  def wrap(cls):
    if not dataclasses.is_dataclass(cls):
      raise ValueError(f"{cls} is not a dataclass.")

    fields = dataclasses.fields(cls)
    if type_key is not None and all(field.name != id_field for field in fields):
      raise ValueError(f'Id field "{id_field}" not found in the class {cls}.')

    def serialize(self, keyed_obj_map: Dict[str, Any]):
      if type_key is None:
        return _fields_to_dict(self, fields, keyed_obj_map)

      obj_id = getattr(self, id_field)
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
      if type_key is None:
        field_value_map = _dict_to_fields(data, fields, keyed_obj_map,
                                          obj_cache)
        return cls(**field_value_map)

      obj_id = data
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

  # Trick to allow the decoration with `@serializable(...)`. In that case,
  # `serializable` is called without cls and should return a decorator.
  if cls is None:
    return wrap
  return wrap(cls)


def _fields_to_dict(obj, fields: Sequence[dataclasses.Field],
                    keyed_obj_map: Dict[str, Any]) -> Dict[str, Any]:
  return dict((field.name, _serialize(getattr(obj, field.name), keyed_obj_map))
              for field in fields)


def _dict_to_fields(obj_dict, fields: Sequence[dataclasses.Field],
                    keyed_obj_map: Dict[str, Any],
                    obj_cache: Dict[str, Any]) -> Dict[str, Any]:
  return dict(
      (field.name,
       _deserialize(obj_dict[field.name], field.type, keyed_obj_map, obj_cache))
      for field in fields)
