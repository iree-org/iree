from enum import Enum
from typing import Any, Dict, Optional, OrderedDict, Sequence, Set, Tuple
import collections
import dataclasses
import typing

SERIALIZE_FUNC_NAME = "__serialize__"
DESERIALIZE_FUNC_NAME = "__deserialize__"


def serialize(obj, serialized_obj_map: OrderedDict[Tuple[str, str], Any]):
  serialize_func = getattr(obj, SERIALIZE_FUNC_NAME, None)
  if serialize_func is not None:
    return serialize_func(serialized_obj_map)
  elif isinstance(obj, list):
    return [serialize(value, serialized_obj_map) for value in obj]
  elif isinstance(obj, Enum):
    return obj.name
  return obj


def deserialize(data,
                obj_type,
                serialized_obj_map: Dict[Tuple[str, str], Any],
                deserialized_obj_map: Dict[Tuple[str, str], Any] = {}):
  deserialize_func = getattr(obj_type, DESERIALIZE_FUNC_NAME, None)
  if deserialize_func is not None:
    return deserialize_func(data, serialized_obj_map, deserialized_obj_map)
  elif typing.get_origin(obj_type) == list:
    subtype, = typing.get_args(obj_type)
    return [
        deserialize(item, subtype, serialized_obj_map, deserialized_obj_map)
        for item in data
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
  if keyed_obj and type_key is None:
    raise ValueError("type_key must be set if keyed_by_id is true.")

  def wrap(cls):
    fields = dataclasses.fields(cls)

    def serialize(self, serialized_obj_map: OrderedDict[Tuple[str, str], Any]):
      if not keyed_obj:
        return _fields_to_dict(self, fields, serialized_obj_map)

      obj_id = getattr(self, id_field)
      # type_key has been checked to be not None above.
      assert type_key is not None
      obj_key = (type_key, obj_id)
      if obj_key in serialized_obj_map:
        # If the value in the map is None, it means we have visited this object
        # before but not yet finished serializing it. This will only happen if
        # there is a circular reference.
        if serialized_obj_map[obj_key] is None:
          raise ValueError(f"Circular reference is not supported: {obj_key}.")
        return obj_id

      # Populate the serialized_obj_map with None first to detect circular
      # reference.
      serialized_obj_map[obj_key] = None
      obj_dict = _fields_to_dict(self, fields, serialized_obj_map)
      serialized_obj_map[obj_key] = obj_dict
      return obj_id

    def deserialize(data, serialized_obj_map: Dict[Tuple[str, str], Any],
                    deserialized_obj_map: Dict[Tuple[str, str], Any]):
      if not keyed_obj:
        field_value_map = _dict_to_fields(data, fields, serialized_obj_map,
                                          deserialized_obj_map)
        return cls(**field_value_map)

      # type_key has been checked to be not None above.
      assert type_key is not None
      obj_key = (type_key, data)
      if obj_key in deserialized_obj_map:
        return deserialized_obj_map[obj_key]

      field_value_map = _dict_to_fields(serialized_obj_map[obj_key], fields,
                                        serialized_obj_map,
                                        deserialized_obj_map)
      derialized_obj = cls(**field_value_map)
      deserialized_obj_map[obj_key] = derialized_obj
      return derialized_obj

    setattr(cls, SERIALIZE_FUNC_NAME, serialize)
    setattr(cls, DESERIALIZE_FUNC_NAME, deserialize)
    return cls

  if cls is None:
    return wrap
  return wrap(cls)


def _fields_to_dict(
    obj, fields: Sequence[dataclasses.Field],
    serialized_obj_map: OrderedDict[Tuple[str, str],
                                    Any]) -> OrderedDict[str, Any]:
  return collections.OrderedDict(
      (field.name, serialize(getattr(obj, field.name), serialized_obj_map))
      for field in fields)


def _dict_to_fields(
    obj_dict, fields: Sequence[dataclasses.Field],
    serialized_obj_map: Dict[Tuple[str, str], Any],
    deserialized_obj_map: Dict[Tuple[str, str], Any]) -> Dict[str, Any]:
  return dict((field.name,
               deserialize(obj_dict[field.name], field.type, serialized_obj_map,
                           deserialized_obj_map)) for field in fields)
