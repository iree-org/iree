from enum import Enum
from typing import Any, Optional, OrderedDict
import collections
import dataclasses
import json

SERIALIZE_FUNC_NAME = "__serialize__"


class CustomEncoder(json.JSONEncoder):

  def __init__(self, container_map, **kwargs):
    self._container_map = container_map
    super().__init__(**kwargs)

  def default(self, obj):
    serialize_func = getattr(obj, SERIALIZE_FUNC_NAME, None)
    if serialize_func:
      return serialize_func(self._container_map)
    elif isinstance(obj, Enum):
      return obj.name
    else:
      return json.JSONEncoder.default(self, obj)


def serializable(cls=None,
                 id_field: Optional[str] = None,
                 container_key: Optional[str] = None):

  if container_key is not None and id_field is None:
    raise ValueError("container_key requires id_field to be set.")

  def wrap(cls):
    field_names = [field.name for field in dataclasses.fields(cls)]

    def serialize(self, container_map: OrderedDict[str, OrderedDict[str, Any]]):
      if container_key is not None and id_field is not None:
        if container_key not in container_map:
          container_map[container_key] = collections.OrderedDict()

        obj_id = getattr(self, id_field)
        if obj_id in container_map[container_key]:
          return obj_id

        dump_dict = collections.OrderedDict()
        for field_name in field_names:
          dump_dict[field_name] = getattr(self, field_name)

        container_map[container_key][obj_id] = None
        result = json.dumps(dump_dict,
                            cls=CustomEncoder,
                            container_map=container_map)
        container_map[container_key][obj_id] = result

        return obj_id

      dump_dict = collections.OrderedDict()
      for field_name in field_names:
        dump_dict[field_name] = getattr(self, field_name)
      return dump_dict

    setattr(cls, SERIALIZE_FUNC_NAME, serialize)
    return cls

  if cls is None:
    return wrap
  return wrap(cls)
