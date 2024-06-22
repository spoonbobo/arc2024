from arc_types import *

from typing import Any, get_origin, get_args, List, Tuple, Set, FrozenSet, Union, get_type_hints, Dict

def typeguard(inp: Any, return_type: type):
    origin = get_origin(return_type)
    args = get_args(return_type)
    
    if origin is list:
        if isinstance(inp, list):
            inner_type = args[0]
            return tuple(typeguard(i, inner_type) for i in inp)
    elif origin is tuple:
        if isinstance(inp, (list, tuple)):
            return tuple(typeguard(i, args[0]) for i in inp)
    elif origin is set:
        if isinstance(inp, set):
            inner_type = args[0]
            return frozenset(typeguard(i, inner_type) for i in inp)
    elif origin is frozenset:
        if isinstance(inp, set):
            inner_type = args[0]
            return frozenset(typeguard(i, inner_type) for i in inp)
    elif origin is Union:
        for arg in args:
            try:
                return typeguard(inp, arg)
            except:
                continue
    elif return_type == Grid:
        if isinstance(inp, (list, tuple)):
            return tuple(tuple(i) if isinstance(i, list) else i for i in inp)
    elif return_type == IntegerList:
        if isinstance(inp, (list, tuple)):
            return tuple(inp)
    elif return_type == IntegerSet:
        if isinstance(inp, set):
            return frozenset(inp)
    elif return_type == Cell:
        if isinstance(inp, (list, tuple)):
            return tuple(inp)
    elif return_type == Object:
        if isinstance(inp, set):
            return frozenset(inp)
    elif return_type == Objects:
        if isinstance(inp, set):
            return frozenset(inp)
    elif return_type == Indices:
        if isinstance(inp, set):
            return frozenset(inp)
    elif return_type == IndicesSet:
        if isinstance(inp, set):
            return frozenset(inp)
    elif return_type == Patch:
        if isinstance(inp, (set, list, tuple)):
            return frozenset(inp) if isinstance(inp, set) else tuple(inp)
    elif return_type == Element:
        if isinstance(inp, (set, list, tuple)):
            return frozenset(inp) if isinstance(inp, set) else tuple(inp)
    elif return_type == Piece:
        if isinstance(inp, (set, list, tuple)):
            return frozenset(inp) if isinstance(inp, set) else tuple(inp)
    elif return_type == ListList:
        if isinstance(inp, (list, tuple)):
            return tuple(tuple(i) if isinstance(i, list) else i for i in inp)
    elif return_type == ContainerContainer:
        if isinstance(inp, Container):
            return inp
    return inp
