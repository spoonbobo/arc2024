from typing import (
    Tuple,
    Union,
    Any,
    Container,
    Callable,
    FrozenSet,
    Iterable,
    TypeVar
)

import typing

Boolean = bool
Integer = int
IntegerList = Tuple[Integer, ...]
Numerical = Union[Integer, IntegerList]
IntegerSet = FrozenSet[Integer]
Grid = Tuple[Tuple[Integer, ...], ...]
Cell = Tuple[Union[Integer, IntegerList], ...]
Object = FrozenSet[Cell]
Objects = FrozenSet[Object]
Indices = FrozenSet[IntegerList]
IndicesSet = FrozenSet[Indices]
Patch = Union[Object, Indices]
Element = Union[Object, Grid]
Piece = Union[Grid, Patch]
ListList = Tuple[Tuple[Any, ...], ...]
ContainerContainer = Container[Container]

param_mapping = {
    'grid': Grid,
    'cell': Cell,
    'object': Object,
    'objects': Objects,
    'indices': Indices,
    'indices_set': IndicesSet,
    'patch': Patch,
    'element': Element,
    'piece': Piece,
}