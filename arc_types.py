from typing import (
    Tuple,
    Union,
    Any,
    List,
    Container,
    Callable,
    FrozenSet,
    Iterable,
    TypeVar
)

import typing

Boolean = bool
Integer = int
IntegerTuple = Tuple[Integer, Integer]
Numerical = Union[Integer, IntegerTuple]
IntegerSet = FrozenSet[Integer]
Grid = Tuple[Tuple[Integer]]
Cell = Tuple[Integer, IntegerTuple]
Object = FrozenSet[Cell]
Objects = FrozenSet[Object]
Indices = FrozenSet[IntegerTuple]
IndicesSet = FrozenSet[Indices]
Patch = Union[Object, Indices]
Element = Union[Object, Grid]
Piece = Union[Grid, Patch]
TupleTuple = Tuple[Tuple]
ContainerContainer = Container[Container]


param_mapping = {
    'boolean': Boolean,
    'integer': Integer,
    'integer_tuple': IntegerTuple,
    'numerical': Numerical,
    'integer_set': IntegerSet,
    'grid': Grid,
    'cell': Cell,
    'object': Object,
    'objects': Objects,
    'indices': Indices,
    'indices_set': IndicesSet,
    'patch': Patch,
    'element': Element,
    'piece': Piece,
    'tuple_tuple': TupleTuple,
    'container_container': ContainerContainer,
}