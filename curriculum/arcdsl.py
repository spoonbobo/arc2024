from typing import (
    Any,
    Tuple,
    FrozenSet,
    Callable,
    Container,
    Union
)

def identity(
    x: Any
) -> Any:
    """ identity function """
    return x


def add(
    a: Union[int, Tuple[int, int]],
    b: Union[int, Tuple[int, int]]
) -> Union[int, Tuple[int, int]]:
    """ addition """
    if isinstance(a, int) and isinstance(b, int):
        return a + b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] + b[0], a[1] + b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a + b[0], a + b[1])
    return (a[0] + b, a[1] + b)


def subtract(
    a: Union[int, Tuple[int, int]],
    b: Union[int, Tuple[int, int]]
) -> Union[int, Tuple[int, int]]:
    """ subtraction """
    if isinstance(a, int) and isinstance(b, int):
        return a - b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] - b[0], a[1] - b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a - b[0], a - b[1])
    return (a[0] - b, a[1] - b)


def multiply(
    a: Union[int, Tuple[int, int]],
    b: Union[int, Tuple[int, int]]
) -> Union[int, Tuple[int, int]]:
    """ multiplication """
    if isinstance(a, int) and isinstance(b, int):
        return a * b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] * b[0], a[1] * b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a * b[0], a * b[1])
    return (a[0] * b, a[1] * b)
    

def divide(
    a: Union[int, Tuple[int, int]],
    b: Union[int, Tuple[int, int]]
) -> Union[int, Tuple[int, int]]:
    """ floor division """
    if isinstance(a, int) and isinstance(b, int):
        return a // b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] // b[0], a[1] // b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a // b[0], a // b[1])
    return (a[0] // b, a[1] // b)


def invert(
    n: Union[int, Tuple[int, int]]
) -> Union[int, Tuple[int, int]]:
    """ inversion with respect to addition """
    return -n if isinstance(n, int) else (-n[0], -n[1])


def even(
    n: int
) -> bool:
    """ evenness """
    return n % 2 == 0


def double(
    n: Union[int, Tuple[int, int]]
) -> Union[int, Tuple[int, int]]:
    """ scaling by two """
    return n * 2 if isinstance(n, int) else (n[0] * 2, n[1] * 2)


def halve(
    n: Union[int, Tuple[int, int]]
) -> Union[int, Tuple[int, int]]:
    """ scaling by one half """
    return n // 2 if isinstance(n, int) else (n[0] // 2, n[1] // 2)


def flip(
    b: bool
) -> bool:
    """ logical not """
    return not b


def equality(
    a: Any,
    b: Any
) -> bool:
    """ equality """
    return a == b


def contained(
    value: Any,
    container: Container
) -> bool:
    """ element of """
    return value in container


def combine(
    a: Container,
    b: Container
) -> Container:
    """ union """
    return type(a)((*a, *b))


def intersection(
    a: FrozenSet,
    b: FrozenSet
) -> FrozenSet:
    """ returns the intersection of two containers """
    return a & b


def difference(
    a: FrozenSet,
    b: FrozenSet
) -> FrozenSet:
    """ set difference """
    return type(a)(e for e in a if e not in b)


def dedupe(
    tup: Tuple
) -> Tuple:
    """ remove duplicates """
    return tuple(e for i, e in enumerate(tup) if tup.index(e) == i)


def order(
    container: Container,
    compfunc: Callable
) -> Tuple:
    """ order container by custom key """
    return tuple(sorted(container, key=compfunc))


def repeat(
    item: Any,
    num: int
) -> Tuple:
    """ repetition of item within vector """
    return tuple(item for i in range(num))


def greater(
    a: int,
    b: int
) -> bool:
    """ greater """
    return a > b


def size(
    container: Container
) -> int:
    """ cardinality """
    return len(container)


def merge(
    containers: Container[Container]
) -> Container:
    """ merging """
    return type(containers)(e for c in containers for e in c)


def maximum(
    container: FrozenSet[int]
) -> int:
    """ maximum """
    return max(container, default=0)


def minimum(
    container: FrozenSet[int]
) -> int:
    """ minimum """
    return min(container, default=0)


def valmax(
    container: Container,
    compfunc: Callable
) -> int:
    """ maximum by custom function """
    return compfunc(max(container, key=compfunc, default=0))


def valmin(
    container: Container,
    compfunc: Callable
) -> int:
    """ minimum by custom function """
    return compfunc(min(container, key=compfunc, default=0))


def argmax(
    container: Container,
    compfunc: Callable
) -> Any:
    """ largest item by custom order """
    return max(container, key=compfunc)


def argmin(
    container: Container,
    compfunc: Callable
) -> Any:
    """ smallest item by custom order """
    return min(container, key=compfunc)


def mostcommon(
    container: Container
) -> Any:
    """ most common item """
    return max(set(container), key=container.count)


def leastcommon(
    container: Container
) -> Any:
    """ least common item """
    return min(set(container), key=container.count)


def initset(
    value: Any
) -> FrozenSet:
    """ initialize container """
    return frozenset({value})


def both(
    a: bool,
    b: bool
) -> bool:
    """ logical and """
    return a and b


def either(
    a: bool,
    b: bool
) -> bool:
    """ logical or """
    return a or b


def increment(
    x: Union[int, Tuple[int, int]]
) -> Union[int, Tuple[int, int]]:
    """ incrementing """
    return x + 1 if isinstance(x, int) else (x[0] + 1, x[1] + 1)


def decrement(
    x: Union[int, Tuple[int, int]]
) -> Union[int, Tuple[int, int]]:
    """ decrementing """
    return x - 1 if isinstance(x, int) else (x[0] - 1, x[1] - 1)


def crement(
    x: Union[int, Tuple[int, int]]
) -> Union[int, Tuple[int, int]]:
    """ incrementing positive and decrementing negative """
    if isinstance(x, int):
        return 0 if x == 0 else (x + 1 if x > 0 else x - 1)
    return (
        0 if x[0] == 0 else (x[0] + 1 if x[0] > 0 else x[0] - 1),
        0 if x[1] == 0 else (x[1] + 1 if x[1] > 0 else x[1] - 1)
    )


def sign(
    x: Union[int, Tuple[int, int]]
) -> Union[int, Tuple[int, int]]:
    """ sign """
    if isinstance(x, int):
        return 0 if x == 0 else (1 if x > 0 else -1)
    return (
        0 if x[0] == 0 else (1 if x[0] > 0 else -1),
        0 if x[1] == 0 else (1 if x[1] > 0 else -1)
    )


def positive(
    x: int
) -> bool:
    """ positive """
    return x > 0


def toivec(
    i: int
) -> Tuple[int, int]:
    """ vector pointing vertically """
    return (i, 0)


def tojvec(
    j: int
) -> Tuple[int, int]:
    """ vector pointing horizontally """
    return (0, j)


def sfilter(
    container: Container,
    condition: Callable
) -> Container:
    """ keep elements in container that satisfy condition """
    return type(container)(e for e in container if condition(e))


def mfilter(
    container: Container,
    function: Callable
) -> FrozenSet:
    """ filter and merge """
    return merge(sfilter(container, function))


def extract(
    container: Container,
    condition: Callable
) -> Any:
    """ first element of container that satisfies condition """
    return next(e for e in container if condition(e))


def totuple(
    container: FrozenSet
) -> Tuple:
    """ conversion to tuple """
    return tuple(container)


def first(
    container: Container
) -> Any:
    """ first item of container """
    return next(iter(container))


def last(
    container: Container
) -> Any:
    """ last item of container """
    return max(enumerate(container))[1]


def insert(
    value: Any,
    container: FrozenSet
) -> FrozenSet:
    """ insert item into container """
    return container.union(frozenset({value}))


def remove(
    value: Any,
    container: Container
) -> Container:
    """ remove item from container """
    return type(container)(e for e in container if e != value)


def other(
    container: Container,
    value: Any
) -> Any:
    """ other value in the container """
    return first(remove(value, container))


def interval(
    start: int,
    stop: int,
    step: int
) -> Tuple:
    """ range """
    return tuple(range(start, stop, step))


def astuple(
    a: int,
    b: int
) -> Tuple[int, int]:
    """ constructs a tuple """
    return (a, b)


def product(
    a: Container,
    b: Container
) -> FrozenSet:
    """ cartesian product """
    return frozenset((i, j) for j in b for i in a)


def pair(
    a: Tuple,
    b: Tuple
) -> Tuple[Tuple]:
    """ zipping of two tuples """
    return tuple(zip(a, b))


def branch(
    condition: bool,
    a: Any,
    b: Any
) -> Any:
    """ if else branching """
    return a if condition else b


def compose(
    outer: Callable,
    inner: Callable
) -> Callable:
    """ function composition """
    return lambda x: outer(inner(x))


def chain(
    h: Callable,
    g: Callable,
    f: Callable,
) -> Callable:
    """ function composition with three functions """
    return lambda x: h(g(f(x)))


def matcher(
    function: Callable,
    target: Any
) -> Callable:
    """ construction of equality function """
    return lambda x: function(x) == target


def rbind(
    function: Callable,
    fixed: Any
) -> Callable:
    """ fix the rightmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda x: function(x, fixed)
    elif n == 3:
        return lambda x, y: function(x, y, fixed)
    else:
        return lambda x, y, z: function(x, y, z, fixed)


def lbind(
    function: Callable,
    fixed: Any
) -> Callable:
    """ fix the leftmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda y: function(fixed, y)
    elif n == 3:
        return lambda y, z: function(fixed, y, z)
    else:
        return lambda y, z, a: function(fixed, y, z, a)


def power(
    function: Callable,
    n: int
) -> Callable:
    """ power of function """
    if n == 1:
        return function
    return compose(function, power(function, n - 1))


def fork(
    outer: Callable,
    a: Callable,
    b: Callable
) -> Callable:
    """ creates a wrapper function """
    return lambda x: outer(a(x), b(x))


def apply(
    function: Callable,
    container: Container
) -> Container:
    """ apply function to each item in container """
    return type(container)(function(e) for e in container)


def rapply(
    functions: Container,
    value: Any
) -> Container:
    """ apply each function in container to value """
    return type(functions)(function(value) for function in functions)


def mapply(
    function: Callable,
    container: Container[Container]
) -> FrozenSet:
    """ apply and merge """
    return merge(apply(function, container))


def papply(
    function: Callable,
    a: Tuple,
    b: Tuple
) -> Tuple:
    """ apply function on two vectors """
    return tuple(function(i, j) for i, j in zip(a, b))


def mpapply(
    function: Callable,
    a: Tuple,
    b: Tuple
) -> Tuple:
    """ apply function on two vectors and merge """
    return merge(papply(function, a, b))


def prapply(
    function,
    a: Container,
    b: Container
) -> FrozenSet:
    """ apply function on cartesian product """
    return frozenset(function(i, j) for j in b for i in a)


def mostcolor(
    element: Union[FrozenSet[Tuple[int, Tuple[int, int]]], Tuple[Tuple[int]]]
) -> int:
    """ most common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return max(set(values), key=values.count)
    

def leastcolor(
    element: Union[FrozenSet[Tuple[int, Tuple[int, int]]], Tuple[Tuple[int]]]
) -> int:
    """ least common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return min(set(values), key=values.count)


def height(
    piece: Union[Tuple[Tuple[int]], Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]]]
) -> int:
    """ height of grid or patch """
    if len(piece) == 0:
        return 0
    if isinstance(piece, tuple):
        return len(piece)
    return lowermost(piece) - uppermost(piece) + 1


def width(
    piece: Union[Tuple[Tuple[int]], Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]]]
) -> int:
    """ width of grid or patch """
    if len(piece) == 0:
        return 0
    if isinstance(piece, tuple):
        return len(piece[0])
    return rightmost(piece) - leftmost(piece) + 1


def shape(
    piece: Union[Tuple[Tuple[int]], Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]]]
) -> Tuple[int, int]:
    """ height and width of grid or patch """
    return (height(piece), width(piece))


def portrait(
    piece: Union[Tuple[Tuple[int]], Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]]]
) -> bool:
    """ whether height is greater than width """
    return height(piece) > width(piece)


def colorcount(
    element: Union[FrozenSet[Tuple[int, Tuple[int, int]]], Tuple[Tuple[int]]],
    value: int
) -> int:
    """ number of cells with color """
    if isinstance(element, tuple):
        return sum(row.count(value) for row in element)
    return sum(v == value for v, _ in element)


def colorfilter(
    objs: FrozenSet[FrozenSet[Tuple[int, Tuple[int, int]]]],
    value: int
) -> FrozenSet[FrozenSet[Tuple[int, Tuple[int, int]]]]:
    """ filter objects by color """
    return frozenset(obj for obj in objs if next(iter(obj))[0] == value)


def sizefilter(
    container: Container,
    n: int
) -> FrozenSet:
    """ filter items by size """
    return frozenset(item for item in container if len(item) == n)


def asindices(
    grid: Tuple[Tuple[int]]
) -> FrozenSet[Tuple[int, int]]:
    """ indices of all grid cells """
    return frozenset((i, j) for i in range(len(grid)) for j in range(len(grid[0])))


def ofcolor(
    grid: Tuple[Tuple[int]],
    value: int
) -> FrozenSet[Tuple[int, int]]:
    """ indices of all grid cells with value """
    return frozenset((i, j) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value)


def ulcorner(
    patch: Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]]
) -> Tuple[int, int]:
    """ index of upper left corner """
    return tuple(map(min, zip(*toindices(patch))))


def urcorner(
    patch: Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]]
) -> Tuple[int, int]:
    """ index of upper right corner """
    return tuple(map(lambda ix: {0: min, 1: max}[ix[0]](ix[1]), enumerate(zip(*toindices(patch)))))


def llcorner(
    patch: Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]]
) -> Tuple[int, int]:
    """ index of lower left corner """
    return tuple(map(lambda ix: {0: max, 1: min}[ix[0]](ix[1]), enumerate(zip(*toindices(patch)))))


def lrcorner(
    patch: Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]]
) -> Tuple[int, int]:
    """ index of lower right corner """
    return tuple(map(max, zip(*toindices(patch))))


def crop(
    grid: Tuple[Tuple[int]],
    start: Tuple[int, int],
    dims: Tuple[int, int]
) -> Tuple[Tuple[int]]:
    """ subgrid specified by start and dimension """
    return tuple(r[start[1]:start[1]+dims[1]] for r in grid[start[0]:start[0]+dims[0]])


def toindices(
    patch: Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]]
) -> FrozenSet[Tuple[int, int]]:
    """ indices of object cells """
    if len(patch) == 0:
        return frozenset()
    if isinstance(next(iter(patch))[1], tuple):
        return frozenset(index for value, index in patch)
    return patch


def recolor(
    value: int,
    patch: Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]]
) -> FrozenSet[Tuple[int, Tuple[int, int]]]:
    """ recolor patch """
    return frozenset((value, index) for index in toindices(patch))


def shift(
    patch: Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]],
    directions: Tuple[int, int]
) -> Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]]:
    """ shift patch """
    if len(patch) == 0:
        return patch
    di, dj = directions
    if isinstance(next(iter(patch))[1], tuple):
        return frozenset((value, (i + di, j + dj)) for value, (i, j) in patch)
    return frozenset((i + di, j + dj) for i, j in patch)


def normalize(
    patch: Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]]
) -> Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]]:
    """ moves upper left corner to origin """
    if len(patch) == 0:
        return patch
    return shift(patch, (-uppermost(patch), -leftmost(patch)))


def dneighbors(
    loc: Tuple[int, int]
) -> FrozenSet[Tuple[int, int]]:
    """ directly adjacent indices """
    return frozenset({(loc[0] - 1, loc[1]), (loc[0] + 1, loc[1]), (loc[0], loc[1] - 1), (loc[0], loc[1] + 1)})


def ineighbors(
    loc: Tuple[int, int]
) -> FrozenSet[Tuple[int, int]]:
    """ diagonally adjacent indices """
    return frozenset({(loc[0] - 1, loc[1] - 1), (loc[0] - 1, loc[1] + 1), (loc[0] + 1, loc[1] - 1), (loc[0] + 1, loc[1] + 1)})


def neighbors(
    loc: Tuple[int, int]
) -> FrozenSet[Tuple[int, int]]:
    """ adjacent indices """
    return dneighbors(loc) | ineighbors(loc)


def objects(
    grid: Tuple[Tuple[int]],
    univalued: bool,
    diagonal: bool,
    without_bg: bool
) -> FrozenSet[FrozenSet[Tuple[int, Tuple[int, int]]]]:
    """ objects occurring on the grid """
    bg = mostcolor(grid) if without_bg else None
    objs = set()
    occupied = set()
    h, w = len(grid), len(grid[0])
    unvisited = asindices(grid)
    diagfun = neighbors if diagonal else dneighbors
    for loc in unvisited:
        if loc in occupied:
            continue
        val = grid[loc[0]][loc[1]]
        if val == bg:
            continue
        obj = {(val, loc)}
        cands = {loc}
        while len(cands) > 0:
            neighborhood = set()
            for cand in cands:
                v = grid[cand[0]][cand[1]]
                if (val == v) if univalued else (v != bg):
                    obj.add((v, cand))
                    occupied.add(cand)
                    neighborhood |= {
                        (i, j) for i, j in diagfun(cand) if 0 <= i < h and 0 <= j < w
                    }
            cands = neighborhood - occupied
        objs.add(frozenset(obj))
    return frozenset(objs)


def partition(
    grid: Tuple[Tuple[int]]
) -> FrozenSet[FrozenSet[Tuple[int, Tuple[int, int]]]]:
    """ each cell with the same value part of the same object """
    return frozenset(
        frozenset(
            (v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value
        ) for value in palette(grid)
    )


def fgpartition(
    grid: Tuple[Tuple[int]]
) -> FrozenSet[FrozenSet[Tuple[int, Tuple[int, int]]]]:
    """ each cell with the same value part of the same object without background """
    return frozenset(
        frozenset(
            (v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value
        ) for value in palette(grid) - {mostcolor(grid)}
    )


def uppermost(
    patch: Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]]
) -> int:
    """ row index of uppermost occupied cell """
    return min(i for i, j in toindices(patch))


def lowermost(
    patch: Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]]
) -> int:
    """ row index of lowermost occupied cell """
    return max(i for i, j in toindices(patch))


def leftmost(
    patch: Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]]
) -> int:
    """ column index of leftmost occupied cell """
    return min(j for i, j in toindices(patch))


def rightmost(
    patch: Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]]
) -> int:
    """ column index of rightmost occupied cell """
    return max(j for i, j in toindices(patch))


def square(
    piece: Union[Tuple[Tuple[int]], Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]]]
) -> bool:
    """ whether the piece forms a square """
    return len(piece) == len(piece[0]) if isinstance(piece, tuple) else height(piece) * width(piece) == len(piece) and height(piece) == width(piece)


def vline(
    patch: Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]]
) -> bool:
    """ whether the piece forms a vertical line """
    return height(patch) == len(patch) and width(patch) == 1


def hline(
    patch: Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]]
) -> bool:
    """ whether the piece forms a horizontal line """
    return width(patch) == len(patch) and height(patch) == 1


def hmatching(
    a: Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]],
    b: Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]]
) -> bool:
    """ whether there exists a row for which both patches have cells """
    return len(set(i for i, j in toindices(a)) & set(i for i, j in toindices(b))) > 0


def vmatching(
    a: Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]],
    b: Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]]
) -> bool:
    """ whether there exists a column for which both patches have cells """
    return len(set(j for i, j in toindices(a)) & set(j for i, j in toindices(b))) > 0


def manhattan(
    a: Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]],
    b: Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]]
) -> int:
    """ closest manhattan distance between two patches """
    return min(abs(ai - bi) + abs(aj - bj) for ai, aj in toindices(a) for bi, bj in toindices(b))


def adjacent(
    a: Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]],
    b: Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]]
) -> bool:
    """ whether two patches are adjacent """
    return manhattan(a, b) == 1


def bordering(
    patch: Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]],
    grid: Tuple[Tuple[int]]
) -> bool:
    """ whether a patch is adjacent to a grid border """
    return uppermost(patch) == 0 or leftmost(patch) == 0 or lowermost(patch) == len(grid) - 1 or rightmost(patch) == len(grid[0]) - 1


def centerofmass(
    patch: Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]]
) -> Tuple[int, int]:
    """ center of mass """
    return tuple(map(lambda x: sum(x) // len(patch), zip(*toindices(patch))))


def palette(
    element: Union[FrozenSet[Tuple[int, Tuple[int, int]]], Tuple[Tuple[int]]]
) -> FrozenSet[int]:
    """ colors occurring in object or grid """
    if isinstance(element, tuple):
        return frozenset({v for r in element for v in r})
    return frozenset({v for v, _ in element})


def numcolors(
    element: Union[FrozenSet[Tuple[int, Tuple[int, int]]], Tuple[Tuple[int]]]
) -> FrozenSet[int]:
    """ number of colors occurring in object or grid """
    return len(palette(element))


def color(
    obj: FrozenSet[Tuple[int, Tuple[int, int]]]
) -> int:
    """ color of object """
    return next(iter(obj))[0]


def toobject(
    patch: Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]],
    grid: Tuple[Tuple[int]]
) -> FrozenSet[Tuple[int, Tuple[int, int]]]:
    """ object from patch and grid """
    h, w = len(grid), len(grid[0])
    return frozenset((grid[i][j], (i, j)) for i, j in toindices(patch) if 0 <= i < h and 0 <= j < w)


def asobject(
    grid: Tuple[Tuple[int]]
) -> FrozenSet[Tuple[int, Tuple[int, int]]]:
    """ conversion of grid to object """
    return frozenset((v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r))


def rot90(
    grid: Tuple[Tuple[int]]
) -> Tuple[Tuple[int]]:
    """ quarter clockwise rotation """
    return tuple(row for row in zip(*grid[::-1]))


def rot180(
    grid: Tuple[Tuple[int]]
) -> Tuple[Tuple[int]]:
    """ half rotation """
    return tuple(tuple(row[::-1]) for row in grid[::-1])


def rot270(
    grid: Tuple[Tuple[int]]
) -> Tuple[Tuple[int]]:
    """ quarter anticlockwise rotation """
    return tuple(tuple(row[::-1]) for row in zip(*grid[::-1]))[::-1]


def hmirror(
    piece: Union[Tuple[Tuple[int]], Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]]]
) -> Union[Tuple[Tuple[int]], Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]]]:
    """ mirroring along horizontal """
    if isinstance(piece, tuple):
        return piece[::-1]
    d = ulcorner(piece)[0] + lrcorner(piece)[0]
    if isinstance(next(iter(piece))[1], tuple):
        return frozenset((v, (d - i, j)) for v, (i, j) in piece)
    return frozenset((d - i, j) for i, j in piece)


def vmirror(
    piece: Union[Tuple[Tuple[int]], Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]]]
) -> Union[Tuple[Tuple[int]], Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]]]:
    """ mirroring along vertical """
    if isinstance(piece, tuple):
        return tuple(row[::-1] for row in piece)
    d = ulcorner(piece)[1] + lrcorner(piece)[1]
    if isinstance(next(iter(piece))[1], tuple):
        return frozenset((v, (i, d - j)) for v, (i, j) in piece)
    return frozenset((i, d - j) for i, j in piece)


def dmirror(
    piece: Union[Tuple[Tuple[int]], Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]]]
) -> Union[Tuple[Tuple[int]], Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]]]:
    """ mirroring along diagonal """
    if isinstance(piece, tuple):
        return tuple(zip(*piece))
    a, b = ulcorner(piece)
    if isinstance(next(iter(piece))[1], tuple):
        return frozenset((v, (j - b + a, i - a + b)) for v, (i, j) in piece)
    return frozenset((j - b + a, i - a + b) for i, j in piece)


def cmirror(
    piece: Union[Tuple[Tuple[int]], Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]]]
) -> Union[Tuple[Tuple[int]], Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]]]:
    """ mirroring along counterdiagonal """
    if isinstance(piece, tuple):
        return tuple(zip(*(r[::-1] for r in piece[::-1])))
    return vmirror(dmirror(vmirror(piece)))


def fill(
    grid: Tuple[Tuple[int]],
    value: int,
    patch: Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]]
) -> Tuple[Tuple[int]]:
    """ fill value at indices """
    h, w = len(grid), len(grid[0])
    grid_filled = list(list(row) for row in grid)
    for i, j in toindices(patch):
        if 0 <= i < h and 0 <= j < w:
            grid_filled[i][j] = value
    return tuple(tuple(row) for row in grid_filled)


def paint(
    grid: Tuple[Tuple[int]],
    obj: FrozenSet[Tuple[int, Tuple[int, int]]]
) -> Tuple[Tuple[int]]:
    """ paint object to grid """
    h, w = len(grid), len(grid[0])
    grid_painted = list(list(row) for row in grid)
    for value, (i, j) in obj:
        if 0 <= i < h and 0 <= j < w:
            grid_painted[i][j] = value
    return tuple(tuple(row) for row in grid_painted)


def underfill(
    grid: Tuple[Tuple[int]],
    value: int,
    patch: Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]]
) -> Tuple[Tuple[int]]:
    """ fill value at indices that are background """
    h, w = len(grid), len(grid[0])
    bg = mostcolor(grid)
    g = list(list(r) for r in grid)
    for i, j in toindices(patch):
        if 0 <= i < h and 0 <= j < w:
            if g[i][j] == bg:
                g[i][j] = value
    return tuple(tuple(r) for r in g)


def underpaint(
    grid: Tuple[Tuple[int]],
    obj: FrozenSet[Tuple[int, Tuple[int, int]]]
) -> Tuple[Tuple[int]]:
    """ paint object to grid where there is background """
    h, w = len(grid), len(grid[0])
    bg = mostcolor(grid)
    g = list(list(r) for r in grid)
    for value, (i, j) in obj:
        if 0 <= i < h and 0 <= j < w:
            if g[i][j] == bg:
                g[i][j] = value
    return tuple(tuple(r) for r in g)


def hupscale(
    grid: Tuple[Tuple[int]],
    factor: int
) -> Tuple[Tuple[int]]:
    """ upscale grid horizontally """
    g = tuple()
    for row in grid:
        r = tuple()
        for value in row:
            r = r + tuple(value for num in range(factor))
        g = g + (r,)
    return g


def vupscale(
    grid: Tuple[Tuple[int]],
    factor: int
) -> Tuple[Tuple[int]]:
    """ upscale grid vertically """
    g = tuple()
    for row in grid:
        g = g + tuple(row for num in range(factor))
    return g


def upscale(
    element: Union[FrozenSet[Tuple[int, Tuple[int, int]]], Tuple[Tuple[int]]],
    factor: int
) -> Union[FrozenSet[Tuple[int, Tuple[int, int]]], Tuple[Tuple[int]]]:
    """ upscale object or grid """
    if isinstance(element, tuple):
        g = tuple()
        for row in element:
            upscaled_row = tuple()
            for value in row:
                upscaled_row = upscaled_row + tuple(value for num in range(factor))
            g = g + tuple(upscaled_row for num in range(factor))
        return g
    else:
        if len(element) == 0:
            return frozenset()
        di_inv, dj_inv = ulcorner(element)
        di, dj = (-di_inv, -dj_inv)
        normed_obj = shift(element, (di, dj))
        o = set()
        for value, (i, j) in normed_obj:
            for io in range(factor):
                for jo in range(factor):
                    o.add((value, (i * factor + io, j * factor + jo)))
        return shift(frozenset(o), (di_inv, dj_inv))


def downscale(
    grid: Tuple[Tuple[int]],
    factor: int
) -> Tuple[Tuple[int]]:
    """ downscale grid """
    h, w = len(grid), len(grid[0])
    g = tuple()
    for i in range(h):
        r = tuple()
        for j in range(w):
            if j % factor == 0:
                r = r + (grid[i][j],)
        g = g + (r, )
    h = len(g)
    dsg = tuple()
    for i in range(h):
        if i % factor == 0:
            dsg = dsg + (g[i],)
    return dsg


def hconcat(
    a: Tuple[Tuple[int]],
    b: Tuple[Tuple[int]]
) -> Tuple[Tuple[int]]:
    """ concatenate two grids horizontally """
    return tuple(i + j for i, j in zip(a, b))


def vconcat(
    a: Tuple[Tuple[int]],
    b: Tuple[Tuple[int]]
) -> Tuple[Tuple[int]]:
    """ concatenate two grids vertically """
    return a + b


def subgrid(
    patch: Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]],
    grid: Tuple[Tuple[int]]
) -> Tuple[Tuple[int]]:
    """ smallest subgrid containing object """
    return crop(grid, ulcorner(patch), shape(patch))


def hsplit(
    grid: Tuple[Tuple[int]],
    n: int
) -> Tuple:
    """ split grid horizontally """
    h, w = len(grid), len(grid[0]) // n
    offset = len(grid[0]) % n != 0
    return tuple(crop(grid, (0, w * i + i * offset), (h, w)) for i in range(n))


def vsplit(
    grid: Tuple[Tuple[int]],
    n: int
) -> Tuple:
    """ split grid vertically """
    h, w = len(grid) // n, len(grid[0])
    offset = len(grid) % n != 0
    return tuple(crop(grid, (h * i + i * offset, 0), (h, w)) for i in range(n))


def cellwise(
    a: Tuple[Tuple[int]],
    b: Tuple[Tuple[int]],
    fallback: int
) -> Tuple[Tuple[int]]:
    """ cellwise match of two grids """
    h, w = len(a), len(a[0])
    resulting_grid = tuple()
    for i in range(h):
        row = tuple()
        for j in range(w):
            a_value = a[i][j]
            value = a_value if a_value == b[i][j] else fallback
            row = row + (value,)
        resulting_grid = resulting_grid + (row, )
    return resulting_grid


def replace(
    grid: Tuple[Tuple[int]],
    replacee: int,
    replacer: int
) -> Tuple[Tuple[int]]:
    """ color substitution """
    return tuple(tuple(replacer if v == replacee else v for v in r) for r in grid)


def switch(
    grid: Tuple[Tuple[int]],
    a: int,
    b: int
) -> Tuple[Tuple[int]]:
    """ color switching """
    return tuple(tuple(v if (v != a and v != b) else {a: b, b: a}[v] for v in r) for r in grid)


def center(
    patch: Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]]
) -> Tuple[int, int]:
    """ center of the patch """
    return (uppermost(patch) + height(patch) // 2, leftmost(patch) + width(patch) // 2)


def position(
    a: Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]],
    b: Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]]
) -> Tuple[int, int]:
    """ relative position between two patches """
    ia, ja = center(toindices(a))
    ib, jb = center(toindices(b))
    if ia == ib:
        return (0, 1 if ja < jb else -1)
    elif ja == jb:
        return (1 if ia < ib else -1, 0)
    elif ia < ib:
        return (1, 1 if ja < jb else -1)
    elif ia > ib:
        return (-1, 1 if ja < jb else -1)


def index(
    grid: Tuple[Tuple[int]],
    loc: Tuple[int, int]
) -> int:
    """ color at location """
    i, j = loc
    h, w = len(grid), len(grid[0])
    if not (0 <= i < h and 0 <= j < w):
        return None
    return grid[loc[0]][loc[1]] 


def canvas(
    value: int,
    dimensions: Tuple[int, int]
) -> Tuple[Tuple[int]]:
    """ grid construction """
    return tuple(tuple(value for j in range(dimensions[1])) for i in range(dimensions[0]))


def corners(
    patch: Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]]
) -> FrozenSet[Tuple[int, int]]:
    """ indices of corners """
    return frozenset({ulcorner(patch), urcorner(patch), llcorner(patch), lrcorner(patch)})


def connect(
    a: Tuple[int, int],
    b: Tuple[int, int]
) -> FrozenSet[Tuple[int, int]]:
    """ line between two points """
    ai, aj = a
    bi, bj = b
    si = min(ai, bi)
    ei = max(ai, bi) + 1
    sj = min(aj, bj)
    ej = max(aj, bj) + 1
    if ai == bi:
        return frozenset((ai, j) for j in range(sj, ej))
    elif aj == bj:
        return frozenset((i, aj) for i in range(si, ei))
    elif bi - ai == bj - aj:
        return frozenset((i, j) for i, j in zip(range(si, ei), range(sj, ej)))
    elif bi - ai == aj - bj:
        return frozenset((i, j) for i, j in zip(range(si, ei), range(ej - 1, sj - 1, -1)))
    return frozenset()


def cover(
    grid: Tuple[Tuple[int]],
    patch: Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]]
) -> Tuple[Tuple[int]]:
    """ remove object from grid """
    return fill(grid, mostcolor(grid), toindices(patch))


def trim(
    grid: Tuple[Tuple[int]]
) -> Tuple[Tuple[int]]:
    """ trim border of grid """
    return tuple(r[1:-1] for r in grid[1:-1])


def move(
    grid: Tuple[Tuple[int]],
    obj: FrozenSet[Tuple[int, Tuple[int, int]]],
    offset: Tuple[int, int]
) -> Tuple[Tuple[int]]:
    """ move object on grid """
    return paint(cover(grid, obj), shift(obj, offset))


def tophalf(
    grid: Tuple[Tuple[int]]
) -> Tuple[Tuple[int]]:
    """ upper half of grid """
    return grid[:len(grid) // 2]


def bottomhalf(
    grid: Tuple[Tuple[int]]
) -> Tuple[Tuple[int]]:
    """ lower half of grid """
    return grid[len(grid) // 2 + len(grid) % 2:]


def lefthalf(
    grid: Tuple[Tuple[int]]
) -> Tuple[Tuple[int]]:
    """ left half of grid """
    return rot270(tophalf(rot90(grid)))


def righthalf(
    grid: Tuple[Tuple[int]]
) -> Tuple[Tuple[int]]:
    """ right half of grid """
    return rot270(bottomhalf(rot90(grid)))


def vfrontier(
    location: Tuple[int, int]
) -> FrozenSet[Tuple[int, int]]:
    """ vertical frontier """
    return frozenset((i, location[1]) for i in range(30))


def hfrontier(
    location: Tuple[int, int]
) -> FrozenSet[Tuple[int, int]]:
    """ horizontal frontier """
    return frozenset((location[0], j) for j in range(30))


def backdrop(
    patch: Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]]
) -> FrozenSet[Tuple[int, int]]:
    """ indices in bounding box of patch """
    if len(patch) == 0:
        return frozenset({})
    indices = toindices(patch)
    si, sj = ulcorner(indices)
    ei, ej = lrcorner(patch)
    return frozenset((i, j) for i in range(si, ei + 1) for j in range(sj, ej + 1))


def delta(
    patch: Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]]
) -> FrozenSet[Tuple[int, int]]:
    """ indices in bounding box but not part of patch """
    if len(patch) == 0:
        return frozenset({})
    return backdrop(patch) - toindices(patch)


def gravitate(
    source: Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]],
    destination: Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]]
) -> Tuple[int, int]:
    """ direction to move source until adjacent to destination """
    si, sj = center(source)
    di, dj = center(destination)
    i, j = 0, 0
    if vmatching(source, destination):
        i = 1 if si < di else -1
    else:
        j = 1 if sj < dj else -1
    gi, gj = i, j
    c = 0
    while not adjacent(source, destination) and c < 42:
        c += 1
        gi += i
        gj += j
        source = shift(source, (i, j))
    return (gi - i, gj - j)


def inbox(
    patch: Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]]
) -> FrozenSet[Tuple[int, int]]:
    """ inbox for patch """
    ai, aj = uppermost(patch) + 1, leftmost(patch) + 1
    bi, bj = lowermost(patch) - 1, rightmost(patch) - 1
    si, sj = min(ai, bi), min(aj, bj)
    ei, ej = max(ai, bi), max(aj, bj)
    vlines = {(i, sj) for i in range(si, ei + 1)} | {(i, ej) for i in range(si, ei + 1)}
    hlines = {(si, j) for j in range(sj, ej + 1)} | {(ei, j) for j in range(sj, ej + 1)}
    return frozenset(vlines | hlines)


def outbox(
    patch: Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]]
) -> FrozenSet[Tuple[int, int]]:
    """ outbox for patch """
    ai, aj = uppermost(patch) - 1, leftmost(patch) - 1
    bi, bj = lowermost(patch) + 1, rightmost(patch) + 1
    si, sj = min(ai, bi), min(aj, bj)
    ei, ej = max(ai, bi), max(aj, bj)
    vlines = {(i, sj) for i in range(si, ei + 1)} | {(i, ej) for i in range(si, ei + 1)}
    hlines = {(si, j) for j in range(sj, ej + 1)} | {(ei, j) for j in range(sj, ej + 1)}
    return frozenset(vlines | hlines)


def box(
    patch: Union[FrozenSet[Tuple[int, Tuple[int, int]]], FrozenSet[Tuple[int, int]]]
) -> FrozenSet[Tuple[int, int]]:
    """ outline of patch """
    if len(patch) == 0:
        return patch
    ai, aj = ulcorner(patch)
    bi, bj = lrcorner(patch)
    si, sj = min(ai, bi), min(aj, bj)
    ei, ej = max(ai, bi), max(aj, bj)
    vlines = {(i, sj) for i in range(si, ei + 1)} | {(i, ej) for i in range(si, ei + 1)}
    hlines = {(si, j) for j in range(sj, ej + 1)} | {(ei, j) for j in range(sj, ej + 1)}
    return frozenset(vlines | hlines)


def shoot(
    start: Tuple[int, int],
    direction: Tuple[int, int]
) -> FrozenSet[Tuple[int, int]]:
    """ line from starting point and direction """
    return connect(start, (start[0] + 42 * direction[0], start[1] + 42 * direction[1]))


def occurrences(
    grid: Tuple[Tuple[int]],
    obj: FrozenSet[Tuple[int, Tuple[int, int]]]
) -> FrozenSet[Tuple[int, int]]:
    """ locations of occurrences of object in grid """
    occs = set()
    normed = normalize(obj)
    h, w = len(grid), len(grid[0])
    oh, ow = shape(obj)
    h2, w2 = h - oh + 1, w - ow + 1
    for i in range(h2):
        for j in range(w2):
            occurs = True
            for v, (a, b) in shift(normed, (i, j)):
                if not (0 <= a < h and 0 <= b < w and grid[a][b] == v):
                    occurs = False
                    break
            if occurs:
                occs.add((i, j))
    return frozenset(occs)


def frontiers(
    grid: Tuple[Tuple[int]]
) -> FrozenSet[FrozenSet[Tuple[int, Tuple[int, int]]]]:
    """ set of frontiers """
    h, w = len(grid), len(grid[0])
    row_indices = tuple(i for i, r in enumerate(grid) if len(set(r)) == 1)
    column_indices = tuple(j for j, c in enumerate(dmirror(grid)) if len(set(c)) == 1)
    hfrontiers = frozenset({frozenset({(grid[i][j], (i, j)) for j in range(w)}) for i in row_indices})
    vfrontiers = frozenset({frozenset({(grid[i][j], (i, j)) for i in range(h)}) for j in column_indices})
    return hfrontiers | vfrontiers


def compress(
    grid: Tuple[Tuple[int]]
) -> Tuple[Tuple[int]]:
    """ removes frontiers from grid """
    ri = tuple(i for i, r in enumerate(grid) if len(set(r)) == 1)
    ci = tuple(j for j, c in enumerate(dmirror(grid)) if len(set(c)) == 1)
    return tuple(tuple(v for j, v in enumerate(r) if j not in ci) for i, r in enumerate(grid) if i not in ri)


def hperiod(
    obj: FrozenSet[Tuple[int, Tuple[int, int]]]
) -> int:
    """ horizontal periodicity """
    normalized = normalize(obj)
    w = width(normalized)
    for p in range(1, w):
        offsetted = shift(normalized, (0, -p))
        pruned = frozenset({(c, (i, j)) for c, (i, j) in offsetted if j >= 0})
        if pruned.issubset(normalized):
            return p
    return w


def vperiod(
    obj: FrozenSet[Tuple[int, Tuple[int, int]]]
) -> int:
    """ vertical periodicity """
    normalized = normalize(obj)
    h = height(normalized)
    for p in range(1, h):
        offsetted = shift(normalized, (-p, 0))
        pruned = frozenset({(c, (i, j)) for c, (i, j) in offsetted if i >= 0})
        if pruned.issubset(normalized):
            return p
    return h
