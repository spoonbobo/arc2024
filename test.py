from arc_types import *

def typeguard(inp: Any, return_type: type):
    if return_type == Grid:
        if isinstance(inp, list) or isinstance(inp, tuple):
            return tuple(tuple(i) if isinstance(i, list) else i for i in inp)

# Cases
print(typeguard(([],[],[]), Grid))  # (((), (), ()))
print(typeguard(([],(),[]), Grid))  # (((), ()), ())
print(typeguard([[], (), []], Grid))  # (((), ()), ())
print(typeguard([(),(),()], Grid))  # (((), (), ()))