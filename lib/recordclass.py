"""
Compatibility shim for recordclass — provides mutable named tuples.

recordclass objects support attribute access AND index access (used by osr2mp4).
"""


def recordclass(name, fields):
    """Create a mutable named-tuple-like class with __slots__."""
    if isinstance(fields, str):
        fields = fields.replace(",", " ").split()

    class RC:
        __slots__ = tuple(fields)

        def __init__(self, *args, **kwargs):
            for i, field in enumerate(self.__slots__):
                if i < len(args):
                    setattr(self, field, args[i])
                elif field in kwargs:
                    setattr(self, field, kwargs[field])
                else:
                    setattr(self, field, None)

        def __getitem__(self, index):
            return getattr(self, self.__slots__[index])

        def __setitem__(self, index, value):
            setattr(self, self.__slots__[index], value)

        def __repr__(self):
            items = ", ".join(
                f"{f}={getattr(self, f)!r}" for f in self.__slots__
            )
            return f"{name}({items})"

        def __iter__(self):
            for f in self.__slots__:
                yield getattr(self, f)

        def __len__(self):
            return len(self.__slots__)

    RC.__name__ = name
    RC.__qualname__ = name
    return RC
