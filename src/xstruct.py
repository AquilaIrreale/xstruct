from enum import Enum, auto
from struct import pack, unpack, calcsize
from contextlib import suppress
from functools import wraps
from ast import (
        Assign, Attribute, Constant, FunctionDef, Load, Module,
        Name, Store, Subscript, arg, arguments, fix_missing_locations)

try:
    import bson
except ImportError:
    _have_bson = False
else:
    _have_bson = True


__version__ = "0.2.0"


class StructError(Exception):
    pass


class StructDeclarationError(StructError):
    pass


class StructSizeMismatch(StructError):
    pass


class StructSizeUnknown(StructError):
    pass


class Endianess(Enum):
    Native = "="
    Little = "<"
    Big    = ">"


class Types(Enum):
    Int8    = auto()
    Int16   = auto()
    Int32   = auto()
    Int64   = auto()
    UInt8   = auto()
    UInt16  = auto()
    UInt32  = auto()
    UInt64  = auto()
    Float   = auto()
    Double  = auto()
    Char    = auto()
    CString = auto()
    BSON    = auto()


globals().update(Endianess.__members__)
globals().update(Types.__members__)


def _endianess_code(endianess):
    if not isinstance(endianess, Endianess):
        raise TypeError(f"{endianess!r} is not a valid endianess")
    return endianess.value


def _simple_unpacker(fmt):
    size = calcsize(fmt)
    def unpacker(buf, endianess=Native):
        fmt_with_endianess = f"{_endianess_code(endianess)}{fmt}"
        value, = unpack(fmt_with_endianess, buf[:size])
        return value, buf[size:]
    return unpacker


def _string_unpack(buf, endianess=None):
    string, sep, tail = buf.partition(b"\0")
    if not sep:
        raise ValueError("Unterminated string in buffer")
    return string, tail


def _bson_unpack(buf, endianess=None):
    size, = unpack("<i", buf[:4])
    return bson.decode(buf[:size]), buf[size:]


def _substruct_unpacker(cls):
    def unpacker(buf, endianess=None):
        ret = cls(buf, exact=False)
        return ret, buf[sizeof(ret):]
    return unpacker


def _optional_unpacker(base_unpacker, default):
    def unpacker(buf, endianess=Native):
        if not buf:
            return default, buf
        return base_unpacker(buf, endianess)
    return unpacker


def _simple_packer(fmt):
    def packer(value, endianess=Native):
        fmt_with_endianess = f"{_endianess_code(endianess)}{fmt}"
        return pack(fmt_with_endianess, value)
    return packer


def _string_pack(value, endianess=None):
    if not isinstance(value, bytes):
        raise TypeError("C string must be a bytes object")
    if b"\0" in value:
        raise ValueError("Null bytes in C string")
    return value + b"\0"


def _bson_pack(value, endianess=None):
    return bson.encode(value)


def _substruct_packer(cls):
    def packer(obj, endianess=None):
        return obj.pack()
    return unpacker


_base_unpackers = {
    Int8:    _simple_unpacker("b"),
    Int16:   _simple_unpacker("h"),
    Int32:   _simple_unpacker("i"),
    Int64:   _simple_unpacker("q"),
    UInt8:   _simple_unpacker("B"),
    UInt16:  _simple_unpacker("H"),
    UInt32:  _simple_unpacker("I"),
    UInt64:  _simple_unpacker("Q"),
    Float:   _simple_unpacker("f"),
    Double:  _simple_unpacker("d"),
    Char:    _simple_unpacker("c"),
    CString: _string_unpack,
    BSON:    _bson_unpack
}

_base_packers = {
    Int8:    _simple_packer("b"),
    Int16:   _simple_packer("h"),
    Int32:   _simple_packer("i"),
    Int64:   _simple_packer("q"),
    UInt8:   _simple_packer("B"),
    UInt16:  _simple_packer("H"),
    UInt32:  _simple_packer("I"),
    UInt64:  _simple_packer("Q"),
    Float:   _simple_packer("f"),
    Double:  _simple_packer("d"),
    Char:    _simple_packer("c"),
    CString: _string_pack,
    BSON:    _bson_pack
}

_fixed_size_types = {
    Int8:   1,
    Int16:  2,
    Int32:  4,
    Int64:  8,
    UInt8:  1,
    UInt16: 2,
    UInt32: 4,
    UInt64: 8,
    Float:  4,
    Double: 8,
    Char:   1,
}


def is_struct_class(cls):
    return isinstance(cls, type) and hasattr(cls, "_struct_members")


def sizeof(x):
    with suppress(KeyError):
        return _fixed_size_types[x]
    if is_struct_class(x):
        if x._struct_predicted_size is not None:
            return x._struct_predicted_size
        raise StructSizeUnknown("Struct template contains members of non-fixed size, cannot deduce total struct size before utilization")
    if hasattr(x, "_struct_decoded_size"):
        return x._struct_decoded_size
    raise TypeError("x must be a struct class, struct object or a fixed size member type designator")


def add_method(cls):
    def decorator(f):
        setattr(cls, f.__name__, f)
    return decorator


def constructor(f):
    @wraps(f)
    @classmethod
    def wrapper(cls, *args, **kwargs):
        obj = cls.__new__(cls)
        super(cls, obj).__init__()
        f(obj, *args, **kwargs)
        return obj
    return wrapper


def _make_args(names):
    return [arg(arg=name) for name in names]


def _make_defaults(names):
    return [
        Subscript(
          value=Name(id='defaults', ctx=Load()),
          slice=Constant(value=name),
          ctx=Load())
        for name in names]


def _make_body(names):
    return [
        Assign(
          targets=[
            Attribute(
              value=Name(id='self', ctx=Load()),
              attr=name,
              ctx=Store())],
          value=Name(id=name, ctx=Load()))
        for name in names]


def _make_init_method(members, defaults):
    ast = Module(
      body=[
        FunctionDef(
          name='__init__',
          args=arguments(
            posonlyargs=[arg(arg='self')],
            args=_make_args(members),
            kwonlyargs=[],
            kw_defaults=[],
            defaults=_make_defaults(defaults.keys())),
          body=_make_body(members),
          decorator_list=[])],
      type_ignores=[])

    fix_missing_locations(ast)

    code = compile(ast, "<xstruct: __init__ ast>", "exec")
    namespace = {"defaults": defaults}
    exec(code, namespace)
    return namespace["__init__"]


def struct(endianess=Native):
    cls = None
    if isinstance(endianess, type):
        cls = endianess
        endianess = Native

    def decorator(cls):
        annotations = getattr(cls, "__annotations__", None)
        if not annotations:
            raise StructDeclarationError("Struct has no members (did you forget to write type annotations?)")
        cls._struct_members = {}
        cls._struct_defaults = {}
        cls._struct_predicted_size = 0
        processing_optionals = False
        for name, type_ in annotations.items():
            if cls._struct_predicted_size is not None:
                try:
                    cls._struct_predicted_size += _fixed_size_types[type_]
                except KeyError:
                    cls._struct_predicted_size = None

            if is_struct_class(type_):
                unpacker = _substruct_unpacker(type_)
                packer = _substruct_packer(type_)
            elif type_ in _base_unpackers:
                if type_ is BSON and not _have_bson:
                    raise StructDeclarationError("BSON support is not available (try installing pymongo)")
                unpacker = _base_unpackers[type_]
                packer = _base_packers[type_]
            else:
                raise TypeError(f"{type_!r} is not a valid type designator for struct member {name}") from e

            if hasattr(cls, name):
                default = getattr(cls, name)
                delattr(cls, name)
                cls._struct_defaults[name] = default
                unpacker = _optional_unpacker(unpacker, default)
                processing_optionals = True
            elif processing_optionals:
                raise StructDeclarationError("Optional members, if present, must be specified after all required ones")

            cls._struct_members[name] = unpacker, packer

        cls._struct_endianess = endianess

        add_method(cls)(
                _make_init_method(
                    cls._struct_members.keys(),
                    cls._struct_defaults))

        @add_method(cls)
        def __repr__(self):
            members = (f"{name}={getattr(self, name)!r}" for name in cls._struct_members)
            return f"{cls.__name__}({', '.join(members)})"

        @add_method(cls)
        @constructor
        def unpack(self, buf, exact=False):
            starting_buf_size = len(buf)
            for name, (unpacker, _) in self._struct_members.items():
                value, buf = unpacker(buf, self._struct_endianess)
                setattr(self, name, value)
            if buf and exact:
                raise StructSizeMismatch("Struct unpacking did not consume all of provided data")
            self._struct_decoded_size = starting_buf_size - len(buf)

        @add_method(cls)
        def pack(self):
            def _pack_each():
                for name, (_, packer) in self._struct_members.items():
                    value = getattr(self, name)
                    if value is not None:
                        yield packer(value)
            return b"".join(_pack_each())

        return cls

    if cls is None:
        return decorator
    else:
        return decorator(cls)


__all__ = [
    "struct", "sizeof", "is_struct_class",
    *Endianess.__members__.keys(),
    *Types.__members__.keys()
]


if __name__ == "__main__":
    data = b"*\x00\x00\x00Hello world!\x00\x18-DT\xfb!\t@"

    @struct(endianess=Little)
    class MyStruct:
        answer:   UInt32
        greeting: CString
        pi:       Double

    s = MyStruct.unpack(data, exact=True)

    print("Original data:", data)
    print("Decoded object:", s)
    print("Bytes decoded:", sizeof(s))
    print("Re-encoding:", s.pack())
    print("Matches original?", "Yes" if s.pack() == data else "No")
