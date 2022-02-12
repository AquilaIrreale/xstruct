
  ############################################################################
  # xstruct - Serialize and deserialize binary data in a declarative way     #
  # Copyright (C) 2021,2022  Simone Cimarelli a.k.a. AquilaIrreale           #
  #                                                                          #
  # This program is free software: you can redistribute it and/or modify     #
  # it under the terms of the GNU General Public License as published by     #
  # the Free Software Foundation, either version 3 of the License, or        #
  # (at your option) any later version.                                      #
  #                                                                          #
  # This program is distributed in the hope that it will be useful,          #
  # but WITHOUT ANY WARRANTY; without even the implied warranty of           #
  # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            #
  # GNU General Public License for more details.                             #
  #                                                                          #
  # You should have received a copy of the GNU General Public License        #
  # along with this program.  If not, see <https://www.gnu.org/licenses/>.   #
  ############################################################################


import sys

from enum import Enum, auto
from struct import pack, unpack, calcsize
from contextlib import suppress
from functools import wraps
from operator import attrgetter
from ast import (
        Assign, Attribute, Constant, FunctionDef, Load, Module,
        Name, Store, Subscript, arg, arguments, fix_missing_locations)

try:
    import bson
except ImportError:
    have_bson = False
else:
    have_bson = True


__version__ = "1.3.2"


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


byteorder = {
    Big: "big",
    Little: "little",
    Native: sys.byteorder
}


def make_length_extractor(length):
    if callable(length) or length is None:
        return length
    elif isinstance(length, str):
        return attrgetter(length)
    elif isinstance(length, int):
        return lambda _: length
    else:
        raise StructDeclarationError(f"Invalid length expression {length!r}")


class Array:
    def __init__(self, elt_type, length=None):
        self.elt_type = elt_type
        self.length_extractor = make_length_extractor(length)


class Bytes:
    def __init__(self, length=None):
        self.length_extractor = make_length_extractor(length)


class CustomMember:
    def unpack(self, obj, buf, endianess):
        raise NotImplementedError

    def pack(self, value, endianess):
        raise NotImplementedError


def endianess_code(endianess):
    if not isinstance(endianess, Endianess):
        raise TypeError(f"{endianess!r} is not a valid endianess")
    return endianess.value


def simple_unpacker(fmt):
    size = calcsize(fmt)
    def unpacker(obj, buf, endianess=Native):
        fmt_with_endianess = f"{endianess_code(endianess)}{fmt}"
        value, = unpack(fmt_with_endianess, buf[:size])
        return value, buf[size:]
    return unpacker


def string_unpack(obj, buf, endianess=None):
    string, sep, tail = buf.partition(b"\0")
    if not sep:
        raise ValueError("Unterminated string in buffer")
    return string, tail


def bson_unpack(obj, buf, endianess=None):
    size, = unpack("<i", buf[:4])
    return bson.decode(buf[:size]), buf[size:]


def bytes_unpacker(length_extractor):
    if length_extractor is not None:
        def unpacker(obj, buf, endianess=None):
            length = length_extractor(obj)
            return buf[:length], buf[length:]
    else:
        def unpacker(obj, buf, endianess=None):
            return buf, b""
    return unpacker


def array_unpacker(base_unpacker, length_extractor):
    if length_extractor is not None:
        def unpacker(obj, buf, endianess=Native):
            length = length_extractor(obj)
            ret = [None] * length
            for i in range(length):
                ret[i], buf = base_unpacker(obj, buf, endianess)
            return ret, buf
    else:
        def unpacker(obj, buf, endianess=Native):
            ret = []
            while buf:
                value, buf = base_unpacker(obj, buf, endianess)
                ret.append(value)
            return ret, buf
    return unpacker


def substruct_unpacker(cls):
    def unpacker(obj, buf, endianess=None):
        ret = cls.unpack(buf, exact=False)
        return ret, buf[sizeof(ret):]
    return unpacker


def optional_unpacker(base_unpacker, default):
    def unpacker(obj, buf, endianess=Native):
        if not buf:
            return default, buf
        return base_unpacker(buf, endianess)
    return unpacker


def simple_packer(fmt):
    def packer(value, endianess=Native):
        fmt_with_endianess = f"{endianess_code(endianess)}{fmt}"
        return pack(fmt_with_endianess, value)
    return packer


def string_pack(value, endianess=None):
    if not isinstance(value, bytes):
        raise TypeError("C string must be a bytes object")
    if b"\0" in value:
        raise ValueError("Null bytes in C string")
    return value + b"\0"


def bson_pack(value, endianess=None):
    return bson.encode(value)


def bytes_pack(value, endianess=None):
    return bytes(value)


def array_packer(base_packer):
    def packer(seq, endianess=None):
        return b"".join(base_packer(x) for x in seq)
    return packer


def substruct_packer(cls):
    def packer(obj, endianess=None):
        return obj.pack()
    return packer


base_unpackers = {
    Int8:    simple_unpacker("b"),
    Int16:   simple_unpacker("h"),
    Int32:   simple_unpacker("i"),
    Int64:   simple_unpacker("q"),
    UInt8:   simple_unpacker("B"),
    UInt16:  simple_unpacker("H"),
    UInt32:  simple_unpacker("I"),
    UInt64:  simple_unpacker("Q"),
    Float:   simple_unpacker("f"),
    Double:  simple_unpacker("d"),
    Char:    simple_unpacker("c"),
    CString: string_unpack,
    BSON:    bson_unpack
}

base_packers = {
    Int8:    simple_packer("b"),
    Int16:   simple_packer("h"),
    Int32:   simple_packer("i"),
    Int64:   simple_packer("q"),
    UInt8:   simple_packer("B"),
    UInt16:  simple_packer("H"),
    UInt32:  simple_packer("I"),
    UInt64:  simple_packer("Q"),
    Float:   simple_packer("f"),
    Double:  simple_packer("d"),
    Char:    simple_packer("c"),
    CString: string_pack,
    BSON:    bson_pack
}


def fixed_size(fmt):
    return calcsize(f"={fmt}")


fixed_size_types = {
    Int8:   fixed_size("b"),
    Int16:  fixed_size("h"),
    Int32:  fixed_size("i"),
    Int64:  fixed_size("q"),
    UInt8:  fixed_size("B"),
    UInt16: fixed_size("H"),
    UInt32: fixed_size("I"),
    UInt64: fixed_size("Q"),
    Float:  fixed_size("f"),
    Double: fixed_size("d"),
    Char:   fixed_size("c"),
}


def is_struct(obj):
    return hasattr(obj, "_struct_members")


def is_struct_class(cls):
    return is_struct(cls) and isinstance(cls, type)


def sizeof(obj):
    with suppress(KeyError):
        return fixed_size_types[obj]
    if is_struct_class(obj):
        if obj._struct_predicted_size is not None:
            return obj._struct_predicted_size
        raise StructSizeUnknown("Struct class contains members of non-fixed size, cannot deduce total struct size before utilization")
    if is_struct(obj):
        return len(obj.pack())
    raise TypeError("obj must be a struct class, struct object or a fixed size member type designator")


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


def make_args(names):
    return [arg(arg=name) for name in names]


def make_defaults(names):
    return [
        Subscript(
          value=Name(id='defaults', ctx=Load()),
          slice=Constant(value=name),
          ctx=Load())
        for name in names]


def make_body(names):
    return [
        Assign(
          targets=[
            Attribute(
              value=Name(id='self', ctx=Load()),
              attr=name,
              ctx=Store())],
          value=Name(id=name, ctx=Load()))
        for name in names]


def make_init_method(members, defaults):
    ast = Module(
      body=[
        FunctionDef(
          name='__init__',
          args=arguments(
            posonlyargs=[arg(arg='self')],
            args=make_args(members),
            kwonlyargs=[],
            kw_defaults=[],
            defaults=make_defaults(defaults.keys())),
          body=make_body(members),
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
                    cls._struct_predicted_size += fixed_size_types[type_]
                except KeyError:
                    cls._struct_predicted_size = None

            if isinstance(type_, Array):
                is_array = True
                length_extractor = type_.length_extractor
                type_ = type_.elt_type
            else:
                is_array = False

            if type_ is Bytes or isinstance(type_, type) and issubclass(type_, CustomMember):
                type_ = type_()

            if is_struct_class(type_):
                unpacker = substruct_unpacker(type_)
                packer = substruct_packer(type_)
            elif isinstance(type_, CustomMember):
                unpacker = type_.unpack
                packer = type_.pack
            elif isinstance(type_, Bytes):
                unpacker = bytes_unpacker(type_.length_extractor)
                packer = bytes_pack
            elif type_ in base_unpackers:
                if type_ is BSON and not have_bson:
                    raise StructDeclarationError("BSON support is not available (try installing pymongo)")
                unpacker = base_unpackers[type_]
                packer = base_packers[type_]
            else:
                raise TypeError(f"{type_!r} is not a valid type designator for struct member {name}")

            if is_array:
                unpacker = array_unpacker(unpacker, length_extractor)
                packer = array_packer(packer)

            if hasattr(cls, name):
                default = getattr(cls, name)
                delattr(cls, name)
                cls._struct_defaults[name] = default
                unpacker = optional_unpacker(unpacker, default)
                processing_optionals = True
            elif processing_optionals:
                raise StructDeclarationError("Optional members, if present, must be specified after all required ones")

            cls._struct_members[name] = unpacker, packer

        cls._struct_endianess = endianess

        add_method(cls)(
                make_init_method(
                    cls._struct_members.keys(),
                    cls._struct_defaults))

        @add_method(cls)
        def __repr__(self):
            members = (f"{name}={getattr(self, name)!r}" for name in cls._struct_members)
            return f"{cls.__name__}({', '.join(members)})"

        @add_method(cls)
        def __bytes__(self):
            return self.pack()

        @add_method(cls)
        @constructor
        def unpack(self, buf, exact=False):
            for name, (unpacker, _) in self._struct_members.items():
                value, buf = unpacker(self, buf, self._struct_endianess)
                setattr(self, name, value)
            if buf and exact:
                raise StructSizeMismatch("Struct unpacking did not consume all of provided data")

        @add_method(cls)
        def pack(self):
            def pack_each():
                for name, (_, packer) in self._struct_members.items():
                    value = getattr(self, name)
                    if value is not None:
                        yield packer(value, self._struct_endianess)
            return b"".join(pack_each())

        return cls

    if cls is None:
        return decorator
    else:
        return decorator(cls)


__all__ = [
    "struct", "sizeof", "is_struct", "is_struct_class",
    "Array", "Bytes", "CustomMember",
    *Endianess.__members__.keys(),
    *Types.__members__.keys()
]


if __name__ == "__main__":
    data = b"*\0\0\0Hello world!\0\x18-DT\xfb!\t@k\xf7\xef\x9c\xc7\xd3/&\xf8A`x\r\x0f\x100\x03Three\0strings\0!\0abcdefg"

    class UInt128(CustomMember):
        def unpack(self, obj, buf, endianess):
            return int.from_bytes(buf[:16], byteorder[endianess], signed=False), buf[16:]

        def pack(self, value, endianess):
            return value.to_bytes(16, byteorder[endianess], signed=False)

    @struct(endianess=Little)
    class MyStruct:
        answer:   UInt32
        greeting: CString
        pi:       Double
        big:      UInt128
        tail_len: UInt8
        tail:     Array(CString, "tail_len")
        tail2:    Bytes(4)
        tail3:    Bytes

    s = MyStruct.unpack(data, exact=True)

    print("Original data:", data)
    print("Decoded object:", s)
    print("Size of object:", sizeof(s))
    print("Re-encoding:", s.pack())
    print("Matches original?", "Yes" if s.pack() == data else "No")
