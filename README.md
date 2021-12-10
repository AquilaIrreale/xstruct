# xstruct

This module provides a solution to serialize, deserialize and represent
packed binary data in a declarative manner. It is built on top and
extends the capabilities of the standard `struct` module while
presenting an higher-level object oriented interface similar to that of
the familiar `@dataclass` decorator. It has optional support for
embedded BSON data structures.

Now available on [PyPI](https://pypi.org/project/xstruct/) and
[GitHub](https://github.com/AquilaIrreale/xstruct)

## Installation

The following command should work on most systems:

```sh
$ pip install xstruct
```

## Usage

This module provides a class decorator with an interface similar to
`dataclasses.dataclass`.

### Basic usage

```python3
from xstruct import struct, UInt16, Big

@struct(endianess=Big)
class UDPHeader:
    src_port: UInt16
    dst_port: UInt16
    length:   UInt16
    checksum: UInt16
```

`xstruct` provides pseudo-types for common signed and unsigned integer
sizes, IEEE-754 floating point numbers, NUL terminated strings and
optionally BSON documents.

Struct objects can be created by decoding binary data from a
`bytes`-like object...

```python3
>>> UDPHeader.unpack(b"\x00\x00 1\x00\x00\x00{\x00\x00\x00L\x00\x00\x00\x00")
UDPHeader(src_port=8241, dst_port=123, length=76, checksum=0)
```

...or through the generated constructor, and can be serialized back to
binary data.

```python3
>>> UDPHeader(src_port=8241, dst_port=123, length=76, checksum=0).pack()
b'\x00\x00 1\x00\x00\x00{\x00\x00\x00L\x00\x00\x00\x00'
```

### Optional members

Default values can be specified for members at the tail end of a struct.

```python
from xstruct import struct, Int32, Little

@struct(endianess=Little)
class SecondsOptional:
    member1: Int32
    member2: Int32 = 0
```

If a buffer ends prematurely during decoding, default values are used in
place of missing struct members.

```python
>>> SecondsOptional.decode(b"*\0\0\0")
SecondsOptional(member1=42, member2=0)
```

Members with a default value can also be omitted when creating a struct
through the generated constructor.

```python
>>> SecondsOptional(42)
SecondsOptional(member1=42, member2=0)
```

### Struct inclusion

Structs can include other structs

```python3
from xstruct import struct, Int32, Int64, CString, Big

@struct(endianess=Big)
class Header:
    src:      Int32
    msg_type: Int32
    upd_time: Int64

@struct(endianess=Big)
class Message:
  header: Header
  msg:    CString
```

Encoding and decoding `Message` will work as expected.

### Struct endianess

The endianess of the numeric members of the struct can optionally be
selected by providing the `endianess` argument to the `struct`
decorator. Valid options are `Little`, `Big` and `Native`. When left
unspecified, endianess defaults to `Native`.

## Future work

As of now, this library is fairly complete and in an usable state.
Future work could add support for:

- [ ] Decoding network addresses (IPv4, IPv6, MAC) to strings
- [ ] Pascal strings
- [ ] Arrays
- [ ] Fixed width embedded binary payloads
- [ ] Tail end padding
- [ ] Decoding total struct size from/to designated member
