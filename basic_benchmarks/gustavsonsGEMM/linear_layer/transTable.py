import struct

for val in range (0, 11):
    ba = bytearray(struct.pack(">f", val))
    my_int = int.from_bytes(ba, "big")
    print(f"{val} -> {str(my_int)}")