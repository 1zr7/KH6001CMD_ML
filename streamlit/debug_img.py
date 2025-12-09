import os
import base64

path = r"c:\Users\h1zr7\OneDrive\Desktop\anti gravity\test\background_magma.jpg"

print(f"Checking path: {path}")
if os.path.exists(path):
    print("File exists.")
    try:
        with open(path, "rb") as f:
            data = f.read()
        print(f"Read {len(data)} bytes.")
        enc = base64.b64encode(data).decode()
        print(f"Encoded length: {len(enc)}")
        # Check first few chars
        print(f"Start: {enc[:50]}")
    except Exception as e:
        print(f"Error reading/encoding: {e}")
else:
    print("File NOT found.")
