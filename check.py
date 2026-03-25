import os

for root, _, files in os.walk("Agents"):
    for f in files:
        if f.endswith(".py"):
            path = os.path.join(root, f)
            with open(path, "rb") as file:
                data = file.read()
                if b"\x00" in data:
                    print("CORRUPTED:", path)