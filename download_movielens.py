import urllib.request
import zipfile
import os

url = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
dest = "ml-25m.zip"

print("Downloading MovieLens 25M (~250 MB)...")
urllib.request.urlretrieve(url, dest)

print("Extracting...")
with zipfile.ZipFile(dest, "r") as z:
    z.extractall(".")

os.remove(dest)
print("Done — ml-25m/ folder created")
