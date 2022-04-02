# Imports the Google Cloud client library
from google.cloud import storage
import io
import os

# Imports the Google Cloud client library
from google.cloud import vision


name = "input_folder/gato.jpg"
# Instantiates a client

print(f"Download blob from bucket {name}")
storage_client = storage.Client()
bucket = storage_client.bucket("ml2_bucket")
blob = bucket.blob(name)

# Instantiates a client
print(f"Apply vision API... {name}")
client = vision.ImageAnnotatorClient()
content = blob.download_as_bytes()
image = vision.Image(content=content)

# Performs label detection on the image file
response = client.label_detection(image=image)
labels = response.label_annotations

result = ""
for label in labels:
    result += str(label.description)
    result += "\n"
print(f"write labels {result}...{name}")
print(f"upload result to bucket {name}")
blob_dst = bucket.blob("output_folder/" + name.split('/')[1].split(".")[0] + ".vision.txt")
blob_dst.upload_from_string(result)

blob.delete()
print("Blob {} deleted.".format(name))