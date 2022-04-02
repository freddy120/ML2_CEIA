# Imports the Google Cloud client library
from google.cloud import storage
import io
import os

# Imports the Google Cloud client library
from google.cloud import vision

def ml2_vision(event, context):
    """Background Cloud Function to be triggered by Cloud Storage.
       This generic function logs relevant data when a file is changed.
    Args:
        event (dict):  The dictionary with data specific to this type of event.
                       The `data` field contains a description of the event in
                       the Cloud Storage `object` format described here:
                       https://cloud.google.com/storage/docs/json_api/v1/objects#resource
        context (google.cloud.functions.Context): Metadata of triggering event.
    Returns:
        None; the output is written to Stackdriver Logging
    """

    print('Event ID: {}'.format(context.event_id))
    print('Event type: {}'.format(context.event_type))
    print('Bucket: {}'.format(event['bucket']))
    print('File: {}'.format(event['name']))
    print('Metageneration: {}'.format(event['metageneration']))
    print('Created: {}'.format(event['timeCreated']))
    print('Updated: {}'.format(event['updated']))

    name = event['name']
    # Instantiates a client

    if 'input_folder' in name:
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