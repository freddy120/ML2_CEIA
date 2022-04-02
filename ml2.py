import os
import io
from google.cloud import vision
import time

while True:
    time.sleep(1)

    os.system("rm input_files_names.txt")
    os.system("gsutil ls gs://ml2_bucket/input_folder >> input_files_names.txt")

    file_to_read = open("input_files_names.txt")

    for line in file_to_read:
        line = line.replace("\n", "")
        if line == "gs://ml2_bucket/input_folder/":
            continue
        else:
            os.system("gsutil cp {} .".format(line))
            line = line.split("/")[-1]
            print("working with {}".format(line))

            # Instantiates a client
            client = vision.ImageAnnotatorClient()

            # The name of the image file to annotate
            file_name = os.path.abspath(line)

            # Loads the image into memory
            with io.open(file_name, 'rb') as image_file:
                content = image_file.read()

            image = vision.Image(content=content)

            # Performs label detection on the image file
            response = client.label_detection(image=image)
            labels = response.label_annotations

            file_to_write = open(line.split(".")[0] + ".txt", "w")
            print('writting:')
            for label in labels:
                file_to_write.write(str(label.description))
                file_to_write.write("\n")
            else:
                print("finished \n \n")

                file_to_write.close()
                # print("cp {} gs://ml2_uba/output_folder/".format(line)")
                os.system("gsutil cp {} gs://ml2_bucket/output_folder/{}".format(line.split(".")[0] + ".txt",
                                                                              line.split(".")[0] + ".txt"))
                # os.system("rm {}".format(line))
    break