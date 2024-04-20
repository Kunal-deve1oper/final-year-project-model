import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
from retinaface import RetinaFace
import json
from PIL import Image
import pandas as pd
from datetime import datetime
import time
from facenet_pytorch import MTCNN, InceptionResnetV1

start_time = time.time()

def save_faces(image_path, face_data, expansion_factor=1.07):
    img = Image.open(image_path)
    for face_id, face_info in face_data.items():
        # Extracting face coordinates
        x1, y1, x2, y2 = map(int, face_info["facial_area"])
        # Calculating expanded bounding box
        width = x2 - x1
        height = y2 - y1
        expand_width = int(width * expansion_factor)
        expand_height = int(height * expansion_factor)
        expanded_x1 = max(0, x1 - (expand_width - width) // 2)
        expanded_y1 = max(0, y1 - (expand_height - height) // 2)
        expanded_x2 = min(img.width, x2 + (expand_width - width) // 2)
        expanded_y2 = min(img.height, y2 + (expand_height - height) // 2)
        # Cropping the expanded face region
        face_img = img.crop((expanded_x1, expanded_y1, expanded_x2, expanded_y2))
        # Saving the cropped face image
        face_img.save(f"./faces/{face_id}.jpg")
        print(f"Face {face_id} saved successfully.")


def verify_face(img1_path, img2_path):
    resnet = InceptionResnetV1(pretrained='casia-webface').eval()

    mtcnn = MTCNN()

    img1 = Image.open(img1_path)
    w1,h1 = img1.size
    if w1 < 60 or h1 < 60:
        return -1
    img2 = Image.open(img2_path)
    w2,h2 = img1.size

    aligned1 = mtcnn(img1)
    if aligned1 == None:
        return -1
    aligned1 = aligned1.unsqueeze(0)
    embeddings1 = resnet(aligned1).detach()
    aligned2 = mtcnn(img2)
    aligned2 = aligned2.unsqueeze(0)
    embeddings2 = resnet(aligned2).detach()

    distance = (embeddings1 - embeddings2).norm().item()

    return distance


image_path = "test6.jpeg"

# Call the layer to detect faces
resp = RetinaFace.detect_faces(image_path)

# Now `resp` contains the detected faces
save_faces(image_path, resp)


newFaces = os.listdir("./faces")
verifiedFaces = os.listdir("./database")

attendance = []

for data in newFaces:
    temp = 5
    s = ""
    for images in verifiedFaces:
        distance = verify_face(f"./faces/{data}",f"./database/{images}")
        if distance == -1:
            s = "no.jpg"
            break
        if distance < temp:
            temp = distance
            s = images
    file_name = s.split('.')[0]
    attendance.append(file_name)
    print(data+ "->" +file_name)

print(attendance)


df = pd.read_csv("attendance.csv")
current_date = datetime.now().strftime('%Y-%m-%d')
df[current_date] = 0
all_students = df.iloc[:,0].tolist()
for data in attendance:
    if data == "no":
        continue
    ind = all_students.index(data)
    df.loc[ind,current_date] = 1
df.to_csv("attendance.csv",index=False)

end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")