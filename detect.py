import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
from retinaface import RetinaFace
import json
from PIL import Image
import pandas as pd
from datetime import datetime
from deepface import DeepFace
from concurrent.futures import ThreadPoolExecutor


def save_faces(image_path, face_data):
    img = Image.open(image_path)
    for face_id, face_info in face_data.items():
        # Extracting face coordinates
        x1, y1, x2, y2 = map(int, face_info["facial_area"])
        # Cropping the face region
        face_img = img.crop((x1, y1, x2, y2))
        # Saving the cropped face image
        face_img.save(f"./faces/{face_id}.jpg")
        print(f"Face {face_id} saved successfully.")


image_path = "test6.jpeg"

# Call the layer to detect faces
resp = RetinaFace.detect_faces(image_path)

# Now `resp` contains the detected faces
save_faces(image_path, resp)


newFaces = os.listdir("./faces")
verifiedFaces = os.listdir("./database")

attendance = []

for data in newFaces:
    temp = 1
    s = ""
    for images in verifiedFaces:
        result = DeepFace.verify(img1_path = f"./faces/{data}", img2_path = f"./database/{images}",enforce_detection=False)
        if result['verified'] and result['distance'] < temp:
            temp = result['distance']
            s = images
    file_name = s.split('.')[0]
    attendance.append(file_name)

print(attendance)

# def verify_faces_batch(newFaces, verifiedFaces, start_index, end_index):
#     attendance_batch = []
#     for data in newFaces[start_index:end_index]:
#         temp = 1
#         s = ""
#         for images in verifiedFaces:
#             result = DeepFace.verify(img1_path=f"./faces/{data}", img2_path=f"./database/{images}", enforce_detection=False)
#             if result['verified'] and result['distance'] < temp:
#                 temp = result['distance']
#                 s = images
#         file_name = s.split('.')[0]
#         attendance_batch.append(file_name)
#     return attendance_batch

# if __name__ == "__main__":
#     newFaces = os.listdir("./faces")
#     verifiedFaces = os.listdir("./database")
#     attendance = []

#     batch_size = len(newFaces) // 2

#     with ThreadPoolExecutor(max_workers=2) as executor:
#         futures = [executor.submit(verify_faces_batch, newFaces, verifiedFaces, i * batch_size, (i + 1) * batch_size) for i in range(2)]

#         for future in futures:
#             attendance.extend(future.result())

#     print(attendance)


df = pd.read_csv("attendance.csv")
current_date = datetime.now().strftime('%Y-%m-%d')
df[current_date] = 0
all_students = df.iloc[:,0].tolist()
for data in attendance:
  ind = all_students.index(data)
  df.loc[ind,current_date] = 1
df.to_csv("attendance.csv",index=False)

