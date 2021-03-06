import face_recognition
import time
import os
import cv2
import pickle

known_faces_dir = "id_folder"

tolerance = 0.5
frame_thickness = 3
font_thickness = 2
model = "cnn"

known_faces = []
known_names = []

print("loading known faces")

for name in os.listdir(known_faces_dir): 
  for filename in os.listdir(f"{known_faces_dir}/{name}"):
    encoding = pickle.load(open(f"{name}/{filename}", "rb"))
    known_faces.append(encoding)
    known_names.append(int(name))
if len(known_names) > 0:
  next_id = max(known_names) + 1
else:
  next_id = 0
  
print("loading video")
video = cv2.VideoCapture('faces.mp4')
while True:
  ret, image = video.read()
  locations = face_recognition.face_locations(image, model=model)
  encodings = face_recognition.face_encodings(image, locations)
  for face_encoding, face_location in zip(encodings, locations):
    results = face_recognition.compare_faces(known_faces, face_encoding, tolerance)
    match = None
    if True in results:
      match = known_names[results.index(True)]
      print(f"match found: {match}")
    else:
      match = str(next_id)
      known_names.append(match)
      known_faces.append(face_encoding)
      next_id += 1
      os.mkdir(f"{known_faces_dir}/{match}")
      
      pickle.dump(face_encoding, open(f"{known_faces_dir}/{match}/{match}-{int(time.time())}.pkl", "wb"))
      
    top_left = (face_location[3], face_location[0])
    bottom_right = (face_location[1], face_location[2])
    color = [0, 255, 0] 
    cv2.rectangle(image, top_left, bottom_right, color, frame_thickness)
    top_left = (face_location[3], face_location[2])
    bottom_right = (face_location[1], face_location[2]+22)
    cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
    cv2.putText(image, match, (face_location[3]+10, face_location[2]+15),
    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), font_thickness)
  cv2.imshow("", image)
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break