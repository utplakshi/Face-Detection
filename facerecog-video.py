import os
import cv2
import face_recognition

#import the video file

video_file=cv2.VideoCapture(os.path.abspath("Video/yjhd.mp4"))

#Capture the length based on the frame.

length=int(video_file.get(cv2.CAP_PROP_FRAME_COUNT))

#Adding all the faces in the video

image_deepika = face_recognition.load_image_file(os.path.abspath("Images/Naina1.png"))
image_ranbir = face_recognition.load_image_file(os.path.abspath("Images/Kabir1.png"))
image_ranaDaggubati = face_recognition.load_image_file(os.path.abspath("Images/Photographer1.png"))

#Generate Face Encoding for the image

deepika_face = face_recognition.face_encodings(image_deepika)[0]
ranbir_face = face_recognition.face_encodings(image_ranbir)[0]
ranaDaggubati_face = face_recognition.face_encodings(image_ranaDaggubati)[0]

#Make a list of all known faces that we want to recognize on basis of the encoding
known_faces=[deepika_face, ranbir_face, ranaDaggubati_face]

facial_points= []
face_encodings=[]
facial_number=0

while True:
    return_value, frame = video_file.read()
    facial_number = facial_number + 1
    
    if not return_value:
        break
    rgb_frame = frame[:, :, ::-1]

    facial_points = face_recognition.face_locations(rgb_frame, model="cnn")
    face_encodings = face_recognition.face_encodings(rgb_frame, facial_points)

    facial_names = []
    for encoding in face_encodings:
        match = face_recognition.compare_faces(known_faces, encoding, tolerance=0.50)
        # deepika_face, ranbir_face, ranaDaggubati_face


        name = ""
        if match[0]:
            name = "Deepika"
        if match[1]:
            name = "Ranbir"
        if match[2]:
            name = "Rana Daggubati"

        facial_names.append(name)

    for (top, right, bottom, left), name in zip(facial_points, facial_names):
        # Enclose the face with the box - Red color 
        # top, right, bottom, left - 129, 710, 373, 465
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Name the characters in the Box created above
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    codec = int(video_file.get(cv2.CAP_PROP_FOURCC))
    fps = int(video_file.get(cv2.CAP_PROP_FPS))
    frame_width = int(video_file.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_file.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_movie = cv2.VideoWriter("output_{}.mp4".format(facial_number), codec, fps, (frame_width,frame_height))
    print("Writing frame {} / {}".format(facial_number, length))
    output_movie.write(frame)

video_file.release()
output_movie.release()
cv2.destroyAllWindows()