from twilio.rest import Client
import keys
from ultralytics import YOLO
from datetime import datetime
now = datetime.now()

#Un comment this to perform detection on image or video
# model = YOLO("drone_model.pt")

# src=['im3','im4']

# for i in src:
#     op=model.predict(source=i+".jpg",show=True,conf=0.1)
#     res=op[0].boxes
    
#     if(res.shape[0]==0):
#         print('No Drone')

#     else:
#         msg="Drone detected near your place "
#         current_time = now.strftime("%H:%M:%S")
#         msg= msg+ "at "+str(current_time)

#         client = Client(keys.account_sid, keys.auth_token)
#         msg=client.messages.create(body=msg, from_=keys.twilio_number, to=keys.target_number3)
#         print("Drone Detected")


#This is for real time detection through camera
import cv2
import numpy as np

model = YOLO("drone_model.pt") 

video_capture = cv2.VideoCapture(0)  

if not video_capture.isOpened():
    print("Failed to open video capture")
    exit()

detected=False
try:
    while True:

        ret, frame = video_capture.read()

        if not ret:
            print("Failed to read frame from video capture")
            break

        # frame = preprocess(frame)


        detections = model.predict(source=frame,show=True,conf=0.1)  # Replace with your own detection logic
        print(detections[0].boxes.shape)

        if(detections[0].boxes.shape[0]==0):
            pass

        else:
            if(not detected):
                #detected=True
                msg="Drone detected near your place "
                current_time = now.strftime("%H:%M:%S")
                msg= msg+ "at "+str(current_time)

                client = Client(keys.account_sid, keys.auth_token)
                msg=client.messages.create(body=msg, from_=keys.twilio_number, to=keys.target_number3)
                print("Drone Detected")


        # for detection in detections:
        #     pass
            # bbox, class_label, confidence = extract_info(detection)

        # draw_bbox(frame, bbox, class_label, confidence)

        cv2.imshow('Live Stream', frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


except KeyboardInterrupt:
    video_capture.release()
    cv2.destroyAllWindows()
