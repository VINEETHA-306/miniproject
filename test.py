import cv2
import os
import numpy as np
import mini as mn
test_img=cv2.imread('C:/Users/Vineetha/Pictures/IMG-20190323-WA0005.jpg')
faces_detected,gray_img=mn.faceDetection(test_img)
print("faces_detected:",faces_detected)

for (x,y,w,h) in faces_detected:
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=5)

resized_img=cv2.resize(test_img,(1000,700))
cv2.imshow("face detected",resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows

faces,faceID=mn.labels_for_training_data('C:/Users/Vineetha/Pictures/prj')
face_recognizer=mn.train_classifier(faces,faceID)
face_recognizer.save('tariningData.xml')
name={1:"priyanka",2:"prabhas"}

cap=cv2.VideoCapture(0)

while True:
    ret,test_img=cap.read()
   faces_detected,gray_img=mn.faceDetection(test_img)

    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=3)
    resized_img=cv2.resize(test_img,(1000,700))
    cv2.waitKey(10)
    
    for face in faces_detected:
        (x,y,w,h)=face
        roi_gray=gray_img[y:y+h,x:x+w]
        label,confidence=face_recognizer.predict(roi_gray)
        print("confidence:",confidence)
        print("label:",label)
        mn.time_frame()
        mn.draw_rect(test_img,face)
        predicted_name=name[label]
        mn.put_text(test_img,predicted_name,x,y)
    resized_img=cv2.resize(test_img,(1000,700))
    cv2.imshow("face detection",resized_img)
    if cv2.waitKey(10)==ord('q'):
        break
cap.release()
cv2.destroyAllWindows
