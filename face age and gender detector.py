import cv2
from random import randrange




face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)


MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']


vid = cv2.VideoCapture(0)



while True:
    frame_read, frame = vid.read(0) 

    padding = 20

    blc_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    coord = face_data.detectMultiScale(frame)  
    
    blob=cv2.dnn.blobFromImage(frame, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
    
    genderNet.setInput(blob)
    genderPreds=genderNet.forward()
    gender=genderList[genderPreds[0].argmax()]
    print(f'Gender: {gender}')

    ageNet.setInput(blob)
    agePreds=ageNet.forward()
    age=ageList[agePreds[0].argmax()]
    print(f'Age: {age[1:-1]} years') 

    


    color = (randrange(255), randrange(255), randrange(255))
    

    for [x, y, w, h,] in coord:
        cv2.rectangle(frame, (x, y), (x+w , y+h), (randrange(255), randrange(255), randrange(255)), 4)
        data = (x, y)
        cv2.putText(frame, f'{gender}, {age}', data, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


    
    
    cv2.imshow('Face Detector', frame)
    k=cv2.waitKey(1)

    if k == ord('q'):
     break

    print(coord)

vid.release()
cv2.destroyAllWindows()
    

    
    


    














