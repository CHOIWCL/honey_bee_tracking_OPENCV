import numpy as np, cv2

cap = cv2.VideoCapture('1.mp4')
fps = cap.get(cv2.CAP_PROP_FPS) # 프레임 수 구하기

delay = int(1000/fps)
# 배경 제거 객체 생성 --- ①
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # 배경 제거 마스크 계산 --- ②
    fgmask = fgbg.apply(frame)
    
    contours, hier = cv2.findContours(fgmask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    
    
    for i in range(len(contours)):
        img = cv2.drawContours(frame, contours, i, (0,255,255), 2)
        
        x,y,w,h = cv2.boundingRect(contours[i])
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,255), 2)
    
    cv2.imshow('frame',frame)
    cv2.imshow('bgsub',fgmask)
    if cv2.waitKey(1) & 0xff == 27:
        break
cap.release()
cv2.destroyAllWindows()
