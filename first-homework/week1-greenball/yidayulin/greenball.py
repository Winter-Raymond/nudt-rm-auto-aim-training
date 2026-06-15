import cv2 as cv
import numpy as np
from collections import deque
cap = cv.VideoCapture('nudt-rm-auto-aim-training/first-homework/first-homework.mp4')

if not cap.isOpened():
    print("YIDAYULIN ERROR:NOT OPEN")
    exit()

history = deque(maxlen=10)  
prev_center = None

while True:
    ret, frame = cap.read()
    

    if not ret:
        break
    result_frame = frame.copy()

    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    lower_green = np.array([35, 110, 50])
    upper_green = np.array([85, 255, 255])
    
    green_mask = cv.inRange(hsv_frame, lower_green, upper_green)
    
    green_mask = cv.erode(green_mask, None, iterations=2)
    green_mask = cv.dilate(green_mask, None, iterations=2)
    
   
    contours, _ = cv.findContours(green_mask, cv.RETR_EXTERNAL, 
                                    cv.CHAIN_APPROX_SIMPLE)
    green_mask_ball=np.zeros_like(green_mask)

    current_center = None
    
    for contour in contours:
        area = cv.contourArea(contour)
        #perimeter = cv.arcLength(contour, True)  //maybe not a circle

        if(area <100):
            continue

        ellipse = cv.fitEllipse(contour)
        (cx, cy), (major, minor), angle = ellipse
    
        ratio = max(major, minor) / (min(major, minor) + 1e-6)
        #if 4*np.pi*area/(perimeter**2)>7.1:
        if ratio <1.2:
            cv.drawContours(green_mask_ball, [contour], -1, 255, -1)
            cv.drawContours(result_frame, [contour], -1, (0, 255, 0), 2)
                
            current_center = (int(cx), int(cy))  # cx, cy 来自 cv.fitEllipse()
            cv.circle(result_frame, current_center, 5, (0, 0, 255), -1)
            
            if len(history) >= 2:
                    
                if len(history) >= 5:
                    dx = current_center[0] - history[-5][0]
                    dy = current_center[1] - history[-5][1]
                else:
                    dx = current_center[0] - history[-1][0]
                    dy = current_center[1] - history[-1][1]
                    
                speed = np.sqrt(dx**2 + dy**2)
                    
                    
                if speed > 0.01: 
                    arrow_length = min(80, int(speed * 2))
                        
                       
                    direction_x = dx / speed
                    direction_y = dy / speed
                        
                        
                    end_point = (int(current_center[0] + direction_x * arrow_length),
                                    int(current_center[1] + direction_y * arrow_length))
                        
                        
                    cv.arrowedLine(result_frame, current_center, end_point,(255, 0, 0), 2, cv.LINE_AA, tipLength=0.3)
                           
            history.append(current_center)
            

    cv.imshow('video', result_frame)
    cv.imshow('green mask', green_mask_ball)
    
    if cv.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()