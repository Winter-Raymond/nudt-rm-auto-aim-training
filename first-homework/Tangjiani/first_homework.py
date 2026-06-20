import cv2
import math

cap = cv2.VideoCapture("nudt-rm-auto-aim-training-ef1c6d117a2e580f59cf4653a77919456bc17b05/first-homework/first-homework.mp4")

old_x = None
old_y = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 60, 50), (85, 255, 255))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    balls = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue

        (x, y), r = cv2.minEnclosingCircle(cnt)
        if r <= 0:
            continue

        circle_area = math.pi * r * r
        ratio = area / circle_area
        if ratio < 0.70:
            continue

        x, y, r = int(x), int(y), int(r)
        balls.append((ratio, x, y, r))

    if balls:
        balls.sort(reverse=True)
        _, x, y, r = balls[0]
        cv2.circle(frame, (x, y), r, (0, 0, 255), 2)

        if old_x is not None:
            dx = x - old_x
            dy = y - old_y
            speed = math.sqrt(dx * dx + dy * dy)

            scale = 2.5
            end_x = int(x + dx * scale)
            end_y = int(y + dy * scale)
            cv2.arrowedLine(frame, (x, y), (end_x, end_y),
                            (0, 0, 255), 2, tipLength=0.3)

        old_x, old_y = x, y

    cv2.imshow("Video", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty("Video", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
