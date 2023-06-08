import math

import autopy.mouse as automouse
import autopy.screen as autoscreen
import cv2
import mediapipe.python.solutions.drawing_utils as mp_draw
import mediapipe.python.solutions.hands as mp_hands
import numpy as np
import pyautogui

width, height = 640, 480
wScr, hScr = autoscreen.size()
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)
pyautogui.FAILSAFE = False


def get_X_Y(idlist):
    return idlist[1:]  # [4(thumb), 263(x), 298(y)]


def instruct(frame):
    cv2.putText(
        frame,
        "Touch your index finger joint with thumb to click",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
    )
    cv2.putText(
        frame,
        "Touch your index finger joint with thumb to click",
        (8, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (128, 0, 200),
        2,
    )
    cv2.putText(
        frame,
        "Press 'x' to exit",
        (455, 470),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )
    return frame


def main():
    preX, preY = 0, 0
    currX, currY = 0, 0

    with mp_hands.Hands(
        max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5
    ) as hands:
        while True:
            _, frame = cap.read()
            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            tips = [4, 5, 8, 10, 12]
            fingers = []

            if results.multi_hand_landmarks:
                myhand = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, myhand, mp_hands.HAND_CONNECTIONS)

                for i, id in enumerate(tips):
                    lm = myhand.landmark[id]
                    h, w, _ = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    fingers.append([i, cx, cy])

                x1, y1 = get_X_Y(fingers[0])
                x2, y2 = get_X_Y(fingers[1])
                x3, y3 = get_X_Y(fingers[2])
                x4, y4 = get_X_Y(fingers[3])
                x5, y5 = get_X_Y(fingers[4])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                cv2.circle(frame, (x1, y1), 10, (255, 255, 0), 5)
                cv2.circle(frame, (x2, y2), 10, (255, 255, 0), 5)
                cv2.circle(frame, (x3, y3), 10, (255, 255, 0), 5)
                cv2.circle(frame, (x5, y5), 10, (255, 255, 0), 5)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                length = math.hypot(x2 - x1, y2 - y1)
                if 15 < length < 25:
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 225), -1)
                    cv2.circle(frame, (cx, cy), 10, (0, 255, 255), 5)
                    if y5 < y4:
                        pyautogui.click(button="right")
                    else:
                        pyautogui.click()

                scrollen = math.hypot(x5 - x3, y5 - y3)
                if scrollen < 20 and y5 < y4:
                    pyautogui.scroll(30, _pause=False)
                elif scrollen < 20:
                    pyautogui.scroll(-30, _pause=False)
                else:
                    con_index_x = np.interp(x3, (100, width - 100), (0, wScr))
                    con_index_y = np.interp(y3, (100, height - 100), (0, hScr))

                    currX = preX + (con_index_x - preX) / 13
                    currY = preY + (con_index_y - preY) / 13
                    automouse.move(currX, currY)
                    preX, preY = currX, currY
                    cv2.rectangle(
                        frame, (100, 100), (width - 100, height - 100), (0, 0, 255), 5
                    )

            frame = instruct(frame)
            cv2.imshow("AI Virtual Mouse", frame)
            if cv2.waitKey(10) == ord("x"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
