import cv2
import mediapipe as mp
import time
import math


class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon,
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None

    def findHands(self, img, draw=True):
        """Detect hands and optionally draw the skeleton."""
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if draw and self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        """Return a list of (id, x, y) pixel positions for ONE hand.
        Call findHands() first on the same frame."""
        lmList = []
        if self.results and self.results.multi_hand_landmarks:
            if handNo < len(self.results.multi_hand_landmarks):
                myHand = self.results.multi_hand_landmarks[handNo]
                lmList = self._landmarksToPixels(img, myHand, draw)
        return lmList

    def findAllPositions(self, img, draw=True):
        """Return a list with one entry per detected hand:
            {"type": "Left" or "Right", "lmList": [(id, x, y), ...]}
        Call findHands() first on the same frame."""
        allHands = []
        if self.results and self.results.multi_hand_landmarks:
            # zip allows us to iterate over both lists in parallel
            for handLms, handedness in zip(self.results.multi_hand_landmarks, 
                                           self.results.multi_handedness):
                handType = handedness.classification[0].label  # "Left" / "Right"
                lmList = self._landmarksToPixels(img, handLms, draw) # Get pixel positions for this hand
                allHands.append({"type": handType, "lmList": lmList})

                if draw:
                    # Label the hand near the wrist
                    _, wx, wy = lmList[0]
                    cv2.putText(img, handType, (wx - 30, wy + 30),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
        return allHands

    def _landmarksToPixels(self, img, handLms, draw):
        """Convert one hand's normalized landmarks to (id, x, y) pixels."""
        lmList = []
        h, w, c = img.shape
        for id, lm in enumerate(handLms.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append((id, cx, cy))
            if draw:
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                cv2.putText(img, str(id), (cx + 5, cy - 5),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        return lmList

    def heart(self, right_hand, left_hand):
        """Check if the hands are making a heart shape."""
        if not right_hand or not left_hand:
            return False

        _, rx_pointer, ry_pointer = right_hand["lmList"][8]  # Right pointer fingertip
        _, lx_pointer, ly_pointer = left_hand["lmList"][8]   # Left pointer fingertip

        _, rx_thumb, ry_thumb = right_hand["lmList"][4]  # Right thumb tip
        _, lx_thumb, ly_thumb = left_hand["lmList"][4]   # Left thumb tip

        pointer_distance = math.hypot(rx_pointer - lx_pointer, ry_pointer - ly_pointer)  # Distance between pointer fingertips
        thumb_distance = math.hypot(rx_thumb - lx_thumb, ry_thumb - ly_thumb)  # Distance between thumb tips

        pointers_above_thumbs = ry_pointer < ry_thumb and ly_pointer < ly_thumb  # Check if both pointer fingertips are above their respective thumb tips

        return pointer_distance < 50 and thumb_distance < 80 and pointers_above_thumbs

    def thumbsUp(self, hand):
        """Check if a single hand is giving a thumbs up."""
        if not hand:
            return False

        landmarks = hand["lmList"]
        _, wrist_x, wrist_y = landmarks[0]
        wrist_position = (wrist_x, wrist_y)

        thumb_tip_y = landmarks[4][2]
        thumb_middle_joint_y = landmarks[3][2]
        thumb_base_y = landmarks[2][2]

        # Thumb points up: tip above its middle joint, which is above its base
        is_thumb_pointing_up = thumb_tip_y < thumb_middle_joint_y < thumb_base_y

        # Thumb tip is the highest point on the whole hand
        is_thumb_highest_point = all(thumb_tip_y <= landmarks[landmark_id][2]
                                     for landmark_id in range(21))

        # Other four fingers curled: each fingertip is closer to the wrist than its middle knuckle
        are_fingers_curled = all(
            math.dist(landmarks[fingertip_id][1:], wrist_position)
            < math.dist(landmarks[fingertip_id - 2][1:], wrist_position)
            for fingertip_id in (8, 12, 16, 20)
        )

        return is_thumb_pointing_up and is_thumb_highest_point and are_fingers_curled
    


def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=2)

    while True:
        success, img = cap.read()
        if not success:
            break

        # Mirror the image so it behaves like a mirror and
        # MediaPipe's "Left"/"Right" labels match your real hands
        img = cv2.flip(img, 1)

        img = detector.findHands(img)
        hands = detector.findAllPositions(img)

        right_hand = next((hand for hand in hands if hand["type"] == "Right"), None)
        left_hand = next((hand for hand in hands if hand["type"] == "Left"), None)

        if detector.heart(right_hand, left_hand):
            cv2.putText(img, "Hands are making a heart", (10, 150),
                        cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

        for hand in hands:

            _, pointer_x, pointer_y = hand["lmList"][8]
            _, thumb_x, thumb_y = hand["lmList"][4]

            color = (0, 255, 0) if hand["type"] == "Right" else (0, 0, 255)
            cv2.circle(img, (pointer_x, pointer_y), 12, color, cv2.FILLED)
            cv2.circle(img, (thumb_x, thumb_y), 12, color, cv2.FILLED)

            if detector.thumbsUp(hand):
                cv2.putText(img, "Thumbs up!", (thumb_x - 60, thumb_y - 30),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)


        # FPS counter
        cTime = time.time()
        fps = 1 / (cTime - pTime) if pTime else 0
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)

        # Quit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # macOS needs this to actually close the window


if __name__ == "__main__":
    main()