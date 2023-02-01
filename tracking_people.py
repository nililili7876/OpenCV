import cv2
import mediapipe as mp
 
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
 
 
 
def video():
    # cap = cv2.VideoCapture(0) # 用于摄像头追踪
    cap = cv2.VideoCapture('')  # 用于视频文件追踪
 
 
 
# Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
 
            ret, frame = cap.read()
 
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
 
            # Make detection
            results = pose.process(image)
 
            # Recolor image to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
 
            # Extrack landmarks
 
            if results.pose_landmarks is None:
                continue
 
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imshow("Mediapipe Feed", image)
 
            if cv2.waitKey(1) & 0xFF == ord('q'): # 按 q 键退出
                break
 
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
 
 
if __name__ == '__main__':
    video()
    print("End!")