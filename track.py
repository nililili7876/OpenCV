from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2

#设置输入选择
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf",help="OpenCV object tracker type")
args = vars(ap.parse_args())


OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        #"boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        #"tld": cv2.TrackerTLD_create,
        #"medianflow": cv2.TrackerMedianFlow_create,
        #"mosse": cv2.TrackerMOSSE_create
    }
tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()


# 抓取相机或读本地视频
if not args.get("video", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)
else:
    vs = cv2.VideoCapture(args["video"])

#初始化目标框和第一帧
initBB = None
fps = None


while True:
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame
    if frame is None:
        break
    #resize the frame
    frame = imutils.resize(frame, width=500)
    (H, W) = frame.shape[:2]
    
    #开始追踪
    if initBB is not None:
        (success, box) = tracker.update(frame)
        #画框
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h),(0, 255, 0), 2)
        fps.update()
        fps.stop()
        info = [
            ("Tracker", args["tracker"]),
            ("Success", "Yes" if success else "No"),
            ("FPS", "{:.2f}".format(fps.fps())),
        ]
        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF 
    # 按s进行目标选取
    if key == ord("s"):
        # select ROI
        initBB = cv2.selectROI("Frame", frame, fromCenter=False,showCrosshair=True)
        # start OpenCV object tracker using the supplied bounding box coordinates
        tracker.init(frame, initBB)
        fps = FPS().start()
        
    #ESC退出
    elif key == 27:
        break

#释放
if not args.get("video", False):
    vs.stop()
else:
    vs.release()
cv2.destroyAllWindows()