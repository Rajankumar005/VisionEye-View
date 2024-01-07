#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
#                                                VisionEye Object Mapping                                                        #
#                                                                                                                                #
#      Ultralytics YOLOv8 VisionEye offers the capability for computers to identify and pinpoint objects,                        #
#      simulating the observational precision of the human eye. This functionality enables computers to discern                  #
#      and focus on specific objects, much like the way the human eye observes details from a particular viewpoint.              #
#                                                                                                                                #         
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
#                                                                                                                                #
#                                                       Import Libary from Ultralytics                                           #
#                                                                                                                                #
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#

from ultralytics import YOLO
from ultralytics.solutions import heatmap
from ultralytics.utils.plotting import colors, Annotator
import cv2
try:
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
except:pass

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
#                                                                                                                                #
#                                                       Load Model From Ultralytics                                              #
#                                                       Load Video as an Inputs                                                  #
#                                                                                                                                #
# Model Link : https://docs.ultralytics.com/models/yolov8/#supported-tasks-and-modes                                             #
# Video Link : https://www.pexels.com                                                                                            #
#                                                                                                                                #
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("bycycle.mp4")

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
#                                                                                                                                #
#                                                        Video Writer                                                            #
#                                                                                                                                #
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#

out = cv2.VideoWriter('visioneye-pinpoint.avi', cv2.VideoWriter_fourcc(*'MJPG'),
                      30, (int(cap.get(3)), int(cap.get(4))))

center_point = (-10, int(cap.get(4)))

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
#                                                                                                                                #
#                                                       Read Frame from Inputs                                                   #
#                                                                                                                                #
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#

while True:
    ret, im0 = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    annotator = Annotator(im0, line_width=2)

    results = model.track(im0, persist=True)
    boxes = results[0].boxes.xyxy.cpu()

    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            annotator.box_label(box, label=str(track_id), color=colors(int(track_id)))
            annotator.visioneye(box, center_point)

    out.write(im0)
    cv2.imshow("visioneye-pinpoint", im0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
#                                                                                                                                #
#                                           Release All Window After Complete Execution                                          #
#                                                                                                                                #
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
    
out.release()
cap.release()
cv2.destroyAllWindows()

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
#                                                                                                                                #
#                                           Special Thanks For *** Ultralytics ***                                               #
#                                                                                                                                #
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#