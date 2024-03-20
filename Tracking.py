import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.solutions import speed_estimation
from collections import defaultdict
from time import time
import easyocr
from Comp import calculate_optical_flow as Optical_Flow
import Util_Methods as um

# Load the YOLOv8 model
model = YOLO('Models/yolov8x.pt')
model = YOLO('Models/plate_req.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)

names = model.model.names

# Open the video file
video_path = "https://camstream.kahramanmaras.bel.tr/live/ulucami.stream/playlist.m3u8"
video_path = "Video/khrmmrs.avi"
video_path = "Video/trafficFlow.webm"
video_path = "Video/plate_req.mp4"
cap = cv2.VideoCapture(video_path)

cv2.namedWindow('AI-KGSY', cv2.WINDOW_NORMAL)
cv2.setWindowProperty("AI-KGSY", cv2.WND_PROP_FULLSCREEN, cv2.WND_PROP_AUTOSIZE)
fps = cap.get(cv2.CAP_PROP_FPS)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

print(f"{width}x{height}")
# Store the track history
track_history = defaultdict(lambda: [])
count_of_frame = 0
left_line = {}
Get_Photo_ID = []
Get_Photo_Guilty_ID = []
cut_out_license = None
prev_frame = None

# Create an OCR reader instance
reader = easyocr.Reader(['en'])

Draw_Track_Dots = True
frame_skip_count = 1 

ROI1x1,ROI1y1,ROI1x2,ROI1y2,ROI1x3,ROI1y3,ROI1x4,ROI1y4 = 850, 350, 890, 305, 1050, 305, 1035, 370


ROI1x1,ROI1y1,ROI1x2,ROI1y2,ROI1x3,ROI1y3,ROI1x4,ROI1y4 = 1060, 30, 1125, 30, 1120, 75, 1035, 75

ROI1x1,ROI1y1,ROI1x2,ROI1y2,ROI1x3,ROI1y3,ROI1x4,ROI1y4 = 770, 600, 1280, 600, 1280, 720, 770, 720


ROI1x1,ROI1y1,ROI1x2,ROI1y2,ROI1x3,ROI1y3,ROI1x4,ROI1y4,ROI1x5,ROI1y5 = 570, 675, 920, 800, 900, 1080, 690, 1070, 490, 950 #### Bottom Walk-Way ROI


ROI2x1,ROI2y1,ROI2x2,ROI2y2,ROI2x3,ROI2y3,ROI2x4,ROI2y4 = 570, 675, 575, 605, 865, 605, 920, 800 #### Bottom Walk-Way Upper-Side ROI


ROI3x1,ROI3y1,ROI3x2,ROI3y2,ROI3x3,ROI3y3,ROI3x4,ROI3y4 = 900, 1080, 690, 1070, 490, 950, 450, 1080 #### Bottom Walk-Way Upper-Side ROI

Bottom_Walk_Way_ROI_np = np.array([(ROI1x1,ROI1y1),(ROI1x2,ROI1y2),(ROI1x3,ROI1y3),(ROI1x4,ROI1y4)], dtype=np.int32)

Bottom_Walk_Way_ROI_np = np.array([(ROI1x1,ROI1y1),(ROI1x2,ROI1y2),(ROI1x3,ROI1y3),(ROI1x4,ROI1y4),(ROI1x5,ROI1y5)], dtype=np.int32)

Bottom_Walk_Way_Upper_Side_ROI_np = np.array([(ROI2x1,ROI2y1),(ROI2x2,ROI2y2),(ROI2x3,ROI2y3),(ROI2x4,ROI2y4)], dtype=np.int32)
Bottom_Walk_Way_Bottom_Side_ROI_np = np.array([(ROI3x1,ROI3y1),(ROI3x2,ROI3y2),(ROI3x3,ROI3y3),(ROI3x4,ROI3y4)], dtype=np.int32)


um.empty_folder("Photos")
um.create_folder_if_not_exists("Photos")

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    count_of_frame += 1
    if count_of_frame % frame_skip_count != 0:
        continue
    
    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        base_frame = frame.copy()
        if prev_frame is not None and False:
            optical_Flow_frame = Optical_Flow(prev_frame,frame,maxCorners=1000,qualityLevel=0.1,minDistance=1)
            cv2.imshow("AI-KGSY-Optical-Flow", optical_Flow_frame)
        
        results = model.track(frame, persist=True,verbose=False,)
        human_use_walkway = False
        drivers_not_letting_pedestrians_pass = False
        drivers_use_walkway = False
        #result = results[0]
        #annotated_frame = results[0].plot()
        if len(results) > 0:
            r = results[0]
        #if class_id == "person":
        if not len(r.boxes) == 0 and r.boxes.id is not None:
            # Get the boxes and track IDs
            #boxes = results[0].boxes.xywh.cpu()
            track_ids = r.boxes.id.int().cpu().tolist()
            # Plot the tracks
            for box, track_id in zip(r.boxes, track_ids):

                color_of_dots = um.generate_random_color_bgr(track_id)
                
                class_id = r.names[box.cls[0].item()]
                conf = int(round(box.conf[0].item(), 2)*100)
                
                if conf <40:
                    continue
                
                cords = box.xyxy[0].tolist()
                cords = [round(x) for x in cords]

                # Assuming current_coords is a tuple or list with four elements (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, cords)

                cv2.rectangle(frame, (x1,y1),(x2,y2), color=(0, 255, 0), thickness=2)
                
                #x, y, w, h = box
                
                x_scale_A,y_scale_A = 1,1

                x1_scaled, y1_scaled = int(x1 * x_scale_A), int(y1 * y_scale_A)
                x2_scaled, y2_scaled = int(x2 * x_scale_A), int(y2 * y_scale_A)
                
                center_x = int ( x1_scaled + (  (x2_scaled - x1_scaled) / 2 ) )
                center_y = int ( y1_scaled + (  (y2_scaled - y1_scaled) / 2 ) )
                
                #if track_id not in Get_Photo_ID and len(track_history[track_id]) > fps*1:
                    #um.create_folder_if_not_exists(f"Photos/{class_id}")
                    #Get_Photo_ID.append(track_id)
                    #processed_frame = frame.copy()
                    #
                    #cv2.circle(processed_frame, (x_center,y_center), radius=5, color=color_of_dots, thickness=-1)
                    #cv2.rectangle(processed_frame,(x,y),(w,h),color_of_dots,2)
                    #cut_out_frame = um.cut_out_section(frame,x,y,w,h)
                    #hh,ww,_ = cut_out_frame.shape
                    #if hh>120 and ww>120 and False:
                    #   results_cut_out = model(cut_out_frame,verbose=False)
                    #   result_cut_out = results_cut_out[0]
                    #   if not len(result_cut_out.boxes) == 0 and result_cut_out[0].boxes.id is not None:
                    #       # Plot the tracks
                    #       for Inner_box in (result_cut_out.boxes):
                    #           Inner_class_id = result_cut_out.names[Inner_box.cls[0].item()]

                    #           if Inner_class_id == class_id:
                    #           
                    #               Inner_cords = box.xyxy[0].tolist()
                    #               Inner_cords = [round(x) for x in Inner_cords]

                    #               # Assuming current_coords is a tuple or list with four elements (x1, y1, x2, y2)
                    #               Inner_x, Inner_y, Inner_w, Inner_h = map(int, Inner_cords)
                    #               #x, y, w, h = box

                    #               Inner_x_center = Inner_x + Inner_w // 2
                    #               Inner_y_center = Inner_y + Inner_h // 2

                    #               cv2.circle(cut_out_frame, (Inner_x_center,Inner_y_center), radius=5, color=color_of_dots, thickness=-1)
                    #               cv2.rectangle(cut_out_frame,(Inner_x, Inner_y),(Inner_w, Inner_h),color_of_dots,1)
                    #           
                    #cv2.imwrite(f'Photos/{class_id}/{track_id}.png',cut_out_frame)
                
                
                track = track_history[track_id]
                track.append((float(x1), float(y1)))  # x, y center point
                if len(track) > fps*3:  # retain 90 tracks for 90 frames
                    track.pop(0)
                ## Draw the tracking lines
                bearing = um.calculate_bearing(track[0],track[-1])
                Direction = um.bearing_to_direction(bearing)
                cv2.putText(frame, f'Direction : {Direction}', (x1+20,y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,255,0],1)
                if Draw_Track_Dots:
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    for point in points:
                        cv2.circle(frame, tuple(point[0]), radius=3, color=color_of_dots, thickness=-1)
                #cv2.putText(frame, f'{track_id}', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, [35,125,255],2)
                
                is_in_center_roi = cv2.pointPolygonTest(Bottom_Walk_Way_ROI_np,(center_x,center_y),True)
                is_in_upper_roi = cv2.pointPolygonTest(Bottom_Walk_Way_Upper_Side_ROI_np,(center_x,center_y),True)
                is_in_bottom_roi = cv2.pointPolygonTest(Bottom_Walk_Way_Bottom_Side_ROI_np,(center_x,center_y),True)
                
                if class_id not in left_line:
                    # If not, create a new list for that class_id
                    left_line[class_id] = []
                
                #speed = calculate_speed(track,fps,height)
                #
                #cv2.putText(frame, f'{speed} KM/H', (x-10,y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,255,0],2)
                #cv2.putText(frame, f'{conf}%', (x-20,y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [255,0,0],1)
                if class_id in ["person"]:
                    if (is_in_center_roi > 0 or (is_in_upper_roi > 0 and Direction in ["down-right","down","down-left","stay"]) or (is_in_bottom_roi > 0 and Direction in ["up","up-right","up-left","stay"])):
                        human_use_walkway = True 
                        
                        
                if class_id in ["car","truck","motorcycle","bus"]:
                    if is_in_center_roi > 0:
                        drivers_use_walkway = True

                        if human_use_walkway and drivers_use_walkway and not Direction in ["stay"]:
                            drivers_not_letting_pedestrians_pass = True
                            if track_id not in Get_Photo_Guilty_ID:
                                um.create_folder_if_not_exists(f"Photos/drivers_not_letting_pedestrians_pass/{class_id}")
                                Get_Photo_Guilty_ID.append(track_id)
                                #processed_frame = frame.copy()
#
                                #cv2.circle(processed_frame, (center_x,center_y), radius=5, color=color_of_dots, thickness=-1)
                                #cv2.rectangle(processed_frame,(x1,y1),(x2,y2),color_of_dots,2)
                                cut_out_frame = um.cut_out_section(base_frame,x1,y1,x2,y2)
                                
                                
                                
                                cv2.imwrite(f'Photos/drivers_not_letting_pedestrians_pass/{class_id}/{track_id}.png',cut_out_frame)
                                
                    
                    
                if track_id not in Get_Photo_ID and len(track_history[track_id]) > fps*1:
                    um.create_folder_if_not_exists(f"Photos/Normal/{class_id}")
                    Get_Photo_ID.append(track_id)
                    #processed_frame = frame.copy()
#
                    #cv2.circle(processed_frame, (center_x,center_y), radius=5, color=color_of_dots, thickness=-1)
                    #cv2.rectangle(processed_frame,(x1,y1),(x2,y2),color_of_dots,2)
                    cut_out_frame = um.cut_out_section(base_frame,x1,y1,x2,y2)
                    cv2.imwrite(f'Photos/Normal/{class_id}/{track_id}.png',cut_out_frame)
                cv2.circle(frame, (center_x,center_y), radius=3, color=color_of_dots, thickness=-1)
                              
                if is_in_center_roi > 0:
                    if track_id not in left_line[class_id]:
                        # Append the track_id to the list associated with the class_id
                        left_line[class_id].append(track_id)

                if class_id in ["car","truck","motorcycle","bus"] and False:
                    cut_out_frame = cut_out_section(frame,x,y,w,h)

                    # Perform OCR on an image file
                    result_of_ocr = reader.readtext(cut_out_frame) 
                    # Print the OCR results
                    if result_of_ocr:
                        text= result_of_ocr[0][1]
                        cv2.putText(frame, f'Plate : {text}', (x+20,y-20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, [0,255,0],1)
                
                
                
            # get finished tracks and do some logic with them
            finished_tracks = track_history.keys() - track_ids

            for ft_id in finished_tracks:
                ft = track_history.pop(ft_id)
                # do some logic with ft here......... 
                    
        puty = 30
        for key, value in left_line.items():
            cv2.putText(frame, f'{key}:{len(left_line[key])}', (10,puty), cv2.FONT_HERSHEY_SIMPLEX, 1.0, [125,125,255],3)
            puty += 30
        cv2.putText(frame, f'Pedestrians-Use-Walkway : {human_use_walkway}', (width-600,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, [125,125,255],2)
        cv2.putText(frame, f'Drivers-Use-Walkway : {drivers_use_walkway}', (width-600,60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, [125,125,255],2)
        cv2.putText(frame, f'Drivers-Ignore-Pedestrians : {drivers_not_letting_pedestrians_pass}', (width-600,90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, [125,125,255],2)
        # Display the annotated frame
        
        if drivers_not_letting_pedestrians_pass:
            cv2.polylines(frame, [Bottom_Walk_Way_ROI_np.astype(int)], isClosed=True, color=(0, 0, 255), thickness=2)
        else:
            cv2.polylines(frame, [Bottom_Walk_Way_ROI_np.astype(int)], isClosed=True, color=(255, 255, 255), thickness=2)
        cv2.polylines(frame, [Bottom_Walk_Way_Upper_Side_ROI_np.astype(int)], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.polylines(frame, [Bottom_Walk_Way_Bottom_Side_ROI_np.astype(int)], isClosed=True, color=(0, 255, 0), thickness=2)
        
       
        cv2.imshow("AI-KGSY", frame)
        prev_frame = base_frame
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()