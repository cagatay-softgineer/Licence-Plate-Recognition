import cv2
import threading
import time
import  os
from datetime import datetime
recording_hour = 1 # hours
video_minutes = 60 # minutes
recording_resolution = (1920,1080)

camera_index = [4]

camera_urls = {
1: "rtsp://AISOFT:Ai5oft*20@176.88.154.106",
2: "rtsp://AISOFT:Ai5oft*20@176.88.154.107",
3: "rtsp://AISOFT:Ai5oft*20@176.88.154.108",
4: "https://camstream.kahramanmaras.bel.tr/live/ulucami.stream/playlist.m3u8"

}

def handle_camera(kamera_url, cam_folder_path):

    end_time = time.time() + (recording_hour * 3600)
    while time.time() < end_time:
        kamera = cv2.VideoCapture(kamera_url)
        width = int(kamera.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(kamera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(kamera.get(cv2.CAP_PROP_FPS))

        frames_to_skip = fps - 1
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        
       
        current_time = datetime.now().strftime("%H-%M") 
        save_path = os.path.join(cam_folder_path, f"CAM_{os.path.basename(cam_folder_path)}_{current_time}")
        out = cv2.VideoWriter(f'{save_path}.avi', fourcc, 12 , recording_resolution)
        print(save_path)

        start_time = time.time()
        while time.time() - start_time < (video_minutes*60):  
            ret, frame = kamera.read()
            if ret:

                frame = cv2.resize(frame, recording_resolution)
                out.write(frame)

                """dor _ in range(frames_to_skip):
                    kamera.grab()"""
            else:
                print(f"Problem in streaming CAM {kamera_url[34:38]}")  
                break
  
        out.release()
        kamera.release() 

    kamera.release()



global_path = "data"
dt_string = datetime.now().strftime("%d-%m-%Y_%H-%M")
sub_folder_path = os.path.join(global_path,dt_string)
os.makedirs(sub_folder_path, exist_ok=True)

threads = []
camera_urls_record = [camera_urls[i] for i in camera_index]
for url in camera_urls_record:
    print("url: ",url[35:38])
    cam_folder_path = os.path.join(sub_folder_path, url[35:38])
    os.makedirs(cam_folder_path, exist_ok=True)
    thread = threading.Thread(target=handle_camera, args=(url, cam_folder_path))
    thread.start()
    threads.append(thread)

for thread in threads:
    thread.join()


"""
tatas olan: 9[7,:ItQvQ6o


"""