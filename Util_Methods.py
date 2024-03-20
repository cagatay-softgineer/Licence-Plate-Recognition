import math
import os
import numpy as np
import random

def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def calculate_bearing(point1, point2,threshold=5):
        if calculate_distance(point1, point2) > threshold:
            delta_x = point2[0] - point1[0]
            delta_y = point2[1] - point1[1]
            bearing = math.atan2(delta_y, delta_x)
            return math.degrees(bearing)
        else:
            return "stay"
            

def bearing_to_direction(bearing):
    if bearing == "stay":
        return bearing
    else:
        # Define directional sectors
        arrows = ['→', '↘', '↓', '↗', '←', '↖', '↑', '↙', '→']

        arrow_to_direction = {
        '↑': 'up',
        '↗': 'up-right',
        '→': 'right',
        '↘': 'down-right',
        '↓': 'down',
        '↙': 'down-left',
        '←': 'left',
        '↖': 'up-left'
        }

        # Convert the bearing to the range [0, 360) degrees
        normalized_bearing = (bearing + 360) % 360

        # Determine the index of the corresponding direction
        index = int((normalized_bearing + 22.5) // 45)

        return [arrow_to_direction[arrows[index]]]

def create_folder_if_not_exists(folder_path):
    try:
        # Try to create the folder
        os.makedirs(folder_path)
    except FileExistsError:
        return
        

def empty_folder(folder_path):
    # List all files and subdirectories in the folder
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for name in files:
            # Delete files
            os.remove(os.path.join(root, name))
        for name in dirs:
            # Delete subdirectories
            os.rmdir(os.path.join(root, name))


def calculate_speed_without_perspective(tracks, frame_rate):
    speed = 0     

    if len(tracks) > 2:
        # Calculate distances between consecutive positions

        first_position = np.array(tracks[0])
        last_position = np.array(tracks[-1])
        
        position_diff = last_position - first_position
    
        # Calculate the distance using the Euclidean norm
        distance = np.sqrt(np.sum(position_diff**2))

        # Calculate time elapsed between consecutive frames
        time_elapsed = len(tracks) / frame_rate

        # Calculate speeds
        speed = (distance / time_elapsed)
    
    return int(speed)


def calculate_speed(tracks, frame_rate, height):
    speed = 0
    
    # Define a threshold for movement detection (adjust as needed)
    movement_threshold = 5  # Adjust as needed
    
    if len(tracks) > 2:
        # Calculate distances between consecutive positions

        first_position = np.array(tracks[0])
        last_position = np.array(tracks[-1])
        
        position_diff = last_position - first_position
        
        # Check if the object is stationary
        if np.linalg.norm(position_diff) < movement_threshold:
            return speed
        
        # Calculate the distance using the Euclidean norm
        distance = np.linalg.norm(position_diff)
        
        # Apply perspective scaling factor
        scaled_tracks = []
        for i, position in enumerate(tracks):
            x, y = position
            scaled_y = y * (height / (i + 1))  # Apply scaling factor to y-coordinate
            scaled_tracks.append((x, scaled_y))
        
        scaled_tracks = np.array(scaled_tracks)
        scaled_first_position = scaled_tracks[0]
        scaled_last_position = scaled_tracks[-1]
        
        scaled_position_diff = scaled_last_position - scaled_first_position
        
        # Calculate time elapsed between consecutive frames
        time_elapsed = len(tracks) / frame_rate

        # Calculate speeds in pixels per second
        speed_pixels_per_second = (np.linalg.norm(scaled_position_diff) / time_elapsed)
        
        # Convert speed to km/h (adjust as needed)
        # Assuming 1 pixel = 0.01 meters (adjust as needed)
        meters_per_pixel = 0.0003
        km_per_meter = 0.001
        seconds_per_hour = 3600
        speed = (speed_pixels_per_second * meters_per_pixel * seconds_per_hour * km_per_meter)
    
    return int(speed)

def generate_random_color_rgb(seed):
    random.seed(seed)
    red = random.randint(0, 255)
    green = random.randint(0, 255)
    blue = random.randint(0, 255)
    return (red, green, blue)


def generate_random_color_bgr(seed):
    return generate_random_color_rgb(seed)[::-1]

def cut_out_section(image, x, y, w, h):
    # Ensure integer values for coordinates
    x, y, w, h = int(x), int(y), int(w), int(h)
    
    # Extract the section from the image using array slicing
    section = image[y:y+h, x:x+w]
    section = image[y:h, x:w]
    return section