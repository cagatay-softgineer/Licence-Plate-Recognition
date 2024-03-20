import cv2
import pytesseract
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import re
import statistics

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Initialize parameters
blur_value = 5
brightness = 0
contrast = 1
threshold_min = 0
threshold_max = 255
threshold_black_min_1 = 0
threshold_black_min_2 = 0
threshold_black_min_3 = 0

threshold_black_max_1 = 255
threshold_black_max_2 = 255
threshold_black_max_3 = 255

LicenceMinLenght = 2
LicenceMaxLenght = 8

possible_string = []

checkbox_checked = False

image_index = 0

scaling_factor = 8

base_image = cv2.imread("Images/1.png")

def sharpen_image(image):
    kernel = np.array([[-1, -1, -1],
                       [-1, 15, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)

def filter_letters_numbers(input_string):
    # Use regular expression to filter only letters and numbers
    filtered_string = re.sub(r'[^a-zA-Z0-9]', '', input_string)
    return filtered_string

def checkbox_changed():
    global checkbox_checked
    if checkbox_var.get():
        checkbox_checked = True
    else:
        checkbox_checked = False
        
def predict_license_plate(string1, string2, string3, min_length=4, max_length=8):
    if string1 == string2 == string3:
        return [string1] if min_length <= len(string1) <= max_length else []
    else:
        # Determine the longest string
        max_length_str = max(string1, string2, string3, key=len)
        
        # Initialize a list to store predicted license plate numbers
        predicted_plates = []

        # Generate combinations of substrings with differing letters from different parts of the longest string
        for i in range(len(max_length_str)):
            for j in range(i + 1, len(max_length_str) + 1):
                for k in range(len(string1)):
                    for l in range(k + 1, len(string1) + 1):
                        for m in range(len(string2)):
                            for n in range(m + 1, len(string2) + 1):
                                for o in range(len(string3)):
                                    for p in range(o + 1, len(string3) + 1):
                                        # Combine substrings from different parts of the strings
                                        candidate_plate = max_length_str[i:j] + string1[k:l] + string2[m:n] + string3[o:p]

                                        # Check if the candidate plate contains only differing letters
                                        if len(set(candidate_plate)) == len(candidate_plate):
                                            if min_length <= len(candidate_plate) <= max_length:
                                                predicted_plates.append(candidate_plate)

        return predicted_plates

# Function to process the image with current parameters
def process_image():
    global blur_value
    global detected_plate_id
    global brightness
    global contrast
    global threshold_min
    global threshold_max
    
    global threshold_black_min_1
    global threshold_black_min_2
    global threshold_black_min_3
    
    global threshold_black_max_1
    global threshold_black_max_2
    global threshold_black_max_3
    
    global LicenceMinLenght
    global LicenceMaxLenght
    
    global possible_string
    
    global checkbox_checked
    # Read the base image
    global base_image

    # Define the scaling factor
    global scaling_factor   # Change as needed

    # Upscale the image using interpolation

    image = cv2.resize(base_image, None, fx=(scaling_factor ** (1/4)), fy=(scaling_factor ** (1/4)), interpolation=cv2.INTER_NEAREST_EXACT)
    image = cv2.resize(image, None, fx=(scaling_factor ** (1/4)), fy=(scaling_factor ** (1/4)), interpolation=cv2.INTER_AREA)
    image = cv2.resize(image, None, fx=(scaling_factor ** (1/4)), fy=(scaling_factor ** (1/4)), interpolation=cv2.INTER_BITS2)
    image = cv2.resize(image, None, fx=(scaling_factor ** (1/4)), fy=(scaling_factor ** (1/4)), interpolation=cv2.INTER_LANCZOS4)
    
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define lower and upper bounds for black color in HSV
    lower_black = np.array([threshold_black_min_1, threshold_black_min_2, threshold_black_min_3])
    upper_black = np.array([threshold_black_max_1, threshold_black_max_2, threshold_black_max_3])  # Adjust the upper bound for better results

    # Create a binary mask for black color
    mask = cv2.inRange(hsv, lower_black, upper_black)

    # Apply the mask to the original image
    black_filtered_image = cv2.bitwise_and(image, image, mask=mask)

    # Adjust brightness and contrast
    adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

    gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)

    # Performing OTSU threshold
    #ret, thresh1 = cv2.threshold(gray, threshold_min, threshold_max, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    ret, thresh1 = cv2.threshold(gray, threshold_min, threshold_max, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    
    # Applying dilation on the threshold image
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)

    blurred = cv2.GaussianBlur(thresh1, (blur_value, blur_value),0)
    
    bloomed_image = cv2.addWeighted(dilation, 1.5, blurred, -0.5, 0)
    
    
    # Convert the image to HSV color space
    hsv2 = cv2.cvtColor(cv2.cvtColor(dilation, cv2.COLOR_GRAY2BGR),cv2.COLOR_BGR2HSV)

    # Create a binary mask for black color
    mask2 = cv2.inRange(hsv2, lower_black, upper_black)

    # Apply the mask to the original image
    black_filtered_image2 = cv2.bitwise_and(image, image, mask=mask2)
    
    bloomed_image = sharpen_image(bloomed_image)
    black_filtered_image = sharpen_image(black_filtered_image)
    black_filtered_image2 = sharpen_image(black_filtered_image2)
    
    bloomed_image = cv2.convertScaleAbs(bloomed_image, alpha=contrast, beta=brightness)
    black_filtered_image = cv2.convertScaleAbs(black_filtered_image, alpha=contrast, beta=brightness)
    black_filtered_image2 = cv2.convertScaleAbs(black_filtered_image2, alpha=contrast, beta=brightness)
    
    
    plate_id = pytesseract.image_to_string(bloomed_image, config='--psm 8')
    plate_id = filter_letters_numbers(plate_id.strip())
    detected_plate_id.set("Detected Plate ID: " + plate_id.strip())
    
    plate_id2 = pytesseract.image_to_string(black_filtered_image, config='--psm 8')
    plate_id2 = filter_letters_numbers(plate_id2.strip())
    detected_plate_id2.set("Detected Plate ID: " + plate_id2.strip())
    
    plate_id3 = pytesseract.image_to_string(black_filtered_image2, config='--psm 8')
    plate_id3 = filter_letters_numbers(plate_id3.strip())
    detected_plate_id3.set("Detected Plate ID: " + plate_id3.strip())

    #possible_string = predict_license_plates([plate_id,plate_id2],LicenceMinLenght,LicenceMaxLenght)
    if checkbox_checked:
        possible_string = predict_license_plate(plate_id,plate_id2,plate_id3,LicenceMinLenght,LicenceMaxLenght)
        dropdown['values'] = possible_string
        if len(possible_string)>0:
            dropdown.set(possible_string[0])
            mode_value = statistics.mode(possible_string)
            detected_plate_id4.set("Most Possible Plate ID: " + mode_value)
    
    
    # Display the processed image
    img = Image.fromarray(cv2.cvtColor(bloomed_image, cv2.COLOR_BGR2RGB))
    img.thumbnail((frame_width, frame_height))
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.config(image=imgtk)
    label.place(x=frame_x, y=frame_y)
    
    img2 = Image.fromarray(cv2.cvtColor(black_filtered_image, cv2.COLOR_BGR2RGB))
    img2.thumbnail((frame_width, frame_height))
    imgtk2 = ImageTk.PhotoImage(image=img2)
    label2.imgtk = imgtk2
    label2.config(image=imgtk2)
    label2.place(x=frame_x2, y=frame_y2)
    
    img3 = Image.fromarray(cv2.cvtColor(black_filtered_image2, cv2.COLOR_BGR2RGB))
    img3.thumbnail((frame_width, frame_height))
    imgtk3 = ImageTk.PhotoImage(image=img3)
    label3.imgtk = imgtk3
    label3.config(image=imgtk3)
    label3.place(x=frame_x3, y=frame_y3)
    
    
    
# Function to update parameters
def update_parameters(_=None):
    global blur_value
    blur_value = blur_scale.get()
    global brightness
    brightness = brightness_scale.get() - 50
    global contrast
    contrast = contrast_scale.get() / 50.0
    global threshold_min
    threshold_min = threshold_min_scale.get()
    global threshold_max
    threshold_max = threshold_max_scale.get()
    
    global threshold_black_min_1
    threshold_black_min_1 = threshold_black_min_1_scale.get()
    global threshold_black_min_2
    threshold_black_min_2 = threshold_black_min_2_scale.get()
    global threshold_black_min_3
    threshold_black_min_3 = threshold_black_min_3_scale.get()
    
    global threshold_black_max_1
    threshold_black_max_1 = threshold_black_max_1_scale.get()
    global threshold_black_max_2
    threshold_black_max_2 = threshold_black_max_2_scale.get()
    global threshold_black_max_3
    threshold_black_max_3 = threshold_black_max_3_scale.get()
    
    global scaling_factor 
    scaling_factor = scaling_factor_scale.get()
    
    global LicenceMinLenght
    LicenceMinLenght = LicenceMinLenghtScale.get()
    
    global LicenceMaxLenght
    LicenceMaxLenght = LicenceMaxLenghtScale.get()
    process_image()

def update_Image(_=None):
    global base_image,image_index
    
    images = ["Images/1.png","Images/2.png","Images/3.png","Images/4.png","Images/5.jpg","Images/7.jpg"]

    image_index = (image_index + 1) % len(images)
    base_image = cv2.imread(f"{images[image_index]}")
    process_image()
    

# Define window size
window_width = 1600
window_height = 900

# Define image frame size and position
frame_width = window_width // 2  # quarter of window width
frame_height = window_height // 2  # quarter of window height

frame_x = 0  # center of the window horizontally
frame_y = 0

frame_x2 = window_width - window_width // 4 - window_width // 12 # center of the window horizontally
frame_y2 = 0

frame_x3 = 0 # center of the window horizontally
frame_y3 = window_height // 2

# Create main window
root = tk.Tk()
root.title("License Plate Detection")

# Create sliders to adjust parameters
blur_scale = tk.Scale(root, from_=1, to=101, orient=tk.HORIZONTAL, label="Blur", resolution=2)
blur_scale.set(1)
blur_scale.pack()

brightness_scale = tk.Scale(root, from_=-255, to=255, orient=tk.HORIZONTAL, label="Brightness")
brightness_scale.set(60)
brightness_scale.pack()

contrast_scale = tk.Scale(root, from_=0, to=255, orient=tk.HORIZONTAL, label="Contrast")
contrast_scale.set(120)
contrast_scale.pack()

threshold_min_scale = tk.Scale(root, from_=0, to=255, orient=tk.HORIZONTAL, label="Threshold Min")
threshold_min_scale.set(0)
threshold_min_scale.pack()

threshold_max_scale = tk.Scale(root, from_=0, to=255, orient=tk.HORIZONTAL, label="Threshold Max")
threshold_max_scale.set(255)
threshold_max_scale.pack()

threshold_black_min_1_scale = tk.Scale(root, from_=0, to=255, orient=tk.HORIZONTAL, label="Threshold Black Min 1")
threshold_black_min_1_scale.set(0)
threshold_black_min_1_scale.pack()

threshold_black_min_2_scale = tk.Scale(root, from_=0, to=255, orient=tk.HORIZONTAL, label="Threshold Black Min 2")
threshold_black_min_2_scale.set(0)
threshold_black_min_2_scale.pack()

threshold_black_min_3_scale = tk.Scale(root, from_=0, to=255, orient=tk.HORIZONTAL, label="Threshold Black Min 3")
threshold_black_min_3_scale.set(0)
threshold_black_min_3_scale.pack()


threshold_black_max_1_scale = tk.Scale(root, from_=0, to=255, orient=tk.HORIZONTAL, label="Threshold Black Max 1")
threshold_black_max_1_scale.set(255)
threshold_black_max_1_scale.pack()

threshold_black_max_2_scale = tk.Scale(root, from_=0, to=255, orient=tk.HORIZONTAL, label="Threshold Black Max 2")
threshold_black_max_2_scale.set(255)
threshold_black_max_2_scale.pack()

threshold_black_max_3_scale = tk.Scale(root, from_=0, to=255, orient=tk.HORIZONTAL, label="Threshold Black Max 3")
threshold_black_max_3_scale.set(255)
threshold_black_max_3_scale.pack()

scaling_factor_scale = tk.Scale(root, from_=0, to=8, orient=tk.HORIZONTAL, label="Scaling Factor", resolution = 0.1)
scaling_factor_scale.set(255)
scaling_factor_scale.pack()

LicenceMinLenghtScale = tk.Scale(root, from_=0, to=12, orient=tk.HORIZONTAL, label="Threshold For Plate Predict")
LicenceMinLenghtScale.set(7)
LicenceMinLenghtScale.pack()

LicenceMaxLenghtScale = tk.Scale(root, from_=0, to=12, orient=tk.HORIZONTAL, label="Max Predict Plate ID Lenght")
LicenceMaxLenghtScale.set(7)
LicenceMaxLenghtScale.pack()

button = tk.Button(root, text="Change Settings", command=update_parameters)
button.pack()

button = tk.Button(root, text="Change Image", command=update_Image)
button.pack()

# Create a label to display the detected plate ID
detected_plate_id = tk.StringVar()
plate_id_label = tk.Label(root, textvariable=detected_plate_id)
plate_id_label.pack()

detected_plate_id2 = tk.StringVar()
plate_id_label2 = tk.Label(root, textvariable=detected_plate_id2)
plate_id_label2.pack()

detected_plate_id3 = tk.StringVar()
plate_id_label3 = tk.Label(root, textvariable=detected_plate_id3)
plate_id_label3.pack()

detected_plate_id4 = tk.StringVar()
plate_id_label4 = tk.Label(root, textvariable=detected_plate_id4)
plate_id_label4.pack()

label = tk.Label(root)
label.pack()

label2 = tk.Label(root)
label2.pack()

label3 = tk.Label(root)
label3.pack()

# Create a variable to store checkbox state
checkbox_var = tk.BooleanVar()

# Create a checkbox
checkbox = tk.Checkbutton(root, text="Check me", variable=checkbox_var, command=checkbox_changed)
checkbox.pack()

# Create a Combobox widget
dropdown = ttk.Combobox(root, values=possible_string)
dropdown.pack()


root.geometry(f"{window_width}x{window_height}")

# Process the initial image
process_image()

# Start the GUI main loop
root.mainloop()