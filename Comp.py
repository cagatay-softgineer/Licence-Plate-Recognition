import cv2

# Load two consecutive frames
def calculate_optical_flow(prev_frame,next_frame,maxCorners=500,qualityLevel=0.3, minDistance=5):
    

    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    # Define parameters for the pyramid and optical flow algorithm
    lk_params = dict(winSize=(15, 15),
                    maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Calculate optical flow
    prev_points = cv2.goodFeaturesToTrack(prev_gray, maxCorners, qualityLevel, minDistance)
    next_points, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, prev_points, None, **lk_params)

    # Select only the points that are successfully tracked
    prev_points = prev_points[status == 1]
    next_points = next_points[status == 1]

    # Display optical flow
    for (prev_x, prev_y), (next_x, next_y) in zip(prev_points, next_points):
        cv2.line(prev_frame, (int(prev_x), int(prev_y)), (int(next_x), int(next_y)), (0, 255, 0), 2)
        cv2.circle(prev_frame, (int(next_x), int(next_y)), 5, (0, 0, 255), -1)
        cv2.circle(prev_frame, (int(prev_x), int(prev_y)), 2, (255, 0, 0), -1)

    return(prev_frame)

# Display the frame with optical flow

def main():
    prev_frame = cv2.imread('Assets/prev.jpg')
    next_frame = cv2.imread('Assets/next.jpg')
    
    cv2.imshow('Optical Flow', calculate_optical_flow(prev_frame,next_frame))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()