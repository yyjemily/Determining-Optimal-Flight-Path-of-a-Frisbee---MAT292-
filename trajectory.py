import cv2
import numpy as np
import time

# --- 1. CONFIGURATION ---
video_path = "frisbee_real_data/IMG_5059.MOV"
# ADJUST THESE: Example for a Bright Orange/Yellow frisbee
# Use an HSV picker if your frisbee is a different color
color_lower = np.array([5, 100, 100]) 
color_upper = np.array([25, 255, 255])

# Lucas-Kanade Parameters (Increased window to account for background noise)
lk_params = dict(winSize=(21, 21), 
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=10, 
                      qualityLevel=0.3, 
                      minDistance=7, 
                      blockSize=7)

# --- 2. INITIALIZATION & ROI SELECTION ---
video = cv2.VideoCapture(video_path)
ret, frame = video.read()
if not ret:
    print("Failed to load video")
    exit()

x_min, y_min, x_max, y_max = 36000, 36000, 0, 0

def coordinat_chooser(event, x, y, flags, param):
    global x_min, y_min, x_max, y_max, frame
    if event == cv2.EVENT_RBUTTONDOWN:
        x_min, y_min = min(x, x_min), min(y, y_min)
        x_max, y_max = max(x, x_max), max(y, y_max)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    if event == cv2.EVENT_MBUTTONDOWN:
        x_min, y_min, x_max, y_max = 36000, 36000, 0, 0

cv2.namedWindow('Select_Frisbee')
cv2.setMouseCallback('Select_Frisbee', coordinat_chooser)

print("Right-click two corners of the frisbee, then press ESC.")
while True:
    cv2.imshow("Select_Frisbee", frame)
    if cv2.waitKey(1) & 0xFF == 27: break
cv2.destroyAllWindows()

# --- 3. TRACKING LOOP ---
cap = cv2.VideoCapture(video_path)
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Initialize p0 within the selected ROI
mask_init = np.zeros_like(old_gray)
mask_init[y_min:y_max, x_min:x_max] = 255
p0 = cv2.goodFeaturesToTrack(old_gray, mask=mask_init, **feature_params)

trajectory_mask = np.zeros_like(old_frame)
height, width = old_frame.shape[:2]

while True:
    ret, frame = cap.read()
    if not ret: break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Color Mask to filter out crowd/stadium noise
    color_mask = cv2.inRange(hsv, color_lower, color_upper)
    color_mask = cv2.medianBlur(color_mask, 5) # Clean up speckle noise

    if p0 is not None:
        # Calculate Optical Flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        
        good_new = []
        good_old = []

        # Validate points against the Color Mask
        if p1 is not None:
            for i, (new, old) in enumerate(zip(p1, p0)):
                if st[i] == 1:
                    nx, ny = map(int, new.ravel())
                    # Only keep point if it's within frame AND matches frisbee color
                    if 0 <= ny < height and 0 <= nx < width:
                        if color_mask[ny, nx] > 0:
                            good_new.append(new)
                            good_old.append(old)

        good_new = np.array(good_new).reshape(-1, 1, 2)
        good_old = np.array(good_old).reshape(-1, 1, 2)

        if len(good_new) > 0:
            # Draw trajectory
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                trajectory_mask = cv2.line(trajectory_mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

            display_img = cv2.add(frame, trajectory_mask)
            p0 = good_new
        else:
            # If color match is lost, try redetecting in a small area around last known point
            p0 = None
            display_img = frame
    else:
        # Re-detecting logic: Limit search to the color mask to avoid grabbing people
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=color_mask, **feature_params)
        display_img = frame

    cv2.imshow('Frisbee Trajectory Tracking', display_img)
    old_gray = frame_gray.copy()

    if cv2.waitKey(25) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()