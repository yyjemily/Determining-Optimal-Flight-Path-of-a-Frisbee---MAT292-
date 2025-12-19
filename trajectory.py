import cv2
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

#config 
video_path = "frisbee_real_data/IMG_5059.MOV"
color_lower = np.array([0, 0, 80])  
color_upper = np.array([180, 80, 255])

# Lucas-Kanade Parameters (Increased window to account for background noise)
lk_params = dict(winSize=(21, 21), 
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=10, 
                      qualityLevel=0.3, 
                      minDistance=7, 
                      blockSize=7)

trajectory_log = [] #store path to later be used for plotting + data extraction 
#set scaling factor for frisbee
frisbee_diameter_m = 0.274 # meters

#Initialize & Rectangle of Interest Selection
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

#calculate scaling factor
pix_convert = x_max - x_min 
scale_fac = frisbee_diameter_m/pix_convert

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
frame_idx = 0 
while True:
    ret, frame = cap.read()
    if not ret: break
    frame_idx += 1

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv, color_lower, color_upper)

    if p0 is not None:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        
        good_new = []
        if p1 is not None:
            for i, (new, old) in enumerate(zip(p1, p0)):
                if st[i] == 1:
                    nx, ny = map(int, new.ravel())
                    if 0 <= ny < height and 0 <= nx < width:
                        if color_mask[ny, nx] > 0:
                            good_new.append(new)
                        else:
                            # This helps you know if the color filter is the culprit
                            print(f"Point dropped: Color at ({nx},{ny}) not in range")

        if len(good_new) > 0:
            good_new = np.array(good_new).reshape(-1, 1, 2)
            
            # Add this inside your 'if len(good_new) > 0:' block
            avg_x = np.mean(good_new[:, 0, 0])
            avg_y = np.mean(good_new[:, 0, 1])

            # Calculate how far the frisbee 'jumped' since the last frame
            if len(trajectory_log) > 0:
                last_x = trajectory_log[-1][1]
                last_y = trajectory_log[-1][2]
                distance_jumped = np.sqrt((avg_x - last_x)**2 + (avg_y - last_y)**2)
                
                # If the point jumped more than 100 pixels, it's probably noise (like a stadium light)
                if distance_jumped > 100:
                    p0 = None # Force reset
                    print("Jumped too far! Likely tracked a stadium light or line.")
                    continue 

            trajectory_log.append([frame_idx, avg_x, avg_y])

            # Draw for real-time visualization
            for pt in good_new:
                a, b = pt.ravel()
                cv2.circle(frame, (int(a), int(b)), 3, (0, 0, 255), -1)
            
            p0 = good_new
        else:
            p0 = None
    
    cv2.imshow('Tracking', frame)
    old_gray = frame_gray.copy()
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()

# --- 4. DATA EXPORT & PLOTTING ---
if trajectory_log:
    df = pd.DataFrame(trajectory_log, columns=['Frame', 'X_px', 'Y_px'])
    fps = 30 # Standard iPhone FPS, adjust if 60
    df['Time_s'] = df['Frame'] / fps
    df['X_m'] = (df['X_px'] - df['X_px'].iloc[0]) * scale_fac
    df['Y_m'] = (df['Y_px'] - df['Y_px'].iloc[0]) * scale_fac * -1 #define up as positivie
    
    #save as csv 
    csv_filename = "frisbee_trajectory_data.csv"
    df.to_csv(csv_filename, index=False)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(df['Time_s'], df['X_m'], label='X Distance')
    ax1.set_title('Horizontal Distance vs Time')
    ax2.plot(df['Time_s'], df['Y_m'], color='orange', label='Y Distance')
    ax2.set_title('Vertical Distance vs Time')
    plt.tight_layout()

    #save the plot

    plot_filename = "frisbee_plot.png"
    plt.savefig(plot_filename, dpi=300) # dpi=300 makes it high resolution for reports
    print(f"Plot successfully saved as {plot_filename}")
    
    plt.show()