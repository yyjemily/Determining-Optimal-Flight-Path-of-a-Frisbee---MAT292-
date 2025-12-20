import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

folder_path = "frisbee_real_data/"
#get videos
video_files = sorted(glob.glob(os.path.join(folder_path, "*.MOV")))

#output directories
os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)

#track settings
color_lower = np.array([0, 0, 80])  
color_upper = np.array([180, 80, 255]) 
frisbee_diameter_m = 0.274 
lk_params = dict(winSize=(31, 31), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
feature_params = dict(maxCorners=20, qualityLevel=0.01, minDistance=7, blockSize=7)

for video_path in video_files:
    base_name = os.path.basename(video_path).split('.')[0]
    print(f"--- Processing: {base_name} ---")
    
    trajectory_log = []
    
    #ROI SELECTION
    video = cv2.VideoCapture(video_path)
    ret, frame = video.read()
    if not ret: continue

    #temp window for ROI selection
    x_min, y_min, x_max, y_max = 36000, 36000, 0, 0
    def coordinat_chooser(event, x, y, flags, param):
        global x_min, y_min, x_max, y_max
        if event == cv2.EVENT_RBUTTONDOWN:
            x_min, y_min = min(x, x_min), min(y, y_min)
            x_max, y_max = max(x, x_max), max(y, y_max)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    cv2.namedWindow('Select_Frisbee')
    cv2.setMouseCallback('Select_Frisbee', coordinat_chooser)
    while True:
        cv2.imshow("Select_Frisbee", frame)
        if cv2.waitKey(1) & 0xFF == 27: break
    cv2.destroyAllWindows()

    scale_fac = frisbee_diameter_m / max(1, (x_max - x_min))

    #Tracking set up and initialization
    cap = cv2.VideoCapture(video_path)
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    
    mask_init = np.zeros_like(old_gray)
    mask_init[y_min:y_max, x_min:x_max] = 255
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=mask_init, **feature_params)

    frame_idx = 0 
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, color_lower, color_upper)

        if p0 is None:
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=color_mask, **feature_params)

        if p0 is not None:
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            if p1 is not None:
                good_new = [new for i, (new, old) in enumerate(zip(p1, p0)) if st[i] == 1 and color_mask[int(new.ravel()[1]), int(new.ravel()[0])] > 0]
                
                if len(good_new) > 0:
                    good_new = np.array(good_new).reshape(-1, 1, 2)
                    avg_x, avg_y = np.mean(good_new[:, 0, 0]), np.mean(good_new[:, 0, 1])
                    trajectory_log.append([frame_idx, avg_x, avg_y])
                    p0 = good_new
                else: p0 = None

        cv2.imshow('Tracking', frame)
        old_gray = frame_gray.copy()
        if cv2.waitKey(1) & 0xFF == 27: break
    
    cap.release()
    cv2.destroyAllWindows()

#export file if the trajectory log is not empty
    if trajectory_log:
        df = pd.DataFrame(trajectory_log, columns=['Frame', 'X_px', 'Y_px'])
        df['Time_s'] = df['Frame'] / 30
        df['X_m'] = (df['X_px'] - df['X_px'].iloc[0]) * scale_fac
        df['Y_m'] = (df['Y_px'] - df['Y_px'].iloc[0]) * scale_fac * -1

        df.to_csv(os.path.join("results", f"{base_name}_results.csv"), index=False)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax1.plot(df['Time_s'], df['X_m'], color='blue')
        ax1.set_title(f"Displacement: {base_name}")
        ax2.plot(df['Time_s'], df['Y_m'], color='orange')
        plt.tight_layout()
        plt.savefig(os.path.join("plots", f"{base_name}_plot.png"))
        plt.close(fig) 

