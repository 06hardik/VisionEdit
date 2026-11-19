"""
create_sample_video.py - Generates a synthetic test video from dataset images
=============================================================================
Creates a sample video (e.g. 5-10 seconds) containing faces with different emotions
so you can test the video inference and recording pipeline immediately.
"""

import os
import cv2
import numpy as np
import config

def create_sample_video(output_video_path="sample_test_video.mp4", duration_per_emotion=1.5, fps=20):
    test_dir = config.TEST_DIR
    emotions = ['happy', 'surprise', 'neutral', 'sad', 'angry']
    
    frames_per_emotion = int(duration_per_emotion * fps)
    video_frames = []
    
    print("[*] Gathering sample face images from test dataset...")
    for emotion in emotions:
        folder = os.path.join(test_dir, emotion)
        if not os.path.exists(folder):
            continue
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.jpg', '.png'))]
        if not files:
            continue
            
        # Select up to 3 different faces for this emotion
        sample_files = files[:3]
        for img_path in sample_files:
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Place the 48x48 / 128x128 face in the center of a standard 640x480 frame
            # simulating a webcam face
            canvas = np.full((480, 640, 3), 40, dtype=np.uint8) # Dark neutral background
            
            # Resize face to a realistic face bounding box size (e.g. 240x240)
            face_large = cv2.resize(img, (240, 240))
            
            # Center coordinates
            y_offset = (480 - 240) // 2
            x_offset = (640 - 240) // 2
            canvas[y_offset:y_offset+240, x_offset:x_offset+240] = face_large
            
            # Add subtitle
            cv2.putText(canvas, f"Dataset Ground Truth: {emotion.upper()}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            
            # Repeat frame
            repeat_count = max(1, frames_per_emotion // len(sample_files))
            for _ in range(repeat_count):
                video_frames.append(canvas)
                
    if not video_frames:
        print("[!] No images found to build sample video.")
        return None
        
    print(f"[*] Writing {len(video_frames)} frames to {output_video_path}...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, float(fps), (640, 480))
    for f in video_frames:
        writer.write(f)
    writer.release()
    print(f"[+] Sample test video created successfully: {output_video_path}")
    return output_video_path

if __name__ == '__main__':
    create_sample_video()

