import os
import cv2
import random

def get_random_frame(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get the total number of frames
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    # Set a random frame position
    random_frame_position = random.randint(0, total_frames)
    
    # Set the frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_position)
    
    # Read the frame
    ret, frame = cap.read()
    
    # Release the video capture object
    cap.release()
    
    # Check if the frame is read successfully
    if ret:
        return frame
    else:
        return None

# Directory containing video files
video_directory = os.path.expanduser('~/sakura_science_intern_dataset/videos')

# Output directory for saving frames
output_directory = os.path.expanduser('~/sakura_science_intern_dataset/videos_jpg')

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Loop through all files in the directory
for filename in os.listdir(video_directory):
    if filename.endswith('.mp4') or filename.endswith('.avi') or filename.endswith('.mkv'):
        # Construct the full path to the video file
        video_path = os.path.join(video_directory, filename)
        
        # Get a random frame from the video
        random_frame = get_random_frame(video_path)
        
        # Save the random frame as JPEG
        if random_frame is not None:
            output_filename = os.path.splitext(filename)[0] + '_random_frame.jpg'
            output_path = os.path.join(output_directory, output_filename)
            cv2.imwrite(output_path, random_frame)
            #print(f"Saved random frame from {filename} as {output_filename}")
        else:
            print(f"Failed to retrieve random frame from {filename}.")
