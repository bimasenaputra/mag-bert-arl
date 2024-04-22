import os
import pandas as pd
from deepface import DeepFace

# Directory containing the images
directory = os.path.expanduser("~/sakura_science_intern_dataset/videos_jpg")

# Initialize an empty list to store results
results = []

# Iterate over the files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        img_path = os.path.join(directory, filename)
        # Analyze the image using DeepFace
        try:
            objs = DeepFace.analyze(img_path=img_path, actions=['race'], enforce_detection=False)
            #print(objs)
            race = objs[0]['dominant_race']
            results.append({'Filename': "_".join(filename.split("_")[:3])+".jpg", 'Race': race})
        except Exception as e:
            results.append({'Filename': "_".join(filename.split("_")[:3])+".jpg", 'Race': 'white'})
            #print(f"Error analyzing {filename}: {type(objs)} {len(objs)}")

# Convert the results to a DataFrame
df = pd.DataFrame(results)

# Save the DataFrame to a CSV file
df.to_csv('race_results.csv', index=False)
