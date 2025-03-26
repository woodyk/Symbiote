#!/usr/bin/env python3
#
# text-to-video.py

import torch
import os
import cv2
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

# Define prompt
prompt = "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance."

# Initialize pipeline
pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-2b",
    torch_dtype=torch.float16
)

pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

# Generate video frames
video = pipe(
    prompt=prompt,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=49,
    guidance_scale=6,
    generator=torch.Generator(device="cuda").manual_seed(42),
).frames[0]

# Export to video file
output_file = "output.mp4"
export_to_video(video, output_file, fps=8)

# Play video using OpenCV
cap = cv2.VideoCapture(output_file)

if not cap.isOpened():
    print("Error: Could not open video.")
else:
    # Play the video
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Video Playback', frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):  # Press 'q' to quit playback
                break
        else:
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    # Remove the video file after playback
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"File {output_file} removed.")

