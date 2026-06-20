import cv2

video = cv2.VideoCapture("videos/sample.mp4")

total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = video.get(cv2.CAP_PROP_FPS)

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

duration = total_frames / fps

print("Video Information")
print("-" * 30)

print("Frames:", total_frames)
print("FPS:", fps)
print("Resolution:", width, "x", height)
print("Duration:", round(duration, 2), "seconds")

video.release()