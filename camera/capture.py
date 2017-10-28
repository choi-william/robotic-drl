import cv2


def get_image():
    retval, im = camera.read()
    return im


# Camera settings
camera_port = 0
frame_width = 640
frame_height = 480
# Frames before capture to allow for automatic white balance
ramp_frames = 30

#
# Main
#
camera = cv2.VideoCapture(camera_port)

# Set camera parameters
# NOTE: these parameters seem to set the camera to predefined frame sizes NOT exact dimensions
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)

if not camera.isOpened():
    print("Error: No camera connected at port " + camera_port.__str__())
    exit(-1)

print("Starting capture")

# Discard derpy frames
for i in range(ramp_frames):
    temp = get_image()

# Capture actual image
print("Taking image...")

filename = "test.png"
camera_capture = get_image()

cv2.imwrite(filename, camera_capture)

camera.release()
