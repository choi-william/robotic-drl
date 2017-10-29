import cv2
import numpy as np

# Camera settings
camera_port = 0
frame_width = 640
frame_height = 480
# Frames before capture to allow for automatic white balance
ramp_frames = 30


def init_camera():

    camera = cv2.VideoCapture(camera_port)

    # Set camera parameters
    # NOTE: these parameters seem to set the camera to predefined frame sizes NOT exact dimensions
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)

    if not camera.isOpened():
        print("Error: No camera connected at port " + camera_port.__str__())
        exit(-1)

    return camera


def camera_skip_frames(camera):
    # Discard derpy frames
    for i in range(ramp_frames):
        capture_frame(camera)

    print("Starting capture")


def capture_frame(camera):
    print("Taking image...")

    retval, im = camera.read()
    return im


def undistort(img, camera_matrix, dist_coefs):

    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))

    out = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)

    # crop
    x, y, w, h = roi
    out = out[y:y + h, x:x + w]

    return out

#
# Main
#
# Captures and image when no flags
# '-u' undistorts an image
# '-d' detects a yellow object

if __name__ == '__main__':
    import sys
    import getopt

    args, img_mask = getopt.getopt(sys.argv[1:], 'ud')
    args = dict(args)

    file = "capture"
    ext = ".png"

    # Hardcode with sample values for now
    camera_matrix = np.array([[532.79534539, 0., 342.4582558],
                              [0., 532.91926174, 233.90061186],
                              [0., 0., 1.0]])

    dist_coefs = np.array([-2.81085949e-01,
                           2.72559023e-02,
                           1.21667006e-03,
                           -1.34205532e-04,
                           1.58517893e-01])

    u_flag = args.get('-u')
    d_flag = args.get('-d')

    if u_flag is not None:
        infile = cv2.imread(file + ext)
        output = undistort(infile, camera_matrix, dist_coefs)
        cv2.imwrite(file + "_undistorted" + ext, output)
    if d_flag is not None:
        myCamera = init_camera()
        camera_skip_frames(myCamera)
        while True:
            camera_capture = capture_frame(myCamera)
            #TODO implement object detection

            cv2.imshow('img', camera_capture)
            cv2.waitKey(1)

    else:
        index = 0
        infile = cv2.imread(file + index.__str__() + ext)

        while infile is not None:
            index += 1
            infile = cv2.imread(file + index.__str__() + ext)

        myCamera = init_camera()
        camera_skip_frames(myCamera)

        camera_capture = capture_frame(myCamera)

        cv2.imwrite(file + index.__str__() + ext, camera_capture)

        cv2.imshow('img', camera_capture)
        cv2.waitKey(1000)

        myCamera.release()
