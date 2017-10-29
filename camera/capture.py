import cv2
import numpy as np

# Camera settings
camera_port = 0
frame_width = 640
frame_height = 480
# Frames before capture to allow for automatic white balance
ramp_frames = 30

# Canny edge detection parameters
minEdgeVal = 100
maxEdgeVal = 200


### Initialize the opencv camera
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


### Discard frames to allow for camera to set its white balance
def camera_skip_frames(camera):
    # Discard derpy frames
    for i in range(ramp_frames):
        capture_frame(camera)

    print("Starting capture")


### Capture a single fram
def capture_frame(camera):
    #print("Taking image...")

    retval, im = camera.read()
    return im


### Remove lens distortion from a captured camera frame
def undistort(img, camera_matrix, dist_coefs):

    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))

    out = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)

    # crop
    x, y, w, h = roi
    out = out[y:y + h, x:x + w]

    return out


### Dummy function for updating positions of hsv bars
def update_bars(pos):
    pass


### Create hsv bars
def create_trackbars():
    vis = np.zeros((1, frame_height, 3), np.uint8)
    cv2.imshow('bars', vis)
    cv2.createTrackbar('hue min', 'bars', 0, 255, update_bars)
    cv2.createTrackbar('hue max', 'bars', 255, 255, update_bars)
    cv2.createTrackbar('saturation min', 'bars', 0, 255, update_bars)
    cv2.createTrackbar('saturation max', 'bars', 255, 255, update_bars)
    cv2.createTrackbar('value min', 'bars', 0, 255, update_bars)
    cv2.createTrackbar('value max', 'bars', 255, 255, update_bars)


### Draw a target icon on the position of the object
def draw_object(img, x, y):

    colour = (0,255,0)

    cv2.circle(img, (x,y), 20, colour, 2)
    if (y - 25 > 0):
        cv2.line(img, (x, y), (x, y - 25), colour, 2)
    else:
        cv2.line(img, (x, y), (x, 0), colour, 2)
    if (y + 25 < frame_height):
        cv2.line(img, (x, y), (x, y + 25), colour, 2)
    else:
        cv2.line(img, (x, y), (x, frame_height), colour, 2)
    if (x - 25 > 0):
        cv2.line(img, (x, y), (x - 25, y), colour, 2)
    else:
        cv2.line(img, (x, y), (0, y), colour, 2)
    if (x + 25 < frame_width):
        cv2.line(img, (x, y), (x + 25, y), colour, 2)
    else:
        cv2.line(img, (x, y), (frame_width, y), colour, 2)

    cv2.putText(img, x.__str__() + "," + y.__str__(), (x, y + 30), 1, 1, colour, 2)


### Returns x and y coordinates in pixels from the top left corner and an angle in degrees clockwise from the up direction
def track_object(img):
    x = 0
    y = 0
    angle = 0

    ## Change from RGB to HSV

    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    ## Colour mask in HSV
    h_min = cv2.getTrackbarPos('hue min', 'bars')
    h_max = cv2.getTrackbarPos('hue max', 'bars')
    s_min = cv2.getTrackbarPos('saturation min', 'bars')
    s_max = cv2.getTrackbarPos('saturation max', 'bars')
    v_min = cv2.getTrackbarPos('value min', 'bars')
    v_max = cv2.getTrackbarPos('value max', 'bars')
    colourMaskLower = np.array([h_min,s_min,v_min])
    colourMaskUpper = np.array([h_max,s_max,v_max])

    colourMask = cv2.inRange(imgHsv, colourMaskLower, colourMaskUpper)

    imgColourMasked = cv2.bitwise_and(img, img, mask=colourMask)

    cv2.imshow('imgColourMasked', imgColourMasked)

    ## Use opening (combined erosion and dilation) to remove noise

    kernel = np.ones((7,7), np.uint8)
    opening = cv2.morphologyEx(imgColourMasked, cv2.MORPH_OPEN, kernel)

    cv2.imshow('opening', opening)

    ## Find contours
    imgGray = cv2.cvtColor(opening, cv2.COLOR_BGR2GRAY)

    image, contours, hierarchy = cv2.findContours(imgGray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if (len(contours) is 0):
        print("no objects found")
        return (x, y, angle)

    if len(contours) < 10:
        for i in range(len(contours)):
            M = cv2.moments(contours[i])
            x = int(M['m10'] / M['m00'])
            y = int(M['m01'] / M['m00'])
            area = cv2.contourArea(contours[i])
            draw_object(img, x, y)
    else:
        print("Too many objects found!")

    cv2.putText(img, 'Objects found: ' + len(contours).__str__(), (0, 30), 2, 1, (0, 255, 0), 2)

    ## Other code used for experimenting, don't delete just yet

    # imgGray = cv2.cvtColor(imgColourMasked, cv2.COLOR_BGR2GRAY)
    #
    # edges = cv2.Canny(imgGray, minEdgeVal, maxEdgeVal)
    #
    # cv2.imshow('edges',edges)
    # cv2.waitKey(1)
    #
    # imgGrayAll = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(imgGrayAll, 127, 255, 0)
    #
    # cv2.imshow('imgGrayAll', imgGrayAll)
    # cv2.waitKey(1)
    #
    # cv2.imshow('thresh', thresh)
    # cv2.waitKey(1)
    #
    # image, cnt, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #
    # epsilon = 0.1 * cv2.arcLength(cnt[0], True)
    # approx = cv2.approxPolyDP(cnt[0], epsilon, True)
    #
    # img3 = cv2.drawContours(image, approx, -1, (128,255,0), 3)
    #
    # img2 = cv2.drawContours(image, cnt, -1, (128,255,0), 3)
    #
    # cv2.imshow('contours', img2)
    # cv2.waitKey(1)
    #
    # cv2.imshow('approx', img3)
    # cv2.waitKey(1)

    return (x, y, angle)

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
        create_trackbars()
        while True:
            camera_capture = capture_frame(myCamera)
            track_object(camera_capture)

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
