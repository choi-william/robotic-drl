import cv2
import numpy as np
import json
import cmath

# Global definitions
epsilon = 1e-4
max_objects = 10
num_actuators = 5


class CameraConfig:
    camera_port = 1
    frame_width = 640
    frame_height = 480
    ramp_frames = 30  # Frames before capture to allow for automatic white balance

    # Base2 values as default:
    camera_matrix = [[786.08925465, 0., 327.54944649],
                     [0., 789.50593105, 223.37587648],
                     [0., 0., 1.]]
    dist_coefs = [4.10344222e-01,
                  -3.49531706e+00,
                  -1.44480264e-03,
                  -4.57970740e-03,
                  7.27633486e+00]

    def to_dict(self):
        return {'camera_port': self.camera_port,
                'frame_width': self.frame_width,
                'frame_height': self.frame_height,
                'ramp_frames': self.ramp_frames,
                'camera_matrix': self.camera_matrix,
                'dist_coefs': self.dist_coefs}

    def from_dict(self, dictionary):
        self.camera_port = dictionary['camera_port']
        self.frame_width = dictionary['frame_width']
        self.frame_height = dictionary['frame_height']
        self.ramp_frames = dictionary['ramp_frames']
        self.camera_matrix = dictionary['camera_matrix']
        self.dist_coefs = dictionary['dist_coefs']


class TrackParams:
    h_min = 0
    h_max = 255
    s_min = 0
    s_max = 255
    v_min = 0
    v_max = 255
    kernel_size = 7

    def to_dict(self):
        return {'h_min': self.h_min,
                'h_max': self.h_max,
                's_min': self.s_min,
                's_max': self.s_max,
                'v_min': self.v_min,
                'v_max': self.v_max,
                'kernel_size': self.kernel_size}

    def from_dict(self, dictionary):
        self.h_min = dictionary['h_min']
        self.h_max = dictionary['h_max']
        self.s_min = dictionary['s_min']
        self.s_max = dictionary['s_max']
        self.v_min = dictionary['v_min']
        self.v_max = dictionary['v_max']
        self.kernel_size = dictionary['kernel_size']


# Initialize the opencv camera
def init_camera(camera_conf):

    camera = cv2.VideoCapture(camera_conf.camera_port)

    # Set camera parameters
    # NOTE: these parameters seem to set the camera to predefined frame sizes NOT exact dimensions
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_conf.frame_height)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_conf.frame_width)
    camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    camera.set(cv2.CAP_PROP_EXPOSURE, 0.0001)
    camera.set(cv2.CAP_PROP_CONTRAST, 0.5)

    # camera.set(cv2.CAP_PROP_XI_AUTO_WB, False) - no work
    # camera.set(cv2.CAP_PROP_CONTRAST,255)  # works, but exposure is still a problem

    if not camera.isOpened():
        print("Error: No camera connected at port " + camera_conf.camera_port.__str__())
        exit(-1)

    return camera


# Discard frames to allow for camera to set its exposure and white balance
def camera_skip_frames(camera, camera_conf):
    # Discard derpy frames
    for i in range(camera_conf.ramp_frames):
        capture_frame(camera)

    print("Starting capture")


# Capture a single frame
def capture_frame(camera):
    # print("Taking image...")

    retval, im = camera.read()
    return im


# Remove lens distortion from a captured camera frame
def undistort(img, camera_matrix, dist_coefs):

    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))

    out = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)

    # crop
    x, y, w, h = roi
    out = out[y:y + h, x:x + w]

    return out


# Dummy function for updating positions of hsv bars
def update_bars():
    pass


# Create hsv bars
def create_trackbars(camera_conf, track_params):
    vis = np.zeros((1, camera_conf.frame_height, 3), np.uint8)
    cv2.imshow('bars', vis)
    cv2.createTrackbar('hue min', 'bars', track_params.h_min, 255, update_bars)
    cv2.createTrackbar('hue max', 'bars', track_params.h_max, 255, update_bars)
    cv2.createTrackbar('saturation min', 'bars', track_params.s_min, 255, update_bars)
    cv2.createTrackbar('saturation max', 'bars', track_params.s_max, 255, update_bars)
    cv2.createTrackbar('value min', 'bars', track_params.v_min, 255, update_bars)
    cv2.createTrackbar('value max', 'bars', track_params.v_max, 255, update_bars)
    cv2.createTrackbar('kernel size', 'bars', track_params.kernel_size, 20, update_bars)


# Draw a target icon on the position of the object
def draw_crosshair(img, x, y, colour, camera_conf, size=1.0, make_label=True):

    cv2.circle(img, (x, y), int(20*size), colour, int(2*size))
    if y - int(25*size) > 0:
        cv2.line(img, (x, y), (x, y - int(25*size)), colour, int(2*size))
    else:
        cv2.line(img, (x, y), (x, 0), colour, int(2*size))
    if y + int(25*size) < camera_conf.frame_height:
        cv2.line(img, (x, y), (x, y + int(25*size)), colour, int(2*size))
    else:
        cv2.line(img, (x, y), (x, camera_conf.frame_height), colour, int(2*size))
    if x - int(25*size) > 0:
        cv2.line(img, (x, y), (x - int(25*size), y), colour, int(2*size))
    else:
        cv2.line(img, (x, y), (0, y), colour, int(2*size))
    if x + int(25*size) < camera_conf.frame_width:
        cv2.line(img, (x, y), (x + int(25*size), y), colour, int(2*size))
    else:
        cv2.line(img, (x, y), (camera_conf.frame_width, y), colour, int(2*size))

    if make_label:
        draw_label(img, x.__str__() + ", " + y.__str__(), x, y, colour, camera_conf, size)


def draw_label(img, text, x, y, colour, camera_conf, size=1.0):
    cv2.putText(img, text, (x, y + 35), 1, 1*size, colour, int(2*size))


def draw_box(img, x1, y1, x2, y2, colour, thickness):
    cv2.rectangle(img,(x1,y1),(x2,y2),colour,thickness)


def find_largest(object_list):
    if len(object_list) == 0:
        return ()
    largest_area = -1
    largest_object = -1

    for i in range(len(object_list)):
        area = object_list[i][2]
        if area > largest_area:
            largest_area = area
            largest_object = i

    return object_list[largest_object]

def find_largest_in_area(object_list, x1, y1, x2, y2):
    if len(object_list) == 0:
        return ()
    largest_area = -1
    largest_object = -1

    for i in range(len(object_list)):
        if object_list[i][0] < x1 or object_list[i][0] > x2:
            continue
        if object_list[i][1] < y1 or object_list[i][1] > y2:
            continue
        area = object_list[i][2]
        if area > largest_area:
            largest_area = area
            largest_object = i

    if largest_object == -1:
        return ()

    return object_list[largest_object]


# returns angle in degrees clockwise from 'up' direction aka positive y axis)
def get_angle(top, bottom):
    dx = top[0] - bottom[0]
    dy = top[1] - bottom[1]
    if dx != 0:
        if dx < 0:
            return 270 + 180.0 / cmath.pi * cmath.atan(((float)(dy)) / dx)
        else:
            return 90 + 180.0 / cmath.pi * cmath.atan(((float)(dy)) / dx)
    else:
        if dx > 0:
            return 180
        else:
            return 0


# Finds and returns objects found in the given image. Only returns a maximum number of objects
# Returns a tuple of:
#  x (in pixels from the bottom right corner)
#  y (in pixels from the bottom right corner)
#  area (in pixels)
def track_objects(img, track_params, show_opening=False):

    # Change from RGB to HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Colour mask in HSV
    # Handle going above 180:
    if track_params.h_max > 180:
        colour_mask_lower1 = np.array([0, track_params.s_min, track_params.v_min])
        colour_mask_upper1 = np.array([track_params.h_max % 180, track_params.s_max, track_params.v_max])
        colour_mask_lower2 = np.array([track_params.h_min % 180, track_params.s_min, track_params.v_min])
        colour_mask_upper2 = np.array([180, track_params.s_max, track_params.v_max])

        colour_mask1 = cv2.inRange(img_hsv, colour_mask_lower1, colour_mask_upper1)
        colour_mask2 = cv2.inRange(img_hsv, colour_mask_lower2, colour_mask_upper2)
        colour_mask = cv2.bitwise_or(colour_mask1, colour_mask2)

    else:
        colour_mask_lower = np.array([track_params.h_min, track_params.s_min, track_params.v_min])
        colour_mask_upper = np.array([track_params.h_max, track_params.s_max, track_params.v_max])

        colour_mask = cv2.inRange(img_hsv, colour_mask_lower, colour_mask_upper)

    img_colour_masked = cv2.bitwise_and(img, img, mask=colour_mask)

    cv2.imshow('img_colour_masked', img_colour_masked)

    # Use opening (combined erosion and dilation) to remove noise
    kernel = np.ones((track_params.kernel_size, track_params.kernel_size), np.uint8)
    opening = cv2.morphologyEx(img_colour_masked, cv2.MORPH_OPEN, kernel)

    if show_opening:
        cv2.imshow('opening', opening)

    # Find contours
    img_gray = cv2.cvtColor(opening, cv2.COLOR_BGR2GRAY)

    image, contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) is 0:
        # print("No objects found")
        return []

    object_list = list()

    # Trim number of objects found
    contours = contours[:max_objects]

    for i in range(len(contours)):
        m = cv2.moments(contours[i])
        if m['m00'] < epsilon:
            continue
        x = int(m['m10'] / m['m00'])
        y = int(m['m01'] / m['m00'])
        area = cv2.contourArea(contours[i])
        object_list.append((x, y, area))

    # Other code used for experimenting, don't delete just yet

    # img_gray = cv2.cvtColor(img_colour_masked, cv2.COLOR_BGR2GRAY)
    #
    # edges = cv2.Canny(img_gray, minEdgeVal, maxEdgeVal)
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
    return object_list


#
# Main
#
# Captures and image when no flags
# '-u' undistorts a single image
# '-t' tracks an object with specified sliders, use '-u' together to also undistort
#     -> while in this mode, press 's' to output the current parameters to 'tracking/object#.json'
#     -> 'r' to reset trackbars
#     -> 'q' to quit
#     -> '--load_params' - load previous parameters
# '-a' runs code for the arena
# '--camera_config <filename>' use a custom camera configuration
#
# Params for ivan's testing:
# -u -t --camera_config config/camera_ivan.json
# -a --camera_config config/camera_ivan.json
if __name__ == '__main__':
    import sys
    import getopt

    args, img_mask = getopt.getopt(sys.argv[1:], 'uta', ['camera_config=', 'load_params=', 'ouc_params=',
                                                         'ouc_top_params=', 'ouc_bottom_params=', 'actuator_params='])
    args = dict(args)
    args.setdefault('--camera_config', 'config/camera_arena.json')
    # args.setdefault('--ouc_params', 'tracking/paper_green.json')
    # args.setdefault('--ouc_top_params', 'tracking/paper_blue.json')
    # args.setdefault('--ouc_bottom_params', 'tracking/paper_red.json')
    # args.setdefault('--actuator_params', 'tracking/paper_purple.json')
    args.setdefault('--ouc_params', 'tracking/white_ping.json')
    args.setdefault('--ouc_top_params', 'tracking/green_block_top.json')
    args.setdefault('--ouc_bottom_params', 'tracking/red_block_bottom.json')
    args.setdefault('--actuator_params', 'tracking/blue_actuator.json')

    camera_config = CameraConfig()
    try:
        f_cam_conf = open(args.get('--camera_config'), 'r')
        camera_config.from_dict(json.load(f_cam_conf))
        f_cam_conf.close()
        print('Loaded camera config from \'' + args.get('--camera_config') + '\'')
    except IOError:
        print('Camera config not found, using default config\n')
        # print(json.dumps(camera_config.to_dict(), sort_keys=True, indent=4))

    image_filename = "capture"
    image_ext = ".png"
    params_filename = "tracking/object"
    params_ext = ".json"

    u_flag = args.get('-u')
    t_flag = args.get('-t')
    a_flag = args.get('-a')

    if t_flag is not None:
        # Calibrate tracking with colour trackbars
        track_params = TrackParams()
        if args.get('--load_params') is not None:
            try:
                track_params_f = open(args.get('--load_params'), 'r')
                track_params.from_dict(json.load(track_params_f))
                track_params_f.close()
                print('Loaded starting parameters from \'' + args.get('--load_params') + '\'')
            except IOError:
                print('Could not load starting parameters from \'' + args.get('--load_params') + '\'\n')
                exit(1)

        myCamera = init_camera(camera_config)
        create_trackbars(camera_config, track_params)

        while True:
            camera_capture = capture_frame(myCamera)
            if u_flag is not None:
                camera_capture = undistort(camera_capture,
                                           np.array(camera_config.camera_matrix), np.array(camera_config.dist_coefs))
            # Set trackbar parameters
            track_params.h_min = cv2.getTrackbarPos('hue min', 'bars')
            track_params.h_max = cv2.getTrackbarPos('hue max', 'bars')
            track_params.s_min = cv2.getTrackbarPos('saturation min', 'bars')
            track_params.s_max = cv2.getTrackbarPos('saturation max', 'bars')
            track_params.v_min = cv2.getTrackbarPos('value min', 'bars')
            track_params.v_max = cv2.getTrackbarPos('value max', 'bars')
            track_params.kernel_size = cv2.getTrackbarPos('kernel size', 'bars')

            object_list = track_objects(camera_capture, track_params, show_opening=True)

            largest_area = 0
            largest_object = 0
            x_main = 0
            y_main = 0

            # Analyze objects to find the largest
            for i in range(len(object_list)):
                area = object_list[i][2]
                if area > largest_area:
                    largest_area = area
                    largest_object = i

            # Draw objects
            for i in range(len(object_list)):
                if i == largest_object:
                    draw_crosshair(camera_capture, object_list[i][0], object_list[i][1], (0, 255, 0), camera_config)
                    x_main = object_list[i][0]
                    y_main = object_list[i][1]
                else:
                    draw_crosshair(camera_capture, object_list[i][0], object_list[i][1], (0, 0, 255), camera_config)

            cv2.putText(camera_capture, 'Objects found: ' + len(object_list).__str__(), (0, 30), 2, 1, (0, 255, 0), 2)

            cv2.imshow('img', camera_capture)
            in_read = cv2.waitKey(1) & 0xFF
            if in_read == ord('s'):
                # Make sure we create a new file
                index = 0
                try:
                    while True:
                        infile = open(params_filename + index.__str__() + params_ext, 'r')
                        infile.close()
                        index += 1
                except FileNotFoundError:
                    pass

                filename = params_filename + index.__str__() + params_ext
                outfile = open(filename, 'w')
                json.dump(track_params.to_dict(), outfile, sort_keys=True, indent=4)
                outfile.close()
                print('saved to: ' + filename)
            elif in_read == ord('r'):
                cv2.setTrackbarPos('hue min', 'bars', 0)
                cv2.setTrackbarPos('hue max', 'bars', 255)
                cv2.setTrackbarPos('saturation min', 'bars', 0)
                cv2.setTrackbarPos('saturation max', 'bars', 255)
                cv2.setTrackbarPos('value min', 'bars', 0)
                cv2.setTrackbarPos('value max', 'bars', 255)
                cv2.setTrackbarPos('kernel size', 'bars', 7)
            elif in_read == ord('q'):
                cv2.destroyAllWindows()
                break

        myCamera.release()
    elif u_flag is not None:
        # Undistort a single image
        infile = cv2.imread(image_filename + '0' + image_ext)
        output = undistort(infile, np.array(camera_config.camera_matrix), np.array(camera_config.dist_coefs))
        cv2.imwrite(image_filename + "_undistorted" + image_ext, output)
    elif a_flag is not None:
        # Arena tracking:

        # Variables used by training:
        ouc_x = 0
        ouc_y = 0
        ouc_angle = 0
        actuators_x = [0, 0, 0, 0, 0]
        actuators_y = [0, 0, 0, 0, 0]

        # Helper variables, tuples are (pos_x, pos_y, angle_deg)
        ouc_top_main = (0, 0, 0)
        ouc_bottom_main = (0, 0, 0)

        # Calculate bounding boxes for actuator positions, assumption: x1 < x2, y1 < y2
        # Current idea: take middle square of size frame height
        # Cut into 5 pieces
        actuator_x1 = [0, 0, 0, 0, 0]
        actuator_x2 = [camera_config.frame_width, camera_config.frame_width,
                       camera_config.frame_width, camera_config.frame_width, camera_config.frame_width]
        actuator_y1 = [0, 0, 0, 0, 0]
        actuator_y2 = [camera_config.frame_height, camera_config.frame_height,
                       camera_config.frame_height, camera_config.frame_height, camera_config.frame_height]

        delta = (int)(camera_config.frame_height / num_actuators)
        offset = (int)((camera_config.frame_width - camera_config.frame_height) / 2)

        for i in range(num_actuators):
            actuator_x1[i] = (int)(offset + i*delta)
            actuator_x2[i] = (int)(offset + (i+1)*delta)
            actuator_y1[i] = 0
            actuator_y2[i] = camera_config.frame_height

        # Load tracking parameters
        ouc_params = TrackParams()
        ouc_top_params = TrackParams()
        ouc_bottom_params = TrackParams()
        actuator_params = TrackParams()
        try:
            ouc_params_f = open(args.get('--ouc_params'), 'r')
            ouc_params.from_dict(json.load(ouc_params_f))
            ouc_params_f.close()
            print('Loaded OUC params from \'' + args.get('--ouc_params') + '\'')
        except IOError:
            print('OUC params not found, stopping...\n')
            exit(1)
        try:
            ouc_top_params_f = open(args.get('--ouc_top_params'), 'r')
            ouc_top_params.from_dict(json.load(ouc_top_params_f))
            ouc_top_params_f.close()
            print('Loaded OUC params from \'' + args.get('--ouc_top_params') + '\'')
        except IOError:
            print('OUC params not found, stopping...\n')
            exit(1)
        try:
            ouc_bottom_params_f = open(args.get('--ouc_bottom_params'), 'r')
            ouc_bottom_params.from_dict(json.load(ouc_bottom_params_f))
            ouc_bottom_params_f.close()
            print('Loaded OUC bottom params from \'' + args.get('--ouc_bottom_params') + '\'')
        except IOError:
            print('OUC bottom params not found, stopping...\n')
            exit(1)
        try:
            actuator_params_f = open(args.get('--actuator_params'), 'r')
            actuator_params.from_dict(json.load(actuator_params_f))
            actuator_params_f.close()
            print('Loaded OUC params from \'' + args.get('--actuator_params') + '\'')
        except IOError:
            print('OUC params not found, stopping...\n')
            exit(1)

        myCamera = init_camera(camera_config)

        while True:
            camera_capture = undistort(capture_frame(myCamera),
                                       np.array(camera_config.camera_matrix), np.array(camera_config.dist_coefs))
            ouc_list = track_objects(camera_capture, ouc_params)
            ouc_top_list = track_objects(camera_capture, ouc_top_params)
            ouc_bottom_list = track_objects(camera_capture, ouc_bottom_params)
            actuator_list = track_objects(camera_capture, actuator_params)

            if len(ouc_list) != 0:
                ouc_main = find_largest(ouc_list)
                ouc_x = ouc_main[0]
                ouc_y = ouc_main[1]

                if len(ouc_top_list) != 0 and len(ouc_bottom_list) != 0:
                    ouc_top_main = find_largest(ouc_top_list)
                    ouc_bottom_main = find_largest(ouc_bottom_list)

                    ouc_angle = get_angle(ouc_top_main, ouc_bottom_main)
                else:
                    pass
                    # print('Angle markers not found!')
            else:
                pass
                # print('No \'ouc\' found!')

            if len(actuator_list) != 0:
                for i in range(num_actuators):
                    index = num_actuators - 1 - i
                    temp = find_largest_in_area(actuator_list, actuator_x1[index], actuator_y1[index],
                                                actuator_x2[index], actuator_y2[index])
                    if temp != ():
                        actuators_x[index] = temp[0]
                        actuators_y[index] = temp[1]
                        actuator_list.remove(temp)
            else:
                pass
                # print('No actuators found')

            # Draw UI
            # OUC
            draw_crosshair(camera_capture, ouc_x, ouc_y, (0, 255, 0), camera_config)
            draw_crosshair(camera_capture, ouc_top_main[0], ouc_top_main[1], (128, 0, 0), camera_config, 0.5)
            draw_crosshair(camera_capture, ouc_bottom_main[0], ouc_bottom_main[1], (0, 0, 128), camera_config, 0.5)

            # Actuators
            for i in range(num_actuators):
                draw_crosshair(camera_capture, actuators_x[i], actuators_y[i], (0, 255, 0), camera_config, 0.5)
                draw_box(camera_capture, actuator_x1[i], actuator_y1[i], actuator_x2[i], actuator_y2[i],
                         (128, 128, 0), 2)

            text = 'Angle: {0.real:.1f}'.format(ouc_angle)

            cv2.putText(camera_capture, text, (0, 30), 2, 1, (0, 255, 0), 2)

            cv2.imshow('img', camera_capture)
            in_read = cv2.waitKey(1) & 0xFF
            if in_read == ord('q'):
                cv2.destroyAllWindows()
                break

        myCamera.release()
    else:
        # Capture single images and save to disk
        myCamera = init_camera(camera_config)
        camera_skip_frames(myCamera, camera_config)

        print('With the focus on the image window, press \'s\' to capture or \'q\' to quit...')

        while True:
            camera_capture = capture_frame(myCamera)

            cv2.imshow('img', camera_capture)

            in_read = cv2.waitKey(1) & 0xFF
            if in_read == ord('s'):  # save on pressing 's'
                index = 0
                infile = cv2.imread(image_filename + index.__str__() + image_ext)
                # Make sure we create a new file
                while infile is not None:
                    index += 1
                    infile = cv2.imread(image_filename + index.__str__() + image_ext)

                filename = image_filename + index.__str__() + image_ext
                cv2.imwrite(filename, camera_capture)
                print('saved to: ' + filename)
            elif in_read == ord('q'):
                cv2.destroyAllWindows()
                break

        myCamera.release()
