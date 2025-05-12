import cv2
import numpy as np

THRESH_STOP = 200
THRESH_SLOW = 120
THRESH_AREA = 2000
TOLERANCE_TURN = 8000
MIN_AREA = 1000
MIN_SOLIDITY = 0.5
MIN_EXTENT = 0.5

""" Segmentation Block """
def get_st_el():
    """
    Returns Trapezoidal Structuring element

    :argument:
        None

    :returns:
        st_el (5x9 np-array): Trapezoidal Structuring Element
    """
    st_el = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 1, 1, 1, 0, 0, 0],
                      [0, 0, 1, 1, 1, 1, 1, 0, 0],
                      [0, 1, 1, 1, 1, 1, 1, 1, 0],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1]], np.uint8)

    return st_el

def get_hue_mask(img):
    """
    Returns hue mask for Red and Pink Hues

    :argument:
        img (3-D np-array): Original Captured Frame
    :returns:
         mask (2-D np-array): Binary Mask for Road
    """

    low_thresh1 = np.array([0, 0, 0])
    high_thresh1 = np.array([10, 255, 255])
    low_thresh2 = np.array([130, 0, 0])
    high_thresh2 = np.array([180, 255, 255])

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(img_hsv, low_thresh1, high_thresh1)
    mask2 = cv2.inRange(img_hsv, low_thresh2, high_thresh2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Visualization
    # cv2.imshow('Hue', img_hsv)
    # cv2.imshow('Mask Red Hue', mask1)
    # cv2.imshow('Mask Pink Hue', mask2)
    # cv2.imshow('Mask', mask)

    return mask

def ext_road(img):
    """
    Extract road by applying hue mask and cleaning it

    :argument:
        img (3-D np-array): Original Captured Frame

    :returns:
        img_road (2-D np-array): Cleaned binary mask of road
    """

    mask_hue = get_hue_mask(frame)
    mask_hue_med = cv2.medianBlur(mask_hue, 9)
    mask_erd = cv2.erode(mask_hue_med, get_st_el())

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_erd, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = np.argmax(areas) + 1
    img_road = np.zeros_like(mask_hue)
    img_road[labels == largest_label] = 255

    # Visualization
    # cv2.imshow('Mask Eroded', mask_erd)
    # cv2.imshow('Mask Hue Med', mask_hue_med)
    # cv2.imshow('Mask Hue', mask_hue)

    return img_road

def ext_lanes(img_or, img_hsv):
    """
    Returns segmented image of lanes

    :argument:
        img_or (3-D np-array): Original Captured Frame
        img_hsv (3-D np-array): Original Frame in HSV

    :returns:
        ret_img (3-D np-array): Segmented Original Frame
    """

    low_thresh = np.array([16, 0, 0])
    high_thresh = np.array([25, 255, 255])

    mask_yellow = cv2.inRange(img_hsv, low_thresh, high_thresh)
    ret_img = cv2.bitwise_and(img_or, img_or, mask = mask_yellow)

    return ret_img

""" Object Detection Block """
def extract_edges(frame):
    """
    Extract the edges from frame through dilation

    :argument:
        frame (2-D np-array): Original Frame in grayscale

    :returns:
        edge_mask (2-D image): Binary Mask of edges
    """

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    dilated = cv2.dilate(frame, kernel, iterations=1)
    diff = cv2.absdiff(dilated, frame)
    _, edge_mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

    # For visualization
    cv2.imshow('Edges through Dilation', diff)
    cv2.imshow('Thresholded edges', edge_mask)

    return edge_mask

def clean_mask(mask):
    """
    Clean the mask through morphological operations

    :argument:
        mask (2-D np.array): Binary Mask

    :returns:
        dilated (2-D np.array): Cleaned Mask
    """

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    dilated = cv2.dilate(closed, kernel_dilate, iterations=1)

    # Visualization
    # cv2.imshow("Cleaned Obj Mask", dilated)

    return dilated

def detect_obstacles(edge_mask, road_mask, frame):
    """
    Find Contour area and filter out false hits based on thresholds

    :argument:
        edge_mask (2-D np-array): Edges in the image/frame
        road_mask (2-D np-array): Road mask

    :returns:
        obstacles (A x 4  array): Coordinates of bounding-boxes
    """

    # Filter the edges to those in region of interest
    edge_roi = cv2.bitwise_and(edge_mask, edge_mask, mask=road_mask)

    # Find Contours in Image
    cnts, _ = cv2.findContours(edge_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    obstacles = []

    # Obstacle detection
    for i, cnt in enumerate(cnts):
        # Extract parameters
        area = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area else 0
        x, y, w, h = cv2.boundingRect(cnt)
        extent = float(area) / (w * h) if (w * h) else 0
        print(f"Contour {i}: area={area:.1f}, solidity={solidity:.2f}, extent={extent:.2f}")

        # Visualization
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Determine if obstacle criteria satisfied
        if area < MIN_AREA:
            print(f" - Rejected by area < {MIN_AREA}")
            continue
        if solidity < MIN_SOLIDITY:
            print(f" - Rejected by solidity < {MIN_SOLIDITY}")
            continue
        if extent < MIN_EXTENT:
            print(f" - Rejected by extent < {MIN_EXTENT}")
            continue
        print(f" - Accepted as obstacle (bounding-box=({x},{y},{w},{h}))")

        # Append to obstacles
        obstacles.append((x, y, w, h))

    # Visualization
    cv2.imshow('Edges in RoI', edge_roi)

    return obstacles

""" Decision Block """
def make_decision(bound_box, img_road, frame):
    """
    Makes Start, Stop, Turning Decisions.

    :argument:
         bound_box (2-D array): Coordinates of bounding boxes
         img_road (2-D np-array): Mask of Road
         frame (3-D np-array): Original Frame of image

    :returns:
        None
    """

    # Assuming that the coordinates of the bounding box are stored in an array bound_box[(x, y, w, h)]
    bound_box_np = np.array(bound_box)

    if len(bound_box_np) != 0:
        # Obstacles are present and decision-making required
        yh_values = bound_box_np[:, 1] + bound_box_np[:, 3]

        # Draw bounding boxes to visualize
        for x, y, w, h in bound_box:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        if np.any(yh_values > THRESH_STOP):
            # Stop condition detected
            print("Stop!")
            cv2.putText(frame, "Stop", (0, 20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 0, 255), 2)
        elif np.any(THRESH_SLOW < yh_values < THRESH_STOP):
            # Speed Slow down and direction maneuvers
            print("Slow Down, obstacles ahead!")
            cv2.putText(frame, "Slow Down, obstacles ahead", (0, 20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 0, 255), 2)

            # Determine if enough space that we can move right of left
            yl, hl, xl, yr, hr, xr = get_extreme_coord(bound_box_np)

            # Form and populate the regions
            left_region = np.zeros_like(img_road)
            right_region = np.zeros_like(img_road)
            left_region[yl: yl + hl, 0 : xl] = img_road[yl: yl + hl, 0 : xl]
            right_region[yr : yr + hr, xr:] = img_road[yr : yr + hr, xr:]

            # Visualization
            cv2.rectangle(frame, (0, yl), (xl, yl + hl), (0, 255, 0), 2)
            cv2.rectangle(frame, (xr, yr), (639, yr + hr), (0, 255, 0), 2)

            # Calculate the left and right areas
            left_area = np.sum(left_region == 255)
            right_area = np.sum(right_region == 255)

            # Determine whether to turn right or left
            if left_area > right_area and left_area > THRESH_AREA:
                print('Turn Left to avoid obstacles')
                cv2.putText(frame, "Turn left to avoid obstacles", (0, 20),
                            cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 0, 255), 2)
            elif right_area > left_area and right_area > THRESH_AREA:
                print('Turn right to avoid obstacles')
                cv2.putText(frame, "Turn Right to avoid obstacles", (0, 20),
                            cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 0, 255), 2)
            else:
                print('Continue straight, not much space to avoid obstacles')
                cv2.putText(frame, "Continue straight, not much space to avoid obstacles", (0, 20),
                            cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 0, 255), 2)
    else:
        # No obstacles, determine whether to turn right or left based on curvature
        M = cv2.moments(img_road)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])  # X-coordinate of centroid
            cy = int(M["m01"] / M["m00"])  # Y-coordinate of centroid
        else:
            cx, cy = 0, 0

        # Visualization
        cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)
        cv2.rectangle(frame, (0, THRESH_SLOW), (cx, THRESH_STOP), (0, 255, 0), 2)
        cv2.rectangle(frame, (cx, THRESH_SLOW), (639, THRESH_STOP), (0, 255, 0), 2)

        # Calculate the areas
        left_area = np.sum(img_road[THRESH_SLOW : THRESH_STOP, 0 : cx]) / 255
        right_area = np.sum(img_road[THRESH_SLOW : THRESH_STOP, cx : ]) / 255
        #print("Area: Left -", left_area, "Area Right -", right_area)

        if np.abs(right_area - left_area) > TOLERANCE_TURN:
            if right_area > left_area:
                print("Turn right, Curvature ahead")
                cv2.putText(frame, "Turn Right, Curvature ahead", (0, 20),
                            cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 0, 255), 2)
            else:
                print("Turn Left, Curvature ahead")
                cv2.putText(frame, "Turn Left, Curvature ahead", (0, 20),
                            cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 0, 255), 2)
        else:
            print("Go")
            cv2.putText(frame, "Go", (0, 20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 0, 255), 2)

def get_extreme_coord(bound_box):
    """
    Returns Coordinates of Extreme Bounding Boxes.

    Arguments:
         bound_box (Ax4 np-array): Array of coordinates of obstacles
    Returns:
         yl, hl, xl (triple): Coordinates of left-most box
         yr, hr, xr+wr (triple): Coordinates of right-most box
    """

    # Get extreme bounding boxes
    left_most = bound_box[np.argmin(bound_box[:, 0])]           # Smallest Value of x, column
    xw_sum = bound_box[:, 0] + bound_box[:, 2]
    right_most = bound_box[np.argmin(xw_sum)]                   # Largest value of x+w

    # Get bounding boxes coordinates
    xl, yl, wl, hl = left_most
    xr, yr, wr, hr = right_most

    return yl, hl, xl, yr, hr, xr+wr
    

'''Raspberry Pi Block'''
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)  
picam2.preview_configuration.main.format = "BGR888"  
picam2.configure("preview")
picam2.start()

# Setup socket server
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('0.0.0.0', 8485))  # Listen on all interfaces, port 8485
server_socket.listen(1)
print("Waiting for laptop connection...")
conn, addr = server_socket.accept()
print(f"Connected by: {addr}")

#reducing fps
fps = 5
frame_interval = 1/fps
last_time = time.time()


while True:
    
    current_time = time.time()
    if (current_time - last_time) >= frame_interval:
        last_time = current_time
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    # back to RGB

        # q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        img_road = ext_road(frame)
        img_lanes = ext_lanes(frame, cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))

        # Obstacle Detection
        img_edges = extract_edges(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        img_edges_clean = clean_mask(img_edges)
        obst_coord = detect_obstacles(img_edges_clean, img_road, frame)

        # Decision-Making
        make_decision(obst_coord, img_road, frame)

        # Visualization
        cv2.imshow('Frame', frame)
        cv2.imshow('Road', img_road)

        # Stream via HTTP
        data = pickle.dumps((frame, img_road))
        message = struct.pack("Q", len(data)) + data
        conn.sendall(message)

cv2.destroyAllWindows()
picam2.stop()
