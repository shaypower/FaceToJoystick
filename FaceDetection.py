import cv2
import numpy as np

# Improved Parameters for Lucas-Kanade Optical Flow for better stability
lk_params = {
    'winSize': (15, 15),  # Smaller window size for better precision
    'maxLevel': 4,  # More pyramid levels for robustness
    'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),  # More iterations for accuracy
    'flags': cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,  # Get eigenvalues for quality assessment
    'minEigThreshold': 1e-4  # Minimum eigenvalue threshold
}

# Load Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(f'{cv2.data.haarcascades}haarcascade_frontalface_default.xml')

# Enhanced preprocessing objects
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # Increased clip limit for better contrast
bilateral_filter_d = 5  # Diameter for bilateral filter
bilateral_sigma_color = 80  # Color space standard deviation
bilateral_sigma_space = 80  # Coordinate space standard deviation

# Load a pre-trained deep learning model for face detection
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

# Variables to store information about the previous frame and detected faces
prev_gray = None  # Previous frame in grayscale
prev_points = None  # Points on the face that we are tracking
previous_faces = []  # List of faces detected in the previous frame

# Tracking quality and smoothing variables
point_history = []  # Store history of tracked points for smoothing
max_history_length = 5  # Maximum number of frames to store in history
tracking_quality_threshold = 0.01  # Minimum quality for tracked points


def preprocess_frame(gray_frame):
    """Enhanced frame preprocessing for better lighting stability."""
    try:
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray_frame, bilateral_filter_d, 
                                     bilateral_sigma_color, bilateral_sigma_space)
        
        # Apply gamma correction for better lighting normalization
        gamma = 1.2
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype("uint8")
        gamma_corrected = cv2.LUT(filtered, table)
        
        # Apply CLAHE for adaptive contrast enhancement
        clahe_applied = clahe.apply(gamma_corrected)
        
        # Gaussian blur for further noise reduction
        final_frame = cv2.GaussianBlur(clahe_applied, (3, 3), 0)
        
        return final_frame
    except Exception:
        print("Failed to apply enhanced preprocessing. Returning original frame.")
        return gray_frame


def detect_faces_dnn(frame):
    """Detect faces in the frame using a deep learning model with tighter bounding boxes."""
    h, w = frame.shape[:2]  # Get the dimensions of the frame
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    faces = []
    # Iterate through detected faces
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # Get the confidence level of detection
        if confidence > 0.5:  # Only consider detections with confidence greater than 0.5
            # Calculate bounding box for detected face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Tighten the bounding box by reducing padding
            face_width = endX - startX
            face_height = endY - startY
            
            # Reduce box size by 20% on each side for tighter fit
            reduction_factor = 0.15
            x_reduction = int(face_width * reduction_factor)
            y_reduction = int(face_height * reduction_factor)
            
            tight_startX = max(0, startX + x_reduction)
            tight_startY = max(0, startY + y_reduction)
            tight_endX = min(w, endX - x_reduction)
            tight_endY = min(h, endY - y_reduction)
            
            # Ensure minimum size
            min_size = 60
            if (tight_endX - tight_startX) >= min_size and (tight_endY - tight_startY) >= min_size:
                faces.append((tight_startX, tight_startY, tight_endX - tight_startX, tight_endY - tight_startY))
    return faces


def detect_faces_with_cascade(gray_frame):
    """Detect faces using Haar Cascade in the grayscale frame with tighter bounding boxes."""
    detected_faces = face_cascade.detectMultiScale(
        gray_frame, 
        scaleFactor=1.05, 
        minNeighbors=12,  # Increased for better accuracy
        minSize=(60, 60),  # Minimum face size
        maxSize=(300, 300)  # Maximum face size to avoid overly large detections
    )
    
    # Tighten the bounding boxes
    tight_faces = []
    for (x, y, w, h) in detected_faces:
        # Reduce box size by 15% on each side for tighter fit
        reduction_factor = 0.12
        x_reduction = int(w * reduction_factor)
        y_reduction = int(h * reduction_factor)
        
        tight_x = max(0, x + x_reduction)
        tight_y = max(0, y + y_reduction)
        tight_w = max(60, w - 2 * x_reduction)  # Ensure minimum width
        tight_h = max(60, h - 2 * y_reduction)  # Ensure minimum height
        
        # Ensure the box doesn't go out of frame bounds
        if tight_x + tight_w <= gray_frame.shape[1] and tight_y + tight_h <= gray_frame.shape[0]:
            tight_faces.append((tight_x, tight_y, tight_w, tight_h))
    
    return tight_faces


def smooth_points(new_points):
    """Apply temporal smoothing to tracked points to reduce jitter."""
    global point_history
    
    if not new_points:
        return []
    
    # Add current points to history
    point_history.append(new_points)
    
    # Keep only recent history
    if len(point_history) > max_history_length:
        point_history.pop(0)
    
    # If we don't have enough history, return current points
    if len(point_history) < 2:
        return new_points
    
    # Apply weighted average smoothing
    smoothed_points = []
    weights = np.linspace(0.1, 1.0, len(point_history))  # More weight to recent points
    weights /= weights.sum()  # Normalize weights
    
    for i, (x, y) in enumerate(new_points):
        avg_x, avg_y = 0, 0
        point_count = 0
        
        # Average across history for this point index
        for j, history_points in enumerate(point_history):
            if i < len(history_points):
                hist_x, hist_y = history_points[i]
                avg_x += hist_x * weights[j]
                avg_y += hist_y * weights[j]
                point_count += weights[j]
        
        if point_count > 0:
            smoothed_points.append((avg_x / point_count, avg_y / point_count))
        else:
            smoothed_points.append((x, y))
    
    return smoothed_points


def detect_face(frame):
    """Detect or track faces in the current frame with improved stability."""
    global prev_gray, prev_points, previous_faces

    # Convert the frame to grayscale and enhance lighting
    gray_frame = preprocess_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    # If this is the first frame or no points to track, perform face detection
    if prev_points is None:
        return detect_faces_control(frame, gray_frame)

    # Try to track the previous points
    good_new, good_old = calculate_optical_flow(gray_frame)
    
    # If tracking fails or quality is poor, fall back to detection
    if len(good_new) < len(prev_points) * 0.5:  # Less than 50% points tracked successfully
        print("Tracking quality degraded, falling back to detection")
        return detect_faces_control(frame, gray_frame)

    # Apply temporal smoothing to reduce jitter
    smoothed_points = smooth_points(good_new)
    
    # Update tracking points and return face bounding boxes
    update_previous_points(smoothed_points, gray_frame)
    return convert_to_bounding_boxes(smoothed_points)


def calculate_center_point(face):
    """Calculate the center point of a detected face bounding box."""
    x, y, w, h = face
    center_x = x + w / 2
    center_y = y + h / 2
    return (center_x, center_y)


def distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def detect_faces_control(frame, gray_frame):
    """Switch between deep learning and Haar Cascade for face detection."""
    faces = detect_faces_dnn(frame) or detect_faces_with_cascade(gray_frame)  # Try both methods

    # Handle the case of multiple detected faces
    if len(faces) > 1:
        if previous_faces:  # If there were faces detected in the last frame
            last_center_point = calculate_center_point(previous_faces[0])  # Center of the last detected face
            # Select the closest face to the last detected one
            faces = [min(faces, key=lambda face: distance(last_center_point, calculate_center_point(face)))]
        else:
            faces = [faces[0]]  # If no previous face, just take the first one

    # Update tracking even if no faces are detected
    if len(faces) > 0:
        update_face_tracking(faces, gray_frame)
    else:
        print("No faces detected in the current frame.")

    return faces


def calculate_optical_flow(gray_frame):
    """Track face movement by calculating optical flow with quality assessment."""
    global prev_gray, prev_points

    # Track points from the previous frame to the current one
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray_frame, prev_points, None, **lk_params)
    
    good_new = []
    good_old = []

    if p1 is not None and st is not None and err is not None:
        # Quality filtering based on status, error, and tracking quality
        for i, ((x_new, y_new), (x_old, y_old), status, error) in enumerate(
            zip(p1.reshape(-1, 2), prev_points.reshape(-1, 2), st.ravel(), err.ravel())):
            
            # Check tracking status and error threshold
            if status == 1 and error < 50:  # Good tracking status and low error
                # Calculate displacement for jump detection
                displacement = np.sqrt((x_new - x_old)**2 + (y_new - y_old)**2)
                
                # Filter out sudden jumps (displacement threshold)
                if displacement < 50:  # Maximum allowed displacement per frame
                    # Check if new point is within any detected face region
                    for (x, y, w, h) in previous_faces:
                        if x <= x_new <= x + w and y <= y_new <= y + h:
                            good_new.append((x_new, y_new))
                            good_old.append((x_old, y_old))
                            break
    
    return good_new, good_old


def update_face_tracking(faces, gray_frame):
    """Update the tracking points based on newly detected faces using good features."""
    global prev_points, prev_gray, previous_faces, point_history

    # Reset point history when redetecting faces
    point_history = []
    
    # Enhanced parameters for goodFeaturesToTrack for tighter face tracking
    feature_params = dict(
        maxCorners=30,       # Reduced for more focused tracking
        qualityLevel=0.02,   # Higher quality threshold for better features
        minDistance=8,       # Closer points allowed for tighter tracking
        blockSize=5          # Smaller block size for finer detail
    )

    all_points = []
    
    for (x, y, w, h) in faces:
        # Create a more focused mask for the central face region
        mask = np.zeros(gray_frame.shape, dtype=np.uint8)
        
        # Focus on the central 80% of the face for better feature selection
        margin_x = int(w * 0.1)
        margin_y = int(h * 0.1)
        inner_x = max(0, x + margin_x)
        inner_y = max(0, y + margin_y)
        inner_w = min(gray_frame.shape[1] - inner_x, w - 2 * margin_x)
        inner_h = min(gray_frame.shape[0] - inner_y, h - 2 * margin_y)
        
        mask[inner_y:inner_y+inner_h, inner_x:inner_x+inner_w] = 255
        
        # Detect good features to track within the focused face region
        corners = cv2.goodFeaturesToTrack(gray_frame, mask=mask, **feature_params)
        
        if corners is not None and len(corners) > 0:
            all_points.extend(corners.reshape(-1, 2))
        else:
            # Fallback strategy: add key facial feature points
            # Center point
            center_x, center_y = x + w // 2, y + h // 2
            all_points.append([center_x, center_y])
            
            # Add points around key facial features for better tracking
            # Upper face (forehead/eyes area)
            all_points.append([center_x - w//4, center_y - h//4])
            all_points.append([center_x + w//4, center_y - h//4])
            
            # Lower face (nose/mouth area)
            all_points.append([center_x, center_y + h//6])
    
    # Prepare points for optical flow tracking
    if all_points:
        prev_points = np.array(all_points, dtype=np.float32).reshape(-1, 1, 2)
    else:
        # Final fallback to face centers
        points = [[x + w / 2, y + h / 2] for x, y, w, h in faces]
        prev_points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    
    prev_gray = gray_frame.copy()
    previous_faces = faces


def reset_tracking():
    """Reset tracking variables to force redetection."""
    global prev_gray, prev_points, previous_faces, point_history
    prev_gray = None
    prev_points = None
    previous_faces = []
    point_history = []


def update_previous_points(good_new, gray_frame):
    """Update the previous points and frame for tracking."""
    global prev_points, prev_gray
    if good_new:
        prev_points = np.array(good_new, dtype=np.float32).reshape(-1, 1, 2)
        prev_gray = gray_frame.copy()


def convert_to_bounding_boxes(good_new):
    """Convert tracked points into accurate, tight bounding boxes for faces."""
    if not good_new:
        return []
    
    faces = []
    
    # Calculate the bounds of all tracked points
    points_array = np.array(good_new)
    min_x = np.min(points_array[:, 0])
    max_x = np.max(points_array[:, 0])
    min_y = np.min(points_array[:, 1])
    max_y = np.max(points_array[:, 1])
    
    # Calculate center and spread of points
    center_x = np.mean(points_array[:, 0])
    center_y = np.mean(points_array[:, 1])
    spread_x = max_x - min_x
    spread_y = max_y - min_y
    
    # If we have previous face information, use it as a reference but adapt size
    if previous_faces and len(previous_faces) > 0:
        prev_w, prev_h = previous_faces[0][2], previous_faces[0][3]
        
        # Adapt size based on point spread but keep it reasonable
        # Use a blend of previous size and current point spread
        adaptive_w = int(0.7 * prev_w + 0.3 * max(spread_x * 2.5, 60))
        adaptive_h = int(0.7 * prev_h + 0.3 * max(spread_y * 2.5, 60))
        
        # Ensure the size doesn't change too drastically
        max_change = 0.3  # Maximum 30% change per frame
        adaptive_w = int(prev_w * (1 - max_change) + adaptive_w * max_change)
        adaptive_h = int(prev_h * (1 - max_change) + adaptive_h * max_change)
        
        # Clamp to reasonable bounds
        adaptive_w = max(60, min(200, adaptive_w))
        adaptive_h = max(60, min(200, adaptive_h))
        
        x = int(center_x - adaptive_w / 2)
        y = int(center_y - adaptive_h / 2)
        faces.append((x, y, adaptive_w, adaptive_h))
    else:
        # No previous face info - create box based on point distribution
        # Make box slightly larger than point spread for better coverage
        box_w = max(80, int(spread_x * 3.0))  # Increased multiplier for better coverage
        box_h = max(80, int(spread_y * 3.0))
        
        # Clamp to reasonable bounds
        box_w = min(150, box_w)
        box_h = min(150, box_h)
        
        x = int(center_x - box_w / 2)
        y = int(center_y - box_h / 2)
        faces.append((x, y, box_w, box_h))
    
    return faces