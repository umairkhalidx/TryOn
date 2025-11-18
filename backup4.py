from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import io
from typing import Dict, List, Tuple

app = Flask(__name__)
CORS(app)

# ---------- EYELASH DETECTION SETUP ----------
mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE_UPPER = [246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
RIGHT_EYE_UPPER = [466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]

LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263

# ---------- EYELASH NAME MAPPING ----------
EYELASH_NAMES = {
    "Drunk In Love": "Drunk In Love",
    "Wedding Day": "Wedding Day",
    "Foxy": "Foxy",
    "Flare": "Flare",
    "Vixen": "Vixen",
    "Other Half 1": "Other Half 1",
    "Other Half 2": "Other Half 2",
    "Staycation": "Staycation",
    "Iconic": "Iconic"
}

# ---------- EYELASH RECOMMENDATION SYSTEM ----------
class EyelashRecommender:
    def __init__(self):
        # Eyelash database with detailed attributes
        self.eyelashes = {
            "Drunk in Love": {
                "style": "cat_eye",
                "intensity": "medium",
                "eye_sizes": ["small", "hooded_small"],
                "look": "glam",
                "description": "Proper Cat eye lash style, medium (not natural not too heavy) perfect for small hooded eyes if they want a glam look.",
                "image_id": "1"
            },
            "Flare": {
                "style": "doll_eye",
                "intensity": "natural",
                "eye_sizes": ["small", "hooded_small"],
                "look": "natural",
                "description": "Natural doll eye, designed for small hooded eyes. For natural looks.",
                "image_id": "2"
            },
            "Foxy": {
                "style": "slight_cat_eye",
                "intensity": "medium",
                "eye_sizes": ["medium", "big"],
                "look": "soft_glam",
                "description": "Slight Cat eye, medium, super wispy gives a soft cat eye look without looking too heavy. Perfect for medium to big eyes.",
                "image_id": "3"
            },
            "Iconic": {
                "style": "slight_cat_eye",
                "intensity": "medium_heavy",
                "eye_sizes": ["small", "medium", "big", "hooded_small", "hooded_medium"],
                "look": "versatile",
                "description": "Slight cat eye, medium-heavy, wispy lash style, suitable for all eye shapes.",
                "image_id": "4"
            },
            "Other Half 1": {
                "style": "half_lash_natural",
                "intensity": "natural",
                "eye_sizes": ["small", "medium", "big", "hooded_small", "hooded_medium"],
                "look": "natural",
                "description": "Most natural, half lash style, applied at the end (corner) of the eye to give cat eye look but very soft. Perfect for hooded eyes too.",
                "image_id": "5"
            },
            "Other Half 2": {
                "style": "half_lash_cat",
                "intensity": "medium",
                "eye_sizes": ["small", "medium", "big", "hooded_small", "hooded_medium"],
                "look": "lifted",
                "description": "Cat eye, medium, half lash, apply at the corner of the eye to give cat eye lifted look. Best for hooded eyes too.",
                "image_id": "6"
            },
            "Staycation": {
                "style": "doll_eye",
                "intensity": "heavy",
                "eye_sizes": ["medium", "big"],
                "look": "dramatic",
                "description": "Heavy, doll eye, super fluffy and thick, best suitable for medium to big eyes. Not good for hooded eyes.",
                "image_id": "7"
            },
            "Vixen": {
                "style": "cat_eye",
                "intensity": "heavy",
                "eye_sizes": ["big", "hooded_medium"],
                "look": "dramatic",
                "description": "Heavy, cat eye, best for big eyes or medium hooded eyes.",
                "image_id": "8"
            },
            "Wedding Day": {
                "style": "doll_eye",
                "intensity": "medium",
                "eye_sizes": ["small", "medium", "big", "hooded_small", "hooded_medium"],
                "look": "versatile",
                "description": "Doll eye, medium, suitable for all eye shapes and sizes.",
                "image_id": "9"
            }
        }
        
        # Key landmarks for eye analysis
        self.LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 157, 173, 246]
        self.RIGHT_EYE_INDICES = [362, 263, 387, 386, 385, 384, 398, 466]
        self.LEFT_EYE_UPPER = [159, 158, 157, 173]
        self.LEFT_EYE_LOWER = [144, 145, 153, 154]
        self.RIGHT_EYE_UPPER = [386, 385, 384, 398]
        self.RIGHT_EYE_LOWER = [373, 374, 380, 381]
        
        # Eyelid landmarks for hood detection
        self.LEFT_UPPER_EYELID = [246, 161, 160, 159, 158, 157, 173, 133]
        self.LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65]
        self.RIGHT_UPPER_EYELID = [466, 388, 387, 386, 385, 384, 398, 362]
        self.RIGHT_EYEBROW = [300, 293, 334, 296, 336, 285, 295]

    def get_euclidean_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points."""
        return np.linalg.norm(point1 - point2)

    def analyze_eye_characteristics(self, landmarks, img_width: int, img_height: int) -> Dict:
        """Analyze eye size, aspect ratio, and hooded characteristics."""
        
        # Convert landmarks to pixel coordinates
        left_eye_points = np.array([[landmarks[i].x * img_width, 
                                     landmarks[i].y * img_height] 
                                    for i in self.LEFT_EYE_INDICES])
        right_eye_points = np.array([[landmarks[i].x * img_width, 
                                      landmarks[i].y * img_height] 
                                     for i in self.RIGHT_EYE_INDICES])
        
        # Calculate eye widths (horizontal distance)
        left_eye_width = self.get_euclidean_distance(left_eye_points[0], left_eye_points[1])
        right_eye_width = self.get_euclidean_distance(right_eye_points[0], right_eye_points[1])
        avg_eye_width = (left_eye_width + right_eye_width) / 2
        
        # Calculate eye heights (vertical distance at center)
        left_upper = np.array([landmarks[159].x * img_width, landmarks[159].y * img_height])
        left_lower = np.array([landmarks[145].x * img_width, landmarks[145].y * img_height])
        left_eye_height = self.get_euclidean_distance(left_upper, left_lower)
        
        right_upper = np.array([landmarks[386].x * img_width, landmarks[386].y * img_height])
        right_lower = np.array([landmarks[374].x * img_width, landmarks[374].y * img_height])
        right_eye_height = self.get_euclidean_distance(right_upper, right_lower)
        
        avg_eye_height = (left_eye_height + right_eye_height) / 2
        
        # Calculate eye aspect ratio (width/height)
        eye_aspect_ratio = avg_eye_width / avg_eye_height if avg_eye_height > 0 else 0
        
        # Calculate face width for normalization (distance between temples)
        face_left = np.array([landmarks[234].x * img_width, landmarks[234].y * img_height])
        face_right = np.array([landmarks[454].x * img_width, landmarks[454].y * img_height])
        face_width = self.get_euclidean_distance(face_left, face_right)
        
        # Normalize eye width relative to face width
        normalized_eye_width = avg_eye_width / face_width if face_width > 0 else 0
        
        # Check for hooded eyes
        is_hooded = self.detect_hooded_eyes(landmarks, img_width, img_height)
        
        # Determine eye size category based on normalized width and aspect ratio
        eye_size = self.categorize_eye_size(normalized_eye_width, eye_aspect_ratio, is_hooded)
        
        return {
            "eye_width": float(avg_eye_width),
            "eye_height": float(avg_eye_height),
            "eye_aspect_ratio": float(eye_aspect_ratio),
            "normalized_eye_width": float(normalized_eye_width),
            "face_width": float(face_width),
            "is_hooded": bool(is_hooded),
            "eye_size": eye_size,
            "eye_size_display": eye_size.replace('_', ' ').title()
        }

    def detect_hooded_eyes(self, landmarks, img_width: int, img_height: int) -> bool:
        """Detect if eyes are hooded by analyzing eyelid visibility."""
        
        # Analyze left eye
        left_upper_lid = np.array([[landmarks[i].x * img_width, 
                                    landmarks[i].y * img_height] 
                                   for i in self.LEFT_UPPER_EYELID])
        left_eyebrow = np.array([[landmarks[i].x * img_width, 
                                  landmarks[i].y * img_height] 
                                 for i in self.LEFT_EYEBROW])
        
        # Calculate vertical distances between eyelid and eyebrow
        left_distances = []
        for lid_point in left_upper_lid[1:-1]:  # Skip corner points
            min_dist = min([self.get_euclidean_distance(lid_point, brow_point) 
                           for brow_point in left_eyebrow])
            left_distances.append(min_dist)
        
        # Analyze right eye
        right_upper_lid = np.array([[landmarks[i].x * img_width, 
                                     landmarks[i].y * img_height] 
                                    for i in self.RIGHT_UPPER_EYELID])
        right_eyebrow = np.array([[landmarks[i].x * img_width, 
                                   landmarks[i].y * img_height] 
                                  for i in self.RIGHT_EYEBROW])
        
        right_distances = []
        for lid_point in right_upper_lid[1:-1]:
            min_dist = min([self.get_euclidean_distance(lid_point, brow_point) 
                           for brow_point in right_eyebrow])
            right_distances.append(min_dist)
        
        # Calculate average eyelid-to-eyebrow distance
        avg_distance = np.mean(left_distances + right_distances)
        
        # Calculate eye height for normalization
        left_upper = np.array([landmarks[159].x * img_width, landmarks[159].y * img_height])
        left_lower = np.array([landmarks[145].x * img_width, landmarks[145].y * img_height])
        eye_height = self.get_euclidean_distance(left_upper, left_lower)
        
        # Normalized ratio
        hood_ratio = avg_distance / eye_height if eye_height > 0 else 0
        
        # Threshold for hooded eyes
        return hood_ratio < 2.5

    def categorize_eye_size(self, normalized_width: float, aspect_ratio: float, is_hooded: bool) -> str:
        """Categorize eye size into small, medium, or big, considering hooded eyes."""
        
        if is_hooded:
            if normalized_width < 0.16:
                return "hooded_small"
            elif normalized_width < 0.185:
                return "hooded_medium"
            else:
                return "hooded_medium"
        else:
            if normalized_width < 0.155:
                return "small"
            elif normalized_width < 0.18:
                return "medium"
            else:
                return "big"

    def recommend_eyelashes(self, eye_characteristics: Dict) -> List[Dict]:
        """Recommend eyelashes based on eye characteristics with confidence scores."""
        
        recommendations = []
        eye_size = eye_characteristics["eye_size"]
        
        for name, details in self.eyelashes.items():
            # Check if this lash is suitable for the detected eye size
            if eye_size in details["eye_sizes"]:
                # Calculate confidence score based on various factors
                confidence = 0.8  # Base confidence
                
                # Boost confidence for exact matches
                if len(details["eye_sizes"]) <= 3:  # More specific lashes
                    confidence += 0.1
                
                # Boost for versatile options
                if "versatile" in details["look"]:
                    confidence += 0.05
                
                recommendations.append({
                    "name": name,
                    "style": details["style"].replace('_', ' ').title(),
                    "intensity": details["intensity"].replace('_', ' ').title(),
                    "look": details["look"].replace('_', ' ').title(),
                    "description": details["description"],
                    "confidence": round(confidence * 100, 0),
                    "image_id": details["image_id"]
                })
        
        # Sort by confidence
        recommendations.sort(key=lambda x: x["confidence"], reverse=True)
        
        return recommendations

    def process_image(self, image_bytes) -> Dict:
        """Process an image and return recommendations."""
        
        # Decode image
        file_bytes = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Could not read image")
        
        img_height, img_width, _ = img.shape
        
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process with Face Mesh
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:
            results = face_mesh.process(image_rgb)
            
            if not results.multi_face_landmarks:
                raise ValueError("No face detected in the image")
            
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Analyze eye characteristics
            eye_chars = self.analyze_eye_characteristics(landmarks, img_width, img_height)
            
            # Get recommendations
            recommendations = self.recommend_eyelashes(eye_chars)
            
            return {
                "eye_characteristics": eye_chars,
                "recommendations": recommendations,
                "total_recommendations": len(recommendations)
            }


# Initialize recommender globally
recommender = EyelashRecommender()


# ---------- HELPER FUNCTIONS FOR TRY-ON ----------
def rotate_image(image, angle):
    """Rotate image around its center"""
    if angle == 0:
        return image
    
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new bounding dimensions
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust rotation matrix to account for translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    # Perform rotation with transparent background
    rotated = cv2.warpAffine(image, M, (new_w, new_h), 
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, 
                            borderValue=(0, 0, 0, 0))
    return rotated


def overlay_transparent_png(background, overlay_img, x, y, width, height, rotation_angle=0):
    """Overlay transparent PNG on background with proper rotation handling"""
    
    # First resize the overlay
    overlay_resized = cv2.resize(overlay_img, (width, height), interpolation=cv2.INTER_AREA)
    
    # Apply rotation if specified
    if rotation_angle != 0:
        overlay_resized = rotate_image(overlay_resized, rotation_angle)
        
        # Calculate the center offset caused by rotation
        # The original center should remain at (x + width//2, y + height//2)
        new_height, new_width = overlay_resized.shape[:2]
        
        # Adjust position to keep the CENTER of the rotated image at the same location
        x = x - (new_width - width) // 2
        y = y - (new_height - height) // 2
        
        # Update dimensions to rotated dimensions
        width = new_width
        height = new_height
    
    # Get dimensions
    bg_h, bg_w = background.shape[:2]
    ov_h, ov_w = overlay_resized.shape[:2]
    
    # Calculate overlay region (clipped to background boundaries)
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bg_w, x + ov_w), min(bg_h, y + ov_h)
    
    # Check if overlay is completely outside background
    if x1 >= x2 or y1 >= y2:
        return background
    
    # Calculate overlay crop coordinates
    crop_x1 = max(0, -x)
    crop_y1 = max(0, -y)
    crop_x2 = crop_x1 + (x2 - x1)
    crop_y2 = crop_y1 + (y2 - y1)
    
    overlay_crop = overlay_resized[crop_y1:crop_y2, crop_x1:crop_x2]
    
    # Extract alpha channel if present
    if overlay_crop.shape[2] == 4:
        alpha = overlay_crop[:, :, 3] / 255.0
        overlay_rgb = overlay_crop[:, :, :3]
    else:
        alpha = np.ones((overlay_crop.shape[0], overlay_crop.shape[1]))
        overlay_rgb = overlay_crop
    
    # Alpha blend
    bg_region = background[y1:y2, x1:x2]
    for c in range(3):
        bg_region[:, :, c] = (alpha * overlay_rgb[:, :, c] + 
                             (1 - alpha) * bg_region[:, :, c])
    
    background[y1:y2, x1:x2] = bg_region
    return background

def get_eye_region_info(landmarks, eye_upper_indices, inner_idx, outer_idx):
    upper_points = np.array([landmarks[i] for i in eye_upper_indices])
    inner = np.array(landmarks[inner_idx])
    outer = np.array(landmarks[outer_idx])
    eye_width = int(np.linalg.norm(outer - inner))
    center_x = int((inner[0] + outer[0]) / 2)
    center_y = int(np.mean(upper_points[:, 1]))
    return {'center_x': center_x, 'center_y': center_y, 'width': eye_width}


def process_eyelash(image_bytes, eyelash_path, vertical_offset=-10, horizontal_offset=0, 
                   size_scale=2.0, height_scale=1.0, rotation_angle=0):
    """Receives image bytes and eyelash path, returns processed image bytes with adjustments"""
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image")

    eyelash_img = cv2.imread(eyelash_path, cv2.IMREAD_UNCHANGED)
    if eyelash_img is None:
        raise ValueError("Eyelash image not found")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    output = img.copy()

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(img_rgb)
        if not results.multi_face_landmarks:
            raise ValueError("No face detected")

        landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in results.multi_face_landmarks[0].landmark]

        left_info = get_eye_region_info(landmarks, LEFT_EYE_UPPER, LEFT_EYE_INNER, LEFT_EYE_OUTER)
        right_info = get_eye_region_info(landmarks, RIGHT_EYE_UPPER, RIGHT_EYE_INNER, RIGHT_EYE_OUTER)

        lash_aspect = eyelash_img.shape[1] / eyelash_img.shape[0]

        # Apply adjustments
        # LEFT EYE
        lw = int(left_info['width'] * size_scale)
        lh = int((lw / lash_aspect) * height_scale)
        lx = left_info['center_x'] - lw // 2 + horizontal_offset
        ly = left_info['center_y'] + vertical_offset - lh // 2

        # RIGHT EYE (negative rotation for mirrored eye)
        rw = int(right_info['width'] * size_scale)
        rh = int((rw / lash_aspect) * height_scale)
        rx = right_info['center_x'] - rw // 2 - horizontal_offset
        ry = right_info['center_y'] + vertical_offset - rh // 2

        # Apply eyelashes with rotation
        output = overlay_transparent_png(output, eyelash_img, rx, ry, rw, rh, -rotation_angle)
        flipped = cv2.flip(eyelash_img, 1)
        output = overlay_transparent_png(output, flipped, lx, ly, lw, lh, rotation_angle)

    _, buffer = cv2.imencode('.jpg', output)
    return buffer.tobytes()


# ---------- API ROUTES ----------

@app.route('/')
def home():
    return jsonify({
        "message": "Eyelash Detection and Recommendation API",
        "endpoints": {
            "try_on": "/upload (POST) - Try on eyelashes with adjustments",
            "adjust": "/adjust (POST) - Adjust eyelash placement",
            "recommend": "/recommend (POST) - Get eyelash recommendations based on eye analysis",
            "settings": "/settings (GET) - Get default settings and valid ranges",
            "eyelashes": "/eyelashes (GET) - Get all available eyelashes"
        },
        "available_eyelashes": list(EYELASH_NAMES.keys())
    })


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    if 'choice' not in request.form:
        return jsonify({"error": "No eyelash choice provided"}), 400

    image_file = request.files['image']
    choice = request.form['choice']

    if choice not in EYELASH_NAMES:
        return jsonify({"error": f"Invalid choice. Must be one of: {list(EYELASH_NAMES.keys())}"}), 400

    # Get adjustment parameters with default values
    vertical_offset = int(request.form.get('vertical_offset', -10))
    horizontal_offset = int(request.form.get('horizontal_offset', 0))
    size_scale = float(request.form.get('size_scale', 2.0))
    height_scale = float(request.form.get('height_scale', 1.0))
    rotation_angle = float(request.form.get('rotation_angle', 0))

    # Validate ranges
    if not (-50 <= vertical_offset <= 50):
        return jsonify({"error": "Vertical offset must be between -50 and 50"}), 400
    if not (-50 <= horizontal_offset <= 50):
        return jsonify({"error": "Horizontal offset must be between -50 and 50"}), 400
    if not (0.5 <= size_scale <= 3.0):
        return jsonify({"error": "Size scale must be between 0.5 and 3.0"}), 400
    if not (0.5 <= height_scale <= 2.0):
        return jsonify({"error": "Height scale must be between 0.5 and 2.0"}), 400
    if not (-45 <= rotation_angle <= 45):
        return jsonify({"error": "Rotation angle must be between -45 and 45 degrees"}), 400

    eyelash_path = f"eyelashes/{choice}.png"
    try:
        result_bytes = process_eyelash(
            image_file.read(), 
            eyelash_path, 
            vertical_offset=vertical_offset,
            horizontal_offset=horizontal_offset,
            size_scale=size_scale,
            height_scale=height_scale,
            rotation_angle=rotation_angle
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    response = make_response(result_bytes)
    response.headers.set('Content-Type', 'image/jpeg')
    response.headers.set('Content-Disposition', 'inline; filename=result.jpg')
    return response


@app.route('/adjust', methods=['POST'])
def adjust_eyelashes():
    """Alternative endpoint specifically for adjustments"""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    data = request.form
    image_file = request.files['image']
    
    # Required parameters
    choice = data.get('choice')
    if not choice or choice not in EYELASH_NAMES:
        return jsonify({"error": f"Invalid choice. Must be one of: {list(EYELASH_NAMES.keys())}"}), 400

    # Adjustment parameters with defaults
    vertical_offset = int(data.get('vertical_offset', -10))
    horizontal_offset = int(data.get('horizontal_offset', 0))
    size_scale = float(data.get('size_scale', 2.0))
    height_scale = float(data.get('height_scale', 1.0))
    rotation_angle = float(data.get('rotation_angle', 0))

    # Validate ranges
    if not (-50 <= vertical_offset <= 50):
        return jsonify({"error": "Vertical offset must be between -50 and 50"}), 400
    if not (-50 <= horizontal_offset <= 50):
        return jsonify({"error": "Horizontal offset must be between -50 and 50"}), 400
    if not (0.5 <= size_scale <= 3.0):
        return jsonify({"error": "Size scale must be between 0.5 and 3.0"}), 400
    if not (0.5 <= height_scale <= 2.0):
        return jsonify({"error": "Height scale must be between 0.5 and 2.0"}), 400
    if not (-45 <= rotation_angle <= 45):
        return jsonify({"error": "Rotation angle must be between -45 and 45 degrees"}), 400

    eyelash_path = f"eyelashes/{choice}.png"
    try:
        result_bytes = process_eyelash(
            image_file.read(), 
            eyelash_path, 
            vertical_offset=vertical_offset,
            horizontal_offset=horizontal_offset,
            size_scale=size_scale,
            height_scale=height_scale,
            rotation_angle=rotation_angle
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    response = make_response(result_bytes)
    response.headers.set('Content-Type', 'image/jpeg')
    response.headers.set('Content-Disposition', 'inline; filename=result.jpg')
    return response


@app.route('/recommend', methods=['POST'])
def recommend_eyelashes():
    """Analyze eye characteristics and recommend suitable eyelashes"""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    image_file = request.files['image']
    
    try:
        # Process image and get recommendations
        result = recommender.process_image(image_file.read())
        
        return jsonify({
            "success": True,
            "analysis": {
                "eye_size": result["eye_characteristics"]["eye_size_display"],
                "is_hooded": result["eye_characteristics"]["is_hooded"],
                "eye_width": round(result["eye_characteristics"]["eye_width"], 2),
                "eye_height": round(result["eye_characteristics"]["eye_height"], 2),
                "aspect_ratio": round(result["eye_characteristics"]["eye_aspect_ratio"], 2)
            },
            "recommendations": result["recommendations"],
            "total_recommendations": result["total_recommendations"]
        })
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route('/settings', methods=['GET'])
def get_default_settings():
    """Endpoint to get default and valid range of settings"""
    return jsonify({
        "default_settings": {
            "vertical_offset": -10,
            "horizontal_offset": 0,
            "size_scale": 2.0,
            "height_scale": 1.0,
            "rotation_angle": 0
        },
        "valid_ranges": {
            "vertical_offset": {"min": -50, "max": 50},
            "horizontal_offset": {"min": -50, "max": 50},
            "size_scale": {"min": 0.5, "max": 3.0},
            "height_scale": {"min": 0.5, "max": 2.0},
            "rotation_angle": {"min": -45, "max": 45}
        }
    })


@app.route('/eyelashes', methods=['GET'])
def get_eyelashes():
    """Get information about all available eyelashes"""
    eyelashes_info = []
    for name, details in recommender.eyelashes.items():
        eyelashes_info.append({
            "name": name,
            "style": details["style"].replace('_', ' ').title(),
            "intensity": details["intensity"].replace('_', ' ').title(),
            "look": details["look"].replace('_', ' ').title(),
            "description": details["description"],
            "image_id": details["image_id"],
            "suitable_for": [size.replace('_', ' ').title() for size in details["eye_sizes"]]
        })
    
    return jsonify({
        "success": True,
        "total": len(eyelashes_info),
        "eyelashes": eyelashes_info
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)