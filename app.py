from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import math
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

# ---------- NEW EYELASH RECOMMENDATION SYSTEM ----------
class EyelashRecommendationSystem:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # Eye landmarks indices
        self.RIGHT_EYE = [33, 133, 160, 159, 158, 157, 173, 155, 154, 153, 145, 144, 163, 7]
        self.LEFT_EYE = [263, 362, 387, 386, 385, 384, 398, 382, 381, 380, 374, 373, 390, 249]
        
        # Key points for measurements
        self.RIGHT_EYE_INNER = 133
        self.RIGHT_EYE_OUTER = 33
        self.LEFT_EYE_INNER = 362
        self.LEFT_EYE_OUTER = 263
        
        # Inventory with detailed specifications
        self.inventory = {
            "Cat Eye Styles": {
                "Foxy": {
                    "suitable_sizes": ["Small", "Medium", "Large"],
                    "suitable_shapes": ["Almond", "Round", "Upturned"],
                    "style_type": "Elongating & Dramatic",
                    "description": "Perfect cat eye with outer corner emphasis",
                    "intensity": "Medium",
                    "look": "Soft Glam"
                },
                "Drunk In Love": {
                    "suitable_sizes": ["Small", "Medium"],
                    "suitable_shapes": ["Almond", "Round", "Upturned", "Downturned"],
                    "style_type": "Subtle Cat Eye",
                    "description": "Soft cat eye effect for everyday glamour",
                    "intensity": "Medium",
                    "look": "Glam"
                },
                "Other Half 2": {
                    "suitable_sizes": ["Small"],
                    "suitable_shapes": ["Almond", "Round", "Upturned", "Hooded"],
                    "style_type": "Delicate Cat Eye",
                    "description": "Lightweight cat eye for smaller eyes",
                    "intensity": "Medium",
                    "look": "Lifted"
                },
                "Vixen": {
                    "suitable_sizes": ["Large"],
                    "suitable_shapes": ["Almond", "Round", "Upturned"],
                    "style_type": "Bold Cat Eye",
                    "description": "Dramatic cat eye for larger eyes",
                    "intensity": "Heavy",
                    "look": "Dramatic"
                }
            },
            "Doll Eye Styles": {
                "Iconic": {
                    "suitable_sizes": ["Small", "Medium", "Large"],
                    "suitable_shapes": ["Round", "Almond", "Upturned", "Downturned", "Hooded"],
                    "style_type": "Universal Doll Eye",
                    "description": "Classic doll eye - works for everyone",
                    "intensity": "Medium-Heavy",
                    "look": "Versatile"
                },
                "Wedding Day": {
                    "suitable_sizes": ["Small", "Medium"],
                    "suitable_shapes": ["Round", "Almond", "Upturned", "Downturned", "Hooded"],
                    "style_type": "Romantic Doll Eye",
                    "description": "Soft, romantic doll effect",
                    "intensity": "Medium",
                    "look": "Versatile"
                },
                "Staycation": {
                    "suitable_sizes": ["Large"],
                    "suitable_shapes": ["Round", "Almond", "Upturned"],
                    "style_type": "Voluminous Doll Eye",
                    "description": "Full, dramatic doll eye for larger eyes",
                    "intensity": "Heavy",
                    "look": "Dramatic"
                }
            },
            "Natural Styles": {
                "Flare": {
                    "suitable_sizes": ["Small", "Medium"],
                    "suitable_shapes": ["Almond", "Upturned", "Downturned", "Round"],
                    "style_type": "Natural with Subtle Flare",
                    "description": "Natural look with gentle outer corner lift",
                    "intensity": "Natural",
                    "look": "Natural"
                },
                "Other Half 1": {
                    "suitable_sizes": ["Small"],
                    "suitable_shapes": ["Hooded"],
                    "style_type": "Natural for Hooded Eyes",
                    "description": "Specially designed for small hooded eyes",
                    "intensity": "Natural",
                    "look": "Natural"
                }
            }
        }
        
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def get_eye_measurements(self, landmarks, img_width, img_height):
        """Extract all eye measurements"""
        measurements = {}
        
        # Convert landmarks to pixel coordinates
        right_eye_inner = (landmarks[self.RIGHT_EYE_INNER].x * img_width, 
                          landmarks[self.RIGHT_EYE_INNER].y * img_height)
        right_eye_outer = (landmarks[self.RIGHT_EYE_OUTER].x * img_width,
                          landmarks[self.RIGHT_EYE_OUTER].y * img_height)
        left_eye_inner = (landmarks[self.LEFT_EYE_INNER].x * img_width,
                         landmarks[self.LEFT_EYE_INNER].y * img_height)
        left_eye_outer = (landmarks[self.LEFT_EYE_OUTER].x * img_width,
                         landmarks[self.LEFT_EYE_OUTER].y * img_height)
        
        # Right eye measurements
        right_eye_top = (landmarks[159].x * img_width, landmarks[159].y * img_height)
        right_eye_bottom = (landmarks[145].x * img_width, landmarks[145].y * img_height)
        right_eye_top2 = (landmarks[160].x * img_width, landmarks[160].y * img_height)
        right_eye_bottom2 = (landmarks[144].x * img_width, landmarks[144].y * img_height)
        
        # Left eye measurements
        left_eye_top = (landmarks[386].x * img_width, landmarks[386].y * img_height)
        left_eye_bottom = (landmarks[374].x * img_width, landmarks[374].y * img_height)
        left_eye_top2 = (landmarks[387].x * img_width, landmarks[387].y * img_height)
        left_eye_bottom2 = (landmarks[373].x * img_width, landmarks[373].y * img_height)
        
        # Calculate absolute measurements
        right_eye_width = self.calculate_distance(right_eye_inner, right_eye_outer)
        left_eye_width = self.calculate_distance(left_eye_inner, left_eye_outer)
        
        right_eye_height1 = self.calculate_distance(right_eye_top, right_eye_bottom)
        right_eye_height2 = self.calculate_distance(right_eye_top2, right_eye_bottom2)
        right_eye_height = (right_eye_height1 + right_eye_height2) / 2
        
        left_eye_height1 = self.calculate_distance(left_eye_top, left_eye_bottom)
        left_eye_height2 = self.calculate_distance(left_eye_top2, left_eye_bottom2)
        left_eye_height = (left_eye_height1 + left_eye_height2) / 2
        
        # Inter-eye distance
        inter_eye_distance = self.calculate_distance(right_eye_inner, left_eye_inner)
        
        # Face width (for relative measurements)
        face_left = (landmarks[234].x * img_width, landmarks[234].y * img_height)
        face_right = (landmarks[454].x * img_width, landmarks[454].y * img_height)
        face_width = self.calculate_distance(face_left, face_right)
        
        # Calculate RELATIVE measurements (scale-invariant)
        avg_eye_width = (right_eye_width + left_eye_width) / 2
        
        measurements['right_eye_width_relative'] = right_eye_width / face_width
        measurements['left_eye_width_relative'] = left_eye_width / face_width
        measurements['avg_eye_width_relative'] = avg_eye_width / face_width
        
        # Eye Aspect Ratio (height/width) - naturally relative
        measurements['right_ear'] = right_eye_height / right_eye_width
        measurements['left_ear'] = left_eye_height / left_eye_width
        measurements['avg_ear'] = (measurements['right_ear'] + measurements['left_ear']) / 2
        
        # Inter-eye distance relative to eye width
        measurements['inter_eye_ratio'] = inter_eye_distance / avg_eye_width
        
        # Calculate eye angles (upturned vs downturned)
        right_angle = math.degrees(math.atan2(
            right_eye_outer[1] - right_eye_inner[1],
            right_eye_outer[0] - right_eye_inner[0]
        ))
        left_angle = math.degrees(math.atan2(
            left_eye_inner[1] - left_eye_outer[1],
            left_eye_inner[0] - left_eye_outer[0]
        ))
        
        measurements['right_eye_angle'] = right_angle
        measurements['left_eye_angle'] = left_angle
        measurements['avg_eye_angle'] = (right_angle + left_angle) / 2
        
        # Eyelid visibility (hooded detection)
        right_crease = (landmarks[157].x * img_width, landmarks[157].y * img_height)
        left_crease = (landmarks[384].x * img_width, landmarks[384].y * img_height)
        
        right_lid_visibility = self.calculate_distance(right_crease, right_eye_top) / right_eye_height
        left_lid_visibility = self.calculate_distance(left_crease, left_eye_top) / left_eye_height
        
        measurements['right_lid_visibility'] = right_lid_visibility
        measurements['left_lid_visibility'] = left_lid_visibility
        measurements['avg_lid_visibility'] = (right_lid_visibility + left_lid_visibility) / 2
        
        # Symmetry score
        measurements['symmetry_score'] = 1 - abs(right_eye_width - left_eye_width) / avg_eye_width
        
        return measurements
    
    def classify_eye_shape(self, measurements):
        """Classify eye shape based on measurements"""
        ear = measurements['avg_ear']
        angle = measurements['avg_eye_angle']
        lid_visibility = measurements['avg_lid_visibility']
        
        # Eye shape classification
        if lid_visibility < 0.3:
            shape = "Hooded"
        elif ear > 0.5:
            shape = "Round"
        elif ear < 0.35:
            if angle > 2:
                shape = "Upturned"
            elif angle < -2:
                shape = "Downturned"
            else:
                shape = "Almond"
        else:
            if angle > 3:
                shape = "Upturned"
            elif angle < -3:
                shape = "Downturned"
            else:
                shape = "Almond"
        
        return shape
    
    def classify_eye_size(self, measurements):
        """Classify eye size relative to face"""
        eye_width_ratio = measurements['avg_eye_width_relative']
        
        if eye_width_ratio < 0.15:
            return "Small"
        elif eye_width_ratio > 0.18:
            return "Large"
        else:
            return "Medium"
    
    def classify_eye_spacing(self, measurements):
        """Classify eye spacing"""
        inter_eye_ratio = measurements['inter_eye_ratio']
        
        if inter_eye_ratio < 0.95:
            return "Close-set"
        elif inter_eye_ratio > 1.15:
            return "Wide-set"
        else:
            return "Average-set"
    
    def calculate_match_score(self, shape, size, spacing, product_details):
        """Calculate how well a product matches the eye characteristics"""
        score = 100
        
        # Perfect match for hooded eyes with specific products
        if shape == "Hooded" and "Hooded" in product_details["suitable_shapes"]:
            score += 50
        
        # Prioritize universal products
        if len(product_details["suitable_sizes"]) == 3:  # Universal size
            score += 10
        if len(product_details["suitable_shapes"]) >= 4:  # Works with many shapes
            score += 10
        
        # Cat eye styles work best for elongation
        if shape in ["Round", "Downturned"] and "Cat Eye" in product_details["style_type"]:
            score += 20
        
        # Doll eye styles work best for round enhancement
        if shape in ["Almond", "Upturned"] and "Doll Eye" in product_details["style_type"]:
            score += 15
        
        # Natural styles for subtle looks
        if size == "Small" and "Natural" in product_details["style_type"]:
            score += 15
        
        return score
    
    def recommend_eyelashes(self, shape, size, spacing):
        """Recommend eyelashes based on eye classification from actual inventory"""
        
        # Find matching products
        recommended_products = []
        
        for category, products in self.inventory.items():
            for product_name, details in products.items():
                # Check if product matches eye size and shape
                size_match = size in details["suitable_sizes"]
                shape_match = shape in details["suitable_shapes"]
                
                if size_match and shape_match:
                    recommended_products.append({
                        "name": product_name,
                        "category": category,
                        "style_type": details["style_type"],
                        "description": details["description"],
                        "intensity": details["intensity"],
                        "look": details["look"],
                        "match_score": self.calculate_match_score(shape, size, spacing, details)
                    })
        
        # Sort by match score
        recommended_products.sort(key=lambda x: x["match_score"], reverse=True)
        
        # Get top 3 recommendations
        top_recommendations = recommended_products[:3] if len(recommended_products) >= 3 else recommended_products
        
        # Spacing-based application tips
        spacing_tips = {
            "Close-set": "Focus application on outer 2/3 of lash line to create width",
            "Wide-set": "Focus application on inner 2/3 of lash line to bring eyes closer",
            "Average-set": "Apply evenly across entire lash line for balanced look"
        }
        
        # Shape-based tips
        shape_tips = {
            "Hooded": "Your hooded eyes look best with curled, wispy lashes that lift and open the eye. Avoid heavy styles.",
            "Round": "Elongate your beautiful round eyes with cat-eye styles that emphasize the outer corners.",
            "Almond": "Lucky you! Your almond eyes are versatile and can rock any lash style - go bold!",
            "Downturned": "Lift your eye shape with curled lashes that have extra volume at the outer corners.",
            "Upturned": "Balance your naturally lifted eyes with even length across the lash line."
        }
        
        return {
            "top_picks": top_recommendations,
            "all_suitable": recommended_products,
            "application_tip": spacing_tips.get(spacing, ""),
            "shape_tip": shape_tips.get(shape, ""),
            "total_matches": len(recommended_products)
        }
    
    def analyze_and_recommend(self, image_bytes):
        """Main function to analyze image and provide recommendations"""
        # Decode image
        file_bytes = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Could not load image")
        
        img_height, img_width = image.shape[:2]
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        with self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:
            results = face_mesh.process(rgb_image)
            
            if not results.multi_face_landmarks:
                raise ValueError("No face detected in image")
            
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Get measurements
            measurements = self.get_eye_measurements(landmarks, img_width, img_height)
            
            # Classify eyes
            eye_shape = self.classify_eye_shape(measurements)
            eye_size = self.classify_eye_size(measurements)
            eye_spacing = self.classify_eye_spacing(measurements)
            
            # Get recommendations
            recommendations = self.recommend_eyelashes(eye_shape, eye_size, eye_spacing)
            
            # Compile results
            return {
                "classification": {
                    "eye_shape": eye_shape,
                    "eye_size": eye_size,
                    "eye_spacing": eye_spacing
                },
                "measurements": {
                    "eye_aspect_ratio": round(measurements['avg_ear'], 3),
                    "eye_angle": round(measurements['avg_eye_angle'], 2),
                    "lid_visibility": round(measurements['avg_lid_visibility'], 3),
                    "inter_eye_ratio": round(measurements['inter_eye_ratio'], 3),
                    "eye_width_to_face_ratio": round(measurements['avg_eye_width_relative'], 3),
                    "symmetry_score": round(measurements['symmetry_score'], 3)
                },
                "recommendations": recommendations
            }


# Initialize the new recommender globally
recommender = EyelashRecommendationSystem()


# ---------- HELPER FUNCTIONS FOR TRY-ON (UNCHANGED) ----------
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
        # Process image and get recommendations using NEW system
        result = recommender.analyze_and_recommend(image_file.read())
        
        return jsonify({
            "success": True,
            "analysis": {
                "eye_shape": result["classification"]["eye_shape"],
                "eye_size": result["classification"]["eye_size"],
                "eye_spacing": result["classification"]["eye_spacing"],
                "measurements": {
                    "eye_aspect_ratio": result["measurements"]["eye_aspect_ratio"],
                    "eye_angle": result["measurements"]["eye_angle"],
                    "lid_visibility": result["measurements"]["lid_visibility"],
                    "inter_eye_ratio": result["measurements"]["inter_eye_ratio"],
                    "eye_width_to_face_ratio": result["measurements"]["eye_width_to_face_ratio"],
                    "symmetry_score": result["measurements"]["symmetry_score"]
                }
            },
            "recommendations": {
                "top_picks": result["recommendations"]["top_picks"],
                "all_suitable": result["recommendations"]["all_suitable"],
                "total_matches": result["recommendations"]["total_matches"],
                "application_tip": result["recommendations"]["application_tip"],
                "shape_tip": result["recommendations"]["shape_tip"]
            }
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
    """Get information about all available eyelashes from inventory"""
    eyelashes_info = []
    
    for category, products in recommender.inventory.items():
        for name, details in products.items():
            eyelashes_info.append({
                "name": name,
                "category": category,
                "style": details["style_type"],
                "intensity": details["intensity"],
                "look": details["look"],
                "description": details["description"],
                "suitable_sizes": details["suitable_sizes"],
                "suitable_shapes": details["suitable_shapes"]
            })
    
    return jsonify({
        "success": True,
        "total": len(eyelashes_info),
        "eyelashes": eyelashes_info
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)