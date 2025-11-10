from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS  # ✅ Import CORS
import cv2
import mediapipe as mp
import numpy as np
import io

app = Flask(__name__)
CORS(app)  # ✅ Allow all origins (you can also specify origins=["*"] if needed)

# ---------- EYELASH DETECTION SETUP ----------
mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE_UPPER = [246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
RIGHT_EYE_UPPER = [466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]

LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263


def overlay_transparent_png(background, overlay_img, x, y, width, height):
    overlay_resized = cv2.resize(overlay_img, (width, height), interpolation=cv2.INTER_AREA)
    bg_h, bg_w = background.shape[:2]

    ov_h, ov_w = overlay_resized.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bg_w, x + ov_w), min(bg_h, y + ov_h)

    if x1 >= x2 or y1 >= y2:
        return background

    overlay_crop = overlay_resized[0:y2 - y1, 0:x2 - x1]

    if overlay_crop.shape[2] == 4:
        alpha = overlay_crop[:, :, 3] / 255.0
        overlay_rgb = overlay_crop[:, :, :3]
    else:
        alpha = np.ones((overlay_crop.shape[0], overlay_crop.shape[1]))
        overlay_rgb = overlay_crop

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


def process_eyelash(image_bytes, eyelash_path, vertical_offset=-10, horizontal_offset=0, size_scale=2.0, height_scale=1.0):
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

        # RIGHT EYE
        rw = int(right_info['width'] * size_scale)
        rh = int((rw / lash_aspect) * height_scale)
        rx = right_info['center_x'] - rw // 2 - horizontal_offset  # Opposite direction for right eye
        ry = right_info['center_y'] + vertical_offset - rh // 2

        # Apply eyelashes
        output = overlay_transparent_png(output, eyelash_img, rx, ry, rw, rh)
        flipped = cv2.flip(eyelash_img, 1)
        output = overlay_transparent_png(output, flipped, lx, ly, lw, lh)

    _, buffer = cv2.imencode('.jpg', output)
    return buffer.tobytes()


@app.route('/')
def home():
    return jsonify({"message": "Eyelash Detection and Placement API"})


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    if 'choice' not in request.form:
        return jsonify({"error": "No eyelash choice provided"}), 400

    image_file = request.files['image']
    choice = request.form['choice']

    if choice not in ['1', '2', '3', '4']:
        return jsonify({"error": "Invalid choice. Must be 1, 2, 3, or 4."}), 400

    # Get adjustment parameters with default values
    vertical_offset = int(request.form.get('vertical_offset', -10))
    horizontal_offset = int(request.form.get('horizontal_offset', 0))
    size_scale = float(request.form.get('size_scale', 2.0))
    height_scale = float(request.form.get('height_scale', 1.0))

    # Validate ranges
    if not (-50 <= vertical_offset <= 50):
        return jsonify({"error": "Vertical offset must be between -50 and 50"}), 400
    if not (-50 <= horizontal_offset <= 50):
        return jsonify({"error": "Horizontal offset must be between -50 and 50"}), 400
    if not (0.5 <= size_scale <= 3.0):
        return jsonify({"error": "Size scale must be between 0.5 and 3.0"}), 400
    if not (0.5 <= height_scale <= 2.0):
        return jsonify({"error": "Height scale must be between 0.5 and 2.0"}), 400

    eyelash_path = f"eyelashes/L{choice}.png"
    try:
        result_bytes = process_eyelash(
            image_file.read(), 
            eyelash_path, 
            vertical_offset=vertical_offset,
            horizontal_offset=horizontal_offset,
            size_scale=size_scale,
            height_scale=height_scale
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
    choice = data.get('choice', '1')
    if choice not in ['1', '2', '3', '4']:
        return jsonify({"error": "Invalid choice. Must be 1, 2, 3, or 4."}), 400

    # Adjustment parameters with defaults
    vertical_offset = int(data.get('vertical_offset', -10))
    horizontal_offset = int(data.get('horizontal_offset', 0))
    size_scale = float(data.get('size_scale', 2.0))
    height_scale = float(data.get('height_scale', 1.0))

    # Validate ranges
    if not (-50 <= vertical_offset <= 50):
        return jsonify({"error": "Vertical offset must be between -50 and 50"}), 400
    if not (-50 <= horizontal_offset <= 50):
        return jsonify({"error": "Horizontal offset must be between -50 and 50"}), 400
    if not (0.5 <= size_scale <= 3.0):
        return jsonify({"error": "Size scale must be between 0.5 and 3.0"}), 400
    if not (0.5 <= height_scale <= 2.0):
        return jsonify({"error": "Height scale must be between 0.5 and 2.0"}), 400

    eyelash_path = f"eyelashes/L{choice}.png"
    try:
        result_bytes = process_eyelash(
            image_file.read(), 
            eyelash_path, 
            vertical_offset=vertical_offset,
            horizontal_offset=horizontal_offset,
            size_scale=size_scale,
            height_scale=height_scale
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    response = make_response(result_bytes)
    response.headers.set('Content-Type', 'image/jpeg')
    response.headers.set('Content-Disposition', 'inline; filename=result.jpg')
    return response


@app.route('/settings', methods=['GET'])
def get_default_settings():
    """Endpoint to get default and valid range of settings"""
    return jsonify({
        "default_settings": {
            "vertical_offset": -10,
            "horizontal_offset": 0,
            "size_scale": 2.0,
            "height_scale": 1.0
        },
        "valid_ranges": {
            "vertical_offset": {"min": -50, "max": 50},
            "horizontal_offset": {"min": -50, "max": 50},
            "size_scale": {"min": 0.5, "max": 3.0},
            "height_scale": {"min": 0.5, "max": 2.0}
        }
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)