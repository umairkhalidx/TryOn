from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_image():
    # Check if the request has an image file
    if 'image' not in request.files:
        return jsonify({"error": "No image file found"}), 400

    image = request.files['image']

    # Check if a choice was provided
    choice = request.form.get('choice')
    if not choice:
        return jsonify({"error": "No choice provided"}), 400

    # Validate that choice is 1, 2, 3, or 4
    if choice not in ['1', '2', '3', '4']:
        return jsonify({"error": "Invalid choice. Must be 1, 2, 3, or 4."}), 400

    # (Optional) Save the image if you want to
    # image.save(f"uploads/{image.filename}")

    return jsonify({
        "message": "Input successful",
        "choice": choice,
        "filename": image.filename
    }), 200


@app.route('/')
def home():
    return jsonify({"message": "Eyelash Detection API"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
