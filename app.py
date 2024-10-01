from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__)

# Route to render the HTML form
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle file upload and detection
@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['file']
    if not file:
        return jsonify({'success': False, 'error': 'No file uploaded'})

    # Convert the uploaded file to a numpy array for OpenCV
    image = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Convert the original image to base64
    _, original_buffer = cv2.imencode('.jpg', img)
    original_img_str = base64.b64encode(original_buffer).decode('utf-8')

    # Example detection logic: Face detection using OpenCV
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Convert the processed image (with rectangles) to base64
    _, processed_buffer = cv2.imencode('.jpg', img)
    processed_img_str = base64.b64encode(processed_buffer).decode('utf-8')

    # Return both the original and processed image as base64
    return jsonify({
        'success': True, 
        'original_image': original_img_str, 
        'processed_image': processed_img_str
    })

if __name__ == '__main__':
    app.run(debug=True)
