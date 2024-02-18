from flask import Flask, render_template, request
import FinalFinal
import joblib

model = joblib.load('trained_model.joblib')
app = Flask(__name__)



@app.route('/')
def main():
    return render_template('inner-page.html')


@app.route('/predict', methods=['POST'])
def home():
    sex = request.form['gender']
    pred = FinalFinal.main()
    if pred>13.7 and sex=='male':
        line="You are not Anemic"
    if pred>12.1 and sex=='female':
        line="You are not Anemic"
    else:
        line ="You are Anemic ,please visit neareast doctor soon"
    return render_template('after-page.html', data=pred,gender=sex,val=line)
@app.route('/backend-endpoint')
def backend_endpoint():
    # Any backend initialization code can go here
    return jsonify({'status': 'success', 'message': 'Backend initialized'})

@app.route('/process-image', methods=['POST'])
def process_image():
    try:
        data = request.get_json()
        image_data = data['image'].split(',')[1]  # Extract base64 image data
        img_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Save the original image as face.jpg
        cv2.imwrite('static/images/face.jpg', img)

        # Perform face and eye detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                # Save each detected eye as eye1.jpg
                eye_img = img[y + ey:y + ey + eh, x + ex:x + ex + ew]
                cv2.imwrite('static/images/eye.jpg', eye_img)

        return jsonify({'status': 'success', 'message': 'Image processed'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# if __name__ == "__main__":
#     app.run(debug=True)
