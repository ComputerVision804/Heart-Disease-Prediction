import os
import joblib
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import uuid
from flask import Flask, render_template, request, send_from_directory

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load structured model (replace with your actual model file)
model = joblib.load('models/model.pkl')

# Define CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Load CNN model weights (replace with your actual model file)
cnn_model = SimpleCNN(num_classes=2)
cnn_model.load_state_dict(torch.load('cnn_heart_disease.pth', map_location=torch.device('cpu')))
cnn_model.eval()

# Transformation for image input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Form inputs for structured data
    age = request.form.get('age')
    if not age:
        return render_template('error.html', message="Age is required")
    
    try:
        age = int(age)
    except ValueError:
        return render_template('error.html', message="Invalid age value. Please enter a valid number.")
    
    # Get other structured data
    sex = int(request.form['sex'])
    cp = int(request.form['cp'])
    trestbps = int(request.form['trestbps'])
    chol = int(request.form['chol'])
    fbs = int(request.form['fbs'])
    restecg = int(request.form['restecg'])
    thalach = int(request.form['thalach'])
    exang = int(request.form['exang'])
    oldpeak = float(request.form['oldpeak'])
    slope = int(request.form['slope'])
    ca = int(request.form['ca'])
    thal = int(request.form['thal'])

    # Predict from structured data
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal]])
    structured_prediction = model.predict(features)[0]

    structured_result = {
        'text': "Heart disease risk detected based on form input.",
        'class': "red"
    } if structured_prediction == 1 else {
        'text': "No heart disease risk based on form input.",
        'class': "green"
    }

    # Handle image predictions if images are uploaded
    image_result = []
    annotated_paths = []
    if 'images' in request.files:
        files = request.files.getlist('images')
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        for file in files:
            if file and file.filename != '':
                # Create a unique filename for each image
                file_id = str(uuid.uuid4())[:8]
                filename = f"{file_id}_{file.filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # Process the image for prediction
                image = Image.open(filepath).convert('RGB')
                input_tensor = transform(image).unsqueeze(0)

                with torch.no_grad():
                    output = cnn_model(input_tensor)
                    probs = torch.nn.functional.softmax(output, dim=1)
                    confidence = probs.max().item() * 100
                    _, predicted = torch.max(output, 1)

                label = "Heart Disease" if predicted.item() == 1 else "Normal"
                color = "red" if predicted.item() == 1 else "green"
                label_text = f"{label} ({confidence:.2f}%)"

                # Annotate the image
                fig, ax = plt.subplots()
                ax.imshow(image)
                ax.axis('off')
                ax.text(10, 30, label_text, fontsize=14, color=color,
                        bbox=dict(facecolor='white', alpha=0.8))

                # Save annotated image
                annotated_name = f"annotated_{filename}"
                annotated_path = os.path.join(app.config['UPLOAD_FOLDER'], annotated_name)
                plt.savefig(annotated_path, bbox_inches='tight')
                plt.close()

                # Append paths and labels to the result
                annotated_paths.append(annotated_name)
                image_result.append({
                    'label': label_text,
                    'color': color,
                    'image': annotated_name
                })

    return render_template('result.html',
                           structured_result=structured_result,
                           image_result=image_result)
@app.route('/uploads/<filename>')   
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
