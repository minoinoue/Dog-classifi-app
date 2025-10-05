from flask import Flask, request, render_template, url_for
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from werkzeug.utils import secure_filename

from model_definition import get_model

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model(len(class_names))
model.load_state_dict(torch.load("best_resnet50_dogs.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_breed(image_path):
    """Hàm nhận đường dẫn ảnh, dự đoán và trả về tên giống chó."""
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted_idx = torch.max(outputs, 1)
            predicted_class = class_names[predicted_idx.item()]

        return predicted_class.replace('_', ' ').title()
    except Exception as e:
        print(f"Error predicting: {e}")
        return "Không thể dự đoán"


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    uploaded_image_url = None

    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            return render_template('index.html', error="Bạn chưa chọn file nào.")

        file = request.files['file']

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            prediction_result = predict_breed(filepath)

            uploaded_image_url = url_for('static', filename='uploads/' + filename)

    return render_template('index.html', prediction=prediction_result, image_url=uploaded_image_url)


if __name__ == '__main__':
    # Tạo thư mục uploads nếu chưa có
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)