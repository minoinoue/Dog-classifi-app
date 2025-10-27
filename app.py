from flask import Flask, request, render_template, url_for
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from werkzeug.utils import secure_filename
import numpy as np

try:
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
except Exception as e:
    print(f"Không thể tải mô hình YOLOv5: {e}")
    yolo_model = None

from model_definition import get_model

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_classifier = get_model(len(class_names))
model_classifier.load_state_dict(torch.load("best_resnet50_dogs.pth", map_location=device))
model_classifier.to(device)
model_classifier.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict_breed(image):
    """Hàm nhận ảnh đã crop, dự đoán và trả về tên giống chó."""
    try:
        image_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model_classifier(image_tensor)
            _, predicted_idx = torch.max(outputs, 1)
            predicted_class = class_names[predicted_idx.item()]
        return predicted_class.replace('_', ' ').title()
    except Exception as e:
        print(f"Error predicting breed: {e}")
        return "Không thể dự đoán"


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    uploaded_image_url = None
    error_message = None

    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            error_message = "Bạn chưa chọn file nào."
        else:
            file = request.files['file']
            if file:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                uploaded_image_url = url_for('static', filename='uploads/' + filename)

                if yolo_model is None:
                    error_message = "Lỗi: Không thể tải mô hình phát hiện đối tượng YOLOv5."
                else:
                    try:
                        img = Image.open(filepath).convert('RGB')

                        # Chạy mô hình YOLOv5 để phát hiện đối tượng
                        yolo_results = yolo_model(img)

                        # Lọc kết quả để tìm các đối tượng "dog"
                        # class ID của 'dog' trong COCO là 16
                        detections = yolo_results.xyxy[0][yolo_results.xyxy[0][:, -1] == 16]

                        if len(detections) > 0:
                            # Lấy đối tượng "dog" có độ tin cậy cao nhất
                            best_detection = detections[torch.argmax(detections[:, 4])]
                            x1, y1, x2, y2 = [int(x.item()) for x in best_detection[:4]]

                            # Cắt ảnh theo bounding box
                            cropped_img = img.crop((x1, y1, x2, y2))

                            # Dự đoán giống chó trên ảnh đã cắt
                            prediction_result = predict_breed(cropped_img)
                        else:
                            error_message = "Không tìm thấy chó trong ảnh. Vui lòng thử một ảnh khác."

                    except Exception as e:
                        print(f"Error during object detection: {e}")
                        error_message = "Có lỗi xảy ra trong quá trình xử lý ảnh."

    return render_template('index.html', prediction=prediction_result, image_url=uploaded_image_url,
                           error=error_message)


if __name__ == '__main__':
    # Tạo thư mục uploads nếu chưa có
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
