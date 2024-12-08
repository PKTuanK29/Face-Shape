import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Kiểm tra nếu tệp mô hình đã có trong thư mục cục bộ
model_path = 'MyModel.keras'
if not os.path.exists(model_path):
    st.error(f"Mô hình không tìm thấy tại {model_path}. Vui lòng tải lại mô hình.")

# Load mô hình đã huấn luyện
model = tf.keras.models.load_model(model_path)

# Nhãn của các lớp
class_labels = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']

# Tiền xử lý ảnh
def preprocess_image(image_file):
    img = Image.open(image_file)  # Đọc từ BytesIO
    img = img.resize((224, 224))  # Resize ảnh
    img_array = np.array(img) / 255.0  # Chuẩn hóa giá trị pixel
    img_array = np.expand_dims(img_array, axis=0)  # Thêm chiều batch size
    return img_array

# Dự đoán ảnh
def predict_image(image_file, model, class_labels):
    img_array = preprocess_image(image_file)  # Tiền xử lý ảnh
    predictions = model.predict(img_array)  # Dự đoán
    predicted_class = np.argmax(predictions, axis=1)[0]  # Lấy lớp có xác suất cao nhất
    predicted_label = class_labels[predicted_class]  # Lấy tên lớp từ nhãn
    predicted_prob = np.max(predictions)  # Lấy xác suất của lớp dự đoán
    return predictions, predicted_label, predicted_prob

# Tiêu đề của trang web
st.title("Dự đoán Hình Dạng Khuôn Mặt")
st.markdown("Chọn một bức ảnh khuôn mặt để dự đoán hình dạng.")

# Tải ảnh lên
uploaded_file = st.file_uploader("Tải ảnh của bạn lên", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Hiển thị ảnh đã tải lên
    img = Image.open(uploaded_file)
    st.image(img, caption="Ảnh đã tải lên", use_column_width=True)
    st.write("")
    
    # Dự đoán ảnh
    predictions, predicted_label, predicted_prob = predict_image(uploaded_file, model, class_labels)

    # Hiển thị kết quả dự đoán
    st.write(f"Dự đoán: {predicted_label} với xác suất {predicted_prob:.2f}")
    
    # Hiển thị đồ thị về kết quả dự đoán
    st.subheader("Đồ thị dự đoán")
    fig, ax = plt.subplots()
    ax.bar(class_labels, predictions[0])  # Dùng predictions đã có sẵn
    ax.set_ylabel('Xác suất')
    ax.set_xlabel('Hình dáng khuôn mặt')
    ax.set_title('Dự đoán xác suất của từng lớp')
    st.pyplot(fig)
