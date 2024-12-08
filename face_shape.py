import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# Kiểm tra nếu mô hình đã có sẵn
try:
    model = tf.keras.models.load_model(r'C:\Users\HP\Downloads\MyModel.keras')
except Exception as e:
    st.error(f"Không thể tải mô hình: {e}")
    model = None

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
    
    # Kiểm tra mô hình và dự đoán
    if model is not None:
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
    else:
        st.error("Không thể tải mô hình, vui lòng thử lại sau.")
