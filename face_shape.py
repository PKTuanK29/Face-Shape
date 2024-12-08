import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
import tensorflow as tf

# Cài đặt thư viện Streamlit
st.set_page_config(page_title="Face Shape Prediction", layout="wide")

# Tải mô hình đã huấn luyện trước đó (chỉ cần tải một lần)
@st.cache_resource
def load_pretrained_model():
    model = load_model('MyModel.keras')  # Đảm bảo rằng mô hình của bạn đã được tải lên hoặc có sẵn trên hệ thống
    return model

model = load_pretrained_model()

# Đọc dữ liệu từ thư mục
DATA_DIR = '/path/to/your/dataset'
TRAIN_DIR = os.path.join(DATA_DIR, 'training_set')
TEST_DIR = os.path.join(DATA_DIR, 'testing_set')

# Hiển thị thông tin tổng quan về bộ dữ liệu
def display_data_overview():
    # Hiển thị thông tin về bộ dữ liệu
    st.title('Dự đoán hình dạng khuôn mặt')
    
    classes = [class_name for class_name in os.listdir(TRAIN_DIR)]
    st.write(f"Number of classes: {len(classes)}")

    count = []
    for class_name in classes:
        count.append(len(os.listdir(os.path.join(TRAIN_DIR, class_name))))
    
    # Vẽ biểu đồ số lượng mẫu mỗi lớp
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=classes, y=count, ax=ax, palette='viridis')
    ax.set_title('Số lượng mẫu mỗi lớp', fontsize=16)
    ax.set_xlabel('Nhóm', fontsize=12)
    ax.set_ylabel('Số lượng', fontsize=12)
    st.pyplot(fig)

# Hiển thị hình ảnh ngẫu nhiên từ tập huấn luyện
def display_sample_images():
    st.subheader("Một số hình ảnh mẫu")
    df_unique = pd.DataFrame([(os.path.join(TRAIN_DIR, class_name, img_name), class_name) 
                              for class_name in os.listdir(TRAIN_DIR) 
                              for img_name in os.listdir(os.path.join(TRAIN_DIR, class_name))])
    
    fig, axes = plt.subplots(ncols=5, figsize=(15, 5))
    for i, (img_path, label) in enumerate(df_unique.sample(5).values):
        img = Image.open(img_path).resize((224, 224))
        axes[i].imshow(img)
        axes[i].set_title(label)
        axes[i].axis('off')
    st.pyplot(fig)

# Chức năng dự đoán hình ảnh
def preprocess_image(image):
    img = Image.open(image).resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Xử lý dữ liệu người dùng upload và dự đoán
uploaded_file = st.file_uploader("Chọn hình ảnh để dự đoán", type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    st.image(uploaded_file, caption="Hình ảnh đầu vào", use_column_width=True)
    preprocessed_image = preprocess_image(uploaded_file)
    predictions = model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions)
    st.write(f"Nhóm dự đoán: {predicted_class}")

# Hiển thị thống kê và kết quả huấn luyện
def display_training_results():
    # Ví dụ về việc hiển thị kết quả huấn luyện trong Streamlit
    history = model.history  # Đảm bảo rằng bạn lưu lại `history` khi huấn luyện mô hình
    st.subheader("Đồ thị kết quả huấn luyện")
    
    # Đồ thị mất mát
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'], label='Mất mát (Training)', color='b')
    ax.plot(history.history['val_loss'], label='Mất mát (Validation)', color='r')
    ax.set_title('Mất mát qua các epoch')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    st.pyplot(fig)

    # Đồ thị độ chính xác
    fig, ax = plt.subplots()
    ax.plot(history.history['accuracy'], label='Độ chính xác (Training)', color='b')
    ax.plot(history.history['val_accuracy'], label='Độ chính xác (Validation)', color='r')
    ax.set_title('Độ chính xác qua các epoch')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.legend()
    st.pyplot(fig)

# Xử lý hiển thị confusion matrix
def display_confusion_matrix():
    st.subheader("Confusion Matrix")
    # Giả sử bạn có một bộ dữ liệu test và hàm để đánh giá mô hình
    y_true = np.array([])  # Nhãn thực tế
    y_pred = np.array([])  # Dự đoán của mô hình
    
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    st.pyplot(fig)

# Hiển thị tất cả các phần của ứng dụng
if st.button("Hiển thị tổng quan bộ dữ liệu"):
    display_data_overview()

if st.button("Hiển thị một số hình ảnh mẫu"):
    display_sample_images()

if st.button("Hiển thị kết quả huấn luyện"):
    display_training_results()

if st.button("Hiển thị Confusion Matrix"):
    display_confusion_matrix()
