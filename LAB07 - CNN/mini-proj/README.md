# Mini-Project: Convolutional Neural Network Image Classification (Lion vs Tiger)
**Version:** 5.0.0

## Project Overview
โปรเจกต์นี้เป็นการพัฒนาระบบจำแนกรูปภาพสัตว์ (Tiger vs Lion) ด้วยโครงข่ายประสาทเทียมแบบคอนโวลูชัน **Convolutional Neural Network (CNN)** โดยครอบคลุมกระบวนการ Machine Learning Pipeline แบบครบวงจร ตั้งแต่การโหลดรูปภาพ, การประมวลผลเบื้องต้น (Preprocessing), การแบ่งชุดข้อมูล (Train/Val/Test Split), การสร้างและเทรนโมเดล CNN, การประเมินประสิทธิภาพโมเดล (Evaluation), การทดลองเปรียบเทียบสถาปัตยกรรม CNN Configurations และจำนวน Epochs, ตลอดจนการทดสอบการทำนายผลลัพธ์ (Inference & Visualization)

---

## References & Sources

- **Dataset Source :**  
  [Tiger vs Lion Image Classification Dataset (Kaggle)](https://www.kaggle.com/datasets/sonalshinde123/tiger-vs-lion-image-classification-dataset)  
  ชุดข้อมูลรูปภาพของสิงโต (Lion) และเสือ (Tiger) รวม 460 รูปภาพ

- **Code Reference :**  
  อ้างอิงและพัฒนาต่อยอดจากสถาปัตยกรรม Pipeline ใน `LAB06 - NN` โดยปรับเปลี่ยนโครงสร้างโมเดลจาก Pure Multi-Layer Perceptron (MLP) มาเป็น Convolutional Neural Network (Conv2D, MaxPooling2D, Flatten, Dense) พร้อมการทดลองเปรียบเทียบเชิงลึก

---

## Project Structure

```text
mini-proj/
│
├── main.py                            # สคริปต์หลักสำหรับรันกระบวนการทั้งหมด (End-to-End Pipeline 7 ขั้นตอน)
├── data_loader.py                     # ฟังก์ชันสำหรับอ่านรูปภาพจากโฟลเดอร์และคัดกรองไฟล์เสีย
├── preprocessing.py                   # แปลงระบบสี BGR -> RGB และปรับขนาดภาพ (Resize 100x100)
├── split_data.py                      # แบ่งชุดข้อมูลเป็น Training, Validation และ Test Set
├── cnn_model.py                       # โครงสร้างโมเดล Convolutional Neural Network (CNN), เทรนโมเดล และทำนายผล
├── evaluate.py                        # คำนวณ Metrics, Confusion Matrix, History และ Prediction sample
├── compare_experiments.py             # ฟังก์ชันและสคริปต์เปรียบเทียบ CNN Configurations และ Epochs
├── test_cnn.py                        # สคริปต์สุ่มทดสอบโมเดลกับภาพตัวอย่าง 4 รูป (2x2 Grid)
│
└── outputs/                           # โฟลเดอร์เก็บผลลัพธ์จากการทำงาน
    ├── features.npy                   # ข้อมูล Array ของฟีเจอร์รูปภาพทั้งหมด
    ├── labels.npy                     # ข้อมูล Array ของ Label ทั้งหมด
    ├── classes.json                   # รายชื่อ Class (Lion, tiger)
    ├── X_train.npy / y_train.npy      # ข้อมูลและ Label สำหรับการฝึกสอน (Train Set)
    ├── X_val.npy / y_val.npy          # ข้อมูลและ Label สำหรับการตรวจสอบ (Validation Set)
    ├── X_test.npy / y_test.npy        # ข้อมูลและ Label สำหรับการทดสอบ (Test Set)
    ├── cnn_model.keras                # ไฟล์โมเดล CNN ที่บันทึกหลังการฝึกสอน
    ├── history.json                   # บันทึกประวัติ Loss และ Accuracy ของแต่ละ Epoch
    ├── confusion_matrix.png           # รูปภาพ Confusion Matrix แสดงผลการจำแนก
    ├── training_history.png           # กราฟเปรียบเทียบ Loss และ Accuracy ตลอดการฝึกสอน
    ├── prediction_sample.png          # รูปภาพตัวอย่างการพยากรณ์พร้อมค่าความเชื่อมั่น (Confidence %)
    ├── comparison_configs.png         # กราฟเปรียบเทียบประสิทธิภาพแต่ละ Configuration
    └── comparison_epochs.png          # กราฟเปรียบเทียบความแม่นยำตามจำนวน Epochs
```

---

## Pipeline Workflow

กระบวนการทำงานของระบบแบ่งออกเป็น 7 ขั้นตอนหลัก:

1. **โหลดชุดข้อมูล (Data Loading - data_loader.py)**:
   - อ่านรูปภาพจากโฟลเดอร์ `Animal/` โดยตรวจจับคลาสย่อยอัตโนมัติ (`Lion`, `tiger`)
   - กรองเฉพาะไฟล์ภาพที่รองรับ (`.jpg`, `.jpeg`, `.png`, `.bmp`) และคัดกรองข้ามไฟล์ที่เสียหาย

2. **เตรียมข้อมูลรูปภาพ (Preprocessing - preprocessing.py)**:
   - แปลงระบบสีของรูปภาพจาก BGR เป็น RGB เพื่อให้สีถูกต้อง
   - ปรับขนาดรูปภาพ (Resize) ให้มีมิติเท่ากันที่ **100x100 พิกเซล**

3. **แบ่งชุดข้อมูล (Data Splitting - split_data.py)**:
   - แบ่งข้อมูลด้วยวิธี Stratified Splitting เพื่อรักษาสัดส่วนของทั้ง 2 คลาส:
     - **Training Set (70%)**: สำหรับให้โมเดลเรียนรู้ (322 ภาพ)
     - **Validation Set (10%)**: สำหรับตรวจสอบระหว่างเทรน (46 ภาพ)
     - **Test Set (20%)**: สำหรับวัดประสิทธิภาพขั้นสุดท้าย (92 ภาพ)

4. **สร้างและฝึกสอนโมเดล (Model Training - cnn_model.py)**:
   - ออกแบบสถาปัตยกรรม **Convolutional Neural Network (CNN)** ประกอบด้วย:
     - ชั้น `Rescaling(1./255)` ปรับสเกลค่าพิกเซลให้อยู่ในช่วง [0, 1]
     - ชั้น `Conv2D(32, 3x3)` + `MaxPooling2D(2x2)` สกัดฟีเจอร์ระดับต่ำ (ขอบและเส้น)
     - ชั้น `Conv2D(64, 3x3)` + `MaxPooling2D(2x2)` สกัดฟีเจอร์ระดับกลาง (พื้นผิวและลวดลาย)
     - ชั้น `Conv2D(128, 3x3)` + `MaxPooling2D(2x2)` สกัดฟีเจอร์ระดับสูง (รูปทรงและใบหน้าสัตว์)
     - ชั้น `Flatten()` แปลงมิติ 2D Feature Map เป็น 1D Feature Vector
     - ชั้น `Dense(128)` พร้อม Activation 'relu' และ L2 Regularization
     - ชั้น `Dropout(0.3)` เพื่อลดปัญหา Overfitting
     - ชั้น Output `Dense(1)` ด้วย Activation 'sigmoid' สำหรับ Binary Classification
   - ฝึกสอนโมเดลด้วย Adam Optimizer พร้อม Callbacks: `EarlyStopping` และ `ReduceLROnPlateau`

5. **ประเมินผลประสิทธิภาพ (Model Evaluation - evaluate.py)**:
   - คำนวณค่า **Accuracy (92.39%), Precision (0.92), Recall (0.92), F1-Score (0.92)**
   - สร้างและบันทึกภาพกราฟวิเคราะห์:
     - **training_history.png**: กราฟแสดงเส้น Loss และ Accuracy เพื่อดูการลู่เข้าและการเรียนรู้
     - **confusion_matrix.png**: ตาราง Heatmap Matrix แสดงความถูกต้องในการจำแนก Lion และ Tiger

6. **การทดลองเปรียบเทียบตามใบงาน (Lab 07 Comparison Experiments - compare_experiments.py / main.py)**:
   - **เปรียบเทียบ CNN Configurations:** เปรียบเทียบความแตกต่างของจำนวน Convolutional Layers, Filters และ Neurons:
     - Config 1: 1 Conv Layer `[32 Filters]` + Dense `64`
     - Config 2: 2 Conv Layers `[32 -> 64 Filters]` + Dense `128`
     - Config 3: 3 Conv Layers `[32 -> 64 -> 128 Filters]` + Dense `128`
     - Config 4: Deep 3 Conv Layers `[64 -> 128 -> 256 Filters]` + 2 Dense `[256, 64]`
     - บันทึกลง **comparison_configs.png**
   - **เปรียบเทียบ Epochs:** เปรียบเทียบผลที่ 10, 20, 30, 50 Epochs บันทึกลง **comparison_epochs.png**

7. **ทดสอบการทำนายผลลัพธ์ (Inference & Sample Test - test_cnn.py)**:
   - สุ่มภาพจาก Test Set มาทดสอบและแสดงผลในรูปแบบ Grid 2x2
   - คำนวณค่าความน่าจะเป็นจริง (**Confidence %**) และแสดงผลภาพพร้อมระบุสถานะ ถูกต้อง (เขียว) / ผิดพลาด (แดง) ในไฟล์ **prediction_sample.png** (ถูกต้อง 4/4 หรือ 100%)

---

## ผลการทดลองเปรียบเทียบ (Lab 07 Summary Results)

### 1. ตารางเปรียบเทียบ CNN Configurations

| Configuration | สถาปัตยกรรม / Conv Layers & Neurons | Train Acc (%) | Val Acc (%) | Test Acc (%) | เวลาฝึกสอน (s) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Config 1 (1 Conv Layer: 32F, 64N)** | 1 Conv `[32]` + Dense `(64)` | 94.72% | 89.13% | **86.96%** | 20.4s |
| **Config 2 (2 Conv Layers: 32->64F, 128N)** | 2 Conv `[32, 64]` + Dense `(128)` | 99.38% | 84.78% | **91.30%** | 24.1s |
| **Config 3 (3 Conv Layers: 32->64->128F, 128N)** | 3 Conv `[32, 64, 128]` + Dense `(128)` | 97.52% | 86.96% | **89.13%** | 20.9s |
| **Config 4 (Deep 3 Conv: 64->128->256F, 2 Dense)** | 3 Conv `[64, 128, 256]` + 2 Dense `[256, 64]` | 98.76% | 84.78% | **94.57%** | 64.6s |

> **ข้อสังเกต:** โมเดล CNN ที่มีความลึกและจำนวนฟิลเตอร์มากขึ้น (Config 4) สามารถสกัดฟีเจอร์รูปภาพที่มีความซับซ้อนได้ดียิ่งขึ้น ส่งผลให้ได้ Test Accuracy สูงสุดถึง **94.57%** ซึ่งเหนือกว่าโมเดล MLP ใน Lab 06 (~61.96%) อย่างมีนัยสำคัญ

### 2. ตารางเปรียบเทียบจำนวน Epochs (บนโมเดล CNN)

| จำนวน Epochs | Train Acc (%) | Val Acc (%) | Test Acc (%) | Test Loss | เวลาฝึกสอน (s) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **10 Epochs** | 94.41% | 86.96% | **88.04%** | 0.3807 | 12.8s |
| **20 Epochs** | 99.07% | 89.13% | **93.48%** | 0.2831 | 24.4s |
| **30 Epochs** | 100.00% | 84.78% | **95.65%** | 0.2647 | 35.2s |
| **50 Epochs** | 100.00% | 89.13% | **93.48%** | 0.2981 | 55.7s |

> **ข้อสังเกต:** ที่จำนวน 30 Epochs โมเดลสามารถเรียนรู้ฟีเจอร์ได้อย่างสมบูรณ์และให้ Test Accuracy สูงที่สุดที่ **95.65%** โดยมี Test Loss ต่ำที่สุดที่ 0.2647 หากเพิ่มเป็น 50 Epochs ประสิทธิภาพบน Test Set จะเริ่มคงที่เนื่องจากโมเดลเข้าสู่จุดอิ่มตัว

---

## สรุปเทคโนโลยีและเครื่องมือที่ใช้ (Tech Stack & Tools)

- **ภาษาที่ใช้พัฒนา (Language):**
  - Python 3

- **ไลบรารีและเฟรมเวิร์กหลัก (Libraries & Frameworks):**
  - **TensorFlow / Keras:** ใช้สร้าง ฝึกสอน และประเมินผลโมเดล Convolutional Neural Network (Conv2D, MaxPooling2D, Flatten, Dense, Dropout)
  - **OpenCV (cv2):** ใช้อ่านไฟล์รูปภาพ จัดการระบบสี (BGR -> RGB) และปรับขนาดภาพ (Resize 100x100)
  - **NumPy:** ใช้จัดการและบันทึกชุดข้อมูลในรูปแบบ Array (.npy)
  - **Scikit-Learn:** ใช้แบ่งชุดข้อมูล (train_test_split with stratify) และคำนวณ Metrics (accuracy_score, classification_report, confusion_matrix)
  - **Matplotlib:** ใช้วาดและบันทึกภาพกราฟผลลัพธ์ (Training Curves, Confusion Matrix, Prediction Sample Grid, Comparison Charts)

- **เทคนิคและอัลกอริทึมที่ใช้ (Key Techniques):**
  - **Convolutional Feature Extraction:** สกัดฟีเจอร์อัตโนมัติผ่าน Kernel Convolution (ขอบ, พื้นผิว, รูปร่าง) ตามหลักการในเอกสาร ML-07-CNN
  - **Spatial Downsampling:** ใช้ MaxPooling2D ลดขนาดมิติและเพิ่ม Spatial Invariance
  - **Regularization:** ใช้ Dropout และ L2 Weight Decay เพื่อป้องกันการเกิด Overfitting
  - **Optimization & Callbacks:** ใช้ Adam Optimizer พร้อม EarlyStopping และ ReduceLROnPlateau
  - **Feature Scaling:** ใช้ชั้น Rescaling(1./255) ปรับสเกลค่าพิกเซลให้อยู่ในช่วง [0, 1] ภายในโมเดล
