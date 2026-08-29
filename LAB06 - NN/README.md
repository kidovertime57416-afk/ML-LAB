# ML-06-Neural Network (NN)

## Project Overview 
โปรเจกต์นี้เป็นการพัฒนาระบบจำแนกรูปภาพสัตว์ (Tiger vs Lion) ด้วยโครงข่ายประสาทเทียม **Convolutional Neural Network (CNN)** โดยครอบคลุมกระบวนการ Machine Learning Pipeline แบบครบวงจร ตั้งแต่การโหลดรูปภาพ, การประมวลผลเบื้องต้น (Preprocessing), การแบ่งชุดข้อมูล (Train/Val/Test Split), การสร้างและเทรนโมเดล, การประเมินประสิทธิภาพโมเดล (Evaluation), ตลอดจนการทดสอบการทำนายผลลัพธ์ (Inference & Visualization)

---

## References & Sources

- **Dataset Source :**  
  [Tiger vs Lion Image Classification Dataset (Kaggle)](https://www.kaggle.com/datasets/sonalshinde123/tiger-vs-lion-image-classification-dataset)  
  ชุดข้อมูลรูปภาพของสิงโต (Lion) และเสือ (Tiger)

- **Code Reference :**  
  [Machine Learning Course - ML-06-NN (GitHub: aproot-en)](https://github.com/aproot-en/Machine-Learning-Course/tree/main/ML-06-NN)  
  นำโครงสร้าง Pipeline และการออกแบบระบบจากคอร์สมา Rewrite และปรับปรุงเป็นโมเดล CNN สำหรับจำแนกรูปภาพ Lion vs Tiger พร้อมทั้งปรับปรุงกราฟและการคำนวณ Confidence ให้ถูกต้องสมบูรณ์

---

## Project Structure

```text
ML-06-NN/
│
├── Animal/                                # โฟลเดอร์เก็บข้อมูลรูปภาพต้นฉบับ
│   ├── Lion/                              # รูปภาพสิงโต (lion_001.jpg ...)
│   └── tiger/                             # รูปภาพเสือ (tiger_001.jpg ...)
│
├── mini-proj/                             # โค้ดโปรแกรมหลักของระบบ
│   ├── main.py                            # สคริปต์หลักสำหรับรันกระบวนการทั้งหมด (End-to-End Pipeline)
│   ├── data_loader.py                     # ฟังก์ชันสำหรับอ่านรูปภาพจากโฟลเดอร์และคัดกรองไฟล์เสีย
│   ├── preprocessing.py                   # แปลงระบบสี BGR -> RGB และปรับขนาดภาพ (Resize)
│   ├── split_data.py                      # แบ่งชุดข้อมูลเป็น Training, Validation และ Test Set
│   ├── nn_model.py                        # โครงสร้างโมเดล CNN, ฟังก์ชันเทรนโมเดล และทำนายผล
│   ├── evaluate.py                        # คำนวณ Accuracy, Classification Report และพล็อตภาพกราฟต่างๆ
│   ├── test_nn.py                         # สคริปต์สุ่มทดสอบโมเดลกับภาพตัวอย่าง 4 รูป
│   │
│   └── outputs/                           # โฟลเดอร์เก็บผลลัพธ์จากการทำงาน
│       ├── features.npy                   # ข้อมูล Array ของฟีเจอร์รูปภาพทั้งหมด
│       ├── labels.npy                     # ข้อมูล Array ของ Label ทั้งหมด
│       ├── classes.json                   # รายชื่อ Class (Lion, tiger)
│       ├── X_train.npy / y_train.npy      # ข้อมูลและ Label สำหรับการฝึกสอน (Train Set)
│       ├── X_val.npy / y_val.npy          # ข้อมูลและ Label สำหรับการตรวจสอบ (Validation Set)
│       ├── X_test.npy / y_test.npy        # ข้อมูลและ Label สำหรับการทดสอบ (Test Set)
│       ├── nn_model.keras                 # ไฟล์โมเดลที่บันทึกหลังการฝึกสอน
│       ├── history.json                   # บันทึกประวัติ Loss และ Accuracy ของแต่ละ Epoch
│       ├── confusion_matrix.png           # รูปภาพ Confusion Matrix แสดงผลการจำแนก
│       ├── training_history.png           # กราฟเปรียบเทียบ Loss และ Accuracy ตลอดการฝึกสอน
│       └── prediction_sample.png          # รูปภาพตัวอย่างการพยากรณ์พร้อมค่าความเชื่อมั่น (Confidence)
│
└── README.md                              # เอกสารอธิบายโปรเจกต์
```

---

## Pipeline Workflow

กระบวนการทำงานของระบบแบ่งออกเป็น 6 ขั้นตอนหลัก:

1. **โหลดชุดข้อมูล (Data Loading - `data_loader.py`)**:
   - อ่านรูปภาพจากโฟลเดอร์ `Animal/` โดยตรวจจับคลาสย่อยอัตโนมัติ (`Lion`, `tiger`)
   - กรองเฉพาะไฟล์ภาพที่รองรับ (`.jpg`, `.jpeg`, `.png`, `.bmp`) และคัดกรองข้ามไฟล์ที่เสียหาย

2. **เตรียมข้อมูลรูปภาพ (Preprocessing - `preprocessing.py`)**:
   - แปลงระบบสีของรูปภาพจาก BGR (ค่าเริ่มต้นของ OpenCV) เป็น RGB เพื่อให้สีถูกต้อง
   - ปรับขนาดรูปภาพ (Resize) ให้มีมิติเท่ากันที่ **100x100 พิกเซล**

3. **แบ่งชุดข้อมูล (Data Splitting - `split_data.py`)**:
   - แบ่งข้อมูลด้วยวิธี Stratified Splitting เพื่อรักษาสัดส่วนของทั้ง 2 คลาส:
     - **Training Set (70%)**: สำหรับให้โมเดลเรียนรู้
     - **Validation Set (10%)**: สำหรับตรวจสอบและปรับค่า Hyperparameters ระหว่างเทรน
     - **Test Set (20%)**: สำหรับวัดประสิทธิภาพขั้นสุดท้าย

4. **สร้างและฝึกสอนโมเดล (Model Training - `nn_model.py`)**:
   - ออกแบบสถาปัตยกรรม **CNN (Convolutional Neural Network)** ประกอบด้วย:
     - ชั้น `Rescaling(1./255)` ปรับค่าพิกเซลให้อยู่ในช่วง [0, 1]
     - ชั้น `Conv2D` และ `MaxPooling2D` เพื่อดึงลักษณะเด่นเชิงพื้นที่ของรูปภาพ
     - ชั้น `Dense` และ `Dropout(0.5)` เพื่อลดโอกาสเกิด Overfitting
     - ชั้น Output ด้วย `Sigmoid` สำหรับการจำแนกประเภทแบบ Binary
   - ฝึกสอนโมเดลด้วย `Adam Optimizer` พร้อม `EarlyStopping` และ `ReduceLROnPlateau`

5. **ประเมินผลประสิทธิภาพ (Model Evaluation - `evaluate.py`)**:
   - คำนวณค่า **Accuracy, Precision, Recall, F1-Score**
   - สร้างและบันทึกภาพกราฟวิเคราะห์:
     - **`training_history.png`**: กราฟแสดงเส้น Loss และ Accuracy เพื่อดูการลู่เข้าและการเรียนรู้
     - **`confusion_matrix.png`**: ตาราง Matrix แสดงความถูกต้องในการทำนายของแต่ละคลาส

6. **ทดสอบการทำนายผลลัพธ์ (Inference & Sample Test - `test_nn.py`)**:
   - สุ่มภาพจาก Test Set มาทดสอบและแสดงผลในรูปแบบ Grid 2x2
   - คำนวณค่าความน่าจะเป็นจริง (**Probability Confidence %**) และแสดงผลภาพพร้อมระบุสถานะ ถูกต้อง (เขียว) / ผิดพลาด (แดง) ในไฟล์ **`prediction_sample.png`**

---

## สรุปเทคโนโลยีและเครื่องมือที่ใช้ (Tech Stack & Tools)

- **ภาษาที่ใช้พัฒนา (Language):**
  - Python 3

- **ไลบรารีและเฟรมเวิร์กหลัก (Libraries & Frameworks):**
  - **TensorFlow / Keras:** ใช้สร้าง ฝึกสอน และประเมินผลโมเดล Convolutional Neural Network (CNN)
  - **OpenCV (`cv2`):** ใช้อ่านไฟล์รูปภาพ จัดการระบบสี (BGR -> RGB) และปรับขนาดภาพ (Resize)
  - **NumPy:** ใช้จัดการและบันทึกชุดข้อมูลในรูปแบบ Array (`.npy`)
  - **Scikit-Learn:** ใช้แบ่งชุดข้อมูล (`train_test_split` with stratify) และคำนวณ Metrics (`accuracy_score`, `classification_report`, `confusion_matrix`)
  - **Matplotlib:** ใช้วาดและบันทึกภาพกราฟผลลัพธ์ (Training Curves, Confusion Matrix, Prediction Sample Grid)

- **เทคนิคและอัลกอริทึมที่ใช้ (Key Techniques):**
  - **Convolutional Neural Network (CNN):** สกัดฟีเจอร์เชิงพื้นที่ของภาพด้วย Conv2D และ MaxPooling2D
  - **Regularization:** ใช้ Dropout (0.5) ป้องกันการเกิด Overfitting
  - **Optimization & Callbacks:** ใช้ Adam Optimizer พร้อม EarlyStopping และ ReduceLROnPlateau
  - **Feature Scaling:** ใช้ชั้น Rescaling(1./255) ปรับสเกลค่าพิกเซลให้อยู่ในช่วง [0, 1] ภายในโมเดล


---
