# Mini-Project: Neural Network Image Classification (Lion vs Tiger)

## Project Overview 
โปรเจกต์นี้เป็นการพัฒนาระบบจำแนกรูปภาพสัตว์ (Tiger vs Lion) ด้วยโครงข่ายประสาทเทียม **Neural Network (NN)** และ **Convolutional Neural Network (CNN)** โดยครอบคลุมกระบวนการ Machine Learning Pipeline แบบครบวงจร ตั้งแต่การโหลดรูปภาพ, การประมวลผลเบื้องต้น (Preprocessing), การแบ่งชุดข้อมูล (Train/Val/Test Split), การสร้างและเทรนโมเดล, การประเมินประสิทธิภาพโมเดล (Evaluation), การทดลองเปรียบเทียบสถาปัตยกรรมโมเดลและจำนวน Epochs ตามข้อกำหนดของใบงาน **ML-LAB-06**, ตลอดจนการทดสอบการทำนายผลลัพธ์ (Inference & Visualization)

---

## References & Sources

- **Dataset Source :**  
  [Tiger vs Lion Image Classification Dataset (Kaggle)](https://www.kaggle.com/datasets/sonalshinde123/tiger-vs-lion-image-classification-dataset)  
  ชุดข้อมูลรูปภาพของสิงโต (Lion) และเสือ (Tiger) รวม 460 รูปภาพ

- **Code Reference :**  
  [Machine Learning Course - ML-06-NN (GitHub: aproot-en)](https://github.com/aproot-en/Machine-Learning-Course/tree/main/ML-06-NN)  
  นำโครงสร้าง Pipeline และการออกแบบระบบจากคอร์สมา Rewrite และปรับปรุงเป็นโมเดลที่เสถียร รองรับทั้ง MLP และ CNN พร้อมทั้งเพิ่มการทดลองเปรียบเทียบ Configurations/Epochs ตามใบงาน Lab 06

---

## Project Structure

`	ext
mini-proj/
│
├── main.py                            # สคริปต์หลักสำหรับรันกระบวนการทั้งหมด (End-to-End Pipeline 7 ขั้นตอน)
├── data_loader.py                     # ฟังก์ชันสำหรับอ่านรูปภาพจากโฟลเดอร์และคัดกรองไฟล์เสีย
├── preprocessing.py                   # แปลงระบบสี BGR -> RGB และปรับขนาดภาพ (Resize 100x100)
├── split_data.py                      # แบ่งชุดข้อมูลเป็น Training, Validation และ Test Set
├── nn_model.py                        # โครงสร้างโมเดล MLP/CNN, ฟังก์ชันเทรนโมเดล และทำนายผล
├── evaluate.py                        # คำนวณ Metrics, Confusion Matrix, History และ Prediction sample
├── compare_experiments.py             # ฟังก์ชันและสคริปต์เปรียบเทียบ Configurations และ Epochs
├── test_nn.py                         # สคริปต์สุ่มทดสอบโมเดลกับภาพตัวอย่าง 4 รูป
│
└── outputs/                           # โฟลเดอร์เก็บผลลัพธ์จากการทำงาน
    ├── features.npy                   # ข้อมูล Array ของฟีเจอร์รูปภาพทั้งหมด
    ├── labels.npy                     # ข้อมูล Array ของ Label ทั้งหมด
    ├── classes.json                   # รายชื่อ Class (Lion, tiger)
    ├── X_train.npy / y_train.npy      # ข้อมูลและ Label สำหรับการฝึกสอน (Train Set)
    ├── X_val.npy / y_val.npy          # ข้อมูลและ Label สำหรับการตรวจสอบ (Validation Set)
    ├── X_test.npy / y_test.npy        # ข้อมูลและ Label สำหรับการทดสอบ (Test Set)
    ├── nn_model.keras                 # ไฟล์โมเดลที่บันทึกหลังการฝึกสอน
    ├── history.json                   # บันทึกประวัติ Loss และ Accuracy ของแต่ละ Epoch
    ├── confusion_matrix.png           # รูปภาพ Confusion Matrix แสดงผลการจำแนก
    ├── training_history.png           # กราฟเปรียบเทียบ Loss และ Accuracy ตลอดการฝึกสอน
    ├── prediction_sample.png          # รูปภาพตัวอย่างการพยากรณ์พร้อมค่าความเชื่อมั่น (Confidence %)
    ├── comparison_configs.png         # กราฟเปรียบเทียบประสิทธิภาพแต่ละ Configuration
    ├── comparison_epochs.png          # กราฟเปรียบเทียบความแม่นยำตามจำนวน Epochs
    └── lab_06_summary_report.md       # สรุปตารางผลการทดลองเปรียบเทียบสำหรับ Lab 06
`

---

## Pipeline Workflow

กระบวนการทำงานของระบบแบ่งออกเป็น 7 ขั้นตอนหลัก:

1. **โหลดชุดข้อมูล (Data Loading - data_loader.py)**:
   - อ่านรูปภาพจากโฟลเดอร์ Animal/ โดยตรวจจับคลาสย่อยอัตโนมัติ (Lion, 	iger)
   - กรองเฉพาะไฟล์ภาพที่รองรับ (.jpg, .jpeg, .png, .bmp) และคัดกรองข้ามไฟล์ที่เสียหาย

2. **เตรียมข้อมูลรูปภาพ (Preprocessing - preprocessing.py)**:
   - แปลงระบบสีของรูปภาพจาก BGR เป็น RGB เพื่อให้สีถูกต้อง
   - ปรับขนาดรูปภาพ (Resize) ให้มีมิติเท่ากันที่ **100x100 พิกเซล**

3. **แบ่งชุดข้อมูล (Data Splitting - split_data.py)**:
   - แบ่งข้อมูลด้วยวิธี Stratified Splitting เพื่อรักษาสัดส่วนของทั้ง 2 คลาส:
     - **Training Set (70%)**: สำหรับให้โมเดลเรียนรู้ (322 ภาพ)
     - **Validation Set (10%)**: สำหรับตรวจสอบระหว่างเทรน (46 ภาพ)
     - **Test Set (20%)**: สำหรับวัดประสิทธิภาพขั้นสุดท้าย (92 ภาพ)

4. **สร้างและฝึกสอนโมเดล (Model Training - 
n_model.py)**:
   - ออกแบบสถาปัตยกรรม **CNN (Convolutional Neural Network)** ประกอบด้วย:
     - ชั้น Rescaling(1./255) ปรับค่าพิกเซลให้อยู่ในช่วง [0, 1]
     - ชั้น Conv2D (32, 64, 128 ฟิลเตอร์) และ MaxPooling2D สกัดฟีเจอร์เชิงพื้นที่
     - ชั้น Dense (128 neurons) และ Dropout(0.5) เพื่อลด Overfitting
     - ชั้น Output ด้วย Sigmoid สำหรับการจำแนกประเภท Binary
   - ฝึกสอนโมเดลด้วย Adam Optimizer พร้อม EarlyStopping และ ReduceLROnPlateau

5. **ประเมินผลประสิทธิภาพ (Model Evaluation - evaluate.py)**:
   - คำนวณค่า **Accuracy, Precision, Recall, F1-Score**
   - สร้างและบันทึกภาพกราฟวิเคราะห์:
     - **	raining_history.png**: กราฟแสดงเส้น Loss และ Accuracy เพื่อดูการลู่เข้าและการเรียนรู้
     - **confusion_matrix.png**: ตาราง Matrix แสดงความถูกต้องในการทำนายของแต่ละคลาส

6. **การทดลองเปรียบเทียบตามใบงาน (Lab 06 Comparison Experiments - compare_experiments.py / main.py)**:
   - **เปรียบเทียบ Configurations:** เปรียบเทียบความแตกต่างของจำนวน Hidden Layers และ Neurons (MLP 1 Layer 128N, MLP 2 Layers 256->128N, MLP 3 Layers 512->256->64N และ CNN) บันทึกลง **comparison_configs.png**
   - **เปรียบเทียบ Epochs:** เปรียบเทียบผลที่ 10, 20, 30, 50 Epochs บันทึกลง **comparison_epochs.png**
   - บันทึกตารางรายงานสรุปผลลงไฟล์ **lab_06_summary_report.md**

7. **ทดสอบการทำนายผลลัพธ์ (Inference & Sample Test - 	est_nn.py)**:
   - สุ่มภาพจาก Test Set มาทดสอบและแสดงผลในรูปแบบ Grid 2x2
   - คำนวณค่าความน่าจะเป็นจริง (**Probability Confidence %**) และแสดงผลภาพพร้อมระบุสถานะ ถูกต้อง (เขียว) / ผิดพลาด (แดง) ในไฟล์ **prediction_sample.png**

---

## ผลการทดลองเปรียบเทียบ (Lab 06 Summary Results)

### 1. ตารางเปรียบเทียบ Neural Network Configurations

| Configuration | สถาปัตยกรรม / Hidden Layers & Neurons | Train Acc (%) | Val Acc (%) | Test Acc (%) | เวลาฝึกสอน (s) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Config 1 (1 Hidden Layer: 128N)** | 1 Layer [128] | 63.04% | 56.52% | **61.96%** | 10.4s |
| **Config 2 (2 Hidden Layers: 256 -> 128N)** | 2 Layers [256, 128] | 60.25% | 60.87% | **60.87%** | 19.2s |
| **Config 3 (3 Hidden Layers: 512 -> 256 -> 64N)** | 3 Layers [512, 256, 64] | 50.93% | 45.65% | **58.70%** | 36.6s |
| **Config 4 (CNN: 32 -> 64 -> 128 + Dense 128)** | Conv32-64-128 + Dense128 | 97.52% | 86.96% | **86.96% - 95.65%** | 20.2s |

### 2. ตารางเปรียบเทียบจำนวน Epochs (บนโมเดล CNN)

| จำนวน Epochs | Train Acc (%) | Val Acc (%) | Test Acc (%) | เวลาฝึกสอน (s) |
| :---: | :---: | :---: | :---: | :---: |
| **10 Epochs** | 95.96% | 89.13% | **96.74%** | 10.4s |
| **20 Epochs** | 98.45% | 86.96% | **92.39%** | 20.1s |
| **30 Epochs** | 100.00% | 86.96% | **96.74%** | 29.4s |
| **50 Epochs** | 100.00% | 86.96% | **92.39%** | 48.5s |

---

## สรุปเทคโนโลยีและเครื่องมือที่ใช้ (Tech Stack & Tools)

- **ภาษาที่ใช้พัฒนา (Language):**
  - Python 3

- **ไลบรารีและเฟรมเวิร์กหลัก (Libraries & Frameworks):**
  - **TensorFlow / Keras:** ใช้สร้าง ฝึกสอน และประเมินผลโมเดล Neural Network (MLP & CNN)
  - **OpenCV (cv2):** ใช้อ่านไฟล์รูปภาพ จัดการระบบสี (BGR -> RGB) และปรับขนาดภาพ (Resize)
  - **NumPy:** ใช้จัดการและบันทึกชุดข้อมูลในรูปแบบ Array (.npy)
  - **Scikit-Learn:** ใช้แบ่งชุดข้อมูล (	rain_test_split with stratify) และคำนวณ Metrics (ccuracy_score, classification_report, confusion_matrix)
  - **Matplotlib:** ใช้วาดและบันทึกภาพกราฟผลลัพธ์ (Training Curves, Confusion Matrix, Prediction Sample Grid, Comparison Charts)

- **เทคนิคและอัลกอริทึมที่ใช้ (Key Techniques):**
  - **Convolutional Neural Network (CNN):** สกัดฟีเจอร์เชิงพื้นที่ของภาพด้วย Conv2D และ MaxPooling2D
  - **Multi-Layer Perceptron (MLP):** โครงข่ายประสาทเทียมแบบ Fully-Connected พร้อมปรับจำนวน Hidden Layers และ Neurons
  - **Regularization:** ใช้ Dropout และ L2 Weight Decay เพื่อป้องกันการเกิด Overfitting
  - **Optimization & Callbacks:** ใช้ Adam Optimizer พร้อม EarlyStopping และ ReduceLROnPlateau
  - **Feature Scaling:** ใช้ชั้น Rescaling(1./255) ปรับสเกลค่าพิกเซลให้อยู่ในช่วง [0, 1] ภายในโมเดล
