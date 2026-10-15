# Machine Learning

รวมโปรเจกต์และแบบฝึกหัด Machine Learning ด้วย Python ตั้งแต่การเตรียมข้อมูล Regression, Classification และ Clustering ไปจนถึง Neural Network และ Object Detection แต่ละแลปมีโค้ดและ README แยกอยู่ในโฟลเดอร์ของตัวเอง

## ภาษาและเครื่องมือ

- **Python** ใช้เตรียมข้อมูล สร้างโมเดล และประเมินผล
- **TensorFlow / Keras** ใช้กับ Neural Network, CNN และโมเดลจำแนกภาพ crop
- **PyTorch / Ultralytics** ใช้เทรนและตรวจจับวัตถุด้วย YOLOv8n
- **Scikit-learn, NumPy, OpenCV และ Matplotlib** ใช้จัดการข้อมูล เตรียมภาพ คำนวณ metrics และวาดกราฟ

## สารบัญแลป

| Lab | หัวข้อ | Dataset |
| --- | --- | --- |
| [LAB02](LAB02/README.md) | Data Preprocessing และ Visualization | Video Game Sales |
| [LAB03](LAB03/README.md) | ทำนายอายุและจำแนกเพศจากภาพใบหน้า | age_gender.csv |
| [LAB04 — KNN](LAB04%20-%20KNN/README.md) | KNN และ Clustering | Automobile |
| [LAB05 — SVM](LAB05%20-%20SVM/README.md) | จำแนกภาพด้วย SVM | Alligator vs Crocodile |
| [LAB06 — NN](LAB06%20-%20NN/README.md) | จำแนกภาพด้วย Feedforward NN / MLP | Lion vs Tiger |
| [LAB07 — CNN](LAB07%20-%20CNN/README.md) | จำแนกภาพด้วย CNN | Lion vs Tiger |
| [LAB08 — DCNN](LAB08%20-%20DCNN/README.md) | เปรียบเทียบ VGG แบบย่อและ epochs | Lion vs Tiger |
| [LAB09 — Object Detection I](LAB09%20-%20Object%20Detection%20I/README.md) | จำแนก crop ด้วยโมเดลแบบ R-CNN และ Fast R-CNN ย่อ | Aquarium 7 คลาส |
| [LAB10 — Object Detection II](LAB10-Object%20Detection%20II/README.md) | ตรวจจับวัตถุด้วย YOLOv8n | Aquarium 7 คลาส |

## ชุดข้อมูลที่ใช้ร่วมกัน

**LAB06–LAB08** ใช้ [Tiger vs Lion Image Classification Dataset](https://www.kaggle.com/datasets/sonalshinde123/tiger-vs-lion-image-classification-dataset) โดย LAB08 อ่านภาพจาก `LAB07 - CNN/Animal/` และใช้ split เดียวกับ LAB07

**LAB09–LAB10** ใช้ [Aquarium Dataset](https://www.kaggle.com/datasets/slavkoprytula/aquarium-data-cots) มี 638 รูป แบ่งเป็น train 448 รูป, validation 127 รูป และ test 63 รูป ได้แก่ `fish`, `jellyfish`, `penguin`, `puffin`, `shark`, `starfish` และ `stingray` โดย LAB09 อ่านข้อมูลจาก `LAB10-Object Detection II/dataset/` โดยตรง

## ผลการทดลองล่าสุด

| Lab / Model | งานที่ประเมิน | ผลบน Test Set |
| --- | --- | --- |
| LAB08 — VGG แบบย่อ | จำแนกภาพ Lion / Tiger | Accuracy 95.65% |
| LAB09 — R-CNN แบบย่อ | จำแนก crop จากกรอบจริง | Accuracy 65.29%, Macro F1 0.4423 |
| LAB09 — Fast R-CNN แบบย่อ | จำแนก crop จากกรอบจริง | Accuracy 78.35%, Macro F1 0.6957 |
| LAB10 — YOLOv8n | ตรวจจับคลาสและตำแหน่งวัตถุ | mAP50 54.92%, mAP50-95 33.57% |

LAB09 ทดลอง baseline เทียบกับ Dropout 0.3 โดยใช้ epochs และ seed เท่ากัน รอบนี้ baseline ได้ validation macro F1 สูงกว่าทั้งสองโมเดล จึงใช้ baseline ประเมิน test

ดูรายละเอียดการทดลอง กราฟ Confusion Matrix และภาพผลลัพธ์ใน README ของแต่ละแลป

## โครงสร้างโฟลเดอร์

```text
ML-LAB/
├── README.md
├── LAB01/
├── LAB02/
├── LAB03/
├── LAB04 - KNN/
├── LAB05 - SVM/
├── LAB06 - NN/
├── LAB07 - CNN/
├── LAB08 - DCNN/
├── LAB09 - Object Detection I/
├── LAB10-Object Detection II/
└── Final-Project/
```
