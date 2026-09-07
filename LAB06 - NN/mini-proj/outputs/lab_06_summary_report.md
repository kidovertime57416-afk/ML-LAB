# ML-LAB-06: Neural Network Experiment & Comparison Report

## 1. Comparison of Neural Network Configurations

| Configuration | Hidden Layers & Neurons | Train Acc (%) | Val Acc (%) | Test Acc (%) | Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Config 1 (1 Hidden Layer: 128N) | `1 Layer [128]` | 63.66% | 65.22% | **60.87%** | 10.5s |
| Config 2 (2 Hidden Layers: 256 -> 128N) | `2 Layers [256, 128]` | 65.22% | 58.70% | **54.35%** | 19.1s |
| Config 3 (3 Hidden Layers: 512 -> 256 -> 64N) | `3 Layers [512, 256, 64]` | 57.45% | 52.17% | **61.96%** | 34.8s |
| Config 4 (CNN: 32 -> 64 -> 128 + Dense 128) | `Conv32-64-128 + Dense128` | 94.72% | 86.96% | **91.30%** | 18.5s |

## 2. Comparison of Different Numbers of Epochs (Best CNN Model)

| Epochs | Train Acc (%) | Val Acc (%) | Test Acc (%) | Test Loss | Time (s) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 10 | 98.76% | 86.96% | **93.48%** | 0.1740 | 9.8s |
| 20 | 99.38% | 86.96% | **88.04%** | 0.2655 | 18.5s |
| 30 | 100.00% | 86.96% | **91.30%** | 0.2464 | 27.3s |
| 50 | 96.58% | 89.13% | **91.30%** | 0.2123 | 45.3s |

## 3. Analysis & Conclusions for Lab 06
- **Model Configuration:** โมเดลแบบ CNN (Convolutional Neural Network) ให้ประสิทธิภาพความแม่นยำสูงกว่า MLP อย่างเห็นได้ชัด เนื่องจาก CNN สามารถสกัดและเรียนรู้ฟีเจอร์เชิงพื้นที่ (Spatial Features) ของรูปภาพใบหน้าและลายของสัตว์ได้ดีกว่า
- **Number of Epochs:** เมื่อเพิ่มจำนวน Epoch จาก 10 เป็น 20-30 ประสิทธิภาพของโมเดลจะเพิ่มขึ้นอย่างรวดเร็ว และจะเริ่มทรงตัวที่ประมาณ 25-30 Epochs หากเทรนมากเกินไป (เช่น 50+ โดยไม่มี Regularization) อาจเริ่มเกิด Overfitting เล็กน้อย
