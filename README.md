# Advanced Deep Learning Driven Geospatial Analysis for GLOF Risk Reduction

### **A Case Study from Pakistan’s Northern Mountain Ranges**

<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="GLOF Risk Reduction">
</p>

#### **Nauman Ali Murad**, **Abinta Mehmood Mir**, **Nazia Shahzadi**

#### **Ghulam Ishaq Khan Institute of Engineering Sciences and Technology, Topi, Pakistan**

[![ResearchGate](https://img.shields.io/badge/ResearchGate-Paper-white.svg)]([YOUR_RESEARCHGATE_LINK](https://www.researchgate.net/publication/388129756_Advanced_Deep_Learning_Driven_Geospatial_Analysis_for_GLOF_Risk_Reduction_A_Case_Study_from_Pakistan's_Northern_Mountain_Ranges)) [![Google Scholar](https://img.shields.io/badge/Google%20Scholar-Profile-green.svg)]([YOUR_GOOGLE_SCHOLAR_LINK](https://scholar.google.com/citations?user=6w77LCsAAAAJ&hl=en)) [![Kaggle](https://img.shields.io/badge/Kaggle-Profile-blue.svg)]([YOUR_KAGGLE_LINK](https://www.kaggle.com/naumanalimurad))



---

## Latest Updates
- **February 2025**: Initial release of the code and paper for glacial lake segmentation using DeepLabV3+, U-Net, and YOLOv8-Seg.

---

## Abstract
Glacial Lake Outburst Floods (GLOFs) pose a severe risk to populations in high-altitude areas, particularly in Pakistan's northern regions. This paper focuses on using Deep Learning (DL) models for detecting and segmenting glacial lakes to mitigate GLOF risks. Using the Glacial Lakes Detection Dataset from High-Mountain Asia, we evaluate DL models like DeepLabV3+, U-Net, and YOLOv8 for lake segmentation and classification. Experimental results demonstrate the proficiency of these models, with DeepLabV3+ (ResNet50 backbone and Dice loss function) achieving the highest IoU score of 77.2%.

---
## Introduction
- **GLOF Risk Reduction**: This project leverages advanced deep learning models to segment and classify glacial lakes in satellite imagery, enabling early detection of GLOF risks.
- **Key Contributions**:
  - Evaluation of state-of-the-art DL models (DeepLabV3+, U-Net, YOLOv8-Seg) for glacial lake segmentation.
  - High IoU scores achieved, with DeepLabV3+ (ResNet50 + Dice Loss) reaching 79.2%.
  - A comprehensive dataset of 1,200 cloud-free Sentinel-2 images with ground truth masks.

---
**Program Flow:**
<p align="center">
    <img src="figures\program flow.png" alt="Glacial Lake Segmentation">
</p>
For architectural diagrams of the models used, check out the figures folder.

---

## Requirements
- **Python 3.8 or higher**
- **TensorFlow 2.x**
- **PyTorch 1.10 or higher**
- **OpenCV**
- **Ultralytics**
- **Scikit-learn**

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/0xnomy/glacier-vision
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
---

## Getting Started  

### Data Access & Preparation  
- The **Glacial Lakes Detection Dataset** can only be accessed upon request at [IEEE Dataport](https://ieee-dataport.org/documents/glacial-lakes-detection-dataset).  
### Training  
- Train the **DeepLabV3+** model using the corresponding Jupyter Notebook:  
  ```bash
  jupyter notebook DeepLabV3+ ResNet + Diceloss GLOF Detection.ipynb
  ```  
- Train the **U-Net** model using the corresponding Jupyter Notebook:  
  ```bash
  jupyter notebook U-Net + EffienctNetB0 GLOF Detection.ipynb
  ```  
- Train the **YOLOv8-Seg** model using the corresponding Jupyter Notebook:  
  ```bash
  jupyter notebook YoloV8-Seg, IoU GLOF Detection.ipynb
  ```  
---

## Acknowledgment
- **Remote Sensing and Spatial Analytics Lab, ITU, Lahore, Pakistan**, for providing the dataset.
- **Ghulam Ishaq Khan Institute of Engineering Sciences and Technology, Topi, Pakistan** for supporting this research.

---

## Citations
Please consider citing our paper in your publications if it helps your research.
```bibtex
@INPROCEEDINGS{10838395,
  author={Murad, Nauman Ali and Mir, Abinta Mehmood and Shahzadi, Nazia},
  booktitle={2024 International Conference on Frontiers of Information Technology (FIT)}, 
  title={Advanced Deep Learning Driven Geospatial Analysis for GLOF Risk Reduction: A Case Study from Pakistan's Northern Mountain Ranges}, 
  year={2024},
  volume={},
  number={},
  pages={1-6},
  keywords={Deep learning;Image segmentation;Adaptation models;Prevention and mitigation;Object detection;Lakes;Predictive models;Sensors;Monitoring;Residual neural networks;Glacial Lake outburst floods (GLOF);remote sensing;geo-spatial data;climate change;semantic segmentation;satellite imagery;deep neural network},
  doi={10.1109/FIT63703.2024.10838395}}
```

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact
For any questions or collaborations, please contact: 
```~ = @```
- Nauman Ali Murad: u2022479 ~ giki.edu.pk
- Abinta Mehmood Mir: abinta.mehmood ~ giki.edu.pk
- Nazia Shahzadi: nazia.shahzadi ~ giki.edu.pk