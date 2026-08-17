# STDF-Net

Official PyTorch implementation of STDF-Net.

---

## ⚙️ Requirements

- Python 3.11  
- PyTorch 2.12.0 (CUDA 12.8)   
- opencv-python  
- numpy  
- scikit-image  
- pytorch-msssim  

---

## 📂 Dataset

We use the following datasets:

### NUDT-MIRSDT
- Download: https://pan.baidu.com/s/1pSN350eurMafLiHBQBnrPA  
- Code: `5whn`

### TSIRMT
- Download: https://drive.google.com/drive/folders/1aWDNdUWkTOuV3fILbgLDEqM2N2erW05n  

⚠️ **Note:**
- The `dataset/` folder in this repository only contains **partial samples**.  
- Please download the **full datasets manually** from the links above.

---

## 📁 Project Structure

```bash
STDF-Net/
├── dataset/        # sample data (not full dataset)
├── checkpoints/    # pre-trained models
├── STDF-Net.py    # network definition
├── train.py        # training script
└── test_xxx.py     # testing script
```

## 🚀 Training & Testing

### Training
python train.py

You can modify training configurations in train.py, including:

- dataset path
- batch size
- learning rate
- number of epochs

### Testing
python test_xxx.py

## ⚠️ Notes:

- Please modify the model path in test_xxx.py before testing.
- Different datasets correspond to different test scripts.
- Select the appropriate test_xxx.py according to the dataset you use.

## 📦 Pre-trained Models

We provide trained models in the checkpoints/ directory:

- NUDT-MIRSDT → NUDT.pth
- TSIRMT → TSIRMT.pth

Please load the corresponding model when testing on different datasets.
