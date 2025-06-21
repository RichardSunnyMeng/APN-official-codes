# Artifact Purification Network
This is the official codes for the paper "Artifact feature purification for cross-domain detection of AI-generated images", published in [Computer Vision and Image Understanding](https://www.sciencedirect.com/science/article/pii/S1077314224001590).

# How to use
1. Install requirements
```
conda create -n APN python==3.7
conda activate APN
pip install -r requirements.txt
```

2. Download pretrained models
[Google Drive](https://drive.google.com/drive/my-drive?dmr=1&ec=wgc-drive-hero-goto)

4. Run code
```
sh scripts/eval.sh
```

4. If you want to train the model
```
sh scripts/run.sh
```

# Results on GenImage
![1750502188358](https://github.com/user-attachments/assets/7b0b91a4-f24c-4567-8848-c8453584e74b)
