# Efficient Multi-Scale Transformer with Convolutional Attention for High-Definition Image Dehazing

This repository provides the **full source code** and **key testing data** for our proposed **Efficient Multi-Scale Transformer** (referred to as **Mulsormer**) for image dehazing. Our method features **Multi-Head Convolutional Attention (MHCA)**, a **Gated Feed-Forward Network (GFFN)**, and a **Multi-Scale Loss Function**, enabling robust high-definition image dehazing while maintaining high efficiency.

Below is a diagram illustrating the basic framework of Mulsormer:
![Model Framework](https://raw.githubusercontent.com/Wang1666570/Efficient-Multi-Scale-Transformer/main/framework/framework.png)




## 1. Repository Structure

```
.
├─ configs/
│   ├─ indoor/               # Configuration files for indoor data experiments
│   ├─ outdoor/              # Configuration files for outdoor data experiments
│   ├─ reside6k/             # Configuration files for RESIDE6K (multiple .json files)
│   │   ├─ default.json
│   │   ├─ dehazeformer-b.json
│   │   ├─ dehazeformer-m.json
│   │   ├─ dehazeformer-s.json
│   │   └─ dehazeformer-t.json
│   └─ rshaze/               # Configuration files for RSHAZE or other custom sets
│
├─ datasets/
│   ├─ __init__.py
│   └─ loader.py             # Data loader and dataset-related utilities
│
├─ models/
│   ├─ __init__.py
│   ├─ dehaze_optim.py       # Optimizer-related modules for the dehazing model
│   └─ utilArch.py           # Model architectures, building blocks, or layers
│
├─ utils/
│   ├─ __init__.py
│   ├─ common.py             # Shared utility functions
│   ├─ data_parallel.py      # Custom data-parallel wrappers
│   └─ metrics.py            # Metrics for image evaluation (PSNR, SSIM, etc.)
│
├─ inference.py              # Single-image inference script
├─ requirements.txt          # Python dependencies
├─ test.py                   # Test/evaluation script for batch data
├─ train.py                  # Main training script

```

## 2. Dependencies and Installation

We recommend **Python 3.7+** with PyTorch ≥ 1.8. The main Python dependencies are listed in `requirements.txt`. Install them via:

```bash
pip install -r requirements.txt
```

## 3. Dataset Preparation

Below is an updated version of the data-preparation section. It unifies the style for **RESIDE-6K** and **RS-Haze**, clarifies the download locations, and shows how to organize each dataset. Feel free to adjust folder names or paths as needed.

### RESIDE-6K
- **Download**: You can get the RESIDE-6K dataset from  [Papers with Code](https://paperswithcode.com/sota/image-dehazing-on-reside-6k) or [Hugging Face](https://huggingface.co/datasets/kmljt/RESIDE-6K).

- **Directory Structure**:  After downloading and extracting, please organize it as:
  ```
  ./dehaze_dataset/
    └─ RESIDE-6K/
        ├─ train/
        └─ test/
  ```
  (Adjust the folder names if needed, but match the path in your config files.)

### RS-Haze
- **Download**:  The RS-Haze dataset is available at [Papers with Code](https://paperswithcode.com/dataset/rs-haze) or [Google Drive](https://drive.google.com/drive/folders/1Yy_GH6_bydYPU6_JJzFQwig4LTh86VI4).

- **Directory Structure**:  After downloading, arrange your folders to mirror the following:
  ```
  ./RS-Haze/
    └─ RESIDE-IN/
        ├─ train/
        │   ├─ GT/
        │   │   └─ ... (ground-truth image files)
        │   └─ hazy/
        │       └─ ... (corresponding hazy image files)
        └─ test/
            └─ ...
  ```
  Be sure the paths in your config (`configs/rshaze/`) point to the correct directory.
  


## 4. Training Procedure

1. **Edit Config Files**  
   In `configs/<exp>/*.json` (e.g., `configs/reside6k/default.json`), you can specify hyperparameters (learning rate, batch size, etc.).  
   We recommend the following defaults for stable training (you can adjust `batch_size` and `lr` if you have limited GPU memory):
   ```json
   {
     "batch_size": 32,
     "patch_size": 256,
     "valid_mode": "test",
     "edge_decay": 0,
     "only_h_flip": false,
     "optimizer": "adamw",
     "lr": 4e-4,
     "epochs": 300,
     "eval_freq": 1
   }
   ```

2. **Run the Training Script**:
   ```bash
   python train.py \
       --model dehazeformer-b \
       --data_dir ./dehaze_dataset \
       --dataset RESIDE-6K \
       --exp reside6k
   ```
   - You may specify `--exp` as one of `reside6k`, `rshaze`, `indoor`, or `outdoor`, based on your dataset and experiment setup.

3. **Checkpoints and Logs**  
   By default, checkpoints and logs go to:
   ```
   ckpt/
   └─ weights/
       └─ reside6k/
           └─ dehazeformer-b_epochX_psnrXX.XXXX.pth
   ```



## 5. Testing / Evaluation

1. **Prepare Model Checkpoints**  
   Make sure your final `.pth` file is in `--save_dir/<exp>/`, for example:  
   ```
   ./saved_models/reside6k/dehazeformer-b.pth
   ```

2. **Run the Testing Script**:
   ```bash
   python test.py \
       --model dehazeformer-b \
       --data_dir ./data \
       --dataset RESIDE-6K \
       --save_dir ./saved_models \
       --result_dir ./results \
       --exp reside6k
   ```

3. **Results and Outputs**  
   Dehazed images and metrics (e.g., PSNR, SSIM) will appear in:
   ```
   results/
   └─ RESIDE-6K/
       └─ dehazeformer-b/
           ├─ imgs/
           └─ XX.XX | 0.XXXX.csv
   ```



## 6. Single-Image Inference

For quick testing on a single image, use:

```bash
python inference.py --image_path /path/to/your_image.jpg --model dehazeformer-b
```

- Adjust `image_path` to point to your local image file.  
- Make sure your desired checkpoint is properly referenced.


## 7. Main Results and Reproducibility

We compare both classic and state-of-the-art methods on **SOTS** and **RTTS**. Below is an example table of our results. See our paper for full details and further experiments.

## Table 1. Quantitative Comparison on SOTS (Indoor/Outdoor/Mixed)

| Methods                  | SOTS-Indoor PSNR | SOTS-Indoor SSIM | SOTS-Outdoor PSNR | SOTS-Outdoor SSIM | SOTS-Mix PSNR | SOTS-Mix SSIM |
|--------------------------|------------------|------------------|-------------------|-------------------|---------------|---------------|
| DCP [1]                  | 16.62            | 0.818            | 19.13             | 0.815             | 17.88         | 0.816         |
| DehazeNet [2]            | 19.82            | 0.821            | 24.75             | 0.927             | 21.02         | 0.870         |
| MSCNN [3]                | 19.84            | 0.833            | 22.06             | 0.908             | 20.31         | 0.857         |
| AOD-Net [4]              | 20.51            | 0.816            | 24.14             | 0.920             | 20.27         | 0.873         |
| GFN [5]                  | 22.30            | 0.880            | 21.55             | 0.844             | 23.52         | 0.905         |
| GridDehazeNet [6]        | 32.16            | 0.984            | 30.06             | 0.982             | 25.86         | 0.946         |
| MSBDN [7]                | 33.67            | 0.976            | 32.37             | 0.982             | 28.59         | 0.973         |
| FFA-Net [8]              | 32.68            | 0.976            | **33.57**         | **0.984**         | 28.97         | 0.973         |
| DehazeFormer [9]         | 35.15            | 0.989            | 33.71             | 0.982             | 30.36         | 0.973         |
| **Ours**                 | **37.49**        | **0.993**        | **34.64**         | **0.984**         | **31.45**     | **0.985**     |

## References
1. [Single Image Haze Removal Using Dark Channel Prior](https://ieeexplore.ieee.org/document/5206515#:~:text=Abstract%3A%20In%20this%20paper%2C%20we%20propose%20a%20simple,to%20remove%20haze%20from%20a%20single%20input%20image.)
2. [DehazeNet: An End-to-End System for Single Image Haze Removal](https://ieeexplore.ieee.org/document/7539399)
3. [Single Image Dehazing via Multi-Scale Convolutional Neural Networks](https://ieeexplore.ieee.org/abstract/document/9006075)
4. [AOD-Net: All-in-One Dehazing Network](https://arxiv.org/pdf/1707.06543)
5. [Gated Fusion Network for Single Image Dehazing](https://arxiv.org/pdf/1804.00213)
6. [GridDehazeNet: Attention-Based Multi-Scale Network for Image Dehazing](https://ieeexplore.ieee.org/document/9010659)
7. [Multi-Scale Boosted Dehazing Network with Dense Feature Fusion](https://arxiv.org/pdf/2004.13388)
8. [FFA-Net: Feature Fusion Attention Network for Single Image Dehazing](https://arxiv.org/pdf/2204.03883)
9. [DehazeFormer: Vision Transformers for Single Image Dehazing](https://ieeexplore.ieee.org/abstract/document/10076399) 

## 8. Contact

If you encounter any issues using the code or replicating the experiments, feel free to contact us:

**Email**: [wzr002234@163.com](mailto:wzr002234@163.com)








