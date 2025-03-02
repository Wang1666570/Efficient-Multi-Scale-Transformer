## Efficient Multi-Scale Transformer with Convolutional Attention for High-Definition Image Dehazing

This repository provides an implementation of our **Efficient Multi-Scale Transformer** for image dehazing, featuring **Multi-Head Convolutional Attention (MHCA)**, a **Gated Feed-Forward Network (GFFN)**, and a **Multi-Scale Loss Function** for robust, high-definition dehazing.

---

## Contribution

- **MHCA (Multi-Head Convolutional Attention):**  Combines convolutional feature extraction with self-attention for efficiency.

- **GFFN (Gated Feed-Forward Network):**  Enhances spatial feature propagation and suppresses unnecessary haze.

- **Multi-Scale Loss Function:**  Integrates perceptual and patch-based loss for refined restoration.

- **State-of-the-Art Performance:**  Achieves **PSNR 37.49 dB** and **SSIM 0.993** on SOTS Indoor, surpassing previous models.

- **Strong Generalization:**  Excels in real-world dehazing tasks (RTTS dataset).

![模型框架](https://raw.githubusercontent.com/Wang1666570/Efficient-Multi-Scale-Transformer/main/framework/framework.png)

---

## 1. Environment

Install the required Python dependencies using:
```bash
pip install -r requirements.txt
```

---

## 2. Dataset

- **RESIDE-6K** (Recommended):  
  - Download from [Papers with Code](https://paperswithcode.com/sota/image-dehazing-on-reside-6k) or other official sources.
  - Organize the dataset directory as follows:
    ```
    D:/workspace/dehaze_dataset/
      └─ RESIDE-6K/
          ├─ train/
          └─ test/
    ```

---

## 3. Training

1. **Edit** the configuration files in `configs/<exp>/*.json` (e.g., `reside6k/default.json`) if needed (e.g., learning rate, batch size).
2. **Run**:
   ```bash
   python train.py \
       --model dehazeformer-b \
       --data_dir D:/workspace/dehaze_dataset \
       --dataset RESIDE-6K \
       --exp reside6k \
       --checkpoint_dir ./ckpt/
   ```
3. **Checkpoints** and logs will be saved to:
   ```
   ckpt/
   └─ weights/
       └─ reside6k/
           └─ dehazeformer-b_epochX_psnrXX.XXXX.pth
   ```

---

## 4. Testing

1. **Ensure** the final `.pth` checkpoint is located in `--save_dir/<exp>/` (e.g., `./saved_models/reside6k/dehazeformer-b.pth`).
2. **Run**:
   ```bash
   python test.py \
       --model dehazeformer-b \
       --data_dir ./data \
       --dataset RESIDE-6K \
       --save_dir ./saved_models \
       --result_dir ./results \
       --exp reside6k
   ```
3. **Outputs**:
   ```
   results/
   └─ RESIDE-6K/
       └─ dehazeformer-b/
           ├─ imgs/              # Dehazed images
           └─ XX.XX | 0.XXXX.csv # PSNR and SSIM results
   ```

---

## 5. Main Results

Below is a quantitative comparison of different methods on various **SOTS** datasets. Our approach achieves the best performance across multiple metrics (PSNR, SSIM).

| Method         | SOTS-Indoor (PSNR) | SOTS-Indoor (SSIM) | SOTS-Outdoor (PSNR) | SOTS-Outdoor (SSIM) | SOTS-mix (PSNR) | SOTS-mix (SSIM) |
|----------------|--------------------|--------------------|---------------------|---------------------|----------------|----------------|
| DCP [4]        | 16.62             | 0.818              | 19.13              | 0.815               | 17.88          | 0.816          |
| DehazeNet [45] | 19.82             | 0.821              | 24.75              | 0.927               | 21.02          | 0.870          |
| MSCNN [8]      | 19.84             | 0.833              | 22.06              | 0.908               | 20.31          | 0.857          |
| AOD-Net [9]    | 20.51             | 0.816              | 24.14              | 0.920               | 20.27          | 0.873          |
| GFN [46]       | 22.30             | 0.880              | 21.55              | 0.844               | 23.52          | 0.905          |
| GridDehazeNet [47] | 32.16         | 0.984              | 30.06              | 0.982               | 25.86          | 0.946          |
| MSBDN [48]     | 33.67             | 0.976              | 32.37              | 0.982               | 28.59          | 0.973          |
| FFA-Net [48]   | 32.68             | 0.976              | 33.57              | 0.984               | 28.97          | 0.973          |
| DehazeFormer [25] | 35.15         | 0.989              | 33.71              | 0.982               | 30.36          | 0.973          |
| **Ours**       | **37.49**         | **0.993**          | **34.64**          | **0.984**           | **31.45**      | **0.985**      |

---

## 6. Contact

For any questions, please contact:  
**Email**: [wzr002234@163.com](mailto:wzr002234@163.com)

---

**Disclaimer**:  
This code is provided for research purposes only. Please review and comply with any relevant licenses and dataset usage policies before use.
```
