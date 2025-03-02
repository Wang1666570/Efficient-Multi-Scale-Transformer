### Efficient Multi-Scale Transformer with Convolutional Attention for High-Definition Image Dehazing

### 1. Environment

```bash
pip install requirements.txt
```


### 2. Dataset
- **RESIDE-6K** (Recommended):  
  - Download from [Papers with Code](https://paperswithcode.com/sota/image-dehazing-on-reside-6k) or other official sources.
  - Organize as:
    ```
    D:/workspace/dehaze_dataset/
      └─ RESIDE-6K/
          ├─ train/
          └─ test/
    ```

### 3. Training
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

### 4. Testing
1. **Ensure** the final `.pth` checkpoint is in `--save_dir/<exp>/` (e.g., `./saved_models/reside6k/dehazeformer-b.pth`).
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

### 5. Contact
- **Email**: wzr002234@163.com


