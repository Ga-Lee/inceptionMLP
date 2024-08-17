# InceptionMLP: A Vision MLP Architecture with Multi-branch Features Fusion (in process)

##  Acc. of Recent MLP-like Models on Imagenet-1K
|     Model     | Sub-model | Parameters | Top 1 Acc. |
|:-------------:|:---------:|:----------:|:----------:|
|   Cycle-MLP	|     T	    |     28M	 |    81.3    |
|      ViP	    |  Small/7	|     25M	 |    81.5    |
|   iMLP	      |     S	    |     20M	 |    82.1    |
|   Hire-MLP    |     S     |     33M    |    82.1    |
|   DynaMixer   |     S     |     26M    |    82.7    |
|   Cycle-MLP	|     S	    |     50M	 |    82.9    |
|      ViP	    |  Medium/7	|     55M	 |    82.7    |
|   Hire-MLP    |     B     |     58M    |    83.2    |
|   DynaMixer   |     M     |     57M    |    83.7    |
|   Cycle-MLP	|     B	    |     88M	 |    83.4    |
|      ViP	    |  Large/7	|     88M	 |    83.2    |
|   Hire-MLP    |     L     |     96M    |    83.8    |

### Pre-trained iMLP now available at: [Baidu Disk])()

## Requirements
### Environment
```
wandb
torch==1.9.0
torchvision>=0.10.0
pyyaml
timm==0.4.12
```
### Data
ImageNet with the following folder structure

```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```
## Training example
Command line for training on 8 GPUs, parameters can be modified with yaml file

```
./distributed_train.sh 8 --amp --sync-bn --pin-mem
```

## Thanks
The code is based on former work **Dynamixer** and **AS-MLP**


