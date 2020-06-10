# Low-light Illumination and Reflectance based Image Enhancement



This is a PyTorch implementation of Low-light Illumination and Reflectance based Image Enhancement.\
[Project Report](https://yolandaxiao.github.io/files/Capstone_Project.pdf)

![Teaser Image](./imgs/teaser.png?raw=true)

## How to Run
### Steps for testing
1. Clone this repository `git clone https://github.com/YolandaXiao/LLIR_Net-lowlight-image-enhancement.git`
1. Download the checkpoints (https://drive.google.com/drive/folders/1K6v-inOyK-i6vq2PhlIKQGxMTOX9F501?usp=sharing)
   1. Place the checkpoints in the folder `./output/[folder_name]/checkpoint/`
1. (Optional) Download the testing datasets (https://drive.google.com/drive/folders/1YihDK8hXa1zPZ0viZTMqnnBB0AfbC4hC?usp=sharing)
   1. Place the datasets in the folder`./datasets/`
1. Run "python3 test.py --dataset [path to testing dataset] --folder_name [folder name] --cuda"
   1. For example, run `python3 test.py --dataset ./datasets/testing_datasets/MEF --folder_name test1 --cuda`
   1. Remove the `--cuda` tag if you're running on CPU
   1. Replace `MEF` with `DICM` or `NPE` to run test on different datasets
   1. Add `--result_path [output sub-folder path]` to specify the name of the output folder path. The default folder is "result", at the path `./output/[folder_name]/result/`.

### Steps for training
1. Clone this repository `git clone https://github.com/YolandaXiao/LLIR_Net-lowlight-image-enhancement.git`
2. Download the training dataset, 'LOLdataset' (https://drive.google.com/drive/folders/1mlX0J0iIDKRmSHRNDfa_EU1bvI7qicFp?usp=sharing) 
3. Run "python3 train.py --dataset [path to training dataset] --folder_name [folder name] --cuda"
   1. For example, run `python3 train.py --dataset ./datasets/LOLdataset --folder_name test1 --cuda`
   1. Remove the `--cuda` tag if you're running on CPU

## Datasets and Checkpoints Downloads
* Training Dataset(LOL dataset): 
  * https://drive.google.com/drive/folders/1mlX0J0iIDKRmSHRNDfa_EU1bvI7qicFp?usp=sharing
* Testing Datasets, including DICM, MEF, and NPE:
  * https://drive.google.com/drive/folders/1YihDK8hXa1zPZ0viZTMqnnBB0AfbC4hC?usp=sharing
* Checkpoints:
  * https://drive.google.com/drive/folders/1K6v-inOyK-i6vq2PhlIKQGxMTOX9F501?usp=sharing
* All resources:
  * https://drive.google.com/drive/folders/1HH-snlvod-uZkaEATwmzmTva-zle4_KY?usp=sharing

## Folder Structure
```bash
├── datasets
│   ├── LOLdataset
│   │   ├── eval15
│   │   │   ├── low
│   │   │   └── high
│   │   └── our485
│   │       ├── low
│   │       └── high
│   └── testing_datasets
│       ├── DICM
│       ├── MEF
│       └── NPE
├── output
│   ├── [folder1_name]
│   │   ├── checkpoint
│   │   │   └──...
│   │   ├── loss_plot
│   │   │   └──...
│   │   └── train_evaluation
│   │       └──...
│   ├── [folder2_name]
│   │   ├── checkpoint
│   │   │   └──...
│   │   ├── loss_plot
│   │   │   └──...
│   │   └── train_evaluation
│   │       └──...
│   └── ...
├── dataset.py
├── networks.py
├── test.py
├── train.py
├── utils.py
├── vgg16.weight
└── README.md
```
