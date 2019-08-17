
# YOUKU-VSRE-2019-49th
We finally got a not bad score and run into the first 50 team.


# 基于反馈网络的超分辨率恢复 

> 说明

该项目源自CVPR的一个开源项目 Feedback Network for Image Super-Resolution [[arXiv]](https://arxiv.org/abs/1903.09814) [CVF] [Poster]
我们团队根据初步调研和对比（EDSR等），发现基于反馈机制的网络在精细化超分的过程中有一定的优势，故基于此项目构建
来形成团队的算法模型

## 主要内容
1. [依赖](#依赖)
2. [运行](#运行)
3. [未来工作](#未来工作)
4. [论文](#论文)

## 1. 依赖

- Ubuntu16.04 or later
- Python 3 (Anaconda is recommended)
- imageio
- Pytorch (Pytorch version >=1.0 is recommended)
- tqdm 
- pandas
- numpy
- cv2 (pip install opencv-python)
- tensorboardX (for visualization)

### 安装依赖

> pip3 install -r requirement.txt


## 2. 运行

### 2.1. 数据准备工作，且对视频转图像和抽帧操作（首次运行）

+  在data/youku_data_list_20190731.txt中有数据下载地址

+  确保ubuntu系统安装了unzip与zip解压包和ffmpeg:
    ```
    sudo apt-get update
    sudo apt-get install zip unzip ffmpeg
    ```
+  分别拷贝相应的视频压缩文件到`../data`各个子目录中
    ```
    |–data
    |-- round1_train_input
    |-- round1_train_label
    |-- round1_val_input
    |-- round1_val_label
    |-- round1_test_input
    ```
+  运行`./prepare_data.sh`将所有数据生成正确的目录（大约20分钟）

### 2.2. 运行SRFBN进行训练和推理

> 本地测试的CUDA版本为10.0, GPU为Nvidia Titan Pascal (12.0G)
由于推理时需要用到增强功能（SRFBN+）,因此，请至少保证显存**大于等于10G**

执行

> ./run.sh

该脚本流程为: **训练 -> 推理 -> 生成视频 -> 生成提交压缩文件**
运行完成后，请上传`../submit/resulit.zip`结果到评测系统

团队在本地训练模型时用到了3个Titan Pascal卡,为了能够在官方环境正常训练模型，默认为使用
1个GPU来训练（P100， 16G），故将训练`batch size`调成`96`,并将模型验证集的`batch size`调为`2`.

> 推理阶段请保存`batch-size=1`,以便可以正确的写入文件夹！

若官方想加快训练，可以将`./options/train/train_youku.json`中的配置改为如下：



**假定官方有N个至少10G显存的GPU**

```yaml
"gpu_ids": [0, 1, 2, N-1], # N为官方使用的GPU个数
...

"datasets": {
    "train": {
        "mode": "LRHR",
        "dataroot_HR": "../data/round1_train_label",
        "dataroot_LR": "../data/round1_train_input",
        "data_type": "img",
        "n_workers": 4,
        "batch_size": 48xN, # 例如GPU数为2，则batch_size=96
        "LR_size": 20,
        "use_flip": true,
        "use_rot": true,
        "noise": "." // ["G", 1.6]
    },
    "val": {
        "mode": "LRHR",
        "n_workers": 1,
        "batch_size": 1xN, # 例如GPU数为2，则batch_size=2
        "dataroot_HR": "../data/round1_val_label",
        "dataroot_LR": "../data/round1_val_input",
        "data_type": "img"
    }
},

```


### 2.3. 模型说明

`code/models`模型文件将包含团队历史最好的模型文件
`SRFBN_x4_YouKu529.pth`,以及本次运行完成后的最新模型文件`best_ckp.pth`

> 如果需要单独评估历史最佳模型，请将模型`SRFBN_x4_YouKu529.pth`重命名为
`best_ckp.pth`并运行`./evaluate_ref.sh`

## 3. 未来工作

SRFBN虽然在精细化恢复方面具有一定的优势，但是速度方面较差，后期团队将结合EDVR
尝试直接针对视频超分，以提高处理速度。

Todo:
- [ ] Ensemble 模型（提高精度）
- [ ] 提高速度



## 4. Result

![result](../imgs/rst.JPG)


## 4.相关论文

```latex
@inproceedings{li2019srfbn,
    author = {Li, Zhen and Yang, Jinglei and Liu, Zheng and Yang, Xiaomin and Jeon, Gwanggil and Wu, Wei},
    title = {Feedback Network for Image Super-Resolution},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year= {2019}
}

@inproceedings{wang2018esrgan,
    author = {Wang, Xintao and Yu, Ke and Wu, Shixiang and Gu, Jinjin and Liu, Yihao and Dong, Chao and Qiao, Yu and Loy, Chen Change},
    title = {ESRGAN: Enhanced super-resolution generative adversarial networks},
    booktitle = {The European Conference on Computer Vision Workshops (ECCVW)},
    year = {2018}
}


```
