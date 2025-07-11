<p align="center">
<h1 align="center"><strong>GLEAM: Learning Generalizable Exploration Policy for Active Mapping in Complex 3D Indoor Scene</strong></h1>
  <p align="center">
      	<strong>ICCV 2025</strong><br>
    <a href='https://xiao-chen.tech/' target='_blank'>Xiao Chen</a>&emsp;
	  <a href='https://tai-wang.github.io/' target='_blank'>Tai Wang</a>&emsp;
    <a href='https://quanyili.github.io/' target='_blank'>Quanyi Li</a>&emsp;
    <a href='https://taohuang13.github.io/' target='_blank'>Tao Huang</a>&emsp;
	  <a href='https://oceanpang.github.io/' target='_blank'>Jiangmiao Pang</a>&emsp;
    <a href='https://tianfan.info/' target='_blank'>Tianfan Xue</a>&emsp;
    <br>
    The Chinese University of Hong Kong&emsp;Shanghai AI Laboratory
    <br>
  </p>
</p>


<div id="top" align="center">

<a href='https://arxiv.org/abs/2505.20294' style='padding-left: 0.5rem;'><img src='https://img.shields.io/badge/arXiv-2505.20294-A42C25?style=flat&logo=arXiv&logoColor=A42C25'></a>
<a href='https://xiao-chen.tech/gleam/' style='padding-left: 0.5rem;'><img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'></a>
<a href='https://github.com/zjwzcx/GLEAM/tree/master/data_gleam' style='padding-left: 0.5rem;'><img src='https://img.shields.io/badge/Data-GLEAMBench-FE7A16?style=flat&logo=google-sheets&logoColor=A42C25'></a>

</div>


## 📋 Contents
1. [About](#-about)
2. [Dataset](#-dataset)
3. [Installation](#-installation)
4. [Training & Evaluation](#-training-evaluation)
5. [Citation](#-citation)
6. [License](#-license)


## 🏠 About
<div style="text-align: center;">
    <img src="assets/overview.png" alt="Dialogue_Teaser" width=100% >
</div>

Generalizable active mapping in complex unknown environments remains a critical challenge for mobile robots. Existing methods, constrained by insufficient training data and conservative exploration strategies, exhibit limited generalizability across scenes with diverse layouts and complex connectivity. To enable scalable training and reliable evaluation, we introduce **GLEAM-Bench**, the first large-scale benchmark designed for generalizable active mapping with 1,152 diverse 3D scenes from synthetic and real-scan datasets. Building upon this foundation, we propose **GLEAM**, a unified generalizable exploration policy for active mapping. Its superior generalizability comes mainly from our semantic representations, long-term navigable goals, and randomized strategies. It significantly outperforms state-of-the-art methods, achieving 66.50% coverage (+9.49%) with efficient trajectories and improved mapping accuracy on 128 unseen complex scenes.
<!-- We propose GLEAM, a unified generalizable exploration policy for active mapping. It significantly outperforms state-of-the-art methods, achieving 66.50% coverage (+9.49%) with efficient trajectories and improved mapping accuracy on 128 unseen complex scenes. -->


## 📊 Dataset

<p align="center">
  <img src="assets/overview_gleambench.png" align="center" width="100%">
</p>
<p align="center">
  <img src="assets/statistic.png" align="center" width="100%">
</p>

**GLEAM-Bench** includes 1,152 diverse 3D scenes from synthetic and real-scan datasets for benchmarking generalizable active mapping policies. These curated scene meshes are characterized by near-watertight geometry, diverse floorplan (≥10 types), and complex interconnectivity. We unify these multi-source datasets through filtering, geometric repair, and task-oriented preprocessing. Please refer to the **[guide](https://github.com/zjwzcx/GLEAM/blob/master/data_gleam/README.md)** for more details and scrips.

We provide all the preprocessed data used in our work, including mesh files (in `obj` folder), ground-truth surface points (in `gt` folder) and asset indexing files (in `urdf` folder). We recommend users fill out the form to access the **download link [[HERE](https://docs.google.com/forms/d/e/1FAIpQLSdq9aX1dwoyBb31nm8L_Mx5FeaVsr5AY538UiwKqg8LPKX9vg/viewform?usp=sharing)]**. The directory structure should be as follows. 


```
GLEAM
├── README.md
├── gleam
│   ├── train
│   ├── test
│   ├── ...
├── data_gleam
│   ├── README.md
│   ├── train_stage1_512
│   │   ├── gt
│   │   ├── obj
│   │   ├── urdf
│   ├── train_stage2_512
│   │   ├── gt
│   │   ├── obj
│   │   ├── urdf
│   ├── eval_128
│   │   ├── gt
│   │   ├── obj
│   │   ├── urdf
├── ...
```


## 🛠️ Installation

We test our code under the following environment:
- NVIDIA RTX 3090/4090 (24GB VRAM)
- NVIDIA Driver: 545.29.02
- Ubuntu 20.04
- CUDA 11.8
- Python 3.8.12
- PyTorch 2.0.0+cu118


1. Clone this repository.

```bash
git clone https://github.com/zjwzcx/GLEAM
cd GLEAM
```

2. Create an environment and install PyTorch.

```bash
conda create -n gleam python=3.8 -y
conda activate gleam
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

```

3. NVIDIA Isaac Gym Installation: https://developer.nvidia.com/isaac-gym/download
```bash
cd isaacgym/python
pip install -e .
```

4. Install GLEAM.

```bash
pip install -r requirements.txt
pip install -e .
```


## 🕹️ Training & Evaluation

[Weights & Bias](https://wandb.ai/site/) (wandb) is highly recommended for analyzing the training logs. If you want to use wandb in our codebase, please paste your wandb API key into `wandb_utils/wandb_api_key_file.txt`. If you don't want to use wandb, please add `--stop_wandb` into the following command. 

We provide the standard checkpoints of GLEAM **[HERE](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155204425_link_cuhk_edu_hk/EiOi5TvbO6JJktArhRGZkLsB8i0ghpwwh-lwFwz4GVASQA?e=Sbhzcm)**. Please use the 40k-step checkpoint as the standard. We also provide the Stage 2 checkpoints excluding the 96 Gibson scenes, as this exclusion made the model more robust and stable overall.

### Training

Please run the following command to reproduce the standard two-stage training of GLEAM.

<!-- ```bash
python gleam/train/train_gleam_stage1.py --sim_device=cuda:0 --num_envs=32 --stop_wandb --headless
``` -->

<!-- And then you need to run the following command to launch training: -->
Stage 1 with 512 scenes:
```bash
python gleam/train/train_gleam_stage1.py --sim_device=cuda:0 --num_envs=32 --headless
```

Stage 2 with additional 512 scenes, continually trained based on the pretrained checkpoint (specified by `--ckpt_path`) from stage 1. Take our released checkpoint as example, `ckpt_path` should be `runs/train_gleam_stage1/models/rl_model_40000000_steps.zip`.

```bash
python gleam/train/train_gleam_stage2.py --sim_device=cuda:0 --num_envs=32 --headless --ckpt_path=${YOUR_CKPT_PATH}$
```

### Customized Training Environments

If you want to customize a novel training environment, you need to create your environment and configuration files in `gleam/env` and then define the task in `gleam/__init__.py`.


### Evaluation

Please run the following command to evaluate the generalization performance of GLEAM on 128 unseen scenes from the test set of GLEAM-Bench. The users should specify the checkpoint via `--ckpt_path`.

```bash
python gleam/test/test_gleam_gleambench.py --sim_device=cuda:0 --num_envs=32 --headless --stop_wandb=True --ckpt_path=${YOUR_CKPT_PATH}$
```


### Main Results

<p align="center">
  <img src="assets/main_result.png" align="center" width="100%">
</p>



## 📝 TODO List
- \[x\] Release GLEAM-Bench (dataset) and the arXiv paper in May 2025.
- \[x\] Release the training code in May 2025.
- \[x\] Release the evaluation code in June 2025.
- \[x\] Release the key scripts in June 2025.
- \[x\] Release the pretrained checkpoint in June 2025.


## 🔗 Citation
If you find our work helpful, please cite it:

```bibtex
@article{chen2025gleam,
  title={GLEAM: Learning Generalizable Exploration Policy for Active Mapping in Complex 3D Indoor Scenes},
  author={Chen, Xiao and Wang, Tai and Li, Quanyi and Huang, Tao and Pang, Jiangmiao and Xue, Tianfan},
  journal={arXiv preprint arXiv:2505.20294},
  year={2025}
}
```

If you use our codebase, dataset, and benchmark, please kindly cite the original datasets involved in our work. BibTex entries are provided below.

<details><summary>Dataset BibTex</summary>

```bibtex
@article{ai2thor,
  author={Eric Kolve and Roozbeh Mottaghi and Winson Han and
          Eli VanderBilt and Luca Weihs and Alvaro Herrasti and
          Daniel Gordon and Yuke Zhu and Abhinav Gupta and
          Ali Farhadi},
  title={{AI2-THOR: An Interactive 3D Environment for Visual AI}},
  journal={arXiv},
  year={2017}
}
```
```bibtex
@inproceedings{chen2024gennbv,
  title={GenNBV: Generalizable Next-Best-View Policy for Active 3D Reconstruction},
  author={Chen, Xiao and Li, Quanyi and Wang, Tai and Xue, Tianfan and Pang, Jiangmiao},
  year={2024}
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
}
```
```bibtex
@inproceedings{rudin2022learning,
  title={Learning to walk in minutes using massively parallel deep reinforcement learning},
  author={Rudin, Nikita and Hoeller, David and Reist, Philipp and Hutter, Marco},
  booktitle={Conference on Robot Learning},
  pages={91--100},
  year={2022},
  organization={PMLR}
}
```
```bibtex
@inproceedings{procthor,
  author={Matt Deitke and Eli VanderBilt and Alvaro Herrasti and
          Luca Weihs and Jordi Salvador and Kiana Ehsani and
          Winson Han and Eric Kolve and Ali Farhadi and
          Aniruddha Kembhavi and Roozbeh Mottaghi},
  title={{ProcTHOR: Large-Scale Embodied AI Using Procedural Generation}},
  booktitle={NeurIPS},
  year={2022},
  note={Outstanding Paper Award}
}
```
```bibtex
@inproceedings{xiazamirhe2018gibsonenv,
  title={Gibson Env: real-world perception for embodied agents},
  author={Xia, Fei and R. Zamir, Amir and He, Zhi-Yang and Sax, Alexander and Malik, Jitendra and Savarese, Silvio},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2018 IEEE Conference on},
  year={2018},
  organization={IEEE}
}
```
```bibtex
@article{khanna2023hssd,
    author={Khanna*, Mukul and Mao*, Yongsen and Jiang, Hanxiao and Haresh, Sanjay and Shacklett, Brennan and Batra, Dhruv and Clegg, Alexander and Undersander, Eric and Chang, Angel X. and Savva, Manolis},
    title={{Habitat Synthetic Scenes Dataset (HSSD-200): An Analysis of 3D Scene Scale and Realism Tradeoffs for ObjectGoal Navigation}},
    journal={arXiv preprint},
    year={2023},
    eprint={2306.11290},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
```bibtex
@article{Matterport3D,
  title={Matterport3D: Learning from RGB-D Data in Indoor Environments},
  author={Chang, Angel and Dai, Angela and Funkhouser, Thomas and Halber, Maciej and Niessner, Matthias and Savva, Manolis and Song, Shuran and Zeng, Andy and Zhang, Yinda},
  journal={International Conference on 3D Vision (3DV)},
  year={2017}
}
```
</details>


## 📄 License
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>
<br />
This work is under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
