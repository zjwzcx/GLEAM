# GLEAM-Bench
<!-- **[Project Page](https://xiao-chen.tech/gleam/) / [arXiv]() / [Code (GLEAM)](https://github.com/zjwzcx/GLEAM)** -->
<!-- [![arXiv](https://img.shields.io/badge/arXiv-2402.16174-blue)](https://arxiv.org/abs/2402.16174)
[![Code](https://img.shields.io/badge/Code-GLEAM-<COLOR>.svg)](https://github.com/zjwzcx/GLEAM)
[![Project](https://img.shields.io/badge/Project-%F0%9F%9A%80-red)](https://xiao-chen.tech/gleam/) -->


<!-- ## Overview -->
<p align="center">
  <img src="../assets/overview_gleambench.png" align="center" width="100%">
</p>
<p align="center">
  <img src="../assets/statistic.png" align="center" width="100%">
</p>


We introduce GLEAM-Bench, a benchmark for generalizable exploration for active mapping in complex 3D indoor scenes.
These scene meshes are characterized by watertight geometry, diverse floorplans (≥10 types), and complex interconnectivity. We unify and refine multi-source datasets through manual filtering, geometric repair, and task-oriented preprocessing. 
To simulate the exploration process, we connect our dataset with NVIDIA Isaac Gym, enabling parallel sensory data simulation and online policy training.


## Download
Please fill in the [form](https://docs.google.com/forms/d/e/1FAIpQLSdq9aX1dwoyBb31nm8L_Mx5FeaVsr5AY538UiwKqg8LPKX9vg/viewform?usp=sharing) to access the download links.

**Consolidated Version (GLEAM)**

We provide all the preprocessed data used in our work, , including mesh files (in `obj` folder), ground-truth surface points (in `gt` folder) and asset indexing files (in `urdf` folder). The directory structure should be as follows.

```
data_gleam
├── train_stage1_512
│   ├── gt
│   ├── obj
│   ├── urdf
├── train_stage2_512
│   ├── gt
│   ├── obj
│   ├── urdf
├── eval_128
│   ├── gt
│   ├── obj
│   ├── urdf
```

The standard training process of GLEAM is divided into two stages (i.e. stage1 and stage2), each involving different 512 training indoor scenes. The evaluation involves 128 unseen testing scenes from ProcTHOR, HSSD, Gibson and Matterport3D (cross-dataset generalization).

<!-- > We integrate and distribute ground-truth data from 1,024 training scenes across two stages, maintaining consistency with GLEAM's training configuration. -->
<!-- >  -->
<!-- > To expedite the training process, we implemented scene compression measures, including the consolidation of entire scenes into unified mesh layers and the removal of textures. In line with our commitment to scalability and community support, full access to the original uncompressed datasets has been openly provided: -->


**Original Separate Version**

The separate meshes are also provided in the download links:
```
data_gleam_raw
├── procthor_12-room_64
│   ├── gt
│   ├── obj
├── gibson_96
│   ├── gt
│   ├── obj
├── hssd_32
│   ├── gt
│   ├── obj
...
```



## Export Meshes from Generated Scenes by ProcTHOR
We also provide the C# script [[HERE](https://github.com/zjwzcx/Batch-Export-ProcTHOR-Meshes)] to export mesh files (.fbx) from generated scenes by ProcTHOR. Note that these generated mesh files have textures and interactive object-level layers.


## Citation

The GLEAM-Bench dataset comes from the GLEAM paper:

- **arXiv**: TODO

- **Code**: https://github.com/zjwzcx/GLEAM

- **BibTex**:
```bibtex
TODO
```


If you use our dataset and benchmark, please kindly cite the original datasets involved in our work. BibTex entries are provided below.

<details><summary>Dataset BibTex</summary>

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
