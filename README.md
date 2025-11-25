# RollPlace

Official code release for the paper:
***“RollPlace: Improving Macro Placement via Monte Carlo Rollout Search”***

---

## Requirements

### Python Package Dependencies

* python == 3.10
* numpy == 1.23.0
* treelib == 1.7.0
* matplotlib == 3.9.4
* tensorboardX == 2.6.1
* scipy == 1.13.1

### Installation

Run the following commands to set up the required Python environment:

```bash
conda create -n RollPlace python==3.10
conda activate RollPlace
pip install -r requirements.txt
```

### hMETIS

RollPlace relies on **hMETIS** for standard-cell clustering.
Download hMETIS latest stable release (1.5.3) from:
[https://karypis.github.io/glaros/software/metis/overview.html](https://karypis.github.io/glaros/software/metis/overview.html)
and extract **shmetis** into the current project directory.

---

## Benchmarks

### ISPD2005 Benchmarks

Please download the full ISPD2005 benchmark suite from:
[https://www.cerc.utexas.edu/~zixuan/ispd2005dp.tar.xz](https://www.cerc.utexas.edu/~zixuan/ispd2005dp.tar.xz)
and extract it into the `benchmarks/ispd2005` directory.

### ChiPBench Benchmarks

Download the ChiPBench dataset following the instructions at:
[https://github.com/MIRALab-USTC/ChiPBench](https://github.com/MIRALab-USTC/ChiPBench)
Then convert the dataset into the corresponding **BookShelf** format and copy the benchmarks into the `benchmarks/ChiPBench` directory.

---

# Macro Placement

After preparing all required environments and datasets, you can run RollPlace using the scripts below.
Placement results will be saved in the `result` directory.

Here are only three examples provided for reference. Run `python RollPlace.py -h` to view all available arguments.

---

## 1. Macro Placement on `adaptec1` (ISPD2005)

Example command for running macro placement on **adaptec1** with grid size 160:

```bash
python RollPlace.py --benchmark_folder ispd2005 --dataset adaptec1 --max_workers 1 --grid_num 160
```

## 2. Macro Placement on `bigblue3` (ISPD2005)

Example with parallel search (3 workers), customized initialization, and search iterations:

```bash
python RollPlace.py --benchmark_folder ispd2005 --dataset bigblue3 --max_workers 3 --grid_num 233 --init_round 50 --search_round 1000 --random_placed_times 50
```

## 3. Macro Placement on `ariane133` (ChiPBench)

Example command for running RollPlace on **ariane133**:

```bash
python RollPlace.py --benchmark_folder ChiPBench --dataset ariane133 --max_workers 1 --grid_num 160
```

---

# OpenROAD Flow

To evaluate the generated macro placement using OpenROAD:

1. Install and set up ChiPBench, which is based on [OpenROAD](https://github.com/The-OpenROAD-Project/OpenROAD) and [OpenROAD-flow-scripts](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts).
2. Copy the placement results from `result/ChiPBench` into the ChiPBench project directory.
3. Use the tools in ChiPBench's `extra` folder to convert the results to **DEF/LEF** formats.
4. Continue running the OpenROAD flow in *Macro Placement Evaluation* mode within ChiPBench.

---

# Citation

If you find this work helpful, please cite:

```
@ARTICLE{RollPlace,
  author={Zhou, Qi and Liu, Guojun and Qi, Guangzhi and Lu, Ming and Liu, Jiechu and Liu, Zhongli and Yang, Jianqun and Li, Xingji},
  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems},
  title={RollPlace: Improving Macro Placement via Monte Carlo Rollout Search},
  year={2025},
  doi={10.1109/TCAD.2025.3635566}}
```