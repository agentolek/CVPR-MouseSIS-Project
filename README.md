# MouseSIS: Space-Time Instance Segmentation of Mice

[![Paper](https://img.shields.io/badge/arXiv-2409.03358-b31b1b.svg)](https://arxiv.org/abs/2409.03358)
[![Dataset](https://img.shields.io/badge/Dataset-GoogleDrive-4285F4.svg)](https://drive.google.com/drive/folders/1TQns9-WZw-n26FaUE3gqdAhGgrlRUzCp?usp=sharing)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This is the official repository for [**MouseSIS: A Frames-and-Events Dataset for Space-Time Instance Segmentation of Mice**](https://arxiv.org/pdf/2409.03358), accepted at the **Workshop on Neuromorphic Vision** in conjunction with **ECCV 2024** by [Friedhelm Hamann](https://friedhelmhamann.github.io/), [Hanxiong Li](), [Paul Mieske](https://scholar.google.de/citations?user=wQPmm6kAAAAJ&hl=de), [Lars Lewejohann](https://www.vetmed.fu-berlin.de/einrichtungen/vph/we11/mitarbeitende/lewejohann_lars3/index.html) and [Guillermo Gallego](http://www.guillermogallego.es).

👀 **This dataset the base for the [SIS Challenge](https://www.codabench.org/competitions/5600/) hosted in conjunction with the [CVPR 2025 Workshop on Event-based Vision](https://tub-rip.github.io/eventvision2025/).**

<p align="center">
  <img src="./image/visualization_seq12_0003.jpg" alt="MouseSIS Visualization" width="600"/>
</p>

## Key Features
- Space-time instance segmentation dataset focused on mice tracking
- Combined frames and event data from neuromorphic vision sensor
- 33 sequences (~20 seconds each, ~600 frames per sequence)
- YouTubeVIS-style annotations
- Baseline implementation and evaluation metrics included

## Versions

- **v1.0.0** (Current, February 2024): Major refactoring and updates, including improved documentation.
- **v0.1.0** (September 2023): Initial release with basic functionality and dataset.

## Table of Contents
- [Quickstart](#quickstart)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
    - [Data](#data)
    - [Pretrained Weights](#pretrained-weights)
    - [Preprocess Events](#preprocess-events)
- [Evaluation](#evaluation)
    - [Quickstart Evaluation](#quickstart-evaluation)
    - [Evaluation on Full Validation Set](#evaluation-on-full-validation-set)
    - [Evaluation on Test Set](#evaluation-on-test-set)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [Additional Resources](#additional-resources)
- [License](#license)

## Quickstart

If you want to work with the dataset the quickest way to access the data and get an idea of it's structure is [downloading one sequence](https://drive.google.com/drive/folders/1amY4kuaZFWdpgHg4RfTrw9Qb-tKrM-8h?usp=drive_link) and the annotations of the according split and visualizing the data, e.g. `seq12.h5`:

```bash
python scripts/visualize_events_frames_and_masks.py --h5_path data/MouseSIS/top/val/seq12.h5 --annotation_path data/MouseSIS/val_annotations.json
```

This requires `h5py, numpy, Pillow, tqdm`. The full dataset structure is explained [here](docs/DATASET.md).

## Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:tub-rip/MouseSIS.git
   cd MouseSIS
   ```

2. Set up the environment:
   ```bash
   conda create --name MouseSIS python=3.8
   conda activate MouseSIS
   ```

3. Install PyTorch (choose a command compatible with your CUDA version from the [PyTorch website](https://pytorch.org/get-started/locally/)), e.g.:
   ```bash
   conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
   ```

4. Install other dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

### Data

1. Create a folder for the original data

    ```bash
    cd <project-root>
    mkdir -p data/MouseSIS
    ```

2. [Download the data and annotation](https://drive.google.com/drive/folders/1amY4kuaZFWdpgHg4RfTrw9Qb-tKrM-8h) and save it in `<project-root>/data/MouseSIS`.
**You do not necessarily need to download the whole dataset, e.g. you can only download the sequences needed for the sequences you want to evaluate on**.
The `data/MouseSIS` folder should be organized as follows:

    ```txt
    data/MouseSIS
    │
    ├── top/
    │   ├── train
    │   │   ├── seq_02.hdf5
    │   │   ├── seq_05.hdf5
    │   │   ├── ...
    │   │   └── seq_33.hdf5
    |   ├── val
    │   │   ├── seq_03.hdf5
    │   │   ├── seq_04.hdf5
    │   │   ├── ...
    │   │   └── seq_25.hdf5
    │   └── test
    │       ├── seq_01.hdf5
    │       ├── seq_07.hdf5
    │       ├── ...
    │       └── seq_32.hdf5
    ├── dataset_info.csv
    ├── val_annotations.json
    └── train_annotations.json
    ```

    - **`top/`**: This directory contains the frame and event data for the Mouse dataset captured from top view, stored as 33 individual `.hdf5` files, each containing approximately 20 seconds of data (around 600 frames), along with temporally aligned events.
    - **`dataset_info.csv`**: This CSV file contains metadata for each sequence, such as recording dates, providing additional context and details about the dataset.
    - **`<split>_annotations.json`**: The annotation file of top view for the respective splits follows a structure similar to MSCOCO's format in JSON, with some modifications. Note that the test annotations are not publicly available. The definition of json files is:

    ```txt
    {
        "info": {
            "description": "string",     // Dataset description
            "version": "string",         // Version identifier
            "date_created": "string"     // Creation timestamp
        },
        "videos": [
            {
                "id": "string",          // Video identifier (range: "01" to "33")
                "width": integer,        // Frame width in pixels (1280)
                "height": integer,       // Frame height in pixels (720)
                "length": integer        // Total number of frames
            }
        ],
        "annotations": [
            {
                "id": integer,           // Unique instance identifier
                "video_id": "string",    // Reference to parent video
                "category_id": integer,  // Object category (1 = mouse)
                "segmentations": [
                    {
                        "size": [height: integer, width: integer],  // Mask dimensions
                        "counts": "string"                          // RLE-encoded segmentation mask
                    }
                ],
                "areas": [float],        // Object area in pixels
                "bboxes": [              // Bounding box coordinates
                    [x_min: float, y_min: float, width: float, height: float]
                ],
                "iscrowd": integer      // Crowd annotation flag (0 or 1)
            }
        ],
        "categories": [
            {
                "id": integer,          // Category identifier
                "name": "string",       // Category name
                "supercategory": "string" // Parent category
            }
        ]
    }
    ```

### Pretrained Weights

Download the [model weights](https://drive.google.com/drive/folders/1-P1HN4FZEy3ETn5rrQiMoDQx3378HLQW?usp=drive_link):
   ```bash
   cd <project-root>
   mkdir models
   # Download yolo_e2vid.pt, yolo_frame.pt, and XMem.pth from the provided link
   # and place them in the models directory
   ```

Afterwards, the `models` folder should be organized as follows:

```txt
models
├── XMem.pth
├── yolo_e2vid.pt
└── yolo_frame.pt
```

### Preprocess Events

This preprocessing step is required only when evaluating the ModelMixSort method from the paper. It relies on e2vid images reconstructed at the grayscale image timesteps.

```bash
python scripts/preprocess_events_to_e2vid_images.py --data_root data/MouseSIS
```

## Evaluation

After downloading the data and model weights, proceed with evaluation. First run inference, e.g. our provided inference script like:

```bash
python3 scripts/inference.py --config <path-to-config-yaml>
```

This saves a file `output/<tracker-name>/final_results.json`. The file contains the predictions in this structure:

```txt
[
  {
    "video_id": int,
    "score": float,
    "instance_id": int,
    "category_id": int,
    "segmentations": [
      null | {
        "size": [int, int],
        "counts": "RLE encoded string"
      },
      ...
    ],
  },
  ...
]
```

Then run the evaluation script like this:

```bash
python src/TrackEval/run_mouse_eval.py --TRACKERS_TO_EVAL <tracker-name> --SPLIT_TO_EVAL <split-name>
```

Below are specific options listed.

### Quickstart Evaluation

This section describes how to run a minimal evaluation workflow on one sequence of the validation set. Only download the sequence `seq_25.hdf5` from the validation set and the according annotations `val_annotations.json`. The resulting folder should look as follows:

```txt
data/MouseSIS
│
├── top/
|   ├── val
│   │   └── seq_25.hdf5
└── val_annotations.json
```

Now you can run inference as

```bash
python3 scripts/inference.py --config configs/predict/quickstart.yaml
```

and then evaluation as

```bash
python scripts/eval.py --TRACKERS_TO_EVAL quickstart --SPLIT_TO_EVAL val
```

This should return the following results

| Sequence |   HOTA   |   MOTA   |   IDF1    |
|----------|----------|----------|-----------|
| 25       | 30.15    | 39.125   | 35.315    |
| **Avg.** | 30.15    | 39.125   | 35.315    |

### Evaluation on Full Validation Set

Similar as for quickstart but download all sequences of the validation set (sequences 3, 4, 12, 25).

```bash
python3 scripts/inference.py --config configs/predict/combined_on_validation.yaml
python scripts/eval.py --TRACKERS_TO_EVAL combined_on_validation --SPLIT_TO_EVAL val
```

Here you should get the following results

| Sequence |   HOTA   |   MOTA   |   IDF1    |
|----------|----------|----------|-----------|
| 3        | 54.679   | 72.432   | 60.212    |
| 4        | 51.717   | 64.942   | 58.36     |
| 12       | 39.497   | 66.049   | 45.431    |
| 25       | 30.15    | 39.125   | 35.315    |
| **Avg.** | 45.256   | 62.097   | 50.459    |

### Evaluation on Test Set Without Sequences 1 & 7 (SIS Challenge)

In this case, download all test sequences and run

```bash
python3 scripts/inference.py --config configs/predict/sis_challenge_baseline.yaml
```

For evaluation you can upload the `final_results.json` to the challenge/benchmark page, which results in the following combined metrics:

| Sequence |   HOTA   |   MOTA   |   IDF1    |
|----------|----------|----------|-----------|
| **Avg.** |  0.43    |  0.45    |  0.5      |

Please note that results vary slightly from the ones reported in the paper after updates for the challenge. Please refer to version v0.1.0 to reproduce the exact paper results.

## Acknowledgements

We greatfully appreciate the following repositories and thank the authors for their excellent work:

- [XMem](https://github.com/hkchengrex/XMem)
- [TrackEval](https://github.com/JonathonLuiten/TrackEval)

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@inproceedings{hamann2024mousesis,
  title={{MouseSIS}: A Frames-and-Events Dataset for Space-Time Instance Segmentation of Mice},
  author={Friedhelm Hamann and Hanxiong Li and Paul Mieske and Lars Lewejohann and Guillermo Gallego},
  booktitle={European Conference on Computer Vision Workshops (ECCVW)},
  year={2024}
}
```

## Additional Resources

* [Recording Software (CoCapture)](https://github.com/tub-rip/CoCapture)
* [Secrets of Event-Based Optical Flow (TPAMI 2024)](https://github.com/tub-rip/event_based_optical_flow)
* [EVILIP: Event-based Image Reconstruction as a Linear Inverse Problem (TPAMI 2022)](https://github.com/tub-rip/event_based_image_rec_inverse_problem)
* [Research page (TU Berlin, RIP lab)](https://sites.google.com/view/guillermogallego/research/event-based-vision)
* [Science Of Intelligence Homepage](https://www.scienceofintelligence.de/)
* [Course at TU Berlin](https://sites.google.com/view/guillermogallego/teaching/event-based-robot-vision)
* [Survey paper](http://rpg.ifi.uzh.ch/docs/EventVisionSurvey.pdf)
* [List of Event-based Vision Resources](https://github.com/uzh-rpg/event-based_vision_resources)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
