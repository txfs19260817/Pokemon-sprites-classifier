# Pokémon-sprites-classifier
A PyTorch implemented Pokémon dot sprites classifier.

## Structure
```text
.
├── README.md
├── dataset  # The training/validating data
├── utils  # a Python package providing helper functions
├── label.csv  # Currently supported species, used for inference
├── labeling_tool.py  # A labeling tool helps crop a rental team screenshot
├── model.py  # Classification CNNs with modified last layer
├── onnx_export.py  # Export a PyTorch model to ONNX format and verify it
├── requirements.txt  # Python package requirements
├── service.py  # Flask-based inference service
├── test.py  # Single image test script
└── train.py  # Model training script
```

## Requirements
* Python 3.7 or above (below are not tested)
* [PyTorch](https://pytorch.org/get-started/locally/)
* Dependencies in requirements.txt
```shell
pip install -r requirements.txt
```

## Usage
### Train
Run `python train.py -h` for help.
```text
usage: train.py [-h] [-d DIR] [-b N] [-e N] [-w N] [--lr LR] [-a ARCH]

Train a Pokemon species classifier.

optional arguments:
  -h, --help            show this help message and exit
  -d DIR, --dataset-root-path DIR
                        root path to dataset (default: ./dataset)
  -b N, --batch-size N  input batch size for training (default: 32)
  -e N, --epochs N      number of epochs to train (default: 200)
  -j N, --num-workers N
                        number of workers to sample data (default: 2)
  --lr LR, --learning-rate LR
                        initial learning rate (default: 0.0001)
  -a ARCH, --arch ARCH  model architecture: alexnet | mobilenetv2 | resnet18
                        (default: mobilenetv2)
```
### Test
Run `python test.py -h` for help.
```text
usage: test.py [-h] [-d DIR] [-a ARCH] FILE [FILE ...]

Test the trained Pokemon species classifier.

positional arguments:
  FILE                  images to be tested

optional arguments:
  -h, --help            show this help message and exit
  -d DIR, --dataset-root-path DIR
                        root path to dataset (default: ./dataset)
  -a ARCH, --arch ARCH  model architecture: alexnet | mobilenetv2 | resnet18
                        (default: mobilenetv2)
```
Example:
```shell
python test.py 1.png 2.png 3.png
```

## Supported Species
Only Pokémon listed in `label.csv` can be recognized by models.

## How to contribute
Please feel free to pull requests to help contribute to this project! Also, please help enlarge the dataset with the provided semi-automatic labeling tool.

### Labeling Tool
The provided labeling tool `labeling_tool.py` helps resize and crop a team preview screenshot and output 6 sprite thumbnails. Then users need to rename the cropped images with Pokémon name manually.

Run `python labeling_tool.py -h` for help.
```text
usage: labeling_tool.py [-h] FILE [FILE ...]
```
Example:
```shell
python labeling_tool.py 1.png 2.png 3.png
```
Labeling procedures:
1. Run the tool as shown in the example, and you will get 3*6=18 (If 3 images as in the example) thumbnails
2. Rename them manually with their Pokémon English name in LOWER CASE and some random characters, connected with `-`. E.g. `pikachu-19260817.png`
3. Move labelled images to the folder `dataset/train` and run `cd ./dataset && python data_gen.py`, then Pull Requests.

**Notes**:
* The input images should be SCREENSHOTS from Nintendo Switch or live-streaming tools (e.g. OBS) rather than photographs taken by a camera
* Some names containing symbols need special dealing. The principle is to remove all symbols directly. Examples: 
  * `tapu fini` -> `tapufini`
  * `ho-oh` -> `hooh`
  * `flabébé` -> `flabebe`
  * `mr.mime` -> `mrmime`
  * `farfetch'd` -> `farfetchd`


## Roadmap
- [x] Release a labeling tool
- [ ] Release pre-trained models
- [ ] Design and deploy an inference service
- [ ] Model compression

## Resources
### Sprites
* [Pokémon Database](https://pokemondb.net/sprites)
* [PokéSprite](https://github.com/msikma/pokesprite)

### Team Preview
* [ScopeLens](https://scopelens.team/)

