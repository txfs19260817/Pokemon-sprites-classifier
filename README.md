# Pokémon-sprites-classifier
A PyTorch implemented Pokémon dot sprites classifier.

## Structure
```text
.
├── Dockerfile  # Inference service Dockerization
├── README.md
├── app.py  # Flask-based inference service
├── configs  # Configuration files for the inference service
├── dataset  # The training/validating data
│   └── label.csv  # Currently supported species, used for inference
├── labeling_tool.py  # A labeling tool helps crop a rental team screenshot
├── model_export.py  # Export a PyTorch model to an ONNX model or a Torch Script
├── requirements.txt  # Python package requirements
├── test.py  # Single image test script
├── train.py  # Model training script
└── utils  # A Python package providing helper functions
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
usage: train.py [-h] [-d DIR] [-b N] [-e N] [-j N] [-lr LR] [-a ARCH] [-c FILE]

Train a Pokemon species classifier in PyTorch.

optional arguments:
  -h, --help            show this help message and exit
  -d DIR, --dataset-root-path DIR
                        root path to dataset (default: ./dataset)
  -b N, --batch-size N  input batch size for training (default: 32)
  -e N, --epochs N      number of epochs to train (default: 300)
  -j N, --num-workers N
                        number of workers to sample data (default: 2)
  -lr LR, --learning-rate LR
                        initial learning rate (default: 0.001)
  -a ARCH, --arch ARCH  model architecture: alexnet | mobilenetv2 | mobilenetv3 | resnet18 | shufflenetv2 (default: shufflenetv2)
  -c FILE, --csv-path FILE
                        label.csv saving path (default: ./dataset/label.csv)
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
  -a ARCH, --arch ARCH  model architecture: alexnet | mobilenetv2 | mobilenetv3 | resnet18 | shufflenetv2 (default: shufflenetv2)
```
Example:
```shell
python test.py 1.png 2.png 3.png
```

### Server
Example:
```shell
python app.py -d
```

## Supported Species
Only Pokémon listed in `label.csv` can be recognized by models.

## Pretrained models
Please refer to the [release](https://github.com/txfs19260817/Pokemon-sprites-classifier/releases) page. Please download the weight files and move them under the root directory.

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
3. Move labelled images to the folder `dataset/train` and run `cd ./dataset && python data_gen.py`, then Pull Requests

**Notes**:
* The input images should be SCREENSHOTS from Nintendo Switch or live-streaming tools (e.g. OBS) rather than photographs taken by a camera
* Some names containing symbols need special dealing. The principle is to remove all symbols directly. Examples: 
  * `tapu fini` -> `tapufini`
  * `ho-oh` -> `hooh`
  * `flabébé` -> `flabebe`
  * `mr.mime` -> `mrmime`
  * `farfetch'd` -> `farfetchd`


## Docker
A Dockerfile is prepared for deploying inference service with gunicorn. Please check `configs` for configuration.
```shell
docker build -t $TAG --build-arg PORT=$PORT --build-arg CERT_PATH=$CERT_PATH .
docker run -it -p $PORT:$PORT -v $CERT_PATH:$CERT_PATH $TAG
```

## Roadmap
- [x] Release a labeling tool
- [x] Release pre-trained models
- [x] Design and deploy an inference service
- [ ] Model compression

## Resources
### Sprites
* [Pokémon Database](https://pokemondb.net/sprites)
* [PokéSprite](https://github.com/msikma/pokesprite)

### Team Preview
* [ScopeLens](https://scopelens.team/)

