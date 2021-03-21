# Pokémon-sprites-classifier
A PyTorch implemented Pokémon dot sprites classifier.

## Structure
```
.
├── README.md
├── dataset  # The training/validating data
├── label.csv  # Currently supported species, used for inference
├── model.py  # Classification CNNs with modified last layer
├── onnx_export.py  # Export a PyTorch model to ONNX format and verify it
├── test.py  # Single image test script
├── train.py  # Model training script
├── transformation.py  # torchvision.transform methods for train and test
└── utils.py
```

## Requirements
* Python 3.7 or above (below are not tested)
* [PyTorch](https://pytorch.org/get-started/locally/)
* Dependencies in requirements.txt
```
pip install -r requirements.txt
```

## Usage
### Train
Run `python train.py -h` for help.
```
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
```
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
```
python test.py 1.png 2.png 3.png
```

## Supported Species
Only Pokémon listed in `label.csv` can be recognized by models.

## How to contribute
TODO

## Roadmap
- [ ] Release a labeling tool
- [ ] Release pre-trained models
- [ ] Design and deploy an inference service
- [ ] Model compression

## Resources
### Sprites
* [Pokémon Database](https://pokemondb.net/sprites)
* [PokéSprite](https://github.com/msikma/pokesprite)

### Team Preview
* [Scopelens](https://scopelens.team/)

