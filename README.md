# Pokémon-sprites-classifier
A PyTorch implemented Pokémon sprites classifier.

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
TODO

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

