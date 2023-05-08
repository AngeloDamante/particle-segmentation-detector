# particle-ViT-segmentation
Challenge resolution by appling segmentation techniques.

[//]: # (descriviamo il problema)
[//]: # (descriviamo l'obiettivo)
[//]: # (Link con le tecniche descritte in un'altra sezione)

Directories layout:
```
particle-ViT-segmentation
├── Dataset
│   ├── Challenge
│   │   ├── ground_truth
│   │   └── VIRUS
│   └── install_dataset.sh
├── Docker
│   ├── docker-compose.yml
│   └── Dockerfile
├── docs
│   └── nmeth.2808.pdf
├── main.py
├── preprocessing
│   ├── analyser.py
│   ├── __init__.py
│   └── Particle.py
├── README.md
├── techniques
│   ├── README.md
│   ├── segmenter
│   │   ├── include
│   │   ├── main.py
│   │   └── src
│   └── utnet
│       ├── include
│       ├── main.py
│       └── src
├── utils
│   └── logger.py
└── verify_requirements.sh
```

## Usage
It's possible train and use the two segmentation techniques directly using the docker container.

### Requirements
```
- Docker >= 19
- gpu nvidia
- nvidia-container-toolkit
```
Make sure you have all the necessary requirements to use this repo.
```
# in particle-ViT-segmentation directory
sh verify_requirements.sh
```
Finally, if you want to install the necessary packages , you can run follow lines
```
pip3 install -r requirements.txt
```

### Docker
`TODO`

## Dataset
The starting datasets for challenge can be found [here](http://www.bioimageanalysis.org/track/).
For your convenience, it can be easily downloaded.
``` bash
# in particle-ViT-segmentation directory
cd Dataset
sh install_dataset.sh
```
[//]: # (successivamente avremo il dataset per train e test)

### Preprocess
In this section we provide utility functions to analyze the original dataset provided by challange.
