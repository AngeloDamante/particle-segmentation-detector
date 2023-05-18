# particle-ViT-segmentation
Challenge resolution by appling segmentation techniques.

[//]: # (descriviamo il problema)
[//]: # (descriviamo l'obiettivo)
[//]: # (Link con le tecniche descritte in un'altra sezione)

Directories layout:
```
particle-ViT-segmentation
├── Dataset
│   └── install_dataset.sh
├── Docker
│   ├── docker-compose.yml
│   └── Dockerfile
├── docs
│   └── nmeth.2808.pdf
├── preprocessing
│   ├── analyser.py
│   ├── Particle.py
│   └── Segmenter.py
├── techniques
│   ├── README.md
│   ├── vit
│   │   ├── include
│   │   ├── main.py
│   │   └── src
│   └── unet
│       ├── include
│       ├── main.py
│       └── src
├── ut
│   ├── ut_analyser.py
│   └── ut_segmenter.pys
├── utils
│   ├── compute_path.py
│   ├── definitions.py
│   ├── logger.py
│   └── Types.py
├── create_segmaps.py
├── main.py
├── requirements.txt
├── README.md
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
In this section we provide utility functions for analyzing and segmenting the original dataset provided by challange.
```
create_segmaps.py [-h] -M {sphere,gauss} [-SNR {snr_1,snr_2,snr_4,snr_7}] [-D {density_high,density_mid,density_low}] [-N NUM] [-IMG SAVE_IMG] [-S SIGMA] [-R RADIUS] [-K KERNEL] [-V VALUE]

optional arguments:
  -h, --help            show this help message and exit
  -M {sphere,gauss}, --mode {sphere,gauss}
                        chose mode to make segmentated map
  -SNR {snr_1,snr_2,snr_4,snr_7}, --snr {snr_1,snr_2,snr_4,snr_7}
                        chose signal to noise ratio
  -D {density_high,density_mid,density_low}, --density {density_high,density_mid,density_low}
                        chose density
  -N NUM, --num NUM     number of images
  -IMG SAVE_IMG, --save_img SAVE_IMG
                        save image
  -S SIGMA, --sigma SIGMA
                        select a sigma in (0.1, 2.0)
  -R RADIUS, --radius RADIUS
                        radius of sphere
  -K KERNEL, --kernel KERNEL
                        kernel dimension
  -V VALUE, --value VALUE
                        value to put on particles coord                       
```
For example
```
python3 create_segmaps.py -M gauss -SNR snr_7 -D density_low -IMG True
```

