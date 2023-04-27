#!/bin/bash
echo "Downloading Original Dataset provided by challange"

wget https://github.com/AngeloDamante/particle-ViT-segmentation/releases/download/v1.0/Challange_dts.tar.xz
tar -xJv -f Challange_dts.tar.xz --directory .
rm Challange_dts.tar.xz