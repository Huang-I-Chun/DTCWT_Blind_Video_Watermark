# PC_error_concealment

## Environment setting
Testing Environment Version:

Ubuntu 20.04

python 3.8

Example install command:
```bach
conda env create --file environment.yml
conda activate watermark
```

## Demo: Single Video Frame Watermark Embedding 
Go to `demo.ipynb` and type `run all`. This notebook will emebed the watermark to frame_0000.png we provide, and create a folder `image` with three output files:

`image/wm_img.png`: the watermarked frame frame generate from frame_0000.png
`image/wm.png`: the key image generate by watermark bitstream
`image/wm_extract.png`: the key image extract from `image/wm_img.png`


## Demo: Video Watermark Embedding 
