## Environment setting
Testing Environment Version:

Ubuntu 20.04

python 3.8

Example install command:
```bach
conda env create --file environment.yml
conda activate watermark
```

## Demo: Single Video Frame Watermark Embedding/Extracting 
Go to `demo.ipynb` and click `run all`. This notebook will emebed/extract the watermark using frame_0000.png we provide, and create a folder `image` with three output files:

`image/wm_img.png`: the watermarked frame frame generate from frame_0000.png

`image/wm.png`: the key image generate by watermark bitstream

`image/wm_extract.png`: the key image extract from `image/wm_img.png`


## Demo: Video Watermark Embedding 
Run following command:

```bach
python embed.py -i [INPUT_VIDEO_PATH] -o [OUTPUT_VIDEO_PATH] -k [KEY] -cl [CODE_LENGTH] -t [CORE_NUM]
```

Example command:

```bach
python embed.py -i video/life_300.mp4 -o video/life_300_wm.mp4 -k 2948  -t 16
```

`-i video/life_300.mp4`: the input raw video path without watermark

`-o video/life_300_wm.mp4`: the output watermarked video path

`-k 2948`: the key value we used to generate the embedding bitstream

`-cl 60`: the generate bitstream length

`-t 16`: use 16 cores to do parallel

For more informations (such as adjusting the strength):

```python embed.py --help```

(Note: All the default parameters are same as the paper)

## Demo: Video Watermark Extracting 
Run following command:

```bach
python extract.py -i [INPUT_VIDEO_PATH] -o [OUTPUT_VIDEO_PATH] -k [KEY] -cl [CODE_LENGTH] -t [CORE_NUM]
```

Example command:

```bach
python extract.py -i video/life_300_wm.mp4 -o result/life_300_wm.json -k 2948 -cl 60 -t 16
```

`-i video/life_300_wm.mp4`: the input watermarked video path

`-o result/life_300_wm.json`: the output json file, cotain 3 keys. `keys` store the final extracted bitstream from this watermarked video, `ans` store the ground truth bitstream of this watermarked video, and `perframe_keys` stores the per-frame extracted keys.

`-k 2948`: the key value we used to generate the embedding bitstream

`-cl 60`: the generate bitstream length

`-t 16`: use 16 cores to do parallel

For more informations (such as adjusting the strength):

```python extract.py --help```

(Note: All the default parameters are same as the paper)
