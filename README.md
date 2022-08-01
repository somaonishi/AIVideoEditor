# AI Video Editor
A repository for Deep AI Video Editing, with the latest SOTA models available for a variety of editing tasks.

## Implemented
- Video interpolation by [IFRNet](https://github.com/ltkong218/IFRNet)

## Requirement
- python >= 3.8.x
- CUDA == 11.3

## Install
#### MacOS or Linux
```bash
python -m venv venv
. venv/bin/activate
pip install requirement.txt
```
#### Windows
```powershell
python -m venv venv
./venv/Scripts/activate
pip install requirement.txt
```

## Video Edit Methods
### Video interpolation by [IFRNet](https://github.com/ltkong218/IFRNet) (Super Slow Motion)
Image interpolation by IFRNet makes the input video super-slow-motion.

#### Command
```bash
python video_super_slow.py INPUT_VIDEO_PATH
```
*option*
* `--output-dir`, `-o`: Path to output dir. [default: outputs/]
* `--num-interp`, `-i`: Number of interpolated images (slow motion x). [default: 2]
* `--batch-size`, `-b`: batch size. [default: 1]
* `--model-size`, `-m`: N is normal model, S is small model, L is large model. [default: N]
* `--remain-img`, `-r`: [flag] remain images. [default: False]

Larger models are more accurate but require more computational resources.
Larger batch sizes are faster, but require more computing resources.

##### example
4x slow motion. batch size is 4. Large model.
```bash
python video_super_slow.py input.mov -i 4 -b 4 -m L
```
