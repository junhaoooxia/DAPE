<div align="center">

# DAPE: Dual-Stage Parameter-Efficient Fine-Tuning for Consistent Video Editing

<p align="center">
  <a href="https://arxiv.org/abs/2505.07057"><img src="https://img.shields.io/badge/arXiv-2505.07057-b31b1b.svg" alt="arXiv"></a>
  <a href="https://junhaoooxia.github.io/DAPE.github.io/"><img src="https://img.shields.io/badge/Project-Page-1e90ff.svg" alt="Project Page"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
</p>

Junhao Xia · Chaoyang Zhang · Yecheng Zhang · Chengyang Zhou · Zhichang Wang · Bochun Liu · Dongshuo Yin

Official PyTorch implementation of *Dual-Stage Parameter-Efficient Fine-Tuning for Consistent Spatial and Temporal Representation* (CVPR 2026 Findings).

</div>

---

## ✨ Examples

<table>
<tr>
  <td align="center" width="22%">
    <img src="https://raw.githubusercontent.com/junhaoooxia/DAPE.github.io/main/static/videos/input_1.gif" width="100%"><br>
    <img src="https://raw.githubusercontent.com/junhaoooxia/DAPE.github.io/main/static/videos/output_1.gif" width="100%"><br>
    <sub><i>"…Van&nbsp;Gogh's starry night style"</i></sub>
  </td>
  <td align="center" width="22%">
    <img src="https://raw.githubusercontent.com/junhaoooxia/DAPE.github.io/main/static/videos/input_2.gif" width="100%"><br>
    <img src="https://raw.githubusercontent.com/junhaoooxia/DAPE.github.io/main/static/videos/output_2.gif" width="100%"><br>
    <sub><i>"…lush mountains and a vibrant sunset sky"</i></sub>
  </td>
  <td align="center" width="22%">
    <img src="https://raw.githubusercontent.com/junhaoooxia/DAPE.github.io/main/static/videos/input_3.gif" width="100%"><br>
    <img src="https://raw.githubusercontent.com/junhaoooxia/DAPE.github.io/main/static/videos/output_3.gif" width="100%"><br>
    <sub><i>"…vibrant ocean reef with colorful corals"</i></sub>
  </td>
  <td align="center" width="22%">
    <img src="https://raw.githubusercontent.com/junhaoooxia/DAPE.github.io/main/static/videos/input_4.gif" width="100%"><br>
    <img src="https://raw.githubusercontent.com/junhaoooxia/DAPE.github.io/main/static/videos/output_4.gif" width="100%"><br>
    <sub><i>"A curious squirrel sits on a wooden surface…"</i></sub>
  </td>
</tr>
<tr>
  <td align="center" width="22%">
    <img src="https://raw.githubusercontent.com/junhaoooxia/DAPE.github.io/main/static/videos/input_5.gif" width="100%"><br>
    <img src="https://raw.githubusercontent.com/junhaoooxia/DAPE.github.io/main/static/videos/output_5.gif" width="100%"><br>
    <sub><i>"…glowing purple aurora, impressionist style"</i></sub>
  </td>
  <td align="center" width="22%">
    <img src="https://raw.githubusercontent.com/junhaoooxia/DAPE.github.io/main/static/videos/input_6.gif" width="100%"><br>
    <img src="https://raw.githubusercontent.com/junhaoooxia/DAPE.github.io/main/static/videos/output_6.gif" width="100%"><br>
    <sub><i>"…cyberpunk cityscape with neon-green accents"</i></sub>
  </td>
  <td align="center" width="22%">
    <img src="https://raw.githubusercontent.com/junhaoooxia/DAPE.github.io/main/static/videos/input_7.gif" width="100%"><br>
    <img src="https://raw.githubusercontent.com/junhaoooxia/DAPE.github.io/main/static/videos/output_7.gif" width="100%"><br>
    <sub><i>"A motorbike in watercolor style"</i></sub>
  </td>
  <td align="center" width="22%">
    <img src="https://raw.githubusercontent.com/junhaoooxia/DAPE.github.io/main/static/videos/input_8.gif" width="100%"><br>
    <img src="https://raw.githubusercontent.com/junhaoooxia/DAPE.github.io/main/static/videos/output_8.gif" width="100%"><br>
    <sub><i>"A marble sculpture of a woman running"</i></sub>
  </td>
</tr>
</table>

<p align="center"><sub><b>Top:</b> input video &nbsp;·&nbsp; <b>Bottom:</b> DAPE edit. See <a href="https://junhaoooxia.github.io/DAPE.github.io/">project page</a> for full-resolution videos and baseline comparisons.</sub></p>

---

## 🧠 Method

<p align="center">
  <img src="https://raw.githubusercontent.com/junhaoooxia/DAPE.github.io/main/static/images/pipeline.png" width="92%">
</p>

DAPE decouples one-shot video adaptation into **two parameter-efficient stages** layered on a frozen text-to-video backbone.

---

## 🚀 Installation

```bash
git clone https://github.com/junhaoooxia/DAPE.git
cd DAPE
pip install -r requirements.txt
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip
```

**Tested with:** Python 3.10, PyTorch 2.0.1, CUDA 11.8.

### Pretrained checkpoint

Download the base T2V model from [CCEdit](https://github.com/RuoyuFeng/CCEdit) and place it at:

```
models/tv2v-no2ndca-depthmidas.ckpt
```

---

## ⚡ Quick start

The repo ships with `data/input_example.mp4` (a snow-capped mountain scene) for a one-command sanity check.

```bash
python main.py -b configs/template.yaml --train True --wandb False
```

On an A800 80G this completes in roughly **15 minutes per video** at 512×512, 17 frames.

---

## 🛠 Configuration

Edit `configs/template.yaml` to point at your own video and prompt:

```yaml
data.params.train.params.video_params:
  video_path: data/your_video.mp4
  caption: "<one-sentence description of the source video>"

inference:
  prompt:
    edit1: "<your edit prompt>"
```

---

## 📝 Citation

```bibtex
@inproceedings{xia2026dape,
  title     = {Dual-Stage Parameter-Efficient Fine-Tuning for Consistent Spatial and Temporal Representation},
  author    = {Xia, Junhao and Zhang, Chaoyang and Zhang, Yecheng and Zhou, Chengyang and
               Wang, Zhichang and Liu, Bochun and Yin, Dongshuo},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

---

## 🙏 Acknowledgements

Built on [CCEdit](https://github.com/RuoyuFeng/CCEdit). We thank the authors for their open-source contribution.

## 📜 License

This project is released under the [MIT License](LICENSE).
