## Depth Anything in 360°

Official PyTorch implementation of:

> [**Depth Anything in 360°: Towards Scale Invariance in the Wild**](https://insta360-research-team.github.io/DA360/)
>
> Hualie Jiang, Ziyang Song, Zhiqiang Lou, Rui Xu, Minglang Tan

<p align="center">
<img src="assets/teaser.jpg" width="980">
</p>

## Upstream Usage

### Installation

```bash
conda env create -f environment.yaml
conda activate da360
pip install -r requirements.txt
```

### Test On Panoramic Images

```bash
python test.py --model_path ./checkpoints/DA360_large.pth --model_name DA360_large
```

### Evaluation

```bash
bash scripts/evaluate.sh
```

Pretrained models are available from the upstream project page and Google Drive links in the original paper repo.

## Maritime Extension

This workspace also contains a custom maritime adaptation for a downward-looking elliptical fisheye camera mounted above a boat.

## Exploratory Status

This repository should be read as an exploratory playground, not as a serious reproduction effort, a validated benchmark, or a paper-quality investigation.

In particular:

- the maritime work here was mainly a quick "vibe check" of how these depth, masking, fusion, and SLAM-like pieces behave on the custom boat camera setup
- it does **not** contain a proper evaluation over a meaningful dataset
- it does **not** contain careful parameter sweeps or robust tuning
- it does **not** claim faithful reproduction of the upstream papers beyond basic local experimentation
- many outputs are useful for intuition and debugging, but should not be treated as strong evidence of model quality or system performance

If this project is taken further, it still needs:

- a cleaner dataset split
- repeatable evaluation metrics
- broader footage coverage
- proper ablations and parameter tuning
- clearer separation between prototypes and validated results

Added locally on top of upstream DA360:

- fisheye-to-ERP preprocessing for the non-standard downward camera geometry
- point-cloud export and browser viewers
- short-window temporal fusion with separate `background` and `boat` clouds
- a VO plus ICP prototype for more SLAM-like temporal alignment
- an alternative DAP depth backend
- automatic boat and water mask benchmarking and SAM2-based propagation

The custom entry points are documented in [scripts/README.md](scripts/README.md).

Project-specific local folders:

- [maritime_input/README.md](maritime_input/README.md)
- [maritime_output/README.md](maritime_output/README.md)

## Publishing Notes

This repository still points to the upstream remote:

- `origin = https://github.com/Insta360-Research-Team/DA360.git`

So it should not be pushed back there with the local maritime changes.

For publishing, the recommended path is:

1. keep upstream DA360 as the base
2. commit only the custom scripts and docs
3. do not commit `.venv`, checkpoints, private footage, or generated outputs
4. push to a new repo or fork under the target organization

## Acknowledgements

This project is partially based on:

- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [PanDA](https://github.com/caozidong/PanDA)
- [UniFuse](https://github.com/alibaba/UniFuse-Unidirectional-Fusion)

## Citation

```bibtex
@article{jiang2025depth,
  title={Depth Anything in $360^\\circ$: Towards Scale Invariance in the Wild},
  author={Jiang, Hualie and Song, Ziyang and Lou, Zhiqiang and Xu, Rui and Tan, Minglang},
  journal={arXiv preprint arXiv:2512.22819},
  year={2025}
}
```
