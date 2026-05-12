# PCDSampling.jl
Draw deterministic samples from multivariate probability distributions.
A Python version of PCD-sampling is available at [https://github.com/KIT-ISAS/PCD_sampling_py](https://github.com/KIT-ISAS/PCD_sampling_py).

## Examples
See the example scripts under `scripts/`.

GPU-acceleration currently only works with `CUDA.jl`.

## Citation
```
@inproceedings{FUSION26_Prossel,
 address = {Trondheim, Norway},
 author = {Dominik Prossel and Zhilun Li and Petr Novikov and Uwe D. Hanebeck},
 booktitle = {Proceedings of the 29th International Conference on Information Fusion (FUSION 2026)},
 month = {June},
 title = {Fast Deterministic Sampling of Gaussian Mixture Densities using Projected Cumulative Distributions},
 year = {2026}
}
```