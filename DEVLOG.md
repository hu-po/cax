# 08.10.2024

## get dependencies working

get it working on ojo and oop

```bash
git clone https://github.com/hu-po/cax.git
cd cax
```

local test on oop works fine

```bash
conda create -n cax python=3.10
conda activate cax
pip install -e '.[dev]'
pytest
```

pick a container for oop

```bash
docker run --gpus all -it --rm \
nvcr.io/nvidia/jax:24.04-py3 \
bash -c "nvidia-smi && python3 -c 'import jax; print(jax.devices())'"
```

build a local docker container and run tests

```bash
docker build -f docker/Dockerfile.gpu -t cax .
docker run --gpus all -it --rm cax
```

or just run in raw

```bash
docker run --gpus all -it --rm \
-v ~/dev/cax:/cax \
nvcr.io/nvidia/jax:24.04-py3 \
bash -c /cax/docker/test.oop.sh
```

set up local environment on ojo
- https://github.com/dusty-nv/jetson-containers/tree/master/packages/ml/jax

```bash
jetson-containers run \
$(autotag jax) \
bash -c "python3 -c 'import jax; print(jax.devices())'"
```

```bash
jetson-containers run \
-v ~/dev/cax:/cax \
$(autotag jax) bash -c /cax/docker/test.ojo.sh
```

tests working on both!

## ideas while reading paper

can a NCA be used as an image encoder? compare a vit, cnn and nca

distill a vlm into an nca?

nca on full arcprize? https://arcprize.org/

