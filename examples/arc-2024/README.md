# 08.10.2024

## oop

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

library tests in container on oop

```bash
docker run --gpus all -it --rm \
-v ~/dev/cax:/cax \
nvcr.io/nvidia/jax:24.04-py3 \
bash -c /cax/examples/arc-2024/test.lib.oop.sh
```

run arc 1d test for oop

```bash
docker run --gpus all -it --rm \
-v ~/dev/cax:/cax \
-p 8888:8888 \
nvcr.io/nvidia/jax:24.04-py3 \
bash -c /cax/examples/arc-2024/test.1d-arc.oop.sh
```

## ojo

- https://github.com/dusty-nv/jetson-containers/tree/master/packages/ml/jax

```bash
jetson-containers run \
$(autotag jax) \
bash -c "python3 -c 'import jax; print(jax.devices())'"
```

```bash
jetson-containers run \
-v ~/dev/cax:/cax \
$(autotag jax) \
bash -c /cax/examples/arc-2024/test.lib.ojo.sh
```

run 1d-arc notebook from inside container

```bash
jetson-containers run \
-v ~/dev/cax:/cax \
-p 8888:8888 \
$(autotag jax) \
bash -c "/cax/examples/arc-2024/test.1d-arc.ojo.sh"
```

run arc-2024

```bash
jetson-containers run \
-v ~/dev/cax:/cax \
-p 8888:8888 \
-e WANDB_API_KEY=$WANDB_API_KEY \
$(autotag jax) \
bash -c "/cax/examples/arc-2024/train.arc-2024.ojo.sh"
```

## ideas while reading paper

can a NCA be used as an image encoder? compare a vit, cnn and nca

distill a vlm into an nca?

nca on full arcprize? https://arcprize.org/

