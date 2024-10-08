# 08.10.2024

get it working on ojo and oop

```bash
git clone https://github.com/hu-po/cax.git
cd cax
```

oop

```bash
conda create -n cax python=3.10
conda activate cax
pip install -e '.[dev]'
pytest
```

set up local environment on ojo
- https://github.com/dusty-nv/jetson-containers/tree/master/packages/ml/jax

```bash
jetson-containers run $(autotag jax) bash -c "python3 -c 'import jax; print(jax.devices())'"
```

```bash
jetson-containers run \
-v ~/dev/cax:/cax \
$(autotag jax) bash -c /cax/ojo.test.sh
```