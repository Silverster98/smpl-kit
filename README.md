# smpl-kit
> Use SMPL more easily! This project aims to provide a simple and easy-to-use interface for SMPL-related human body models, e.g., SMPL, MANO, and SMPL-X. It also provides useful tools, e.g., SMPL parameter transformation, SMPL body mesh visualization, etc.

## Installation

1. Before installing `smplkit`, please make sure that you have installed [pytorch](https://pytorch.org/).


2. To install the `smplkit` package, you can either install it from PyPI or install it from source.

   - To install from PyPI by using `pip`:

   ```bash
   pip install smplkit
   ```

   - Or, clone this repository and install it from source:

   ```bash
   git clone git@github.com:Silverster98/smpl-kit.git
   cd smpl-kit
   pip install .
   ```

## Documentation

### Tutorial

#### 1. Use the `SMPLLayer` to generate a SMPL body mesh with random translation while keeping other parameters as zero:

```python
import torch
import trimesh
from smplkit import SMPLLayer as SMPL

NJOINTS = 23

body_model = SMPL(num_betas=10)

transl = torch.rand((1, 3), dtype=torch.float32)

verts = body_model(transl=transl, return_verts=True)
faces = body_model.faces
verts = verts.cpu().numpy()

mesh = trimesh.Trimesh(vertices=verts[0], faces=body_model.faces)
```

#### 2. Use the `SMPLParam` to optimize the SMPL parameters (**only tranlation**) with a given mesh:

```python
import torch
from smplkit import SMPLLayer as SMPL
from smplkit import SMPLParam
from torch.optim import SGD

bm = SMPL(num_betas=10)

target_transl = torch.tensor([[0, 1, 0]], dtype=torch.float32)
target_verts = bm(transl=target_transl, return_verts=True)

## init param and optimizer
param = SMPLParam(transl=torch.rand(1, 3), requires_grad=[True, False, False, False])
opt = SGD(param.trainable_parameters(), lr=0.1)

for i in range(200):
    opt.zero_grad()
    output = bm(**param._parameters_dict(), return_verts=True)
    
    loss = ((output - target_verts) ** 2).mean()
    
    loss.backward()
    opt.step()

    if (i + 1) % 20 == 0:
        print(f"Optimization Error in Step {i + 1:3d}: {loss.item()}")
```

## License

This project is licensed under the terms of the [MIT](LICENSE) license.

## Acknowledgement

Some codes are borrowed from [SMPL-X](https://github.com/vchoutas/smplx). If your use this code, please consider citing the most relevant [works](https://github.com/vchoutas/smplx#citation).
