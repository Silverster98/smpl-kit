
import torch
import trimesh

EPS = 1e-6

def test1():
    """ Test the difference of outputs between smplkit and smplx
    """
    import smplx
    from smplkit import SMPLLayer as SMPL

    SMPL_NJOINTS = 24

    bm1 = SMPL(num_betas=10)
    bm2 = smplx.create(
        './body_models',
        model_type='smpl',
        num_betas=10,
    )

    transl = torch.rand((2, 3), dtype=torch.float32)
    orient = torch.rand((2, 3), dtype=torch.float32)
    betas = torch.rand((2, 10), dtype=torch.float32)
    body_pose = torch.rand((2, 23 * 3), dtype=torch.float32)

    verts1 = bm1(betas=betas, transl=transl, orient=orient, body_pose=body_pose)
    verts2 = bm2(betas=betas, transl=transl, global_orient=orient, body_pose=body_pose)

    assert torch.abs(verts1.vertices - verts2.vertices).sum() < EPS, 'SMPLLayer and smplx.create produce different results'
    assert torch.abs(verts1.joints[:, :SMPL_NJOINTS, :] - verts2.joints[:, :SMPL_NJOINTS, :]).sum() < EPS, 'SMPLLayer and smplx.create produce different results'

    print('[T1] Pass!\n')

def test2():
    """ Test SMPLParam construction
    """
    from smplkit import SMPLParam

    a = SMPLParam(requires_grad=[True, False, True, False])

    assert a.transl.dtype == torch.float32, f'SMPLParam.transl.dtype is not torch.float32, but {a.transl.dtype}'
    assert str(a.transl.device) == 'cpu', f'SMPLParam.transl.device is not cpu, but {a.transl.device}'
    assert a.transl.shape == (1, 3), f'SMPLParam.transl.shape is not (1, 3), but {a.transl.shape}'

    assert a.transl.requires_grad == False, f'SMPLParam.transl.requires_grad is not False, but {a.transl.requires_grad}'
    assert a.orient.requires_grad == False, f'SMPLParam.orient.requires_grad is not False, but {a.orient.requires_grad}'
    assert a.betas.requires_grad == False, f'SMPLParam.betas.requires_grad is not False, but {a.betas.requires_grad}'
    assert a.body_pose.requires_grad == False, f'SMPLParam.body_pose.requires_grad is not False, but {a.body_pose.requires_grad}'

    assert a._transl.requires_grad == True, f'SMPLParam._transl.requires_grad is not True, but {a.transl.requires_grad}'
    assert a._orient.requires_grad == False, f'SMPLParam._orient.requires_grad is not False, but {a.orient.requires_grad}'
    assert a._betas.requires_grad == True, f'SMPLParam._betas.requires_grad is not True, but {a.betas.requires_grad}'
    assert a._body_pose.requires_grad == False, f'SMPLParam._body_pose.requires_grad is not False, but {a.body_pose.requires_grad}'

    print('[T2] Pass!\n')

def test3():
    """ Test the gradient of SMPLParam
    """
    from smplkit import SMPLParam

    ## init param and optimizer
    v = torch.rand(1, 3)
    param = SMPLParam(transl=v, requires_grad=[True, False, False, False])

    transl, orient, betas, body_pose = param.parameters()
    assert transl.shape==(1, 3), f'SMPLParam.transl.shape is not (1, 3), but {transl.shape}'
    assert orient.shape==(1, 3), f'SMPLParam.orient.shape is not (1, 3), but {orient.shape}'
    assert betas.shape==(1, 10), f'SMPLParam.betas.shape is not (1, 10), but {betas.shape}'
    assert body_pose.shape==(1, 69), f'SMPLParam.body_pose.shape is not (1, 69), but {body_pose.shape}'
    assert transl.requires_grad==False, f'SMPLParam.transl.requires_grad is not False, but {transl.requires_grad}'
    assert orient.requires_grad==False, f'SMPLParam.orient.requires_grad is not False, but {orient.requires_grad}'
    assert betas.requires_grad==False, f'SMPLParam.betas.requires_grad is not False, but {betas.requires_grad}'
    assert body_pose.requires_grad==False, f'SMPLParam.body_pose.requires_grad is not False, but {body_pose.requires_grad}'
    
    _transl, _orient, _betas, _body_pose = param._parameters()
    assert _transl.shape==(1, 3), f'SMPLParam._transl.shape is not (1, 3), but {_transl.shape}'
    assert _orient.shape==(1, 3), f'SMPLParam._orient.shape is not (1, 3), but {_orient.shape}'
    assert _betas.shape==(1, 10), f'SMPLParam._betas.shape is not (1, 10), but {_betas.shape}'
    assert _body_pose.shape==(1, 69), f'SMPLParam._body_pose.shape is not (1, 69), but {_body_pose.shape}'
    assert _transl.requires_grad==True, f'SMPLParam._transl.requires_grad is not True, but {_transl.requires_grad}'
    assert _orient.requires_grad==False, f'SMPLParam._orient.requires_grad is not False, but {_orient.requires_grad}'
    assert _betas.requires_grad==False, f'SMPLParam._betas.requires_grad is not False, but {_betas.requires_grad}'
    assert _body_pose.requires_grad==False, f'SMPLParam._body_pose.requires_grad is not False, but {_body_pose.requires_grad}'
    

    tp = param.trainable_parameters()
    assert len(tp) == 1, f'len(param.trainable_parameters()) is not 1, but {len(tp)}'
    _transl = tp[0]
    assert _transl.shape==(1, 3), f'SMPLParam._transl.shape is not (1, 3), but {_transl.shape}'
    assert _transl.requires_grad==True, f'SMPLParam._transl.requires_grad is not True, but {_transl.requires_grad}'


    for n, p in param.named_parameters():
        if n == 'transl':
            assert p.shape==(1, 3), f'SMPLParam.transl.shape is not (1, 3), but {p.shape}'
            assert p.requires_grad==False, f'SMPLParam.transl.requires_grad is not False, but {p.requires_grad}'
        if n == 'orient':
            assert p.shape==(1, 3), f'SMPLParam.orient.shape is not (1, 3), but {p.shape}'
            assert p.requires_grad==False, f'SMPLParam.orient.requires_grad is not False, but {p.requires_grad}'
        if n == 'betas':
            assert p.shape==(1, 10), f'SMPLParam.betas.shape is not (1, 10), but {p.shape}'
            assert p.requires_grad==False, f'SMPLParam.betas.requires_grad is not False, but {p.requires_grad}'
        if n == 'body_pose':
            assert p.shape==(1, 69), f'SMPLParam.body_pose.shape is not (1, 69), but {p.shape}'
            assert p.requires_grad==False, f'SMPLParam.body_pose.requires_grad is not False, but {p.requires_grad}'
    
    for n, p in param._named_parameters():
        if n == 'transl':
            assert p.shape==(1, 3), f'SMPLParam.transl.shape is not (1, 3), but {p.shape}'
            assert p.requires_grad==True, f'SMPLParam.transl.requires_grad is not True, but {p.requires_grad}'
        if n == 'orient':
            assert p.shape==(1, 3), f'SMPLParam.orient.shape is not (1, 3), but {p.shape}'
            assert p.requires_grad==False, f'SMPLParam.orient.requires_grad is not False, but {p.requires_grad}'
        if n == 'betas':
            assert p.shape==(1, 10), f'SMPLParam.betas.shape is not (1, 10), but {p.shape}'
            assert p.requires_grad==False, f'SMPLParam.betas.requires_grad is not False, but {p.requires_grad}'
        if n == 'body_pose':
            assert p.shape==(1, 69), f'SMPLParam.body_pose.shape is not (1, 69), but {p.shape}'
            assert p.requires_grad==False, f'SMPLParam.body_pose.requires_grad is not False, but {p.requires_grad}'
    
    for n, p in param.trainable_named_parameters():
        assert n == 'transl', f'SMPLParam.named_trainable_parameters() is not transl, but {n}'
        if n == 'transl':
            assert p.shape==(1, 3), f'SMPLParam.transl.shape is not (1, 3), but {p.shape}'
            assert p.requires_grad==True, f'SMPLParam.transl.requires_grad is not True, but {p.requires_grad}'
    

    for n, p in param.parameters_dict().items():
        if n == 'transl':
            assert p.shape==(1, 3), f'SMPLParam.transl.shape is not (1, 3), but {p.shape}'
            assert p.requires_grad==False, f'SMPLParam.transl.requires_grad is not False, but {p.requires_grad}'
        if n == 'orient':
            assert p.shape==(1, 3), f'SMPLParam.orient.shape is not (1, 3), but {p.shape}'
            assert p.requires_grad==False, f'SMPLParam.orient.requires_grad is not False, but {p.requires_grad}'
        if n == 'betas':
            assert p.shape==(1, 10), f'SMPLParam.betas.shape is not (1, 10), but {p.shape}'
            assert p.requires_grad==False, f'SMPLParam.betas.requires_grad is not False, but {p.requires_grad}'
        if n == 'body_pose':
            assert p.shape==(1, 69), f'SMPLParam.body_pose.shape is not (1, 69), but {p.shape}'
            assert p.requires_grad==False, f'SMPLParam.body_pose.requires_grad is not False, but {p.requires_grad}'
    
    for n, p in param._parameters_dict().items():
        if n == 'transl':
            assert p.shape==(1, 3), f'SMPLParam.transl.shape is not (1, 3), but {p.shape}'
            assert p.requires_grad==True, f'SMPLParam.transl.requires_grad is not True, but {p.requires_grad}'
        if n == 'orient':
            assert p.shape==(1, 3), f'SMPLParam.orient.shape is not (1, 3), but {p.shape}'
            assert p.requires_grad==False, f'SMPLParam.orient.requires_grad is not False, but {p.requires_grad}'
        if n == 'betas':
            assert p.shape==(1, 10), f'SMPLParam.betas.shape is not (1, 10), but {p.shape}'
            assert p.requires_grad==False, f'SMPLParam.betas.requires_grad is not False, but {p.requires_grad}'
        if n == 'body_pose':
            assert p.shape==(1, 69), f'SMPLParam.body_pose.shape is not (1, 69), but {p.shape}'
            assert p.requires_grad==False, f'SMPLParam.body_pose.requires_grad is not False, but {p.requires_grad}'
    
    for n, p in param.trainable_parameters_dict().items():
        assert n == 'transl', f'SMPLParam.named_trainable_parameters() is not transl, but {n}'
        if n == 'transl':
            assert p.shape==(1, 3), f'SMPLParam.transl.shape is not (1, 3), but {p.shape}'
            assert p.requires_grad==True, f'SMPLParam.transl.requires_grad is not True, but {p.requires_grad}'

    from smplkit import SMPLLayer as SMPL
    bm = SMPL(num_betas=10)
    verts1 = bm(**param.parameters_dict(), return_verts=True)
    verts2 = bm(transl=v, return_verts=True)
    assert torch.abs(verts1 - verts2).sum() < EPS, f"SMPLLayer(**param.parameters_dict()) != SMPLLayer(transl=v), but {torch.abs(verts1 - verts2).sum()}"

    print('[T3] Pass!\n')

def test4(visualize=False):
    """ Test optimization of SMPLParam """
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

        if (i + 1) % 100 == 0:
            print(f"Optimization Error in Step {i + 1:3d}: {loss.item()}")

    transl, orient, betas, body_pose = param.parameters()
    assert torch.abs(orient).sum() == 0, f"orient.sum() != 0, but {orient.sum()}"
    assert torch.abs(betas).sum() == 0, f"betas.sum() != 0, but {betas.sum()}"
    assert torch.abs(body_pose).sum() == 0, f"body_pose.sum() != 0, but {body_pose.sum()}"

    assert torch.abs(transl - target_transl).mean() < EPS, f"transl != target_transl, but {torch.abs(transl - target_transl).sum()}"

    verts = bm(transl=transl, return_verts=True)
    assert torch.abs(verts - target_verts).mean() < EPS, f"SMPLParam(transl=transl) != target_verts, but {torch.abs(verts - target_verts).sum()}"

    print('[T4] Pass!\n')

    ## visualize
    if visualize:
        S = trimesh.Scene()
        S.add_geometry(trimesh.creation.axis())

        target_mesh = trimesh.Trimesh(vertices=target_verts[0].numpy(), faces=bm.faces)
        target_mesh.visual.vertex_colors = [255, 0, 0, 255]
        S.add_geometry(target_mesh)

        mesh = trimesh.Trimesh(vertices=verts[0].numpy(), faces=bm.faces)
        mesh.visual.vertex_colors = [0, 255, 0, 255]
        S.add_geometry(mesh)

        S.show()
