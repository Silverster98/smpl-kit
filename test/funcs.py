
import torch
import trimesh

EPS = 1e-6

def test1_1():
    """ Test the difference of outputs between smplkit and smplx (SMPL)
    """
    import smplx
    from smplkit import SMPLLayer as SMPL

    SMPL_NJOINTS = 23 + 1

    bm1 = SMPL(num_betas=10)
    bm2 = smplx.create(
        './body_models',
        model_type='smpl',
        num_betas=10,
        batch_size=2,
    )

    transl = torch.rand((2, 3), dtype=torch.float32)
    orient = torch.rand((2, 3), dtype=torch.float32)
    betas = torch.rand((2, 10), dtype=torch.float32)
    body_pose = torch.rand((2, 23 * 3), dtype=torch.float32)

    out1 = bm1(betas=betas, transl=transl, orient=orient, body_pose=body_pose)
    out2 = bm2(betas=betas, transl=transl, global_orient=orient, body_pose=body_pose)

    assert torch.abs(out1.vertices - out2.vertices).sum() < EPS, 'SMPLLayer and smplx.create produce different results'
    assert torch.abs(out1.joints[:, :SMPL_NJOINTS, :] - out2.joints[:, :SMPL_NJOINTS, :]).sum() < EPS, 'SMPLLayer and smplx.create produce different results'


    out1 = bm1(transl=transl, body_pose=body_pose, apply_trans=False)
    out2 = bm2(transl=torch.zeros((2, 3), dtype=torch.float32), body_pose=body_pose)

    assert torch.abs(out1.vertices - out2.vertices).sum() < EPS, 'SMPLLayer and smplx.create produce different results'
    assert torch.abs(out1.joints[:, :SMPL_NJOINTS, :] - out2.joints[:, :SMPL_NJOINTS, :]).sum() < EPS, 'SMPLLayer and smplx.create produce different results'


    out2 = bm2(transl=transl, body_pose=body_pose)

    bm1 = SMPL('./body_models/smpl/SMPL_NEUTRAL.pkl', num_betas=10)
    verts1 = bm1(transl=transl, body_pose=body_pose, return_verts=True)
    assert torch.abs(verts1 - out2.vertices).sum() < EPS, 'SMPLLayer and smplx.create produce different results'


    joints1 = bm1(transl=transl, body_pose=body_pose, return_joints=True)
    assert torch.abs(joints1 - out2.joints[:, :SMPL_NJOINTS, :]).sum() < EPS, 'SMPLLayer and smplx.create produce different results'
    
    print('[T1-1] Pass!\n')

def test1_2():
    """ Test the difference of outputs between smplkit and smplx (SMPL+H)
    """
    import smplx
    from smplkit import SMPLHLayer as SMPLH
    ALL_NUM_NJOINTS = 51 + 1
    NUM_BODY_JOINTS = 21


    bm1 = SMPLH(num_betas=10)
    bm2 = smplx.create(
        './body_models/smplh/SMPLH_NEUTRAL.pkl',
        model_type='smplh',
        num_betas=10,
        ext='pkl',
        batch_size=2,
    )

    transl = torch.rand((2, 3), dtype=torch.float32)
    orient = torch.rand((2, 3), dtype=torch.float32)
    betas = torch.rand((2, 10), dtype=torch.float32)
    body_pose = torch.rand((2, NUM_BODY_JOINTS * 3), dtype=torch.float32)
    left_hand_pose = torch.rand((2, 6), dtype=torch.float32)
    right_hand_pose = torch.rand((2, 6), dtype=torch.float32)

    out1 = bm1(
        betas=betas,
        transl=transl,
        orient=orient,
        body_pose=body_pose,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose
    )

    out2 = bm2(
        betas=betas,
        transl=transl,
        global_orient=orient,
        body_pose=body_pose,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose
    )
    assert torch.abs(out1.vertices - out2.vertices).sum() < EPS, 'SMPLHLayer and smplx.create produce different results'
    assert torch.abs(out1.joints[:, :ALL_NUM_NJOINTS, :] - out2.joints[:, :ALL_NUM_NJOINTS, :]).sum() < EPS, 'SMPLHLayer and smplx.create produce different results'


    out1 = bm1(transl=transl, betas=betas, right_hand_pose=right_hand_pose, apply_trans=False)
    out2 = bm2(transl=torch.zeros((2, 3), dtype=torch.float32), betas=betas, right_hand_pose=right_hand_pose)
    
    assert torch.abs(out1.vertices - out2.vertices).sum() < EPS, 'SMPLHLayer and smplx.create produce different results'
    assert torch.abs(out1.joints[:, :ALL_NUM_NJOINTS, :] - out2.joints[:, :ALL_NUM_NJOINTS, :]).sum() < EPS, 'SMPLHLayer and smplx.create produce different results'

    out2 = bm2(transl=transl, betas=betas, right_hand_pose=right_hand_pose)

    bm1 = SMPLH('./body_models/smplh/SMPLH_NEUTRAL.pkl', num_betas=10)
    verts1 = bm1(transl=transl, betas=betas, right_hand_pose=right_hand_pose, return_verts=True)
    assert torch.abs(verts1 - out2.vertices).sum() < EPS, 'SMPLHLayer and smplx.create produce different results'

    joints1 = bm1(transl=transl, betas=betas, right_hand_pose=right_hand_pose, return_joints=True)
    assert torch.abs(joints1 - out2.joints[:, :ALL_NUM_NJOINTS, :]).sum() < EPS, 'SMPLHLayer and smplx.create produce different results'

    print('[T1-2] Pass!\n')

def test1_3():
    """ Test the difference of outputs between smplkit and smplx (SMPL-X)
    """
    import smplx
    from smplkit import SMPLXLayer as SMPLX
    ALL_NUM_JOINTS = 21 + 15 * 2 + 3 + 1
    NUM_BODY_JOINTS = 21

    bm1 = SMPLX(num_betas=10, num_pca_comps=12)
    bm2 = smplx.create(
        './body_models',
        model_type='smplx',
        num_betas=10,
        num_pca_comps=12,
        ext='npz',
        batch_size=2,
    )

    transl = torch.rand((2, 3), dtype=torch.float32)
    orient = torch.rand((2, 3), dtype=torch.float32)
    betas = torch.rand((2, 10), dtype=torch.float32)
    body_pose = torch.rand((2, NUM_BODY_JOINTS * 3), dtype=torch.float32)
    left_hand_pose = torch.rand((2, 12), dtype=torch.float32)
    right_hand_pose = torch.rand((2, 12), dtype=torch.float32)
    expression = torch.rand((2, 10), dtype=torch.float32)
    jaw_pose = torch.rand((2, 3), dtype=torch.float32)
    leye_pose = torch.rand((2, 3), dtype=torch.float32)
    reye_pose = torch.rand((2, 3), dtype=torch.float32)


    out1 = bm1(
        betas=betas,
        transl=transl,
        orient=orient,
        body_pose=body_pose,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        expression=expression,
        jaw_pose=jaw_pose,
        leye_pose=leye_pose,
        reye_pose=reye_pose
    )

    out2 = bm2(
        betas=betas,
        transl=transl,
        global_orient=orient,
        body_pose=body_pose,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        expression=expression,
        jaw_pose=jaw_pose,
        leye_pose=leye_pose,
        reye_pose=reye_pose
    )

    assert torch.abs(out1.vertices - out2.vertices).sum() < EPS, 'SMPLXLayer and smplx.create produce different results'
    assert torch.abs(out1.joints[:, :ALL_NUM_JOINTS, :] - out2.joints[:, :ALL_NUM_JOINTS, :]).sum() < EPS, 'SMPLXLayer and smplx.create produce different results'
    ## 55 joints, 21 extra body vertices, and 51 landmarks
    assert torch.abs(out1.landmarks - out2.joints[:, 55+21:55+21+51, :]).sum() < EPS, 'SMPLXLayer and smplx.create produce different results'


    out1 = bm1(
        transl=transl,
        betas=betas,
        body_pose=body_pose,
        right_hand_pose=right_hand_pose,
        expression=expression,
        apply_trans=False,
    )

    out2 = bm2(
        betas=betas,
        body_pose=body_pose,
        right_hand_pose=right_hand_pose,
        expression=expression,
    )
    
    assert torch.abs(out1.vertices - out2.vertices).sum() < EPS, 'SMPLXLayer and smplx.create produce different results'
    assert torch.abs(out1.joints[:, :ALL_NUM_JOINTS, :] - out2.joints[:, :ALL_NUM_JOINTS, :]).sum() < EPS, 'SMPLXLayer and smplx.create produce different results'
    ## 55 joints, 21 extra body vertices, and 51 landmarks
    assert torch.abs(out1.landmarks - out2.joints[:, 55+21:55+21+51, :]).sum() < EPS, 'SMPLXLayer and smplx.create produce different results'

    out2 = bm2(
        transl=transl,
        betas=betas,
        body_pose=body_pose,
        right_hand_pose=right_hand_pose,
        expression=expression,
    )

    bm1 = SMPLX('./body_models/smplx/SMPLX_NEUTRAL.npz', num_betas=10, num_pca_comps=12)
    verts1 = bm1(
        transl=transl,
        betas=betas,
        body_pose=body_pose,
        right_hand_pose=right_hand_pose,
        expression=expression,
        return_verts=True
    )
    assert torch.abs(verts1 - out2.vertices).sum() < EPS, 'SMPLXLayer and smplx.create produce different results'

    joints1 = bm1(
        transl=transl,
        betas=betas,
        body_pose=body_pose,
        right_hand_pose=right_hand_pose,
        expression=expression,
        return_joints=True
    )
    assert torch.abs(joints1 - out2.joints[:, :ALL_NUM_JOINTS, :]).sum() < EPS, 'SMPLXLayer and smplx.create produce different results'


    bm1 = SMPLX(flat_hand_mean=True, num_expression_coeffs=12, use_face_contour=True)
    bm2 = smplx.create(
        './body_models',
        model_type='smplx',
        ext='npz',
        flat_hand_mean=True,
        num_expression_coeffs=12,
        use_face_contour=True,
        batch_size=2,
    )

    transl = torch.rand((2, 3), dtype=torch.float32)
    orient = torch.rand((2, 3), dtype=torch.float32)
    betas = torch.rand((2, 10), dtype=torch.float32)
    body_pose = torch.rand((2, NUM_BODY_JOINTS * 3), dtype=torch.float32)
    left_hand_pose = torch.rand((2, 6), dtype=torch.float32)
    right_hand_pose = torch.rand((2, 6), dtype=torch.float32)
    expression = torch.rand((2, 12), dtype=torch.float32)
    jaw_pose = torch.rand((2, 3), dtype=torch.float32)
    leye_pose = torch.rand((2, 3), dtype=torch.float32)
    reye_pose = torch.rand((2, 3), dtype=torch.float32)

    out1 = bm1(
        betas=betas,
        transl=transl,
        orient=orient,
        body_pose=body_pose,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        expression=expression,
        jaw_pose=jaw_pose,
        leye_pose=leye_pose,
        reye_pose=reye_pose
    )

    out2 = bm2(
        betas=betas,
        transl=transl,
        global_orient=orient,
        body_pose=body_pose,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        expression=expression,
        jaw_pose=jaw_pose,
        leye_pose=leye_pose,
        reye_pose=reye_pose
    )

    assert torch.abs(out1.vertices - out2.vertices).sum() < EPS, 'SMPLXLayer and smplx.create produce different results'
    assert torch.abs(out1.joints[:, :ALL_NUM_JOINTS, :] - out2.joints[:, :ALL_NUM_JOINTS, :]).sum() < EPS, 'SMPLXLayer and smplx.create produce different results'
    ## 55 joints, 21 extra body vertices, and 51 + 17 landmarks
    assert torch.abs(out1.landmarks - out2.joints[:, 55+21:55+21+51+17, :]).sum() < EPS, 'SMPLXLayer and smplx.create produce different results'

    print('[T1-3] Pass!\n')

def test2_1():
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

    print('[T2-1] Pass!\n')

def test2_2():
    """ Test SMPLHParam construction
    """
    from smplkit import SMPLHParam

    a = SMPLHParam(requires_grad=[False, True, False, True, False, True], dtype=torch.float64)

    assert a.transl.dtype == torch.float64, f'SMPLHParam.transl.dtype is not torch.float64, but {a.transl.dtype}'
    assert str(a.transl.device) == 'cpu', f'SMPLHParam.transl.device is not cpu, but {a.transl.device}'
    assert a.transl.shape == (1, 3), f'SMPLHParam.transl.shape is not (1, 3), but {a.transl.shape}'

    assert a.transl.requires_grad == False, f'SMPLHParam.transl.requires_grad is not False, but {a.transl.requires_grad}'
    assert a.orient.requires_grad == False, f'SMPLHParam.orient.requires_grad is not False, but {a.orient.requires_grad}'
    assert a.betas.requires_grad == False, f'SMPLHParam.betas.requires_grad is not False, but {a.betas.requires_grad}'
    assert a.body_pose.requires_grad == False, f'SMPLHParam.body_pose.requires_grad is not False, but {a.body_pose.requires_grad}'
    assert a.left_hand_pose.requires_grad == False, f'SMPLHParam.left_hand_pose.requires_grad is not False, but {a.left_hand_pose.requires_grad}'
    assert a.right_hand_pose.requires_grad == False, f'SMPLHParam.right_hand_pose.requires_grad is not False, but {a.right_hand_pose.requires_grad}'

    assert a._transl.requires_grad == False, f'SMPLParam._transl.requires_grad is not False, but {a.transl.requires_grad}'
    assert a._orient.requires_grad == True, f'SMPLParam._orient.requires_grad is not True, but {a.orient.requires_grad}'
    assert a._betas.requires_grad == False, f'SMPLParam._betas.requires_grad is not False, but {a.betas.requires_grad}'
    assert a._body_pose.requires_grad == True, f'SMPLParam._body_pose.requires_grad is not True, but {a.body_pose.requires_grad}'
    assert a._left_hand_pose.requires_grad == False, f'SMPLParam._left_hand_pose.requires_grad is not False, but {a.left_hand_pose.requires_grad}'
    assert a._right_hand_pose.requires_grad == True, f'SMPLParam._right_hand_pose.requires_grad is not True, but {a.right_hand_pose.requires_grad}'

    print('[T2-2] Pass!\n')

def test2_3():
    """ Test SMPLXParam construction
    """
    from smplkit import SMPLXParam

    a = SMPLXParam(requires_grad=True)

    assert a.transl.dtype == torch.float32, f'SMPLXParam.transl.dtype is not torch.float32, but {a.transl.dtype}'
    assert str(a.transl.device) == 'cpu', f'SMPLXParam.transl.device is not cpu, but {a.transl.device}'
    assert a.transl.shape == (1, 3), f'SMPLXParam.transl.shape is not (1, 3), but {a.transl.shape}'

    assert a.transl.requires_grad == False, f'SMPLXParam.transl.requires_grad is not False, but {a.transl.requires_grad}'
    assert a.orient.requires_grad == False, f'SMPLXParam.orient.requires_grad is not False, but {a.orient.requires_grad}'
    assert a.betas.requires_grad == False, f'SMPLXParam.betas.requires_grad is not False, but {a.betas.requires_grad}'
    assert a.body_pose.requires_grad == False, f'SMPLXParam.body_pose.requires_grad is not False, but {a.body_pose.requires_grad}'
    assert a.left_hand_pose.requires_grad == False, f'SMPLXParam.left_hand_pose.requires_grad is not False, but {a.left_hand_pose.requires_grad}'
    assert a.right_hand_pose.requires_grad == False, f'SMPLXParam.right_hand_pose.requires_grad is not False, but {a.right_hand_pose.requires_grad}'
    assert a.expression.requires_grad == False, f'SMPLXParam.expression.requires_grad is not False, but {a.expression.requires_grad}'
    assert a.jaw_pose.requires_grad == False, f'SMPLXParam.jaw_pose.requires_grad is not False, but {a.jaw_pose.requires_grad}'
    assert a.leye_pose.requires_grad == False, f'SMPLXParam.leye_pose.requires_grad is not False, but {a.leye_pose.requires_grad}'
    assert a.reye_pose.requires_grad == False, f'SMPLXParam.reye_pose.requires_grad is not False, but {a.reye_pose.requires_grad}'

    assert a._transl.requires_grad == True, f'SMPLXParam._transl.requires_grad is not True, but {a.transl.requires_grad}'
    assert a._orient.requires_grad == True, f'SMPLXParam._orient.requires_grad is not True, but {a.orient.requires_grad}'
    assert a._betas.requires_grad == True, f'SMPLXParam._betas.requires_grad is not True, but {a.betas.requires_grad}'
    assert a._body_pose.requires_grad == True, f'SMPLXParam._body_pose.requires_grad is not True, but {a.body_pose.requires_grad}'
    assert a._left_hand_pose.requires_grad == True, f'SMPLXParam._left_hand_pose.requires_grad is not True, but {a.left_hand_pose.requires_grad}'
    assert a._right_hand_pose.requires_grad == True, f'SMPLXParam._right_hand_pose.requires_grad is not True, but {a.right_hand_pose.requires_grad}'
    assert a._expression.requires_grad == True, f'SMPLXParam._expression.requires_grad is not True, but {a.expression.requires_grad}'
    assert a._jaw_pose.requires_grad == True, f'SMPLXParam._jaw_pose.requires_grad is not True, but {a.jaw_pose.requires_grad}'
    assert a._leye_pose.requires_grad == True, f'SMPLXParam._leye_pose.requires_grad is not True, but {a.leye_pose.requires_grad}'
    assert a._reye_pose.requires_grad == True, f'SMPLXParam._reye_pose.requires_grad is not True, but {a.reye_pose.requires_grad}'


    a = SMPLXParam(requires_grad=False)

    assert a._transl.requires_grad == False, f'SMPLXParam._transl.requires_grad is not False, but {a.transl.requires_grad}'
    assert a._orient.requires_grad == False, f'SMPLXParam._orient.requires_grad is not False, but {a.orient.requires_grad}'
    assert a._betas.requires_grad == False, f'SMPLXParam._betas.requires_grad is not False, but {a.betas.requires_grad}'
    assert a._body_pose.requires_grad == False, f'SMPLXParam._body_pose.requires_grad is not False, but {a.body_pose.requires_grad}'
    assert a._left_hand_pose.requires_grad == False, f'SMPLXParam._left_hand_pose.requires_grad is not False, but {a.left_hand_pose.requires_grad}'
    assert a._right_hand_pose.requires_grad == False, f'SMPLXParam._right_hand_pose.requires_grad is not False, but {a.right_hand_pose.requires_grad}'
    assert a._expression.requires_grad == False, f'SMPLXParam._expression.requires_grad is not False, but {a.expression.requires_grad}'
    assert a._jaw_pose.requires_grad == False, f'SMPLXParam._jaw_pose.requires_grad is not False, but {a.jaw_pose.requires_grad}'
    assert a._leye_pose.requires_grad == False, f'SMPLXParam._leye_pose.requires_grad is not False, but {a.leye_pose.requires_grad}'
    assert a._reye_pose.requires_grad == False, f'SMPLXParam._reye_pose.requires_grad is not False, but {a.reye_pose.requires_grad}'


    a = SMPLXParam(requires_grad=[False, False, False, False, False, True, True, True, True, True])

    assert a._transl.requires_grad == False, f'SMPLXParam._transl.requires_grad is not False, but {a.transl.requires_grad}'
    assert a._orient.requires_grad == False, f'SMPLXParam._orient.requires_grad is not False, but {a.orient.requires_grad}'
    assert a._betas.requires_grad == False, f'SMPLXParam._betas.requires_grad is not False, but {a.betas.requires_grad}'
    assert a._body_pose.requires_grad == False, f'SMPLXParam._body_pose.requires_grad is not False, but {a.body_pose.requires_grad}'
    assert a._left_hand_pose.requires_grad == False, f'SMPLXParam._left_hand_pose.requires_grad is not False, but {a.left_hand_pose.requires_grad}'
    assert a._right_hand_pose.requires_grad == True, f'SMPLXParam._right_hand_pose.requires_grad is not True, but {a.right_hand_pose.requires_grad}'
    assert a._expression.requires_grad == True, f'SMPLXParam._expression.requires_grad is not True, but {a.expression.requires_grad}'
    assert a._jaw_pose.requires_grad == True, f'SMPLXParam._jaw_pose.requires_grad is not True, but {a.jaw_pose.requires_grad}'
    assert a._leye_pose.requires_grad == True, f'SMPLXParam._leye_pose.requires_grad is not True, but {a.leye_pose.requires_grad}'
    assert a._reye_pose.requires_grad == True, f'SMPLXParam._reye_pose.requires_grad is not True, but {a.reye_pose.requires_grad}'

    print('[T2-3] Pass!\n')

def test3_1():
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

    print('[T3-1] Pass!\n')

def test3_2():
    """ Test the gradient of SMPLHParam
    """
    from smplkit import SMPLHParam

    ## init param and optimizer
    v = torch.rand(1, 3)
    param = SMPLHParam(transl=v, requires_grad=[True, False, False, False, False, True])

    transl, orient, betas, body_pose, left_hand_pose, right_hand_pose = param.parameters()
    assert transl.shape==(1, 3), f'SMPLHParam.transl.shape is not (1, 3), but {transl.shape}'
    assert orient.shape==(1, 3), f'SMPLHParam.orient.shape is not (1, 3), but {orient.shape}'
    assert betas.shape==(1, 10), f'SMPLHParam.betas.shape is not (1, 10), but {betas.shape}'
    assert body_pose.shape==(1, 63), f'SMPLHParam.body_pose.shape is not (1, 63), but {body_pose.shape}'
    assert left_hand_pose.shape==(1, 6), f'SMPLHParam.left_hand_pose.shape is not (1, 6), but {left_hand_pose.shape}'
    assert right_hand_pose.shape==(1, 6), f'SMPLHParam.right_hand_pose.shape is not (1, 6), but {right_hand_pose.shape}'
    assert transl.requires_grad==False, f'SMPLHParam.transl.requires_grad is not False, but {transl.requires_grad}'
    assert orient.requires_grad==False, f'SMPLHParam.orient.requires_grad is not False, but {orient.requires_grad}'
    assert betas.requires_grad==False, f'SMPLHParam.betas.requires_grad is not False, but {betas.requires_grad}'
    assert body_pose.requires_grad==False, f'SMPLHParam.body_pose.requires_grad is not False, but {body_pose.requires_grad}'
    assert left_hand_pose.requires_grad==False, f'SMPLHParam.left_hand_pose.requires_grad is not False, but {left_hand_pose.requires_grad}'
    assert right_hand_pose.requires_grad==False, f'SMPLHParam.right_hand_pose.requires_grad is not False, but {right_hand_pose.requires_grad}'
    
    _transl, _orient, _betas, _body_pose, _left_hand_pose, _right_hand_pose = param._parameters()
    assert _transl.requires_grad==True, f'SMPLHParam._transl.requires_grad is not True, but {_transl.requires_grad}'
    assert _orient.requires_grad==False, f'SMPLHParam._orient.requires_grad is not False, but {_orient.requires_grad}'
    assert _betas.requires_grad==False, f'SMPLHParam._betas.requires_grad is not False, but {_betas.requires_grad}'
    assert _body_pose.requires_grad==False, f'SMPLHParam._body_pose.requires_grad is not False, but {_body_pose.requires_grad}'
    assert _left_hand_pose.requires_grad==False, f'SMPLHParam._left_hand_pose.requires_grad is not False, but {_left_hand_pose.requires_grad}'
    assert _right_hand_pose.requires_grad==True, f'SMPLHParam._right_hand_pose.requires_grad is not True, but {_right_hand_pose.requires_grad}'
    

    tp = param.trainable_parameters()
    assert len(tp) == 2, f'len(param.trainable_parameters()) is not 2, but {len(tp)}'
    _transl, _right_hand_pose = tp
    assert _transl.shape==(1, 3), f'SMPLHParam._transl.shape is not (1, 3), but {_transl.shape}'
    assert _transl.requires_grad==True, f'SMPLHParam._transl.requires_grad is not True, but {_transl.requires_grad}'
    assert _right_hand_pose.shape==(1, 6), f'SMPLHParam._right_hand_pose.shape is not (1, 6), but {_right_hand_pose.shape}'
    assert _right_hand_pose.requires_grad==True, f'SMPLHParam._right_hand_pose.requires_grad is not True, but {_right_hand_pose.requires_grad}'

    for n, p in param.parameters_dict().items():
        if n == 'transl':
            assert p.shape==(1, 3), f'SMPLHParam.transl.shape is not (1, 3), but {p.shape}'
            assert p.requires_grad==False, f'SMPLHParam.transl.requires_grad is not False, but {p.requires_grad}'
        if n == 'orient':
            assert p.shape==(1, 3), f'SMPLHParam.orient.shape is not (1, 3), but {p.shape}'
            assert p.requires_grad==False, f'SMPLHParam.orient.requires_grad is not False, but {p.requires_grad}'
        if n == 'betas':
            assert p.shape==(1, 10), f'SMPLHParam.betas.shape is not (1, 10), but {p.shape}'
            assert p.requires_grad==False, f'SMPLHParam.betas.requires_grad is not False, but {p.requires_grad}'
        if n == 'body_pose':
            assert p.shape==(1, 63), f'SMPLHParam.body_pose.shape is not (1, 63), but {p.shape}'
            assert p.requires_grad==False, f'SMPLHParam.body_pose.requires_grad is not False, but {p.requires_grad}'
        if n == 'left_hand_pose':
            assert p.shape==(1, 6), f'SMPLHParam.left_hand_pose.shape is not (1, 6), but {p.shape}'
            assert p.requires_grad==False, f'SMPLHParam.left_hand_pose.requires_grad is not False, but {p.requires_grad}'
        if n == 'right_hand_pose':
            assert p.shape==(1, 6), f'SMPLHParam.right_hand_pose.shape is not (1, 6), but {p.shape}'
            assert p.requires_grad==False, f'SMPLHParam.right_hand_pose.requires_grad is not False, but {p.requires_grad}'
    
    for n, p in param._parameters_dict().items():
        if n == 'transl':
            assert p.shape==(1, 3), f'SMPLHParam.transl.shape is not (1, 3), but {p.shape}'
            assert p.requires_grad==True, f'SMPLHParam.transl.requires_grad is not True, but {p.requires_grad}'
        if n == 'orient':
            assert p.shape==(1, 3), f'SMPLHParam.orient.shape is not (1, 3), but {p.shape}'
            assert p.requires_grad==False, f'SMPLHParam.orient.requires_grad is not False, but {p.requires_grad}'
        if n == 'betas':
            assert p.shape==(1, 10), f'SMPLHParam.betas.shape is not (1, 10), but {p.shape}'
            assert p.requires_grad==False, f'SMPLHParam.betas.requires_grad is not False, but {p.requires_grad}'
        if n == 'body_pose':
            assert p.shape==(1, 63), f'SMPLHParam.body_pose.shape is not (1, 63), but {p.shape}'
            assert p.requires_grad==False, f'SMPLHParam.body_pose.requires_grad is not False, but {p.requires_grad}'
        if n == 'left_hand_pose':
            assert p.shape==(1, 6), f'SMPLHParam.left_hand_pose.shape is not (1, 6), but {p.shape}'
            assert p.requires_grad==False, f'SMPLHParam.left_hand_pose.requires_grad is not False, but {p.requires_grad}'
        if n == 'right_hand_pose':
            assert p.shape==(1, 6), f'SMPLHParam.right_hand_pose.shape is not (1, 6), but {p.shape}'
            assert p.requires_grad==True, f'SMPLHParam.right_hand_pose.requires_grad is not True, but {p.requires_grad}'
    
    for n, p in param.trainable_parameters_dict().items():
        assert n == 'transl' or n == 'right_hand_pose', f'SMPLHParam.named_trainable_parameters() is not transl or right_hand_pose, but {n}'
        if n == 'transl':
            assert p.shape==(1, 3), f'SMPLHParam.transl.shape is not (1, 3), but {p.shape}'
            assert p.requires_grad==True, f'SMPLHParam.transl.requires_grad is not True, but {p.requires_grad}'
        if n == 'right_hand_pose':
            assert p.shape==(1, 6), f'SMPLHParam.right_hand_pose.shape is not (1, 6), but {p.shape}'
            assert p.requires_grad==True, f'SMPLHParam.right_hand_pose.requires_grad is not True, but {p.requires_grad}'

    from smplkit import SMPLHLayer as SMPLH
    bm = SMPLH(num_betas=10)
    verts1 = bm(**param.parameters_dict(), return_verts=True)
    verts2 = bm(transl=v, return_verts=True)
    assert torch.abs(verts1 - verts2).sum() < EPS, f"SMPLHLayer(**param.parameters_dict()) != SMPLHLayer(transl=v), but {torch.abs(verts1 - verts2).sum()}"

    print('[T3-2] Pass!\n')

def test3_3():
    """ Test the gradient of SMPLXParam
    """
    from smplkit import SMPLXParam

    ## init param and optimizer
    v = torch.rand(1, 3)
    param = SMPLXParam(transl=v, num_pca_comps=0, requires_grad=[True, False, False, False, False, True, False, False, True, True])

    transl, orient, betas, body_pose, left_hand_pose, right_hand_pose, expression, jaw_pose, leye_pose, reye_pose = param.parameters()
    assert transl.shape==(1, 3), f'SMPLXParam.transl.shape is not (1, 3), but {transl.shape}'
    assert orient.shape==(1, 3), f'SMPLXParam.orient.shape is not (1, 3), but {orient.shape}'
    assert betas.shape==(1, 10), f'SMPLXParam.betas.shape is not (1, 10), but {betas.shape}'
    assert body_pose.shape==(1, 63), f'SMPLXParam.body_pose.shape is not (1, 63), but {body_pose.shape}'
    assert left_hand_pose.shape==(1, 45), f'SMPLXParam.left_hand_pose.shape is not (1, 45), but {left_hand_pose.shape}'
    assert right_hand_pose.shape==(1, 45), f'SMPLXParam.right_hand_pose.shape is not (1, 45), but {right_hand_pose.shape}'
    assert expression.shape==(1, 10), f'SMPLXParam.expression.shape is not (1, 10), but {expression.shape}'
    assert jaw_pose.shape==(1, 3), f'SMPLXParam.jaw_pose.shape is not (1, 3), but {jaw_pose.shape}'
    assert leye_pose.shape==(1, 3), f'SMPLXParam.leye_pose.shape is not (1, 3), but {leye_pose.shape}'
    assert reye_pose.shape==(1, 3), f'SMPLXParam.reye_pose.shape is not (1, 3), but {reye_pose.shape}'

    assert transl.requires_grad==False, f'SMPLXParam.transl.requires_grad is not False, but {transl.requires_grad}'
    assert orient.requires_grad==False, f'SMPLXParam.orient.requires_grad is not False, but {orient.requires_grad}'
    assert betas.requires_grad==False, f'SMPLXParam.betas.requires_grad is not False, but {betas.requires_grad}'
    assert body_pose.requires_grad==False, f'SMPLXParam.body_pose.requires_grad is not False, but {body_pose.requires_grad}'
    assert left_hand_pose.requires_grad==False, f'SMPLXParam.left_hand_pose.requires_grad is not False, but {left_hand_pose.requires_grad}'
    assert right_hand_pose.requires_grad==False, f'SMPLXParam.right_hand_pose.requires_grad is not False, but {right_hand_pose.requires_grad}'
    assert expression.requires_grad==False, f'SMPLXParam.expression.requires_grad is not False, but {expression.requires_grad}'
    assert jaw_pose.requires_grad==False, f'SMPLXParam.jaw_pose.requires_grad is not False, but {jaw_pose.requires_grad}'
    assert leye_pose.requires_grad==False, f'SMPLXParam.leye_pose.requires_grad is not False, but {leye_pose.requires_grad}'
    assert reye_pose.requires_grad==False, f'SMPLXParam.reye_pose.requires_grad is not False, but {reye_pose.requires_grad}'
    
    _transl, _orient, _betas, _body_pose, _left_hand_pose, _right_hand_pose, _expression, _jaw_pose, _leye_pose, _reye_pose = param._parameters()
    assert _transl.requires_grad==True, f'SMPLXParam._transl.requires_grad is not True, but {_transl.requires_grad}'
    assert _orient.requires_grad==False, f'SMPLXParam._orient.requires_grad is not False, but {_orient.requires_grad}'
    assert _betas.requires_grad==False, f'SMPLXParam._betas.requires_grad is not False, but {_betas.requires_grad}'
    assert _body_pose.requires_grad==False, f'SMPLXParam._body_pose.requires_grad is not False, but {_body_pose.requires_grad}'
    assert _left_hand_pose.requires_grad==False, f'SMPLXParam._left_hand_pose.requires_grad is not False, but {_left_hand_pose.requires_grad}'
    assert _right_hand_pose.requires_grad==True, f'SMPLXParam._right_hand_pose.requires_grad is not True, but {_right_hand_pose.requires_grad}'
    assert _expression.requires_grad==False, f'SMPLXParam._expression.requires_grad is not False, but {_expression.requires_grad}'
    assert _jaw_pose.requires_grad==False, f'SMPLXParam._jaw_pose.requires_grad is not False, but {_jaw_pose.requires_grad}'
    assert _leye_pose.requires_grad==True, f'SMPLXParam._leye_pose.requires_grad is not True, but {_leye_pose.requires_grad}'
    assert _reye_pose.requires_grad==True, f'SMPLXParam._reye_pose.requires_grad is not True, but {_reye_pose.requires_grad}'
    

    tp = param.trainable_parameters()
    assert len(tp) == 4, f'len(param.trainable_parameters()) is not 4, but {len(tp)}'
    _transl, _right_hand_pose, _leye_pose, _reye_pose = tp
    assert _transl.requires_grad==True, f'SMPLXParam._transl.requires_grad is not True, but {_transl.requires_grad}'
    assert _right_hand_pose.requires_grad==True, f'SMPLXParam._right_hand_pose.requires_grad is not True, but {_right_hand_pose.requires_grad}'
    assert _leye_pose.requires_grad==True, f'SMPLXParam._leye_pose.requires_grad is not True, but {_leye_pose.requires_grad}'
    assert _reye_pose.requires_grad==True, f'SMPLXParam._reye_pose.requires_grad is not True, but {_reye_pose.requires_grad}'

    for n, p in param.parameters_dict().items():
        if n == 'transl':
            assert p.shape==(1, 3), f'SMPLXParam.transl.shape is not (1, 3), but {p.shape}'
            assert p.requires_grad==False, f'SMPLXParam.transl.requires_grad is not False, but {p.requires_grad}'
        if n == 'orient':
            assert p.shape==(1, 3), f'SMPLXParam.orient.shape is not (1, 3), but {p.shape}'
            assert p.requires_grad==False, f'SMPLXParam.orient.requires_grad is not False, but {p.requires_grad}'
        if n == 'betas':
            assert p.shape==(1, 10), f'SMPLXParam.betas.shape is not (1, 10), but {p.shape}'
            assert p.requires_grad==False, f'SMPLXParam.betas.requires_grad is not False, but {p.requires_grad}'
        if n == 'body_pose':
            assert p.shape==(1, 63), f'SMPLXParam.body_pose.shape is not (1, 63), but {p.shape}'
            assert p.requires_grad==False, f'SMPLXParam.body_pose.requires_grad is not False, but {p.requires_grad}'
        if n == 'left_hand_pose':
            assert p.shape==(1, 45), f'SMPLXParam.left_hand_pose.shape is not (1, 45), but {p.shape}'
            assert p.requires_grad==False, f'SMPLXParam.left_hand_pose.requires_grad is not False, but {p.requires_grad}'
        if n == 'right_hand_pose':
            assert p.shape==(1, 45), f'SMPLXParam.right_hand_pose.shape is not (1, 45), but {p.shape}'
            assert p.requires_grad==False, f'SMPLXParam.right_hand_pose.requires_grad is not False, but {p.requires_grad}'
        if n == 'expression':
            assert p.shape==(1, 10), f'SMPLXParam.expression.shape is not (1, 10), but {p.shape}'
            assert p.requires_grad==False, f'SMPLXParam.expression.requires_grad is not False, but {p.requires_grad}'
        if n == 'jaw_pose':
            assert p.shape==(1, 3), f'SMPLXParam.jaw_pose.shape is not (1, 3), but {p.shape}'
            assert p.requires_grad==False, f'SMPLXParam.jaw_pose.requires_grad is not False, but {p.requires_grad}'
        if n == 'leye_pose':
            assert p.shape==(1, 3), f'SMPLXParam.leye_pose.shape is not (1, 3), but {p.shape}'
            assert p.requires_grad==False, f'SMPLXParam.leye_pose.requires_grad is not False, but {p.requires_grad}'
        if n == 'reye_pose':
            assert p.shape==(1, 3), f'SMPLXParam.reye_pose.shape is not (1, 3), but {p.shape}'
            assert p.requires_grad==False, f'SMPLXParam.reye_pose.requires_grad is not False, but {p.requires_grad}'
    
    for n, p in param._parameters_dict().items():
        if n == 'transl':
            assert p.shape==(1, 3), f'SMPLXParam.transl.shape is not (1, 3), but {p.shape}'
            assert p.requires_grad==True, f'SMPLXParam.transl.requires_grad is not True, but {p.requires_grad}'
        if n == 'orient':
            assert p.shape==(1, 3), f'SMPLXParam.orient.shape is not (1, 3), but {p.shape}'
            assert p.requires_grad==False, f'SMPLXParam.orient.requires_grad is not False, but {p.requires_grad}'
        if n == 'betas':
            assert p.shape==(1, 10), f'SMPLXParam.betas.shape is not (1, 10), but {p.shape}'
            assert p.requires_grad==False, f'SMPLXParam.betas.requires_grad is not False, but {p.requires_grad}'
        if n == 'body_pose':
            assert p.shape==(1, 63), f'SMPLXParam.body_pose.shape is not (1, 63), but {p.shape}'
            assert p.requires_grad==False, f'SMPLXParam.body_pose.requires_grad is not False, but {p.requires_grad}'
        if n == 'left_hand_pose':
            assert p.shape==(1, 45), f'SMPLXParam.left_hand_pose.shape is not (1, 45), but {p.shape}'
            assert p.requires_grad==False, f'SMPLXParam.left_hand_pose.requires_grad is not False, but {p.requires_grad}'
        if n == 'right_hand_pose':
            assert p.shape==(1, 45), f'SMPLXParam.right_hand_pose.shape is not (1, 45), but {p.shape}'
            assert p.requires_grad==True, f'SMPLXParam.right_hand_pose.requires_grad is not True, but {p.requires_grad}'
        if n == 'expression':
            assert p.shape==(1, 10), f'SMPLXParam.expression.shape is not (1, 10), but {p.shape}'
            assert p.requires_grad==False, f'SMPLXParam.expression.requires_grad is not False, but {p.requires_grad}'
        if n == 'jaw_pose':
            assert p.shape==(1, 3), f'SMPLXParam.jaw_pose.shape is not (1, 3), but {p.shape}'
            assert p.requires_grad==False, f'SMPLXParam.jaw_pose.requires_grad is not False, but {p.requires_grad}'
        if n == 'leye_pose':
            assert p.shape==(1, 3), f'SMPLXParam.leye_pose.shape is not (1, 3), but {p.shape}'
            assert p.requires_grad==True, f'SMPLXParam.leye_pose.requires_grad is not True, but {p.requires_grad}'
        if n == 'reye_pose':
            assert p.shape==(1, 3), f'SMPLXParam.reye_pose.shape is not (1, 3), but {p.shape}'
            assert p.requires_grad==True, f'SMPLXParam.reye_pose.requires_grad is not True, but {p.requires_grad}'
    
    for n, p in param.trainable_parameters_dict().items():
        assert n in ['transl', 'right_hand_pose', 'leye_pose', 'reye_pose'], f'SMPLXParam.named_trainable_parameters() is not transl, right_hand_pose, leye_pose, or reye_pose, but {n}'
        if n == 'transl':
            assert p.shape==(1, 3), f'SMPLXParam.transl.shape is not (1, 3), but {p.shape}'
            assert p.requires_grad==True, f'SMPLXParam.transl.requires_grad is not True, but {p.requires_grad}'
        if n == 'right_hand_pose':
            assert p.shape==(1, 45), f'SMPLXParam.right_hand_pose.shape is not (1, 45), but {p.shape}'
            assert p.requires_grad==True, f'SMPLXParam.right_hand_pose.requires_grad is not True, but {p.requires_grad}'
        if n == 'leye_pose':
            assert p.shape==(1, 3), f'SMPLXParam.leye_pose.shape is not (1, 3), but {p.shape}'
            assert p.requires_grad==True, f'SMPLXParam.leye_pose.requires_grad is not True, but {p.requires_grad}'
        if n == 'reye_pose':
            assert p.shape==(1, 3), f'SMPLXParam.reye_pose.shape is not (1, 3), but {p.shape}'
            assert p.requires_grad==True, f'SMPLXParam.reye_pose.requires_grad is not True, but {p.requires_grad}'

    from smplkit import SMPLXLayer as SMPLX
    bm = SMPLX(num_betas=10, num_pca_comps=0)
    verts1 = bm(**param.parameters_dict(), return_verts=True)
    verts2 = bm(transl=v, return_verts=True)
    assert torch.abs(verts1 - verts2).sum() < EPS, f"SMPLLayer(**param.parameters_dict()) != SMPLLayer(transl=v), but {torch.abs(verts1 - verts2).sum()}"

    print('[T3-3] Pass!\n')

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
