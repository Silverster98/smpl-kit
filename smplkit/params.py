import torch
import torch.nn as nn
import numpy as np
from typing import Union, Optional, List, Dict, Tuple

class Parameter(object):

    def __init__(self) -> None:
        pass

class SMPLParam(Parameter):

    NUM_JOINTS = 23

    def __init__(
            self,
            transl: Optional[Union[torch.Tensor, np.ndarray]]=None,
            orient: Optional[Union[torch.Tensor, np.ndarray]]=None,
            betas: Optional[Union[torch.Tensor, np.ndarray]]=None,
            body_pose: Optional[Union[torch.Tensor, np.ndarray]]=None,
            num_betas: Optional[int]=10,
            requires_grad: Optional[Union[List,Tuple,bool]]=False,
            **kwargs,
        ) -> None:
        """ SMPLParam constructor

        Args:
            transl: translation parameters, default is None
            orient: orientation parameters, default is None
            betas: shape parameters, default is None
            body_pose: body pose parameters, default is None
            num_betas: number of shape parameters, default is 10
            requires_grad: whether to require gradient, default is False. If true, all the parameters will be converted to nn.Parameter with requires_grad=True
            batch_size: batch size, default is 1
            dtype: data type, default is torch.float32
            device: device, default is 'cpu'
        """
        super(SMPLParam, self).__init__()
        self.num_betas = num_betas

        ## compute default batch size, dtype, and device
        def _get_bs_dtype_device(tensors: List) -> int:
            for t in tensors:
                if isinstance(t, np.ndarray):
                    return t.shape[0], torch.from_numpy(t).dtype, 'cpu'
                if torch.is_tensor(t):
                    return t.shape[0], t.dtype, t.device
            return kwargs.get('batch_size', 1), kwargs.get('dtype', torch.float32), kwargs.get('device', 'cpu')
        batch_size, dtype, device = _get_bs_dtype_device([transl, orient, body_pose])

        torch_kwargs = {'dtype': dtype, 'device': device}
        
        ## initialize the parameters if not given
        if transl is None:
            transl = torch.zeros(batch_size, 3, **torch_kwargs)
        elif torch.is_tensor(transl):
            transl = transl.to(**torch_kwargs)
        else:
            transl = torch.tensor(transl, **torch_kwargs)
        
        if orient is None:
            orient = torch.zeros(batch_size, 3, **torch_kwargs)
        elif torch.is_tensor(orient):
            orient = orient.to(**torch_kwargs)
        else:
            orient = torch.tensor(orient, **torch_kwargs)
        
        if betas is None:
            betas = torch.zeros(batch_size, self.num_betas, **torch_kwargs)
        elif torch.is_tensor(betas):
            betas = betas.to(**torch_kwargs)
        else:
            betas = torch.tensor(betas, **torch_kwargs)
        
        if body_pose is None:
            body_pose = torch.zeros(batch_size, self.NUM_JOINTS*3, **torch_kwargs)
        elif torch.is_tensor(body_pose):
            body_pose = body_pose.to(**torch_kwargs)
        else:
            body_pose = torch.tensor(body_pose, **torch_kwargs)
        
        assert transl.shape[0] == orient.shape[0] == betas.shape[0] == body_pose.shape[0], f"[Shape Error] the batch size of the given parameters are not equal!"
        

        ## set requires_grad
        if isinstance(requires_grad, (list, tuple)):
            assert len(requires_grad) == 4, f"[Length Error] requires_grad must be a list/tuple of length 4!"
            transl_rg, orient_rg, betas_rg, body_pose_rg = requires_grad
        elif isinstance(requires_grad, bool):
            transl_rg, orient_rg, betas_rg, body_pose_rg = requires_grad, requires_grad, requires_grad, requires_grad
        else:
            raise ValueError(f"[Value Error] requires_grad must be a bool or a list/tuple of bools!")

        if transl_rg:
            transl.requires_grad = True
        if orient_rg:
            orient.requires_grad = True
        if betas_rg:
            betas.requires_grad = True
        if body_pose_rg:
            body_pose.requires_grad = True
        
        self._transl = transl
        self._orient = orient
        self._betas = betas
        self._body_pose = body_pose
        
    @property
    def transl(self) -> torch.Tensor:
        """ Returns the copied and detached translational parameters of the SMPL model. """
        return self._transl.clone().detach()
    
    @property
    def orient(self) -> torch.Tensor:
        """ Returns the copied and detached orientation parameters of the SMPL model. """
        return self._orient.clone().detach()
    
    @property
    def betas(self) -> torch.Tensor:
        """ Returns the copied and detached shape parameters of the SMPL model. """
        return self._betas.clone().detach()
    
    @property
    def body_pose(self) -> torch.Tensor:
        """ Returns the copied and detached body pose parameters of the SMPL model. """
        return self._body_pose.clone().detach()
    
    ## parameters list
    def parameters(self) -> List:
        """ Returns the list of copied and detached parameters of the SMPL model. """
        return [self.transl, self.orient, self.betas, self.body_pose]
    
    def _parameters(self) -> List:
        """ Returns the list of parameters of the SMPL model. """
        return [self._transl, self._orient, self._betas, self._body_pose]
    
    def trainable_parameters(self) -> List:
        """ Returns the list of trainable parameters of the SMPL model. """
        return [p for p in self._parameters() if p.requires_grad]
    
    ## parameters dict
    def parameters_dict(self) -> Dict:
        """ Returns the dictionary of copied and detached parameters of the SMPL model. """
        return {'transl': self.transl, 'orient': self.orient, 'betas': self.betas, 'body_pose': self.body_pose}
    
    def _parameters_dict(self) -> Dict:
        """ Returns the dictionary of parameters of the SMPL model. """
        return {'transl': self._transl, 'orient': self._orient, 'betas': self._betas, 'body_pose': self._body_pose}
    
    def trainable_parameters_dict(self) -> Dict:
        """ Returns the dictionary of trainable parameters of the SMPL model. """
        return {n: p for n, p in self._parameters_dict().items() if p.requires_grad}

    ## named parameters
    def named_parameters(self) -> List:
        """ Returns the list of named, copied and detached parameters of the SMPL model. """
        return [('transl', self.transl), ('orient', self.orient), ('betas', self.betas), ('body_pose', self.body_pose)]
    
    def _named_parameters(self) -> List:
        """ Returns the list of named parameters of the SMPL model. """
        return [('transl', self._transl), ('orient', self._orient), ('betas', self._betas), ('body_pose', self._body_pose)]

    def trainable_named_parameters(self) -> List:
        """ Returns the list of trainable, named parameters of the SMPL model. """
        return [(n, p) for n, p in self._named_parameters() if p.requires_grad]
    