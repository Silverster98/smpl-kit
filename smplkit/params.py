import torch
import torch.nn as nn
import numpy as np
from typing import Union, Optional, List, Dict, Tuple

class Parameter(object):

    def __init__(self) -> None:
        pass

class SMPLParam(Parameter):

    NUM_JOINTS = 23
    NUM_BODY_JOINTS = 23

    def __init__(
            self,
            transl: Optional[Union[torch.Tensor, np.ndarray]]=None,
            orient: Optional[Union[torch.Tensor, np.ndarray]]=None,
            betas: Optional[Union[torch.Tensor, np.ndarray]]=None,
            body_pose: Optional[Union[torch.Tensor, np.ndarray]]=None,
            num_betas: int=10,
            requires_grad: Union[List, Tuple, bool]=False,
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
            batch_size: batch size, default is 1 if no parameters are given
            dtype: data type, default is torch.float32
            device: device, default is 'cpu'
        """
        super(SMPLParam, self).__init__()

        batch_size = kwargs['batch_size'] if 'batch_size' in kwargs \
            else self._get_batch_size([transl, orient, body_pose, betas])
        torch_kwargs = {'dtype': kwargs.get('dtype', torch.float32), 'device': kwargs.get('device', 'cpu')}

        self.num_betas = num_betas
        
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
            body_pose = torch.zeros(batch_size, self.NUM_BODY_JOINTS*3, **torch_kwargs)
        elif torch.is_tensor(body_pose):
            body_pose = body_pose.to(**torch_kwargs)
        else:
            body_pose = torch.tensor(body_pose, **torch_kwargs)
        
        assert transl.shape[0] == orient.shape[0] == betas.shape[0] == body_pose.shape[0], f"[Shape Error] the batch size of the given parameters are not equal!"
        

        ## set requires_grad
        if isinstance(requires_grad, (list, tuple)):
            assert len(requires_grad) >= 4, f"[Length Error] For SMPL model, requires_grad must be a list/tuple of length >= 4!"
            transl_rg, orient_rg, betas_rg, body_pose_rg = requires_grad[0:4]
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
    
    def _get_batch_size(self, items: List) -> int:
        """ Compute the batch size from given parameters

        Args:
            items: given parameters
        
        Return:
            The batch size, if there are no parameters given, return 1.
        """
        for item in items:
            if isinstance(item, np.ndarray):
                return item.shape[0]
            if torch.is_tensor(item):
                return item.shape[0]
        
        return 1
        
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

class SMPLHParam(SMPLParam):

    NUM_BODY_JOINTS = SMPLParam.NUM_JOINTS - 2
    NUM_HAND_JOINTS = 15
    NUM_JOINTS = NUM_BODY_JOINTS + 2 * NUM_HAND_JOINTS

    def __init__(
            self,
            transl: Optional[Union[torch.Tensor, np.ndarray]]=None,
            orient: Optional[Union[torch.Tensor, np.ndarray]]=None,
            betas: Optional[Union[torch.Tensor, np.ndarray]]=None,
            body_pose: Optional[Union[torch.Tensor, np.ndarray]]=None,
            left_hand_pose: Optional[Union[torch.Tensor, np.ndarray]]=None,
            right_hand_pose: Optional[Union[torch.Tensor, np.ndarray]]=None,
            num_betas: int=10,
            num_pca_comps: int=6,
            requires_grad: Union[List, Tuple, bool]=False,
            **kwargs,
        ) -> None:
        """ SMPLHParam constructor

        Args:
            transl: translation parameters, default is None
            orient: orientation parameters, default is None
            betas: shape parameters, default is None
            body_pose: body pose parameters, default is None
            left_hand_pose: left hand pose parameters, default is None
            right_hand_pose: right hand pose parameters, default is None
            num_betas: number of shape parameters, default is 10
            num_pca_comps: number of pca components, default is 6; if num_pca_comps <= 0, the hand pose parameters will be used directly
            requires_grad: whether to require gradient, default is False. If true, all the parameters will be converted to nn.Parameter with requires_grad=True
            batch_size: batch size, default is 1 if no parameters are given
            dtype: data type, default is torch.float32
            device: device, default is 'cpu'
        """

        batch_size = kwargs['batch_size'] if 'batch_size' in kwargs \
            else self._get_batch_size([transl, orient, body_pose, left_hand_pose, right_hand_pose, betas])
        kwargs['batch_size'] = batch_size
        torch_kwargs = {'dtype': kwargs.get('dtype', torch.float32), 'device': kwargs.get('device', 'cpu')}

        super(SMPLHParam, self).__init__(
            transl=transl,
            orient=orient,
            betas=betas,
            body_pose=body_pose,
            num_betas=num_betas,
            requires_grad=requires_grad,
            **kwargs
        )

        self.num_pca_comps = num_pca_comps
        self.hand_pose_dim = num_pca_comps if num_pca_comps > 0 else 3 * self.NUM_HAND_JOINTS

        ## initialize the parameters if not given
        if left_hand_pose is None:
            left_hand_pose = torch.zeros(batch_size, self.hand_pose_dim, **torch_kwargs)
        elif torch.is_tensor(left_hand_pose):
            left_hand_pose = left_hand_pose.to(**torch_kwargs)
        else:
            left_hand_pose = torch.tensor(left_hand_pose, **torch_kwargs)
        
        if right_hand_pose is None:
            right_hand_pose = torch.zeros(batch_size, self.hand_pose_dim, **torch_kwargs)
        elif torch.is_tensor(right_hand_pose):
            right_hand_pose = right_hand_pose.to(**torch_kwargs)
        else:
            right_hand_pose = torch.tensor(right_hand_pose, **torch_kwargs)
        
        assert self._transl.shape[0] == left_hand_pose.shape[0] == right_hand_pose.shape[0], f"[Shape Error] the batch size of the given hand pose parameters are not equal!"
        

        ## set requires_grad
        if isinstance(requires_grad, (list, tuple)):
            assert len(requires_grad) >= 6, f"[Length Error] For SMPL+H model, requires_grad must be a list/tuple of length >= 6!"
            left_hand_pose_rg, right_hand_pose_rg = requires_grad[4:6]
        elif isinstance(requires_grad, bool):
            left_hand_pose_rg, right_hand_pose_rg = requires_grad, requires_grad
        else:
            raise ValueError(f"[Value Error] requires_grad must be a bool or a list/tuple of bools!")
        
        if left_hand_pose_rg:
            left_hand_pose.requires_grad = True
        if right_hand_pose_rg:
            right_hand_pose.requires_grad = True
        
        self._left_hand_pose = left_hand_pose
        self._right_hand_pose = right_hand_pose
    
    @property
    def left_hand_pose(self) -> torch.Tensor:
        """ Returns the copied and detached left hand pose parameters of the SMPL+H model. """
        return self._left_hand_pose.clone().detach()
    
    @property
    def right_hand_pose(self) -> torch.Tensor:
        """ Returns the copied and detached right hand pose parameters of the SMPL+H model. """
        return self._right_hand_pose.clone().detach()

    ## parameters list
    def parameters(self) -> List:
        """ Returns the list of copied and detached parameters of the SMPL+H model. """
        return super().parameters() + [self.left_hand_pose, self.right_hand_pose]

    def _parameters(self) -> List:
        """ Returns the list of parameters of the SMPL+H model. """
        return super()._parameters() + [self._left_hand_pose, self._right_hand_pose]

    def trainable_parameters(self) -> List:
        """ Returns the list of trainable parameters of the SMPL+H model. """
        return [p for p in self._parameters() if p.requires_grad]
    
    ## parameters dict
    def parameters_dict(self) -> Dict:
        """ Returns the dictionary of copied and detached parameters of the SMPL+H model. """
        return {**super().parameters_dict(), 'left_hand_pose': self.left_hand_pose, 'right_hand_pose': self.right_hand_pose}
    
    def _parameters_dict(self) -> Dict:
        """ Returns the dictionary of parameters of the SMPL+H model. """
        return {**super()._parameters_dict(), 'left_hand_pose': self._left_hand_pose, 'right_hand_pose': self._right_hand_pose}
    
    def trainable_parameters_dict(self) -> Dict:
        """ Returns the dictionary of trainable parameters of the SMPL+H model. """
        return {n: p for n, p in self._parameters_dict().items() if p.requires_grad}

class SMPLXParam(SMPLHParam):

    NUM_BODY_JOINTS = SMPLHParam.NUM_BODY_JOINTS
    NUM_HAND_JOINTS = 15
    NUM_FACE_JOINTS = 3
    NUM_JOINTS = NUM_BODY_JOINTS + 2 * NUM_HAND_JOINTS + NUM_FACE_JOINTS

    def __init__(
            self,
            transl: Optional[Union[torch.Tensor, np.ndarray]]=None,
            orient: Optional[Union[torch.Tensor, np.ndarray]]=None,
            betas: Optional[Union[torch.Tensor, np.ndarray]]=None,
            body_pose: Optional[Union[torch.Tensor, np.ndarray]]=None,
            left_hand_pose: Optional[Union[torch.Tensor, np.ndarray]]=None,
            right_hand_pose: Optional[Union[torch.Tensor, np.ndarray]]=None,
            expression: Optional[Union[torch.Tensor, np.ndarray]]=None,
            jaw_pose: Optional[Union[torch.Tensor, np.ndarray]]=None,
            leye_pose: Optional[Union[torch.Tensor, np.ndarray]]=None,
            reye_pose: Optional[Union[torch.Tensor, np.ndarray]]=None,
            num_betas: int=10,
            num_pca_comps: int=6,
            num_expression_coeffs: int=10,
            requires_grad: Union[List, Tuple, bool]=False,
            **kwargs,
        ) -> None:
        """ SMPLXParam constructor

        Args:
            transl: translation parameters, default is None
            orient: orientation parameters, default is None
            betas: shape parameters, default is None
            body_pose: body pose parameters, default is None
            left_hand_pose: left hand pose parameters, default is None
            right_hand_pose: right hand pose parameters, default is None
            num_betas: number of shape parameters, default is 10
            num_pca_comps: number of pca components, default is 6; if num_pca_comps <= 0, the hand pose parameters will be used directly
            num_expression_coeffs: number of expression coefficients, default is 10
            requires_grad: whether to require gradient, default is False. If true, all the parameters will be converted to nn.Parameter with requires_grad=True
            batch_size: batch size, default is 1 if no parameters are given
            dtype: data type, default is torch.float32
            device: device, default is 'cpu'
        """

        batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else self._get_batch_size([
            transl, orient, body_pose, left_hand_pose, right_hand_pose, expression, jaw_pose, leye_pose, reye_pose, betas])
        kwargs['batch_size'] = batch_size
        torch_kwargs = {'dtype': kwargs.get('dtype', torch.float32), 'device': kwargs.get('device', 'cpu')}

        super(SMPLXParam, self).__init__(
            transl=transl,
            orient=orient,
            betas=betas,
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            num_betas=num_betas,
            num_pca_comps=num_pca_comps,
            requires_grad=requires_grad,
            **kwargs
        )

        self.num_expression_coeffs = num_expression_coeffs

        ## initialize the parameters if not given
        if expression is None:
            expression = torch.zeros(batch_size, self.num_expression_coeffs, **torch_kwargs)
        elif torch.is_tensor(expression):
            expression = expression.to(**torch_kwargs)
        else:
            expression = torch.tensor(expression, **torch_kwargs)
        
        if jaw_pose is None:
            jaw_pose = torch.zeros(batch_size, 3, **torch_kwargs)
        elif torch.is_tensor(jaw_pose):
            jaw_pose = jaw_pose.to(**torch_kwargs)
        else:
            jaw_pose = torch.tensor(jaw_pose, **torch_kwargs)
        
        if leye_pose is None:
            leye_pose = torch.zeros(batch_size, 3, **torch_kwargs)
        elif torch.is_tensor(leye_pose):
            leye_pose = leye_pose.to(**torch_kwargs)
        else:
            leye_pose = torch.tensor(leye_pose, **torch_kwargs)
        
        if reye_pose is None:
            reye_pose = torch.zeros(batch_size, 3, **torch_kwargs)
        elif torch.is_tensor(reye_pose):
            reye_pose = reye_pose.to(**torch_kwargs)
        else:
            reye_pose = torch.tensor(reye_pose, **torch_kwargs)
        
        assert self._transl.shape[0] == expression.shape[0] == jaw_pose.shape[0] == leye_pose.shape[0] == reye_pose.shape[0], f"[Shape Error] the batch size of the given facial parameters are not equal!"


        ## set requires_grad
        if isinstance(requires_grad, (list, tuple)):
            assert len(requires_grad) >= 10, f"[Length Error] For SMPL-X model, requires_grad must be a list/tuple of length >= 10!"
            expression_rg, jaw_pose_rg, leye_pose_rg, reye_pose_rg = requires_grad[6:10]
        elif isinstance(requires_grad, bool):
            expression_rg, jaw_pose_rg, leye_pose_rg, reye_pose_rg = requires_grad, requires_grad, requires_grad, requires_grad
        else:
            raise ValueError(f"[Value Error] requires_grad must be a bool or a list/tuple of bools!")

        if expression_rg:
            expression.requires_grad = True
        if jaw_pose_rg:
            jaw_pose.requires_grad = True
        if leye_pose_rg:
            leye_pose.requires_grad = True
        if reye_pose_rg:
            reye_pose.requires_grad = True
        
        self._expression = expression
        self._jaw_pose = jaw_pose
        self._leye_pose = leye_pose
        self._reye_pose = reye_pose
    
    @property
    def expression(self) -> torch.Tensor:
        """ Returns the copied and detached expression parameters of the SMPL-X model. """
        return self._expression.clone().detach()
    
    @property
    def jaw_pose(self) -> torch.Tensor:
        """ Returns the copied and detached jaw pose parameters of the SMPL-X model. """
        return self._jaw_pose.clone().detach()
    
    @property
    def leye_pose(self) -> torch.Tensor:
        """ Returns the copied and detached left eye pose parameters of the SMPL-X model. """
        return self._leye_pose.clone().detach()
    
    @property
    def reye_pose(self) -> torch.Tensor:
        """ Returns the copied and detached right eye pose parameters of the SMPL-X model. """
        return self._reye_pose.clone().detach()
    
    ## parameters list
    def parameters(self) -> List:
        """ Returns the list of copied and detached parameters of the SMPL-X model. """
        return super().parameters() + [self.expression, self.jaw_pose, self.leye_pose, self.reye_pose]
    
    def _parameters(self) -> List:
        """ Returns the list of parameters of the SMPL-X model. """
        return super()._parameters() + [self._expression, self._jaw_pose, self._leye_pose, self._reye_pose]
    
    def trainable_parameters(self) -> List:
        """ Returns the list of trainable parameters of the SMPL-X model. """
        return [p for p in self._parameters() if p.requires_grad]
    
    ## parameters dict
    def parameters_dict(self) -> Dict:
        """ Returns the dictionary of copied and detached parameters of the SMPL-X model. """
        return {**super().parameters_dict(), 'expression': self.expression, 'jaw_pose': self.jaw_pose, 'leye_pose': self.leye_pose, 'reye_pose': self.reye_pose}
    
    def _parameters_dict(self) -> Dict:
        """ Returns the dictionary of parameters of the SMPL-X model. """
        return {**super()._parameters_dict(), 'expression': self._expression, 'jaw_pose': self._jaw_pose, 'leye_pose': self._leye_pose, 'reye_pose': self._reye_pose}
    
    def trainable_parameters_dict(self) -> Dict:
        """ Returns the dictionary of trainable parameters of the SMPL-X model. """
        return {n: p for n, p in self._parameters_dict().items() if p.requires_grad}
