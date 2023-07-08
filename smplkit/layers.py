
import os
import pickle
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from typing import Dict, Union, List, Optional

from .misc import *
from .params import *
from .lbs import *

class SMPLLayer(nn.Module):

    NUM_JOINTS = 23
    NUM_BODY_JOINTS = 23
    SHAPE_SPACE_DIM = 300

    def __init__(
            self,
            model_path: Optional[str]=None,
            gender: Optional[str]='neutral',
            num_betas: Optional[int]=10,
            ext: Optional[str]='pkl',
            **kwargs,
        ) -> None:
        """ SMPL layer constructor

        Args:
            model_path: The path of the SMPL model path or folder where the model parameters are stored
            gender: specify the gender of the model, default is 'neutral'
            num_betas: the number of shape coeffients, default is 10
            ext: the extension of the SMPL model file, default is 'pkl'
            dtype: the data type of the model parameters, default is torch.float32
        """
        super(SMPLLayer, self).__init__()

        self.model_path = search_model_path(model_path, 'SMPL')
        self.gender = gender
        self.num_betas = num_betas
        self.ext = ext
        self.dtype = kwargs.get('dtype', torch.float32)

        if os.path.isdir(self.model_path):
            self.model_path = os.path.join(
                self.model_path, f"SMPL_{self.gender}.{self.ext}")
        assert os.path.exists(self.model_path), f"[Not Find] SMPL model: {self,model_path}!"

        ## Load SMPL model parameters
        with open(self.model_path, 'rb') as fp:
            data_struct = Struct(**pickle.load(fp, encoding='latin1'))
        
        shapedirs = data_struct.shapedirs
        max_num_betas = min(shapedirs.shape[-1], self.SHAPE_SPACE_DIM)
        assert self.num_betas <= max_num_betas, f"[Value Error] the given num_betas: {self.num_betas} is greater than supported: {max_num_betas}!"
        shapedirs = shapedirs[:, :, :self.num_betas]

        self.register_buffer(
            'shapedirs', to_tensor(to_numpy(shapedirs), dtype=self.dtype)
        )
        self.register_buffer(
            'faces_tensor', to_tensor(to_numpy(data_struct.f, dtype=np.int64))
        )
        self.register_buffer(
            'v_template', to_tensor(to_numpy(data_struct.v_template), dtype=self.dtype)
        )
        self.register_buffer(
            'J_regressor', to_tensor(to_numpy(data_struct.J_regressor), dtype=self.dtype)
        )
        self.register_buffer(
            'posedirs', rearrange(
                to_tensor(to_numpy(data_struct.posedirs), dtype=self.dtype),
                'v c h -> h (v c)'
            )
        )
        parents = to_tensor(to_numpy(data_struct.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer(
            'parents', parents
        )
        self.register_buffer(
            'lbs_weights', to_tensor(to_numpy(data_struct.weights), dtype=self.dtype)
        )

        # for attr in data_struct.__dict__.keys():
        #     print("data_struct.%s = %r" % (attr, getattr(data_struct, attr)))
    
    @property
    def name(self) -> str:
        """ Return the name of the this SMPL layer """
        return "SMPL"
    
    @property
    def faces(self) -> np.ndarray:
        """ Return the faces of the SMPL model, i.e., a numpy array of shape <num_faces, 3> """
        return self.faces_tensor.cpu().numpy()
    
    @property
    def num_verts(self) -> int:
        """ Return the number of vertices of the SMPL model """
        return self.v_template.shape[0]
    
    @property
    def num_faces(self) -> int:
        """ Return the number of faces of the SMPL model """
        return self.faces_tensor.shape[0]

    def forward(
        self,
        transl: Optional[torch.Tensor]=None,
        orient: Optional[torch.Tensor]=None,
        betas: Optional[torch.Tensor]=None,
        body_pose: Optional[torch.Tensor]=None,
        pose2rot: Optional[bool]=True,
        apply_trans: Optional[bool]=True,
        return_verts: Optional[bool]=False,
        **kwargs,
    ) -> Union[SMPLOutput, torch.Tensor]:
        """ Forward pass of SMPL layer

        Args:
            transl: the translation, a tensor of shape <batch_size, 3>
            orient: the global orientation, it should be represented in axis-angle format, a tensor of shape <batch_size, 3>
            betas: the shape coeffients, torch.Tensor of shape <batch_size, num_betas>
            body_pose: the body pose parameters, it should be represented in axis-angle format, a tensor of shape <batch_size, J*3>
            pose2rot: flag of converting the pose parameters to rotation matrices, default is True
            apply_trans: flag of applying the translation, default is True
            return_verts: flag of only returning the vertices, default is False
            batch_size: batch size of smpl parameters, default is 1, if any of `[transl, orient, body_pose]` is given, this parameter will be ignored
        
        Return:
            This method returns a SMPLOutput object or only returns a tensor of shape <batch_size, num_verts, 3> representing the SMPL body mesh vertices
        """

        torch_kwargs = {'dtype': self.dtype, 'device': self.shapedirs.device}
        
        ## compute the batch size
        def _get_batch_size(tensors: List) -> int:
            for t in tensors:
                if t is not None:
                    return t.shape[0]
            return kwargs.get('batch_size', 1)
        batch_size = _get_batch_size([transl, orient, body_pose])
        
        ## initialize the parameters if not given
        if transl is None:
            transl = torch.zeros(batch_size, 3, **torch_kwargs)
        
        if orient is None:
            orient = torch.zeros(batch_size, 3, **torch_kwargs)
        
        if betas is None:
            betas = torch.zeros(batch_size, self.num_betas, **torch_kwargs)
        else:
            ## expand the betas if the batch size of betas is 1
            if betas.shape[0] != batch_size and betas.shape[0] == 1:
                betas = betas.expand(batch_size, -1)
        
        if body_pose is None:
            body_pose = torch.zeros(batch_size, self.NUM_JOINTS*3, **torch_kwargs)
        
        assert transl.shape[0] == orient.shape[0] == betas.shape[0] == body_pose.shape[0], f"[Shape Error] the batch size of the given parameters are not equal!"

        
        ## linear blend skinning
        full_pose = torch.cat([orient, body_pose], dim=1)
        vertices, joints = lbs(betas, full_pose, self.v_template,
                               self.shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights, pose2rot=pose2rot)
        
        ## apply translation
        if apply_trans:
            joints = joints + transl.unsqueeze(1)
            vertices = vertices + transl.unsqueeze(1)

        ## prepare output
        if return_verts:
            return vertices
        
        return SMPLOutput(
            vertices=vertices,
            faces=self.faces_tensor.clone(),
            joints=joints,
        )
