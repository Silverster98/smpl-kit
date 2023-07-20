
import os
import pickle
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from typing import Dict, Union, List, Optional

from smplkit.lbs import torch
from smplkit.misc import Optional, SMPLHOutput, Union, torch
from smplkit.params import Optional, Union, torch

from .misc import *
from .params import *
from .lbs import *

def load_model_parameters(model: str, name: str, gender: str, ext: str):
    """ Load SMPL model parameters from the given model path

    Args:
        name: model name
        model: model path, can be a directory or file path
        gender: model gender
        ext: the extension of the model file

    Return:
        A dict containing the SMPL model parameters
    """
    if os.path.isdir(model):
        model = os.path.join(model, f"{name}_{gender.upper()}.{ext}")
    assert os.path.exists(model), f"[Not Find] {name} model: {model}!"

    if model.split('.')[-1] == 'pkl':
        with open(model, 'rb') as fp:
            data_struct = Struct(**pickle.load(fp, encoding='latin1'))
    elif model.split('.')[-1] == 'npz':
        data_struct = Struct(**np.load(model, allow_pickle=True))
    else:
        raise Exception(f"[Not Support] the extension of the model file: {model}!")
    
    return data_struct

class SMPLLayer(nn.Module):

    NUM_JOINTS = 23
    NUM_BODY_JOINTS = 23
    SHAPE_SPACE_DIM = 300

    def __init__(
            self,
            model_path: Optional[str]=None,
            gender: str='neutral',
            num_betas: int=10,
            ext: str='pkl',
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

        self.model_path = search_model_path(model_path, self.name)
        self.gender = gender
        self.num_betas = num_betas
        self.ext = ext
        self.dtype = kwargs.get('dtype', torch.float32)

        self.data_struct = load_model_parameters(self.model_path, self.name, self.gender, self.ext)
        
        shapedirs = self.data_struct.shapedirs
        max_num_betas = min(shapedirs.shape[-1], self.SHAPE_SPACE_DIM)
        assert self.num_betas <= max_num_betas, f"[Value Error] the given num_betas: {self.num_betas} is greater than supported: {max_num_betas}!"
        shapedirs = shapedirs[:, :, :self.num_betas]

        self.register_buffer(
            'shapedirs', to_tensor(to_numpy(shapedirs), dtype=self.dtype)
        )
        self.register_buffer(
            'faces_tensor', to_tensor(to_numpy(self.data_struct.f), dtype=torch.long)
        )
        self.register_buffer(
            'v_template', to_tensor(to_numpy(self.data_struct.v_template), dtype=self.dtype)
        )
        self.register_buffer(
            'J_regressor', to_tensor(to_numpy(self.data_struct.J_regressor), dtype=self.dtype)
        )
        self.register_buffer(
            'posedirs', rearrange(
                to_tensor(to_numpy(self.data_struct.posedirs), dtype=self.dtype),
                'v c h -> h (v c)'
            )
        )
        parents = to_tensor(to_numpy(self.data_struct.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer(
            'parents', parents
        )
        self.register_buffer(
            'lbs_weights', to_tensor(to_numpy(self.data_struct.weights), dtype=self.dtype)
        )

        # for attr in self.data_struct.__dict__.keys():
        #     print("self.data_struct.%s = %r" % (attr, getattr(self.data_struct, attr)))
    
    def _get_batch_size(self, items: List, default_batch_size: int) -> int:
        """ Compute the batch size of the given parameters

        Args:
            items: a list of parameters
            default_batch_size: the default batch size
        
        Return:
            The batch size, if there are no parameters given, return the default batch size
        """
        for item in items:
            if item is not None:
                return item.shape[0]
        return default_batch_size
    
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
        pose2rot: bool=True,
        apply_trans: bool=True,
        return_verts: bool=False,
        return_joints: bool=False,
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
            return_verts: flag of **only** returning the vertices, default is False
            return_joints: flag of **only** returning the joints, default is False
            batch_size: batch size of smpl parameters, default is 1, if any of `[transl, orient, body_pose]` is given, this parameter will be ignored
        
        Return:
            This method returns a SMPLOutput object or only returns a tensor of shape <batch_size, num_verts, 3> representing the SMPL body mesh vertices
        """

        torch_kwargs = {'dtype': self.dtype, 'device': self.shapedirs.device}
        batch_size = self._get_batch_size([transl, orient, body_pose, betas], kwargs.get('batch_size', 1))
        
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
            body_pose = torch.zeros(batch_size, self.NUM_BODY_JOINTS*3, **torch_kwargs)
        
        assert transl.shape[0] == orient.shape[0] == betas.shape[0] == body_pose.shape[0], f"[Shape Error] the batch size of the given parameters are not equal!"

        
        ## linear blend skinning
        full_pose = torch.cat([orient, body_pose], dim=1)
        vertices, joints = lbs(betas, full_pose, self.v_template,
                               self.shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights, pose2rot=pose2rot)

        ## only return the vertices for saving computation
        if return_verts:
            if apply_trans:
                vertices = vertices + transl.unsqueeze(1)
            return vertices
        
        ## only return the joints for saving computation
        if return_joints:
            if apply_trans:
                joints = joints + transl.unsqueeze(1)
            return joints
        
        ## apply translation
        if apply_trans:
            joints = joints + transl.unsqueeze(1)
            vertices = vertices + transl.unsqueeze(1)
        
        return SMPLOutput(
            vertices=vertices,
            faces=self.faces_tensor.clone(),
            joints=joints,
            body_pose=body_pose,
        )

class SMPLHLayer(SMPLLayer):

    NUM_BODY_JOINTS = SMPLLayer.NUM_JOINTS - 2
    NUM_HAND_JOINTS = 15
    NUM_JOINTS = NUM_BODY_JOINTS + 2 * NUM_HAND_JOINTS

    def __init__(
            self,
            model_path: Optional[str]=None,
            gender: str='neutral',
            num_betas: int=10,
            num_pca_comps: int=6,
            flat_hand_mean: bool=False,
            ext: str='pkl',
            **kwargs,
        ) -> None:
        """ SMPLH layer constructor
        
        Args:
            model_path: The path of the SMPL model path or folder where the model parameters are stored
            gender: specify the gender of the model, default is 'neutral'
            num_betas: the number of shape coeffients, default is 10
            num_pca_comps: the number of PCA components for the hand pose parameters, default is 6; setting the value as 0 will not use pca.
            flat_hand_mean: flag of using the flat hand mean, default is False
            ext: the extension of the SMPL model file, default is 'pkl'
            dtype: the data type of the model parameters, default is torch.float32
        """
        super(SMPLHLayer, self).__init__(
            model_path=model_path,
            gender=gender,
            num_betas=num_betas,
            ext=ext,
            **kwargs
        )
        
        self.num_pca_comps = num_pca_comps
        self.flat_hand_mean = flat_hand_mean
        self.hand_pose_dim = num_pca_comps if num_pca_comps > 0 else 3 * self.NUM_HAND_JOINTS

        if self.num_pca_comps > 0:
            left_hand_components = self.data_struct.hands_componentsl[:num_pca_comps]
            right_hand_components = self.data_struct.hands_componentsr[:num_pca_comps]
            self.register_buffer(
                'left_hand_components', to_tensor(to_numpy(left_hand_components), dtype=self.dtype)
            )
            self.register_buffer(
                'right_hand_components', to_tensor(to_numpy(right_hand_components), dtype=self.dtype)
            )
        
        if self.flat_hand_mean:
            left_hand_mean = np.zeros_like(self.data_struct.hands_meanl)
            right_hand_mean = np.zeros_like(self.data_struct.hands_meanr)
        else:
            left_hand_mean = self.data_struct.hands_meanl
            right_hand_mean = self.data_struct.hands_meanr
        self.register_buffer(
            'left_hand_mean', to_tensor(to_numpy(left_hand_mean), dtype=self.dtype)
        )
        self.register_buffer(
            'right_hand_mean', to_tensor(to_numpy(right_hand_mean), dtype=self.dtype)
        )

        pose_mean_tensor = self._create_mean_pose()
        self.register_buffer(
            'pose_mean', pose_mean_tensor
        )
    
    def _create_mean_pose(self):
        """ Create the mean pose of SMPLH model

        Return:
            A tensor representing the mean pose of SMPLH model
        """
        pose_mean = torch.cat([
            torch.zeros([3], dtype=self.dtype), # global orientation
            torch.zeros([self.NUM_BODY_JOINTS*3], dtype=self.dtype), # body pose
            self.left_hand_mean, # left hand pose
            self.right_hand_mean # right hand pose
        ], dim=0)

        return pose_mean

    @property
    def name(self):
        return "SMPLH"
    
    def forward(
        self,
        transl: Optional[torch.Tensor]=None,
        orient: Optional[torch.Tensor]=None,
        betas: Optional[torch.Tensor]=None,
        body_pose: Optional[torch.Tensor]=None,
        left_hand_pose: Optional[torch.Tensor]=None,
        right_hand_pose: Optional[torch.Tensor]=None,
        pose2rot: bool=True,
        apply_trans: bool=True,
        return_verts: bool=False,
        return_joints: bool=False,
        **kwargs,
    ) -> Union[SMPLHOutput, torch.Tensor]:
        """ Forward pass of SMPLH layer

        Args:
            transl: the translation, a tensor of shape <batch_size, 3>
            orient: the global orientation, it should be represented in axis-angle format, a tensor of shape <batch_size, 3>
            betas: the shape coeffients, torch.Tensor of shape <batch_size, num_betas>
            body_pose: the body pose parameters, it should be represented in axis-angle format, a tensor of shape <batch_size, J*3>
            left_hand_pose: the left hand pose parameters, it should be represented in axis-angle format, a tensor of shape <batch_size, 15*3>
            right_hand_pose: the right hand pose parameters, it should be represented in axis-angle format, a tensor of shape <batch_size, 15*3>
            pose2rot: flag of converting the pose parameters to rotation matrices, default is True
            apply_trans: flag of applying the translation, default is True
            return_verts: flag of **only** returning the vertices, default is False
            return_joints: flag of **only** returning the joints, default is False
            batch_size: batch size of smplh parameters, default is 1, if any of `[transl, orient, body_pose]` is given, this parameter will be ignored
        
        Return:
            This method returns a SMPLHOutput object or only returns a tensor of shape <batch_size, num_verts, 3> representing the SMPLH body mesh vertices
        """

        torch_kwargs = {'dtype': self.dtype, 'device': self.shapedirs.device}
        batch_size = self._get_batch_size([transl, orient, body_pose, left_hand_pose, right_hand_pose, betas], 
                                        kwargs.get('batch_size', 1))

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
            body_pose = torch.zeros(batch_size, self.NUM_BODY_JOINTS*3, **torch_kwargs)
        
        if left_hand_pose is None:
            left_hand_pose = torch.zeros(batch_size, self.hand_pose_dim, **torch_kwargs)
        
        if right_hand_pose is None:
            right_hand_pose = torch.zeros(batch_size, self.hand_pose_dim, **torch_kwargs)
        
        assert transl.shape[0] == orient.shape[0] == betas.shape[0] == body_pose.shape[0] == left_hand_pose.shape[0] == right_hand_pose.shape[0], f"[Shape Error] the batch size of the given parameters are not equal!"
        
        if self.num_pca_comps > 0:
            left_hand_pose = torch.einsum(
                'bi,ij->bj', [left_hand_pose, self.left_hand_components]
            )
            right_hand_pose = torch.einsum(
                'bi,ij->bj', [right_hand_pose, self.right_hand_components]
            )
        
        full_pose = torch.cat([orient, body_pose, left_hand_pose, right_hand_pose], dim=1)
        full_pose += self.pose_mean

        vertices, joints = lbs(betas, full_pose, self.v_template,
                               self.shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights, pose2rot=pose2rot)
        
        ## only return the vertices for saving computation
        if return_verts:
            if apply_trans:
                vertices = vertices + transl.unsqueeze(1)
            return vertices
        
        ## only return the joints for saving computation
        if return_joints:
            if apply_trans:
                joints = joints + transl.unsqueeze(1)
            return joints
        
        ## apply translation
        if apply_trans:
            joints = joints + transl.unsqueeze(1)
            vertices = vertices + transl.unsqueeze(1)
    
        return SMPLHOutput(
            vertices=vertices,
            faces=self.faces_tensor.clone(),
            joints=joints,
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
        )

class SMPLXLayer(SMPLHLayer):

    NUM_BODY_JOINTS = SMPLHLayer.NUM_BODY_JOINTS
    NUM_HAND_JOINTS = 15
    NUM_FACE_JOINTS = 3
    NUM_JOINTS = NUM_BODY_JOINTS + 2 * NUM_HAND_JOINTS + NUM_FACE_JOINTS
    EXPRESSION_SPACE_DIM = 100
    NECK_IDX = 12

    def __init__(
            self,
            model_path: Optional[str]=None,
            gender: str='neutral',
            num_betas: int=10,
            num_pca_comps: int=6,
            flat_hand_mean: bool=False,
            num_expression_coeffs: int=10,
            use_face_contour: bool=False,
            ext: str='npz',
            **kwargs,
        ) -> None:
        """ SMPLX layer constructor

        Args:
            model_path: The path of the SMPL model path or folder where the model parameters are stored
            gender: specify the gender of the model, default is 'neutral'
            num_betas: the number of shape coeffients, default is 10
            num_pca_comps: the number of PCA components for the hand pose parameters, default is 6; setting the value as 0 will not use pca.
            flat_hand_mean: flag of using the flat hand mean, default is False
            num_expression_coeffs: the number of expression coeffients, default is 10
            use_face_contour: flag of computing the keypoints that form the facial contour, default is False
            ext: the extension of the SMPL model file, default is 'pkl'
            dtype: the data type of the model parameters, default is torch.float32
        """
        super(SMPLXLayer, self).__init__(
            model_path=model_path,
            gender=gender,
            num_betas=num_betas,
            num_pca_comps=num_pca_comps,
            flat_hand_mean=flat_hand_mean,
            ext=ext,
            **kwargs
        )

        shapedirs = self.data_struct.shapedirs
        if shapedirs.shape[-1] < self.SHAPE_SPACE_DIM + self.EXPRESSION_SPACE_DIM:
            ## If the model does not contain enough shape and expression space, we will use the first 10 expression space
            ## Default: 10 shape space + 10 expression space
            num_expression_coeffs = min(num_expression_coeffs, 10)
            expr_start_idx = 10
            expr_end_idx = expr_start_idx + num_expression_coeffs
        else:
            num_expression_coeffs = min(num_expression_coeffs, self.EXPRESSION_SPACE_DIM)
            expr_start_idx = self.SHAPE_SPACE_DIM
            expr_end_idx = self.SHAPE_SPACE_DIM + num_expression_coeffs
        self.num_expression_coeffs = num_expression_coeffs
        self.use_face_contour = use_face_contour

        expr_dirs = self.data_struct.shapedirs[:, :, expr_start_idx:expr_end_idx]
        self.register_buffer(
            'expr_dirs', to_tensor(to_numpy(expr_dirs), dtype=self.dtype)
        )

        lmk_faces_idx = self.data_struct.lmk_faces_idx
        self.register_buffer(
            'lmk_faces_idx', to_tensor(to_numpy(lmk_faces_idx), dtype=torch.int64)
        )
        lmk_bary_coords = self.data_struct.lmk_bary_coords
        self.register_buffer(
            'lmk_bary_coords', to_tensor(to_numpy(lmk_bary_coords), dtype=self.dtype)
        )
        if self.use_face_contour:
            dynamic_lmk_faces_idx = self.data_struct.dynamic_lmk_faces_idx
            self.register_buffer(
                'dynamic_lmk_faces_idx', to_tensor(to_numpy(dynamic_lmk_faces_idx), dtype=torch.int64)
            )

            dynamic_lmk_bary_coords = self.data_struct.dynamic_lmk_bary_coords
            self.register_buffer(
                'dynamic_lmk_bary_coords', to_tensor(to_numpy(dynamic_lmk_bary_coords), dtype=self.dtype)
            )

            neck_kin_chain = find_joint_kin_chain(self.NECK_IDX, self.parents)
            self.register_buffer(
                'neck_kin_chain', to_tensor(to_numpy(neck_kin_chain), dtype=torch.int64)
            )
    
    def _create_mean_pose(self):
        """ Create the mean pose of SMPLX model

        Return:
            A tensor representing the mean pose of SMPLX model
        """
        mean_pose = torch.cat([
            torch.zeros([3], dtype=self.dtype), # global orientation
            torch.zeros([self.NUM_BODY_JOINTS*3], dtype=self.dtype), # body pose
            torch.zeros([3], dtype=self.dtype), # jaw pose
            torch.zeros([3], dtype=self.dtype), # left eye pose
            torch.zeros([3], dtype=self.dtype), # right eye pose
            self.left_hand_mean, # left hand pose
            self.right_hand_mean # right hand pose
        ], dim=0)

        return mean_pose
    
    @property
    def name(self) -> str:
        return "SMPLX"

    def forward(
            self,
            transl: Optional[torch.Tensor]=None,
            orient: Optional[torch.Tensor]=None,
            betas: Optional[torch.Tensor]=None,
            body_pose: Optional[torch.Tensor]=None,
            left_hand_pose: Optional[torch.Tensor]=None,
            right_hand_pose: Optional[torch.Tensor]=None,
            expression: Optional[Tensor]=None,
            jaw_pose: Optional[Tensor]=None,
            leye_pose: Optional[Tensor]=None,
            reye_pose: Optional[Tensor]=None,
            pose2rot: bool=True,
            apply_trans: bool=True,
            return_verts: bool=False,
            return_joints: bool=False,
            **kwargs
        ) -> Union[SMPLXOutput, torch.Tensor]:
        """ Forward pass of SMPLX layer

        Args:
            transl: the translation, a tensor of shape <batch_size, 3>
            orient: the global orientation, it should be represented in axis-angle format, a tensor of shape <batch_size, 3>
            betas: the shape coeffients, torch.Tensor of shape <batch_size, num_betas>
            body_pose: the body pose parameters, it should be represented in axis-angle format, a tensor of shape <batch_size, J*3>
            left_hand_pose: the left hand pose parameters, it should be represented in axis-angle format, a tensor of shape <batch_size, 15*3>
            right_hand_pose: the right hand pose parameters, it should be represented in axis-angle format, a tensor of shape <batch_size, 15*3>
            expression: the expression coeffients, torch.Tensor of shape <batch_size, num_expression_coeffs>
            jaw_pose: the jaw pose parameters, it should be represented in axis-angle format, a tensor of shape <batch_size, 3>
            leye_pose: the left eye pose parameters, it should be represented in axis-angle format, a tensor of shape <batch_size, 3>
            reye_pose: the right eye pose parameters, it should be represented in axis-angle format, a tensor of shape <batch_size, 3>
            pose2rot: flag of converting the pose parameters to rotation matrices, default is True
            apply_trans: flag of applying the translation, default is True
            return_verts: flag of **only** returning the vertices, default is False
            return_joints: flag of **only** returning the joints, default is False
            batch_size: batch size of smplx parameters, default is 1, if any of `[transl, orient, body_pose]` is given, this parameter will be ignored
        
        Return:
            This method returns a SMPLXOutput object or only returns a tensor of shape <batch_size, num_verts, 3> representing the SMPLX body mesh vertices
        """

        torch_kwargs = {'dtype': self.dtype, 'device': self.shapedirs.device}
        batch_size = self._get_batch_size([transl, orient, body_pose, left_hand_pose, right_hand_pose,
                                        expression, jaw_pose, leye_pose, reye_pose, betas], kwargs.get('batch_size', 1))

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
            body_pose = torch.zeros(batch_size, self.NUM_BODY_JOINTS*3, **torch_kwargs)
        
        if left_hand_pose is None:
            left_hand_pose = torch.zeros(batch_size, self.hand_pose_dim, **torch_kwargs)
        
        if right_hand_pose is None:
            right_hand_pose = torch.zeros(batch_size, self.hand_pose_dim, **torch_kwargs)
        
        if expression is None:
            expression = torch.zeros(batch_size, self.num_expression_coeffs, **torch_kwargs)
        
        if jaw_pose is None:
            jaw_pose = torch.zeros(batch_size, 3, **torch_kwargs)
        
        if leye_pose is None:
            leye_pose = torch.zeros(batch_size, 3, **torch_kwargs)
        
        if reye_pose is None:
            reye_pose = torch.zeros(batch_size, 3, **torch_kwargs)
        
        assert transl.shape[0] == orient.shape[0] == betas.shape[0] == body_pose.shape[0] == left_hand_pose.shape[0] == right_hand_pose.shape[0] == expression.shape[0] == jaw_pose.shape[0] == leye_pose.shape[0] == reye_pose.shape[0], f"[Shape Error] the batch size of the given parameters are not equal!"

        if self.num_pca_comps > 0:
            left_hand_pose = torch.einsum(
                'bi,ij->bj', [left_hand_pose, self.left_hand_components]
            )
            right_hand_pose = torch.einsum(
                'bi,ij->bj', [right_hand_pose, self.right_hand_components]
            )
        
        full_pose = torch.cat([
            orient.reshape(-1, 1, 3),
            body_pose.reshape(-1, self.NUM_BODY_JOINTS, 3),
            jaw_pose.reshape(-1, 1, 3),
            leye_pose.reshape(-1, 1, 3),
            reye_pose.reshape(-1, 1, 3),
            left_hand_pose.reshape(-1, self.NUM_HAND_JOINTS, 3),
            right_hand_pose.reshape(-1, self.NUM_HAND_JOINTS, 3)
        ], dim=1).reshape(-1, self.NUM_JOINTS * 3 + 3)
        full_pose += self.pose_mean

        shape_components = torch.cat([betas, expression], dim=-1)
        shapedirs = torch.cat([self.shapedirs, self.expr_dirs], dim=-1)

        vertices, joints = lbs(shape_components, full_pose, self.v_template,
                               shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights, pose2rot=pose2rot)
        
        ## only return the vertices for saving computation
        if return_verts:
            if apply_trans:
                vertices = vertices + transl.unsqueeze(1)
            return vertices

        ## only return the joints for saving computation
        if return_joints:
            if apply_trans:
                joints = joints + transl.unsqueeze(1)
            return joints
        
        ## compute the landmarks
        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0).expand(batch_size, -1).contiguous()
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).repeat(batch_size, 1, 1)
        if self.use_face_contour:
            lmk_idx_and_bcoords = find_dynamic_lmk_idx_and_bcoords(
                vertices, full_pose, self.dynamic_lmk_faces_idx,
                self.dynamic_lmk_bary_coords, self.neck_kin_chain, pose2rot=pose2rot
            )
            dyn_lmk_faces_idx, dyn_lmk_bary_coords = lmk_idx_and_bcoords

            lmk_faces_idx = torch.cat([lmk_faces_idx, dyn_lmk_faces_idx], dim=1)
            lmk_bary_coords = torch.cat([lmk_bary_coords.expand(batch_size, -1, -1), dyn_lmk_bary_coords], dim=1)
        
        landmarks = vertices2landmarks(vertices, self.faces_tensor, lmk_faces_idx, lmk_bary_coords)

        ## apply translation
        if apply_trans:
            joints = joints + transl.unsqueeze(1)
            vertices = vertices + transl.unsqueeze(1)
            landmarks = landmarks + transl.unsqueeze(1)
        
        return SMPLXOutput(
            vertices=vertices,
            faces=self.faces_tensor.clone(),
            joints=joints,
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            landmarks=landmarks,
        )
