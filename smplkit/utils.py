import torch
import numpy as np
from typing import Any, Union, Tuple, List, Dict, Optional
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

from .misc import ModelOutput
from .constants import *

## Some useful functions
def matrix_to_parameter(
        T: torch.Tensor,
        trans: torch.Tensor,
        orient: torch.Tensor,
        pelvis: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Convert vertice transformation matrix to smpl trans and orient parameter. The given T should be column matrices.

    As we use matrix_to_axis_angle in pytorch3d to convert the representations, the rotation angle of output orient may not be in [0, pi] strictly.

    Reference: https://www.dropbox.com/scl/fi/zkatuv5shs8d4tlwr8ecc/Change-parameters-to-new-coordinate-system.paper?dl=0&rlkey=lotq1sh6wzkmyttisc05h0in0

    Args:
        T: target transformation matrix, shape is <4, 4>, should be column matrices
        trans: origin trans of smpl parameters, shape is <N, 3>
        orient: origin orient of smpl parameters, shape is <N, 3>
        pelvis: origin pelvis, shape is <N, 3>
    
    Return:
        Transformed trans and orient parameters
    """
    R = T[0:3, 0:3]
    t = T[0:3, -1]

    pelvis = pelvis - trans
    trans = torch.matmul(R, (trans + pelvis).T).T - pelvis

    rot_mats = axis_angle_to_matrix(orient) # pytorch3d uses row matrices
    rot_mats = torch.matmul(R, rot_mats)
    orient = matrix_to_axis_angle(rot_mats)

    return trans + t, orient

def compute_orient(
    body: Union[torch.Tensor, ModelOutput],
):
    """ Compute the body orientation with given body joints
    """
    raise NotImplementedError

def compute_normal(
    body: Union[torch.Tensor, ModelOutput],
    faces: Optional[torch.Tensor]=None,
):
    """ Compute the body normal with given body vertices and faces
    """
    raise NotImplementedError

def compute_sdf(
    points: torch.Tensor,
    body: Union[torch.Tensor, ModelOutput],
    faces: Optional[torch.Tensor]=None,
):
    """ Signed Distance Function
    """
    raise NotImplementedError

## Some useful classes
class VertexSelector():

    @staticmethod
    def select_vertex(
        body: Union[torch.Tensor, np.ndarray, ModelOutput],
        index: Union[List, Dict],
    ):
        """ Select specific vertices with a given index

        Args:
            body: the output of a body model, can be a tensor or array or ModelOutput. \
                If it is tensor or array, the shape is <..., N, 3>, where N is the number of vertices
            index: the index of vertices to be selected, can be a list or dict. \
                If it is a list, each element is a index of vertices to be selected. \
                If it is a dict, the key is the vertex name and the value is the index of vertices to be selected.
                
        Return:
            The selected vertices tensor or array.
        """
        if isinstance(body, ModelOutput):
            verts = body.vertices
        else:
            verts = body
        
        if isinstance(index, list):
            index = index
        elif isinstance(index, dict):
            index = list(index.values())
        else:
            raise Exception('[Type Error] Index type not supported! Please use list, dict or tuple.')
        
        return verts[..., index, :]

    @staticmethod
    def contact_vertex(
        body: Union[torch.Tensor, np.ndarray, ModelOutput],
        parts: Optional[List]=None,
    ):
        """ Select contact vertices from a given body

        Args:
            body: the output of a body model, can be a tensor or array or ModelOutput. \
                If it is tensor or array, the shape is <..., N, 3>, where N is the number of vertices
            parts: the used parts for selecting contact vertices, default is None, which means all contact, \
                i.e., ['back', 'gluteus', 'L_Hand', 'L_Leg', 'R_Hand', 'R_Leg', 'thighs'], will be used.
        
        Return:
            The selected contact vertices tensor or array.
        """
        if isinstance(body, ModelOutput):
            verts = body.vertices
        else:
            verts = body
        
        parts = CONTACT_PART_NAME if parts is None else parts

        index = []
        if verts.shape[-2] == VERTEX_NUM.SMPL: # SMPL and SMPL+H have same body vertex number
            raise NotImplementedError
            for part in parts:
                index += CONTACT_VERTEX_IDS.SMPL[part]['verts_ind']
        elif verts.shape[-2] == VERTEX_NUM.SMPLX:
            for part in parts:
                index += CONTACT_VERTEX_IDS.SMPLX[part]['verts_ind']
        else:
            raise Exception('[Length Error] The number of vertex is not supported! Please input an array with 6890 vertices for SMPL(+H) or 10475 for SMPL-X.')
        
        return VertexSelector.select_vertex(verts, index)
        
    @staticmethod
    def key_vertex(
        body: Union[torch.Tensor, np.ndarray, ModelOutput],
        index: Optional[Union[List, Dict, Tuple]]=None
    ):
        """ Select key vertices from a given body

        In smplx https://github.com/vchoutas/smplx/blob/main/smplx/vertex_joint_selector.py, the key vertices are selected by VertexJointSelector and are concated with the joints as a output.
        In our implementation, we provide this utility function to select key vertices from a given body, which is more flexible.

        Args:
            body: the output a body model, can be a tensor or array or ModelOutput. \
                If it is tensor or array, the shape is <..., N, 3>, where N is the number of vertices
            index: the index of key vertices, default is None, which means all key vertices will be selected with default KEY_VERTICES_INDEX
        
        Return:
            The selected key vertices tensor or array.
        """
        if isinstance(body, ModelOutput):
            verts = body.vertices
        else:
            verts = body
        
        nverts = verts.shape[-2]
        if nverts == VERTEX_NUM.SMPL:
            if index is None:
                index = KEY_VERTEX_IDS.SMPL
        elif nverts == VERTEX_NUM.SMPLX:
            if index is None:
                index = KEY_VERTEX_IDS.SMPLX
        else:
            raise Exception('[Length Error] The number of vertex is not supported! Please input an array with 6890 vertices for SMPL(+H) or 10475 for SMPL-X.')

        return VertexSelector.select_vertex(verts, index)

class JointSelector():

    @staticmethod
    def select_joint(
        body: Union[torch.Tensor, np.ndarray, ModelOutput],
        index: Union[List, Dict]
    ):
        """ Select specific joints with a given index

        Args:
            body: the output of a body model, can be a tensor or array or ModelOutput. \
                If it is a tensor or array, the shape is <..., J, 3>, where J is the number of joints
            index: the index of vertices to be selected, can be a list or dict. \
                If it is a list, each element is a index of vertices to be selected. \
                If it is a dict, the key is the vertex name and the value is the index of vertices to be selected.
        
        Return:
            The selected joints tensor or array.
        """
        if isinstance(body, ModelOutput):
            joints = body.joints
        else:
            joints = body
        
        if joints.shape[-2] == JOINTS_NUM.SMPL + 1:
            joints_name = JOINTS_NAME.SMPL
        elif joints.shape[-2] == JOINTS_NUM.SMPLH + 1:
            joints_name = JOINTS_NAME.SMPLH
        elif joints.shape[-2] == JOINTS_NUM.SMPLX + 1:
            joints_name = JOINTS_NAME.SMPLX
        else:
            raise Exception('[Length Error] The number of joints is not supported! Please input an array with 23+1 joints for SMPL, 51+1 for SMPL+H or 54+1 for SMPL-X.')
        
        if isinstance(index, list):
            if isinstance(index[0], str):
                index = [joints_name.index(n) for n in index]
            elif isinstance(index[0], int):
                index = index
            else:
                raise Exception('[Type Error] Item type in index not supported! Please use str or int.')
        elif isinstance(index, dict):
            index =  list(index.values())
        else:
            raise Exception('[Type Error] Index type not supported! Please use list, dict or tuple.')
        
        return joints[..., index, :]

def _singleton(cls):
    _instance = {}

    def inner(*args, **kwargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]
    return inner

@_singleton
class BodyModel():
    def __init__(self, model_type: str, *args, **kwargs):
        """ Create a singleton body model. """
        self.reset(model_type, *args, **kwargs)
    
    def reset(self, model_type: Optional[str]=None, *args, **kwargs):
        """ Reset the body model. """
        if model_type is None or model_type == '':
            assert self.model is not None, '[Value Error] Body model type not specified! Please use smpl, smplh or smplx.'
            model_type = self.model.name
        
        if model_type.lower() == 'smpl':
            from .layers import SMPLLayer as SMPL
            self.model = SMPL(*args, **kwargs)
        elif model_type.lower() == 'smplh':
            from .layers import SMPLHLayer as SMPLH
            self.model = SMPLH(*args, **kwargs)
        elif model_type.lower() == 'smplx':
            from .layers import SMPLXLayer as SMPLX
            self.model = SMPLX(*args, **kwargs)
        else:
            raise Exception('[Value Error] Body model type not supported! Please use smpl, smplh or smplx.')
    
    def run(self, *args, **kwargs):
        """ Run the body model to output body meshes. """
        return self.model(*args, **kwargs)
    
    def to(self, **kwargs):
        """ Move the body model to a specific device or dtype. """
        return self.model.to(**kwargs)
