#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH, SH2RGB
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

class GaussianModel: 
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def _set_num_models(self, n_models):
        n_models = int(n_models)
        if n_models < 1:
            raise ValueError(f"num_models must be at least 1, got {n_models}.")
        self.n_models = n_models
        self.M = min(2, self.n_models)

    def __init__(self, dataset): 
        self.active_sh_degree = 0
        self.max_sh_degree = dataset.sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scalar = 0
        self.tmp = 0.2
        self.iteration = 0
        self._feat_unc = torch.empty(0)
        self.model_id = 0

        self._set_num_models(getattr(dataset, "num_models", 6))
        self.pri_std = -6
        self.pri_width = 0.1

        self.pri_opacity_std = 1.85
        self.pri_opacity_mean = 2

        self.lr_scales = [0.1, 0.1, 0.1]

        self.spawn_percent_base = dataset.spawn_percent_base
        self.spawn_min_opacity = dataset.spawn_min_opacity

        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scalar,
            self.n_models,
        )
    
    def restore(self, model_args, training_args):
        if len(model_args) == 13:
            (self.active_sh_degree, 
            self._xyz, 
            self._features_dc, 
            self._features_rest,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self.max_radii2D, 
            xyz_gradient_accum, 
            denom,
            opt_dict, 
            self.spatial_lr_scalar,
            n_models) = model_args
            self._set_num_models(n_models)
        elif len(model_args) == 12:
            (self.active_sh_degree, 
            self._xyz, 
            self._features_dc, 
            self._features_rest,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self.max_radii2D, 
            xyz_gradient_accum, 
            denom,
            opt_dict, 
            self.spatial_lr_scalar) = model_args
        else:
            raise ValueError(
                f"Unsupported GaussianModel checkpoint format with {len(model_args)} entries."
            )
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    def _active_member_ids(self):
        start_id = int(self.model_id) % self.n_models
        member_offsets = torch.arange(self.M, device=self._xyz.device)
        return (start_id + member_offsets) % self.n_models

    def _stable_member_uniform(self, reference_tensor, stream_id):
        if reference_tensor.numel() == 0:
            return torch.empty_like(reference_tensor)

        flat_index = torch.arange(
            reference_tensor.numel(), device=reference_tensor.device, dtype=torch.float32
        )
        seed = float((int(self.model_id) % self.n_models) + 1 + 104729 * stream_id)
        values = torch.sin(flat_index * 12.9898 + seed * 78.233) * 43758.5453
        values = values - torch.floor(values)
        return values.reshape(reference_tensor.shape).to(reference_tensor.dtype)

    def _stable_member_normal(self, reference_tensor, stream_id):
        uniform = self._stable_member_uniform(reference_tensor, stream_id)
        uniform = uniform.to(torch.float32).clamp_(1e-6, 1.0 - 1e-6)
        normal = np.sqrt(2.0) * torch.erfinv(2.0 * uniform - 1.0)
        return normal.to(reference_tensor.dtype)

    @property
    def get_scaling(self): 
        scale = self.compute_scal()
        return scale

    def compute_scal(self): 
        scal = self._scaling
        sample_model_ids = self._active_member_ids()

        width = self.offsets["_scaling_offset"][...,sample_model_ids].mean(dim=-1)
        width = torch.nn.functional.softplus(width).clamp_(1e-2, 1e2)
        left = self.offsets["_scaling_offset"][...,sample_model_ids+self.n_models].mean(dim=-1)

        offset_scal = (width) * self._stable_member_uniform(scal, stream_id=1) + left
        offset_scal = self.mr_list*offset_scal+(1-self.mr_list)* torch.ones_like(offset_scal).cuda().requires_grad_(True)

        scal = scal * offset_scal
        scal = self.scaling_activation(scal)
        return scal

    def compute_kl_uniform_scal(self): 
        sample_model_ids = torch.randperm(self.n_models)[:self.M].cuda().requires_grad_(False).detach()
        width = self.offsets["_scaling_offset"][...,sample_model_ids].mean(dim=-1)
        width = torch.nn.functional.softplus(width).clamp_(1e-2, 1e2)
        left = self.offsets["_scaling_offset"][...,sample_model_ids+self.n_models].mean(dim=-1)

        pri_left = 1-self.pri_width

        right = left + width
        prior_right = 1

        kl = torch.abs(left - pri_left) + torch.abs(right - prior_right)

        return kl.mean()

    @property
    def get_rotation(self):
        return self.compute_rotation()
    def compute_rotation(self): 
        r = self.rotation_activation(self._rotation)

        return r

    @property
    def get_xyz(self):
        xyz = self.compute_xyz()
        return xyz
    def compute_xyz(self):
        sample_model_ids = self._active_member_ids()
        xyz = self._xyz

        std = self.offsets["_xyz_offset"][..., sample_model_ids].mean(dim=-1)
        std = torch.nn.functional.softplus(std)

        mean = self.offsets["_xyz_offset"][..., sample_model_ids+self.n_models].mean(dim=-1)

        offset = self._stable_member_normal(xyz, stream_id=2)
        offset = offset*std+mean

        xyz = xyz + self.mr_list*offset
        return xyz

    def compute_kl_xyz(self): 
        sample_model_ids = torch.randperm(self.n_models)[:self.M].cuda().requires_grad_(False).detach()
        std = self.offsets["_xyz_offset"][..., sample_model_ids].mean(dim=-1)
        std = torch.nn.functional.softplus(std)
        mean = self.offsets["_xyz_offset"][..., sample_model_ids+self.n_models].mean(dim=-1)

        pri_std = torch.nn.functional.softplus(torch.tensor(self.pri_std).float()).item()
        pri_mean, pri_std = torch.zeros_like(mean), pri_std*torch.ones_like(std)

        log_sigma_pri, log_sigma_post = torch.log(pri_std), torch.log(std)
        kl = log_sigma_pri - log_sigma_post + \
        (torch.exp(log_sigma_post)**2 + (mean-pri_mean)**2)/(2*torch.exp(log_sigma_pri)**2) - 0.5
        return kl.mean()

    @property
    def get_features(self):
        return self.compute_features()
    def compute_features(self): 
        features_dc = self._features_dc
        features_rest = self._features_rest

        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self): 
        opacity = self.compute_opacity()
        sample_model_ids = self._active_member_ids()
        std = self.offsets["_opacity_offset"][..., sample_model_ids].mean(dim=-1)
        std = torch.nn.functional.softplus(std)
        mean = self.offsets["_opacity_offset"][..., sample_model_ids+self.n_models].mean(dim=-1)

        p_logit = (std*self._stable_member_normal(opacity, stream_id=3) + mean)
        offset_opc = torch.sigmoid(p_logit / self.tmp)
        offset_opc = self.mr_list*offset_opc+(1-self.mr_list)*torch.ones_like(offset_opc).cuda().requires_grad_(True)

        opacity = opacity * offset_opc

        return opacity

    def compute_kl_opacity(self):
        sample_model_ids = torch.randperm(self.n_models)[:self.M].cuda().requires_grad_(False).detach()
        std = self.offsets["_opacity_offset"][..., sample_model_ids].mean(dim=-1)
        std = torch.nn.functional.softplus(std)
        mean = self.offsets["_opacity_offset"][..., sample_model_ids+self.n_models].mean(dim=-1)

        pri_std = torch.nn.functional.softplus(torch.tensor(self.pri_opacity_std).float()).item()
        pri_mean = self.pri_opacity_mean

        pri_mean, pri_std = pri_mean * torch.ones_like(mean), pri_std * torch.ones_like(std)

        log_sigma_pri, log_sigma_post = torch.log(pri_std), torch.log(std)
        kl = log_sigma_pri - log_sigma_post + \
        (torch.exp(log_sigma_post)**2 + (mean-pri_mean)**2)/(2*torch.exp(log_sigma_pri)**2) - 0.5
        return kl.mean()


    def compute_opacity(self):
        x = self.opacity_activation(self._opacity)
        return x

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def init_offset(self): 
        _xyz_offset = torch.zeros([self._xyz.shape[0], 3, self.n_models*2])
        _xyz_offset[..., :self.n_models] = self.pri_std
        _xyz_offset = nn.Parameter(_xyz_offset.requires_grad_(True).cuda())

        _scaling_offset = torch.zeros([self._xyz.shape[0], 3, self.n_models*2])

        _scaling_offset[..., :self.n_models] = torch.log(torch.exp(1/torch.tensor(self.n_models))-1).item()
        _scaling_offset[..., self.n_models:] = 1-self.pri_width
        _scaling_offset = nn.Parameter(_scaling_offset.requires_grad_(True).cuda())

        _opacity_offset = torch.zeros([self._xyz.shape[0], 1, self.n_models*2])
        _opacity_offset[..., :self.n_models] = self.pri_opacity_std
        _opacity_offset[..., self.n_models:] = self.pri_opacity_mean
        _opacity_offset = nn.Parameter(_opacity_offset.requires_grad_(True).cuda())
        offsets = [
            {"_xyz_offset": _xyz_offset},
            {"_scaling_offset": _scaling_offset}, 
            {"_opacity_offset": _opacity_offset}
        ]

        lr_scales = self.lr_scales
        self.offsets = {}

        for i in range(len(offsets)): 
            if lr_scales[i] != 0.0: 
                self.offsets[list(offsets[i].keys())[0]] = list(offsets[i].values())[0]

        self.mr_list = torch.zeros([self._xyz.shape[0], 1]).cuda().requires_grad_(False)

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scalar : float):
        self.spatial_lr_scalar = spatial_lr_scalar
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scalars = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scalars.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.init_offset()
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scalar, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}, 
        ]

        lr_scales = self.lr_scales
        l_offset = []

        _l_offset = [
            {'params': [], 'lr': lr_scales[0]*training_args.position_lr_init * self.spatial_lr_scalar, "name": "_xyz_offset"}, 
            {'params': [], 'lr': lr_scales[1]*training_args.opacity_lr, "name": "_scaling_offset"}, 
            {'params': [], 'lr': lr_scales[2]*training_args.rotation_lr, "name": "_opacity_offset"}, 
        ]

        j=0
        for i in range(len(_l_offset)): 
            if lr_scales[i] != 0.0: 
                _l_offset[i]['params'] = [self.offsets[list(self.offsets.keys())[j]]]
                l_offset += [_l_offset[i]]
                j+=1
        l = l + l_offset

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scalar,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scalar,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)


    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self, with_offsets=True):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scalar_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))

        if not with_offsets:
            return l

        for i in range(self.offsets["_xyz_offset"].shape[1]*self.offsets["_xyz_offset"].shape[2]):
            l.append('xyz_offset_{}'.format(i))
        for i in range(self.offsets["_scaling_offset"].shape[1]*self.offsets["_scaling_offset"].shape[2]):
            l.append('scaling_offset_{}'.format(i))
        for i in range(self.offsets["_opacity_offset"].shape[1]*self.offsets["_opacity_offset"].shape[2]):
            l.append('opacity_offset_{}'.format(i))

        l.append('mr')

        return l

    def save_ply(self, path, with_offsets=True): 
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(with_offsets=with_offsets)]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)

        if with_offsets: 
            _xyz_offset = self.offsets["_xyz_offset"].detach().flatten(start_dim=1).contiguous().cpu().numpy()
            _scaling_offset = self.offsets["_scaling_offset"].detach().flatten(start_dim=1).contiguous().cpu().numpy()
            _opacity_offset = self.offsets["_opacity_offset"].detach().flatten(start_dim=1).contiguous().cpu().numpy()

            attributes = np.concatenate((attributes, _xyz_offset, _scaling_offset, _opacity_offset), axis=1)
            attributes = np.concatenate((attributes, self.mr_list.detach().cpu().numpy()), axis=1)

        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scalar_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scalars = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scalars[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scalars, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.init_offset()

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.active_sh_degree = self.max_sh_degree


        xyz_offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("xyz_offset")]
        xyz_offset_names = sorted(xyz_offset_names, key = lambda x: int(x.split('_')[-1]))
        xyz_offset = np.zeros((xyz.shape[0], len(xyz_offset_names)))
        for idx, attr_name in enumerate(xyz_offset_names):
            xyz_offset[:, idx] = np.asarray(plydata.elements[0][attr_name])
        xyz_offset = xyz_offset.reshape((xyz_offset.shape[0], 3, len(xyz_offset_names) // 3))

        scaling_offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scaling_offset")]
        scaling_offset_names = sorted(scaling_offset_names, key = lambda x: int(x.split('_')[-1]))
        scaling_offset = np.zeros((xyz.shape[0], len(scaling_offset_names)))
        for idx, attr_name in enumerate(scaling_offset_names):
            scaling_offset[:, idx] = np.asarray(plydata.elements[0][attr_name])
        scaling_offset = scaling_offset.reshape((scaling_offset.shape[0], 3, len(scaling_offset_names) // 3))

        opacity_offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("opacity_offset")]
        opacity_offset_names = sorted(opacity_offset_names, key = lambda x: int(x.split('_')[-1]))
        opacity_offset = np.zeros((xyz.shape[0], len(opacity_offset_names)))
        for idx, attr_name in enumerate(opacity_offset_names):
            opacity_offset[:, idx] = np.asarray(plydata.elements[0][attr_name])
        opacity_offset = opacity_offset.reshape((opacity_offset.shape[0], 1, len(opacity_offset_names)))

        loaded_n_models = xyz_offset.shape[2] // 2
        self._set_num_models(loaded_n_models)

        self.offsets["_xyz_offset"] = nn.Parameter(torch.tensor(xyz_offset, dtype=torch.float, device="cuda").requires_grad_(True))
        self.offsets["_scaling_offset"] = nn.Parameter(torch.tensor(scaling_offset, dtype=torch.float, device="cuda").requires_grad_(True))
        self.offsets["_opacity_offset"] = nn.Parameter(torch.tensor(opacity_offset, dtype=torch.float, device="cuda").requires_grad_(True))

        self.mr_list = torch.tensor(np.asarray(plydata.elements[0]["mr"])[..., np.newaxis]).float().cuda().requires_grad_(False)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        for name in self.offsets.keys(): 
            self.offsets[name] = optimizable_tensors[name]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_offset):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        d = {**d, **new_offset}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        for name in self.offsets.keys(): 
            self.offsets[name] = optimizable_tensors[name]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def spawn(self, extent): 
        percent_base = self.spawn_percent_base
        min_opacity = self.spawn_min_opacity

        mr_mask = torch.norm(self.get_scaling, dim=1) > percent_base * extent
        self.mr_list = mr_mask.int()[...,None]

        transparent_mask = (self.get_opacity < min_opacity)[:,0]
        self.mr_list[transparent_mask] = 0

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)


        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        new_offset = {}        
        for name in self.offsets.keys(): 
            n_dim = len(self.offsets[name].shape)
            new_shape = [1 for i in range(n_dim)]
            new_shape[0] = N
            new_offset[name] = self.offsets[name][selected_pts_mask, ...].repeat(*new_shape)

        self.mr_list = torch.cat([self.mr_list, torch.ones_like(self.mr_list[selected_pts_mask].repeat(N,1))], dim=0)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_offset)

        prune_filter = torch.cat([selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)])

        self.mr_list = self.mr_list[~prune_filter]
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_offset = {}        
        for name in self.offsets.keys(): 
            new_offset[name] = self.offsets[name][selected_pts_mask, ...]

        self.mr_list = torch.cat([self.mr_list, self.mr_list[selected_pts_mask]], dim=0)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_offset)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, ):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        self.mr_list = self.mr_list[~prune_mask]
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        vp_update_filter = update_filter
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[vp_update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
