"""Source code: https://github.com/palver7/CFLPytorch"""

import math
import torch
import argparse
from torch import nn, Tensor
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair
from torch.jit.annotations import Optional, Tuple


def offcalc(batchsize=4):
    # type: (Tensor, Tensor, Tensor, Optional[Tensor], Tuple[int, int], Tuple[int, int], Tuple[int, int]) -> Tensor
    """
    Performs Equirectangular Convolution, described in Corners for Layout : End to End Layout Recovery from 360 Images

    Arguments:
        input (Tensor[batch_size, in_channels, in_height, in_width]): input tensor
        weight (Tensor[out_channels, in_channels // groups, kernel_height, kernel_width]):
            convolution weights, split into groups of size (in_channels // groups)
        bias (Tensor[out_channels]): optional bias of shape (out_channels,). Default: None
        stride (int or Tuple[int, int]): distance between convolution centers. Default: 1
        padding (int or Tuple[int, int]): height/width of padding of zeroes around
            each image. Default: 0
        dilation (int or Tuple[int, int]): the spacing between kernel elements. Default: 1

    Returns:
        output (Tensor[batch_sz, out_channels, out_h, out_w]): result of convolution


    Examples::
        >>> input = torch.rand(1, 3, 10, 10)
        >>> kh, kw = 3, 3
        >>> weight = torch.rand(5, 3, kh, kw)
        >>> # offset should have the same spatial size as the output
        >>> # of the convolution. In this case, for an input of 10, stride of 1
        >>> # and kernel size of 3, without padding, the output size is 8
        >>> offset = torch.rand(5, 2 * kh * kw, 8, 8)
        >>> out = deform_conv2d(input, offset, weight)
        >>> print(out.shape)
        >>> # returns
        >>>  torch.Size([1, 5, 8, 8])
    """

    def rotation_matrix(axis, theta):
        """ code by cfernandez and jmfacil """
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        axis = torch.as_tensor(axis, device='cpu', dtype=torch.float)
        axis = axis / math.sqrt(torch.dot(axis, axis))
        a = math.cos(theta / 2.0)
        b, c, d = -axis * math.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        ROT = torch.tensor([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                         [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                         [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]], device='cpu', dtype=torch.float)
        return ROT

    def equi_coord(pano_W, pano_H, k_W, k_H, u, v):
        """ code by cfernandez and jmfacil """
        fov_w = k_W * math.radians(360./float(pano_W))
        focal = (float(k_W)/2) / math.tan(fov_w/2)
        c_x = 0
        c_y = 0

        u_r, v_r = u, v
        u_r, v_r = u_r-float(pano_W)/2., v_r-float(pano_H)/2.
        phi, theta = u_r/(pano_W) * (math.pi) * 2, -v_r/(pano_H) * (math.pi)

        ROT = rotation_matrix((0, 1, 0), phi)
        ROT = torch.matmul(ROT, rotation_matrix((1, 0, 0), theta))

        h_range = torch.tensor(range(k_H), device='cpu', dtype=torch.float)
        w_range = torch.tensor(range(k_W,), device='cpu', dtype=torch.float)
        w_ones = (torch.ones(k_W, device='cpu', dtype=torch.float))
        h_ones = (torch.ones(k_H, device='cpu', dtype=torch.float))
        h_grid = torch.matmul(torch.unsqueeze(h_range, -1), torch.unsqueeze(w_ones, 0))+0.5-float(k_H)/2
        w_grid = torch.matmul(torch.unsqueeze(h_ones, -1), torch.unsqueeze(w_range, 0))+0.5-float(k_W)/2

        K = torch.tensor([[focal, 0, c_x], [0, focal, c_y], [0., 0., 1.]], device='cpu', dtype=torch.float)
        inv_K = torch.inverse(K)
        rays = torch.stack([w_grid, h_grid, torch.ones(h_grid.shape, device='cpu', dtype=torch.float)],0)
        rays = torch.matmul(inv_K, rays.reshape(3, k_H*k_W))
        rays /= torch.norm(rays, dim=0, keepdim=True)
        rays = torch.matmul(ROT, rays)
        rays = rays.reshape(3, k_H, k_W)

        phi = torch.atan2(rays[0, ...], rays[2, ...])
        theta = torch.asin(torch.clamp(rays[1, ...], -1, 1))

        x = (pano_W)/(2.*math.pi)*phi + float(pano_W)/2.
        y = (pano_H)/(math.pi)*theta + float(pano_H)/2.

        roi_y = h_grid+v_r + float(pano_H)/2.
        roi_x = w_grid+u_r + float(pano_W)/2.

        new_roi_y = (y)
        new_roi_x = (x)

        offsets_x = (new_roi_x - roi_x)
        offsets_y = (new_roi_y - roi_y)

        return offsets_x, offsets_y

    def distortion_aware_map(pano_W, pano_H, k_W, k_H, s_width=1, s_height=1, bs=4):
        """ code by cfernandez and jmfacil """
        offset = torch.zeros(2*k_H*k_W, pano_H, pano_W, device='cpu', dtype=torch.float)
        for v in range(0, pano_H, s_height):
            for u in range(0, pano_W, s_width):
                offsets_x, offsets_y = equi_coord(pano_W, pano_H, k_W, k_H, u, v)
                offsets = torch.cat((torch.unsqueeze(offsets_y, -1), torch.unsqueeze(offsets_x, -1)), dim=-1)
                total_offsets = offsets.flatten()
                offset[:, v, u] = total_offsets

        offset = torch.unsqueeze(offset, 0)
        offset = torch.cat([offset for _ in range(bs)], dim=0)
        offset.requires_grad_(False)
        return offset

    layerdict = {
                0:((256,512),(3,3),(1,1)),
                1:((256,512),(3,3),(1,1)),
                2:((128,256),(7,7),(2,2)),
                3:((64,128),(3,3),(1,1)),
                4:((64,128),(3,3),(1,1)),
                5:((64,128),(3,3),(1,1)),
                6:((64,128),(3,3),(1,1)),
                7:((32,64),(3,3),(2,2)),
                8:((32,64),(3,3),(1,1)),
                9:((32,64),(3,3),(1,1)),
                10:((32,64),(3,3),(1,1)),
                11:((16,32),(3,3),(2,2)),
                12:((16,32),(3,3),(1,1)),
                13:((16,32),(3,3),(1,1)),
                14:((16,32),(3,3),(1,1)),
                15:((8,16),(3,3),(2,2)),
                16:((8,16),(3,3),(1,1)),
                17:((8,16),(3,3),(1,1)),
                18:((8,16),(3,3),(1,1)),
                19:((8,16),(1,1),(1,1)),
                20:((16,32),(1,1),(1,1)),
                21:((16,32),(3,3),(1,1)),
                22:((32,64),(1,1),(1,1)),
                23:((32,64),(3,3),(1,1)),
                24:((64,128),(1,1),(1,1)),
                25:((64,128),(3,3),(1,1)),
                26:((128,256),(1,1),(1,1)),
                27:((128,256),(3,3),(1,1)),
                28:((256,512),(3,3),(1,1)),
                29:((256,512),(1,1),(1,1))
                }

    
    paramlist =[((256,512),(3,3),(1,1)),
                ((256,512),(1,1),(1,1)),
                ((128,256),(7,7),(2,2)),
                ((128,256),(1,1),(1,1)),
                ((128,256),(3,3),(1,1)),
                ((64,128),(3,3),(1,1)),
                ((64,128),(1,1),(1,1)),
                ((32,64),(3,3),(2,2)),
                ((32,64),(3,3),(1,1)),
                ((32,64),(1,1),(1,1)),
                ((16,32),(3,3),(2,2)),
                ((16,32),(3,3),(1,1)),
                ((16,32),(1,1),(1,1)),
                ((8,16),(3,3),(2,2)),
                ((8,16),(3,3),(1,1)),
                ((8,16),(1,1),(1,1))
                ]
    offsetdict={}
    for key in paramlist:
        offsetdict[key] = distortion_aware_map(key[0][1],key[0][0],key[1][1],key[1][0],
        s_width=key[2][1],s_height=key[2][0],bs=batchsize)
    return layerdict, offsetdict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=2, metavar='B',
                        help='size of minibatches (default: 4)')
    parser.add_argument('--mode', type=str, default='train', metavar='M',
                        help='append to the saved dict file either train or test (default: train)')
    layerdict, offsetdict = offcalc(batchsize=2)
    torch.save(layerdict,'./layer_256x512lll.pt')
    torch.save(offsetdict,'./offset_256x512lll.pt')
