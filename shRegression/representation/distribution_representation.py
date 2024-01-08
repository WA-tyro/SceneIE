import cv2
import numpy as np
from PIL import Image
import pickle
import os.path
import imageio
import vtk
# from vtk.util import numpy_support
import vtkmodules.all as vtk
import torch
# import detect_util
import util
import matplotlib.pyplot as plt
from scipy.special import lpmv, factorial
from tqdm import tqdm

imageio.plugins.freeimage.download()


def tonemapping(im):
    power_im = np.power(im, 1 / 2.4)
    # print (np.amax(power_im))
    non_zero = power_im > 0
    if non_zero.any():
        r_percentile = np.percentile(power_im[non_zero], 99)
    else:
        r_percentile = np.percentile(power_im, 99)
    alpha = 0.8 / (r_percentile + 1e-10)
    tonemapped_im = np.multiply(alpha, power_im)

    tonemapped_im = np.clip(tonemapped_im, 0, 1)
    return tonemapped_im

def rgb_to_intenisty(rgb):
    intensity = 0.3 * rgb[..., 0] + 0.59 * rgb[..., 1] + 0.11 * rgb[..., 2]
    return intensity

def polar_to_cartesian(phi_theta):
    phi, theta = phi_theta
    # sin(theta) * cos(phi)
    x = np.sin(theta) * np.cos(phi)
    # sin(theta) * sin(phi)
    y = np.sin(theta) * np.sin(phi)
    # cos(theta)
    z = np.cos(theta)
    return np.array([x, y, z])

def normalize_2_unit_sphere(pts):
    num_pts = pts.GetNumberOfPoints()
    # print("we have #{} pts".format(num_pts))
    for i in range(num_pts):
        tmp = list(pts.GetPoint(i))
        n = vtk.vtkMath.Normalize(tmp)
        pts.SetPoint(i, tmp)

def polyhedron(subdivide=1):

    icosa = vtk.vtkPlatonicSolidSource()
    icosa.SetSolidTypeToIcosahedron()
    icosa.Update()
    subdivided_sphere = icosa.GetOutput()

    for i in range(subdivide):
        linear = vtk.vtkLinearSubdivisionFilter()
        linear.SetInputData(subdivided_sphere)
        linear.SetNumberOfSubdivisions(1)
        linear.Update()
        subdivided_sphere = linear.GetOutput()
        normalize_2_unit_sphere(subdivided_sphere.GetPoints())
        subdivided_sphere.Modified()

    # if save_directions:
    transform = vtk.vtkSphericalTransform()
    transform = transform.GetInverse()
    pts = subdivided_sphere.GetPoints()
    pts_spherical = vtk.vtkPoints()
    transform.TransformPoints(pts, pts_spherical)

    pts_arr = vtk.vtk_to_numpy(pts.GetData())
    # print (as_numpy.shape)
    return pts_arr

class extract_mesh():
    def __init__(self, h=128, w=256, ln=128):
        self.h, self.w = h, w
        steradian = np.linspace(0, h, num=h, endpoint=False) + 0.5
        steradian = np.sin(steradian / h * np.pi)
        steradian = np.tile(steradian.transpose(), (w, 1))
        steradian = steradian.transpose()
        self.steradian = steradian[..., np.newaxis]

        y_ = np.linspace(0, np.pi, num=h)  # + np.pi / h
        x_ = np.linspace(0, 2 * np.pi, num=w)  # + np.pi * 2 / w
        X, Y = np.meshgrid(x_, y_)
        Y = Y.reshape((-1, 1))
        X = X.reshape((-1, 1))
        phi_theta = np.stack((X, Y), axis=1)
        xyz = util.polar_to_cartesian(phi_theta)
        xyz = xyz.reshape((h, w, 3))  # 128, 256, 3
        xyz = np.expand_dims(xyz, axis=2)
        self.xyz = np.repeat(xyz, ln, axis=2)
        self.anchors = util.sphere_points(ln)

        dis_mat = np.linalg.norm((self.xyz - self.anchors), axis=-1)
        self.idx = np.argsort(dis_mat, axis=-1)[:, :, 0]
        self.ln, _ = self.anchors.shape

    def compute(self, hdr):

        hdr = self.steradian * hdr
        hdr_intensity = 0.3 * hdr[..., 0] + 0.59 * hdr[..., 1] + 0.11 * hdr[..., 2]
        max_intensity_ind = np.unravel_index(np.argmax(hdr_intensity, axis=None), hdr_intensity.shape)
        max_intensity = hdr_intensity[max_intensity_ind]
        map = hdr_intensity > (max_intensity * 0.05)
        map = np.expand_dims(map, axis=-1)
        light = hdr * map
        remain = hdr * (1 - map)

        ambient = remain.sum(axis=(0, 1))    #mean(axis=0).mean(axis=0)
        anchors = np.zeros((self.ln, 3))

        for i in range(self.ln):
            mask = self.idx == i
            mask = np.expand_dims(mask, -1)
            anchors[i] = (light * mask).sum(axis=(0, 1))

        anchors_engergy = 0.3 * anchors[..., 0] + 0.59 * anchors[..., 1] + 0.11 * anchors[..., 2]
        distribution = anchors_engergy / anchors_engergy.sum()
        anchors_rgb = anchors.sum(0)   # energy
        intensity = np.linalg.norm(anchors_rgb)
        rgb_ratio = anchors_rgb / intensity
        # distribution = anchors / intensity
        #
        # parametric_lights = {"distribution": distribution,
        #                      'intensity': intensity,
        #                      'rgb_ratio': rgb_ratio,
        #                      'ambient': ambient}

        sh = util.load_SH((hdr_path.replace('warpedHDROutputs', 'SHGT')).replace('.exr', '.txt'))
        sh_image = reconstruction(coeff, sh)
        parametric_lights = {"distribution": distribution,
                             'intensity': intensity,
                             'rgb_ratio': rgb_ratio,
                             'sh_image' : sh_image}
        # parametric_lights = {"anchors": anchors,
        #                      'sh_image' : sh_image}
        return parametric_lights, map


def legendre(n, x):
    # 将x转换为列向量
    x = np.array(x).reshape(-1, 1)
    # 计算L_n(x)
    Lnm = lpmv(np.arange(n + 1), n, x)
    # Lnm = lpmv(n, np.arange(n + 1), x)
    Lnm = Lnm.T
    return Lnm.flatten()


def getSH(N, dirs, basisType):
    """
    Get spherical harmonics up to order N.

    Args:
        N (int): Maximum order of harmonics.
        dirs (ndarray): Angles in radians for each evaluation point, where
            inclination is the polar angle from zenith: inclination = pi/2-elevation.
        basisType (str): Basis type of spherical harmonics, either 'complex' or 'real'.

    Returns:
        ndarray: Spherical harmonics of size (N+1)^2 x Ndirs.

    """

    Ndirs = dirs.shape[0]
    Nharm = (N + 1) ** 2

    Y_N = np.zeros((Nharm, Ndirs), dtype=np.double)
    idx_Y = 0
    for n in range(N + 1):
        m = np.arange(0, n + 1)
        # for m in np.arange(-n, n+1):

        # Vector of unnormalised associated Legendre functions of current order
        Lnm_real = legendre(n, np.cos(dirs[:, 1]).T).reshape((n + 1, Ndirs))

        if n != 0:
            # Cancel the Condon-Shortley phase from the definition of
            # the Legendre functions to result in signless real SH
            condon = (-1) ** np.concatenate([m[:0:-1], m],dtype=np.double)[:, np.newaxis] * np.ones((1, Ndirs))
            # Lnm_real = condon * np.concatenate([Lnm_real[-2::-1, :], Lnm_real], axis=0)
            Lnm_real = condon * np.concatenate([Lnm_real[:0:-1, :], Lnm_real],dtype=np.double)

        # Normalisations
        norm_real = (np.sqrt((2 * n + 1) * factorial(n - m) / (4 * np.pi * factorial(n + m)),dtype=np.double))[:, np.newaxis]

        # Convert to matrix, for direct matrix multiplication with the rest
        Nnm_real = norm_real * np.ones((1, Ndirs))
        if n != 0:
            Nnm_real = np.concatenate([Nnm_real[:0:-1, :], Nnm_real])

        CosSin = np.zeros((2 * n + 1, Ndirs),dtype=np.double)
        # Zero degree
        CosSin[n, :] = np.ones((1, Ndirs))
        # Positive and negative degrees
        if n != 0:
            CosSin[m[1:] + n, :] = np.sqrt(2) * np.cos(m[1:][:, np.newaxis] * dirs[:, 0].T)
            CosSin[-m[:0:-1] + n, :] = np.sqrt(2) * np.sin(m[:0:-1].reshape(len(m[:0:-1]), 1) * dirs[:, 0].T)

        Ynm = (Nnm_real * Lnm_real * CosSin)

        Y_N[idx_Y:idx_Y + (2 * n + 1), :] = Ynm
        idx_Y += 2 * n + 1

    # Transpose to Ndirs x Nharm
    Y_N = Y_N.T

    return Y_N


def reconstruction(coeff, shcoeff):
    h = 128
    w = 256
    # shcoeff = shcoeff * 255.0
    r_coeff = shcoeff[0,:]
    g_coeff = shcoeff[1,:]
    b_coeff = shcoeff[2,:]
    out_r = np.sum(coeff * r_coeff, axis=1)

    out_g = np.sum(coeff * g_coeff, axis=1)

    out_b = np.sum(coeff * b_coeff, axis=1)

    sh_img = np.zeros((128,256,3))
    sh_img[:, :, 0] = np.reshape(out_r, (h, w), order='F')
    sh_img[:, :, 1] = np.reshape(out_g, (h, w), order='F')
    sh_img[:, :, 2] = np.reshape(out_b, (h, w), order='F')
    sh_img = sh_img.clip(0, 255).astype(np.uint8)
    return sh_img

theta = ((np.arange(128) + 0.5) * np.pi) / 128
phi = ((np.arange(256) + 0.5) * 2 * np.pi) / 256
x, y = np.meshgrid(phi, theta)
dirs = np.column_stack((x.ravel('F'), y.ravel('F')))
coeff = getSH(3,dirs)

# train_dir = '/home/fangneng.zfn/datasets/LavalIndoor/nips/'
bs_dir = '/media/wangao/DataDisk/Study/Dataset/warped_Laval_Indoor_HDR_Dataset/'
hdr_dir = bs_dir + 'warpedHDROutputs/test/'
sv_dir = bs_dir + 'pkl/test/'
# crop_dir = bs_dir + 'tpami cxd'
nms = os.listdir(hdr_dir)
# nms = nms[:100]
ln = 128

extractor = extract_mesh(ln=ln)

i = 0
# nms = ['AG8A9899-others-40-1.62409-1.07406.exr']
for nm in tqdm(nms):
    if nm.endswith('.exr'):
        hdr_path = hdr_dir + nm
        h = util.PanoramaHandler()
        hdr = h.read_exr(hdr_path)
        para, map = extractor.compute(hdr)

        # 产生pkl文件
        with open((sv_dir + os.path.basename(hdr_path).replace('exr', 'pickle')), 'wb') as handle:
            pickle.dump(para, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # 
        # im = Image.fromarray(sh_image)
        # nm_ = nm.split('.')[0]
        # im.show()
        # im.save('./sh_image/{}_rec.png'.format(nm_))



        # i += 1
        # print (i)

        # dirs = util.sphere_points(ln)
        # dirs = torch.from_numpy(dirs)
        # dirs = dirs.view(1, ln*3).cuda().float()
        #
        # size = torch.ones((1, ln)).cuda().float() * 0.0025
        # intensity = torch.from_numpy(np.array(para['intensity'])).float().cuda()
        # intensity = intensity.view(1, 1, 1).repeat(1, ln, 3).cuda()
        #
        # rgb_ratio = torch.from_numpy(np.array(para['rgb_ratio'])).float().cuda()
        # rgb_ratio = rgb_ratio.view(1, 1, 3).repeat(1, ln, 1).cuda()
        #
        # distribution = torch.from_numpy(para['distribution']).cuda().float()
        # distribution = distribution.view(1, ln, 1).repeat(1, 1, 3)
        #
        # light_rec = distribution * intensity * rgb_ratio
        # light_rec = light_rec.contiguous().view(1, ln*3)
        #
        # env = util.convert_to_panorama(dirs, size, light_rec)
        # env = env.detach().cpu().numpy()[0]
        # env = util.tonemapping(env) * 255.0
        # im = np.transpose(env, (1, 2, 0))
        # im = Image.fromarray(im.astype('uint8'))
        #
        # nm_ = nm.split('.')[0]
        # # im.show()
        # im.save('./{}_rec.png'.format(nm_))
        #
        # gt = util.tonemapping(hdr) * 255.0
        # gt = Image.fromarray(gt.astype('uint8'))
        # gt.save('./tmp/{}_gt.png'.format(nm_))
        #
        # light = util.tonemapping(hdr) * 255.0 * map
        # light = Image.fromarray(light.astype('uint8'))
        # light.save('./tmp/{}_light.png'.format(nm_))
        # print (1/0)
