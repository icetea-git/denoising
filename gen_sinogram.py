import os
import json
import numpy as np
import odl
from tqdm import tqdm
from skimage.transform import resize
import pydicom
import nrrd
from math import ceil

MU_WATER = 20
MU_AIR = 0.02
MU_MAX = 3071 * (MU_WATER - MU_AIR) / 1000 + MU_WATER

# path to output
SINOGRAM_PATH = '.\\output\\sinogram'
GROUND_TRUTH_PATH = '.\\output\\ground_truth'

FILE_LIST_FILE = 'data.json'

os.makedirs(SINOGRAM_PATH, exist_ok=True)
os.makedirs(GROUND_TRUTH_PATH, exist_ok=True)

with open(FILE_LIST_FILE, 'r') as f:
    file_list = json.load(f)

# ~26cm x 26cm images
MIN_PT = [-0.13, -0.13]
MAX_PT = [0.13, 0.13]


def lidc_idri_gen():
    r = np.random.RandomState(0)
    sort_data = []
    for z in file_list:
        tmp_dicom = pydicom.read_file(z)
        if tmp_dicom.pixel_array.shape == (512, 512):
            sort_data.append(tmp_dicom)

    sort_data.sort(key=lambda x: int(x.InstanceNumber))
    dicom_array = np.stack([s.pixel_array for s in sort_data])

    array = dicom_array.astype(np.float32).T

    # rescale by dicom meta info
    array *= tmp_dicom.RescaleSlope
    array += tmp_dicom.RescaleIntercept

    # add noise to get continuous values from discrete ones
    array += r.uniform(0., 1., size=array.shape)

    array *= (MU_WATER - MU_AIR) / 1000
    array += MU_WATER
    array /= MU_MAX
    np.clip(array, 0., 1., out=array)
    array = np.transpose(array, (2, 1, 0))

    return array


lidc_idri_gen_len = len(file_list)

NUM_ANGLES = 1000
RECO_IM_SHAPE = (512, 512)

# image shape for simulation
IM_SHAPE = (1000, 1000)

reco_space = odl.uniform_discr(min_pt=MIN_PT, max_pt=MAX_PT,
                               shape=RECO_IM_SHAPE, dtype=np.float32)
space = odl.uniform_discr(min_pt=MIN_PT, max_pt=MAX_PT, shape=IM_SHAPE,
                          dtype=np.float64)

reco_geometry = odl.tomo.parallel_beam_geometry(
    reco_space, num_angles=NUM_ANGLES)
geometry = odl.tomo.parallel_beam_geometry(
    space, num_angles=NUM_ANGLES, det_shape=reco_geometry.detector.shape)

IMPL = 'astra_cpu'
reco_ray_trafo = odl.tomo.RayTransform(reco_space, reco_geometry)
ray_trafo = odl.tomo.RayTransform(space, geometry)

PHOTONS_PER_PIXEL = 4096

rs = np.random.RandomState(3)

NUM_SAMPLES_PER_FILE = 64
n_files = ceil(lidc_idri_gen_len / NUM_SAMPLES_PER_FILE)


def ff(im):
    im_resized = resize(im * MU_MAX, IM_SHAPE, order=1)

    # apply forward operator
    data = ray_trafo(im_resized).asarray()

    data *= (-1)
    np.exp(data, out=data)
    data *= PHOTONS_PER_PIXEL
    return data


slices = lidc_idri_gen()
print(len(slices))
it1 = 0
it2 = NUM_SAMPLES_PER_FILE
for filenumber in tqdm(range(n_files)):
    obs_filename = os.path.join(
        SINOGRAM_PATH, 'sinogram_{:03d}.nrrd'.format(filenumber))
    ground_truth_filename = os.path.join(
        GROUND_TRUTH_PATH, 'ground_truth_{:03d}.nrrd'.format(filenumber))

    observation_dataset = []
    ground_truth_dataset = []

    for data in tqdm(slices[it1:it2, ...]):
        ground_truth_dataset.append(data)
        data = ff(data)
        data = rs.poisson(data) / PHOTONS_PER_PIXEL
        np.maximum(0.1 / PHOTONS_PER_PIXEL, data, out=data)
        np.log(data, out=data)
        data /= (-MU_MAX)
        observation_dataset.append(data)

    it1 += NUM_SAMPLES_PER_FILE
    it2 += NUM_SAMPLES_PER_FILE

    if it2 > len(slices):
        it2 = len(slices)
    observation_dataset = np.array(observation_dataset).transpose(1, 2, 0)
    ground_truth_dataset = np.array(ground_truth_dataset).transpose(2, 1, 0)
    print(observation_dataset.shape)
    print(ground_truth_dataset.shape)
    nrrd.write(obs_filename, observation_dataset)
    nrrd.write(ground_truth_filename, ground_truth_dataset)
