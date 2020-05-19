import numpy as np
import nrrd
import os
import astra
from tqdm import tqdm

SINOGRAM_PATH = '.\\output\\sinogram'
RECONSTRUCTED_PATH = '.\\output\\noisy'


def reconstruct(sinogram):
    slices, header = nrrd.read(sinogram)
    rec = []
    for i in tqdm(range(slices.shape[2])):
        vol_geom = astra.create_vol_geom(512, 512)
        proj_geom = astra.create_proj_geom('parallel', 1.0, 727, np.linspace(0, np.pi, 1000));
        sinogram_id = astra.data2d.create('-sino', proj_geom, slices[:, :, i])

        proj_id = astra.create_projector('line', proj_geom, vol_geom);

        rec_id = astra.data2d.create('-vol', vol_geom)

        cfg = astra.astra_dict('FBP');
        cfg['ProjectorId'] = proj_id;
        cfg['ProjectionDataId'] = sinogram_id;
        cfg['ReconstructionDataId'] = rec_id;
        alg_id = astra.algorithm.create(cfg);
        astra.algorithm.run(alg_id, 150);
        rec.append(astra.data2d.get(rec_id));

    rec = np.array(rec).transpose(1, 2, 0)
    rec = np.flipud(rec)
    return rec


dcm_filenames = sorted(list(os.listdir(SINOGRAM_PATH)))
it = 0
for dcm_file in dcm_filenames:
    if dcm_file.endswith('.nrrd'):
        recon_filename = os.path.join(RECONSTRUCTED_PATH, 'noisy_{:03d}.nrrd'.format(it))
        nrrd.write(recon_filename, reconstruct(os.path.join(SINOGRAM_PATH, dcm_file)))
        it += 1
