import numpy as np
import open3d as o3d
import argparse
from tqdm import tqdm
import os
import logging, uuid
import cv2

def main(opt):

    assert os.path.splitext(opt.src)[1] == '.txt', f"src should be .txt file"
    if not opt.save_path:
        opt.save_path = os.path.splitext(opt.src)[0] + '.' + uuid.uuid4().hex
    
    if not opt.sc:
        opt.sc = [0.0, 0.0, 0.0]
    assert len(opt.sc) == 3, f"scannet center should be a [3,] vector"

    if not os.path.isdir(opt.save_path):
        os.mkdir(opt.save_path)

    # log
    logger = logging.getLogger("3D-2D Projection")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(message)s"
    )
    file_handler = logging.FileHandler("%s/logs.txt" % opt.save_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    def log_string(str):
        logger.info(str)
        print(str)
    log_string("opt: {}".format(opt))


    pcd = np.loadtxt(opt.src, comments="//")
    log_string("load txt file. #%d of points: " % len(pcd))

    xyz = pcd[:,:3] - np.array(opt.sc)
    rgb = pcd[:,3:6]
    intn = np.array(pcd[:,6])
    depth = np.sum(xyz ** 2, axis=-1) ** 0.5

    if np.max(rgb[:,0]) > 1:
        rgb /= 255

    # blank canvas
    width, height = (opt.img_size, opt.img_size)
    I_color = np.zeros((height, width, 3), dtype=np.uint8)
    I_depth = -np.ones((height, width), dtype=np.float64)
    I_intn = -np.ones((height, width), dtype=np.float64)

    # Convert the point cloud to a meshgrid 
    im_x = np.arctan2(xyz[:,1], xyz[:,0])
    im_y = np.arctan2(xyz[:,2], (xyz[:,1]**2 + xyz[:,0]**2)**0.5)
    im0_x = ((im_x / np.pi + 1) / 2 * (width-1)).astype(np.int16)
    im0_y = ((-im_y / np.pi + 1) / 2 * (height-1)).astype(np.int16)

    for i in tqdm(range(len(xyz))):
        I_depth[im0_y[i], im0_x[i]] = depth[i]
        I_intn[im0_y[i], im0_x[i]] = intn[i]
        I_color[im0_y[i], im0_x[i], :] = (rgb[i] * 255).astype(np.uint8)


    if opt.debug:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        plt.imshow(I_color)
        plt.imshow(I_color[960:2570, :1920])
        plt.show()


    ## clipping
    I_gray = cv2.cvtColor(I_color, cv2.COLOR_BGR2GRAY)
    mask = I_gray > 10
    ind = np.ix_(mask.any(1),mask.any(0))

    log_string("clipping bounding box")
    log_string("height_min: %d" % ind[0].min())
    log_string("height_max: %d" % ind[0].max())
    log_string("width_min: %d" % ind[1].min())
    log_string("width_max: %d" % ind[1].max())

    I_color_clipped = I_color[ind]

    if opt.v_split < 0:
        ss = os.path.join(opt.save_path, os.path.split(opt.src)[-1] + '.png')
        cv2.imwrite(ss, I_color_clipped[:,:,::-1])
        log_string("img_save: %s" % ss)

    else:
        ss1 = os.path.join(opt.save_path, os.path.split(opt.src)[-1] + '.1.png')
        ss2 = os.path.join(opt.save_path, os.path.split(opt.src)[-1] + '.2.png')

        cv2.imwrite(ss1, I_color_clipped[:,:int(width/2),::-1])
        log_string("img_save: %s" % ss1)

        cv2.imwrite(ss2, I_color_clipped[:,int(width/2):,::-1])
        log_string("img_save: %s" % ss1)



# function to read PLY file into NxF numpy array
def readPLY(filename):
    pcd = o3d.io.read_point_cloud(filename)
    pcd = np.hstack([pcd.points, pcd.colors]).astype(np.float32)
    return pcd

# function to write NxF numpy array point cloud to PLY file
# point cloud should have at least 6 columns: XYZ, RGB [0 - 1]
def writePLY(pcd, filename):
    output_cloud = o3d.geometry.PointCloud()
    output_cloud.points = o3d.utility.Vector3dVector(pcd[:, :3])
    output_cloud.colors = o3d.utility.Vector3dVector(pcd[:, 3:6])
    o3d.io.write_point_cloud(filename, output_cloud)
    print('Saved',len(pcd),'points to',filename)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default="./datasets/Denoise-Indoor1.txt")
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--sc', nargs='+', type= float, help='scanner center coordinate, input: x0 y0 z0')
    parser.add_argument('--v_split', type=int, default= -1)
    parser.add_argument('--img_size', type=int, default=3840)
    parser.add_argument('--debug', action='store_true')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

