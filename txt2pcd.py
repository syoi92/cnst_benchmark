import numpy as np
import open3d as o3d
import argparse
from pathlib import Path
import os


def main(opt):
    if not opt.save_path:
        Path(opt.txt_file)
        ROOT = Path(opt.txt_file).parents[0]
        fn = Path(opt.txt_file).stem + '.ply'
        save_path = os.path.join(ROOT, fn)
        
        assert not os.path.isfile(save_path), "a ply file already exists"
        opt.save_path = save_path
    
    assert Path(opt.save_path).suffix == '.ply'

    pcd = np.loadtxt(opt.txt_file, comments="//")
    xyz = pcd[:,:3]
    rgb = pcd[:,3:6]
    if np.max(rgb[:,0]) > 1:
        rgb /= 255

    pcd_ = np.hstack([xyz, rgb]).astype(np.float32)
    writePLY(pcd_, opt.save_path)



def writePLY(pcd, filename):
    output_cloud = o3d.geometry.PointCloud()
    output_cloud.points = o3d.utility.Vector3dVector(pcd[:, :3])
    output_cloud.colors = o3d.utility.Vector3dVector(pcd[:, 3:6])
    o3d.io.write_point_cloud(filename, output_cloud)
    print('Saved',len(pcd),'points to',filename)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt_file', type=str, default="./datasets/sample.txt")
    parser.add_argument('--save_path', type=str)
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
