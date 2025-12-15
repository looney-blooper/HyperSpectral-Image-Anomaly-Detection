import tifffile as tiff
from scipy.io import savemat
import os 

def convert_tif_to_mat(tif_path, mat_path):
    img = tiff.imread(tif_path)
    savemat(mat_path, {'image': img})

if __name__ == "__main__":
    input_tif = ""  
    output_mat = "kaggle_test.mat"  
    convert_tif_to_mat(input_tif, output_mat)