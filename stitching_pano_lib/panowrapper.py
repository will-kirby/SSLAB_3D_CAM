from __future__ import division, print_function

import os
import time
import re
import numpy as np
import matplotlib.pyplot as plt

try:
    import libpyopenpano as pano
except:
    ValueError(
        "Couldn't import 'libpyopenpano' library. You may need to use the shell "
        "script (*.sh files) to run this module or export LD_LIBRARY_PATH variable.\n"
        "    => Ex: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIB_DIR && python prepare_stitching_data.py"
    )


# DIRECTORY="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# LIB_DIR=$DIRECTORY/libs
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIB_DIR 
# python panowrapper.py

def get_file_list(path, exts = ["jpg", "jpeg", "png", "bmp"]):
    """Get a list of files with the given extension in the given directory."""
    file_list = []
    
    # exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]
    for ext in exts:
        file_list.extend(
            [
                os.path.join(path, filename)
                for filename in os.listdir(path)
                if re.search(r"\." + ext + "$", filename, re.IGNORECASE)
            ]
        )
        
    print(file_list)
    return file_list

if __name__ == "__main__":
    import libpyopenpano as pano

    # help(pano)
    # Test Stitching
    pano_config_file = "config.cfg"
    pano.init_config(pano_config_file)
    pano.print_config()
    print(f" {pano_config_file:_^50}")

    # Get the list of images
    mdata = [
        {
            "input_dir": "/home/UFAD/enghonda/projects/stitching_pano_lib/Campus/CMU0",
            "out_dir": "/home/UFAD/enghonda/projects/stitching_pano_lib/Campus/CMU0/output",
        },
        {
            "input_dir": "/home/UFAD/enghonda/projects/stitching_pano_lib/Campus/CMU0",
            "out_dir": "/home/UFAD/enghonda/projects/stitching_pano_lib/Campus/CMU0/output",
        },
    ]

    id = 0

    out_dir = mdata[id]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    output_result = f"{out_dir}/{id:05d}.jpg"
    print(output_result)

    nb_stitched_img = 0
    stitcher = None
    
    file_list = get_file_list(mdata[id]["input_dir"])
    print(file_list)

    stitcher = None
    try:
        # Instantiate the Stitcher
        stitcher = pano.Stitcher(file_list)
        # Stitch the images
        mat = stitcher.build()
        # Save the result
        pano.write_img(output_result, mat)        
    except:
        print(f"Error: Cannot stitch image [{id}] - [{output_result}]")
        exit(1)

    # Sleep for 1 second
    time.sleep(1)

    # Stitching the next images
    # Assume that the first image is already stitched
    # And the next images are from the same camera and order
    id += 1
    multi_band_blend = 10  # 0 is for linear blending

    # The function will stitch the next images without
    # re-initializing the stitcher and without recompting the features
    mat = stitcher.build_from_new_images(file_list, multi_band_blend)

    # Save the result
    output_result = f"{out_dir}/{id:05d}.jpg"
    pano.write_img(output_result, mat)

    # Convert the image to a numpy array
    p = np.array(mat, copy=False)
    plt.imshow(p)
    plt.show()
