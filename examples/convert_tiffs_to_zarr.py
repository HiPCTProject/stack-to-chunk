"""
An example script to convert a multi-tiff stack to a chunked zarr file.

This script loads the multi-tiff stack from a Google Drive folder into a dask
array, and then saves it as a chunked zarr file.
"""

import shutil

from stack_to_chunk import MultiScaleGroup
from stack_to_chunk.io_helpers import load_env_var_as_path, read_tiff_stack_with_dask

# Paths to the Google Drive folder containing tiffs for all subjects & channels
# and the output folder for the zarr files (both set as environment variables)
input_dir = load_env_var_as_path("ATLAS_PROJECT_TIFF_INPUT_DIR")
output_dir = load_env_var_as_path("ATLAS_PROJECT_ZARR_OUTPUT_DIR")

# Define subject ID and check that the corresponding folder exists
subject_id = "topro35"
assert (input_dir / subject_id).is_dir()

# Define channel (by wavelength) and check that there is exactly one folder
# containing the tiff files for this channel in the subject folder
channel = "488"
channel_dirs = sorted(input_dir.glob(f"{subject_id}/*{channel}*"))
assert len(channel_dirs) == 1
channel_dir = channel_dirs[0]

# Select chunk size
chunk_size = 64


if __name__ == "__main__":

    tiff_files = sorted(channel_dir.glob("*.tif"))
    # Create a folders for the subject and channel in the output directory
    zarr_file_path = output_dir / subject_id / f"{subject_id}_{channel}.zarr"
    if zarr_file_path.exists():
        print(f"Deleting existing {zarr_file_path}")
        shutil.rmtree(zarr_file_path)

    # Create a MultiScaleGroup object (zarr group)
    group = MultiScaleGroup(
        zarr_file_path,
        name=f"{subject_id}_{channel}",
        spatial_unit="micrometer",
        voxel_sizes=(3, 3, 3),
    )

    # Read the tiff stack into a dask array
    da_arr = read_tiff_stack_with_dask(channel_dir)
    da_arr = da_arr.transpose((2, 1, 0))
    da_arr.rechunk(chunks=(da_arr.shape[0], da_arr.shape[1], 1))

    # Add the dask array to the zarr group
    group.add_full_res_data(
        da_arr,
        n_processes=5,
        chunk_size=chunk_size,
        compressor="default",
    )
