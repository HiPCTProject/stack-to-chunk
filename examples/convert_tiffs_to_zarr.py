"""
An example script to convert a multi-tiff stack to a chunked zarr file.

This script loads the multi-tiff stack from a Google Drive folder into a dask
array, and then saves it as a chunked zarr file.
"""

import shutil
from pathlib import Path

from stack_to_chunk import MultiScaleGroup
from stack_to_chunk.tiff_helpers import read_tiff_stack_with_dask

# Define path to the Google Drive folder containing data for all subjects
gdrive_dir = Path("/Users/nsirmpilatze/GDrive")
project_dir = (
    gdrive_dir / "Shared drives/Cummings and O'Connell Collaboration/Tadpoles/"
)
assert project_dir.is_dir()

# Define subject ID and check that the corresponding folder exists
subject_id = "topro35"
assert (project_dir / subject_id).is_dir()

# Define channel (by wavelength) and check that there is exactly one folder
# containing the tiff files for this channel in the subject folder
channel = "488"
channel_dirs = sorted(project_dir.glob(f"{subject_id}/*{channel}*"))
assert len(channel_dirs) == 1
channel_dir = channel_dirs[0]

# Select chunk size
chunk_size = 64


if __name__ == "__main__":

    tiff_files = sorted(channel_dir.glob("*.tif"))

    # Define output directory and create a folder for the subject and channel
    output_dir = Path("/Users/nsirmpilatze/Data/tadpoles")
    assert output_dir.is_dir()
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

    # Add the full resolution data
    stack = read_tiff_stack_with_dask(channel_dir)
    stack = stack.transpose((2, 1, 0))

    group.add_full_res_data(
        stack,
        n_processes=5,
        chunk_size=chunk_size,
        compressor="default",
    )
