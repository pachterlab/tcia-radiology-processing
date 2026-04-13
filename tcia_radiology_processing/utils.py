import glob
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import gzip
import tempfile
import zipfile
import time
import threading
import psutil
from functools import wraps

import highdicom as hd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import nibabel as nib
import numpy as np
import pandas as pd
import pydicom
import scipy.ndimage as ndi
import SimpleITK as sitk
from ipywidgets import interact
from radiomics import featureextractor
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROFILE_PIPELINE = True
PROFILE_PIPELINE_DATA_DIR = None

tcia_kirc_manifest_url = "https://www.cancerimagingarchive.net/wp-content/uploads/TCIA_TCGA-KIRC_09-16-2015.tcia"
tcga_kirc_metadata_url = "https://www.cancerimagingarchive.net/wp-content/uploads/TCIA_TCGA-KIRC_09-16-2015-nbia-digest.xlsx"

BAD_SERIES_KEYWORD = {"localizer", "survey", "asset", "scout", "cal", "mipseries", "pjn", "summary series", "topogram", "mip", "smart prep"}  # skip localizer/survey/asset/calibration/MIP series (will be super low quality)
SERIES_DESCRIPTION_KEYWORDS_EXCLUDE_RADIOMICS = [
    "B5",  # catches B50, B51, etc.
    "B6",  # catches B60
    "B7",  # catches B70
    "B8",  # catches B80
    "bone"
]

seg_mask_number_to_label = {
    0: "Background",
    1: "Organ",
    2: "Tumor",
    3: "Cyst",
}
seg_mask_label_to_number = {v: k for k, v in seg_mask_number_to_label.items()}

def _dir_size_bytes(path):
    try:
        return int(subprocess.check_output(["du", "-sb", path]).split()[0])
    except Exception:
        return None


def measure_time_memory_storage(
    enabled=True,
    disk_path=None,
    interval=0.1,
):

    def decorator(func):

        if not enabled:
            func.last_metrics = None
            return func

        @wraps(func)
        def wrapper(*args, **kwargs):

            # ---- resolve disk path dynamically ----

            resolved_disk_path = disk_path

            if callable(disk_path):
                resolved_disk_path = disk_path()

            # ---- monitoring setup ----

            process = psutil.Process()
            peak_mem = 0
            running = True

            def monitor():
                nonlocal peak_mem
                while running:
                    try:
                        mem = process.memory_info().rss
                        for child in process.children(recursive=True):
                            try:
                                mem += child.memory_info().rss
                            except psutil.NoSuchProcess:
                                pass
                        peak_mem = max(peak_mem, mem)
                    except psutil.NoSuchProcess:
                        pass

                    time.sleep(interval)

            start_disk = _dir_size_bytes(resolved_disk_path) if resolved_disk_path else None
            start_time = time.time()

            monitor_thread = threading.Thread(target=monitor, daemon=True)
            monitor_thread.start()

            try:
                result = func(*args, **kwargs)
            finally:
                running = False
                monitor_thread.join()

            end_time = time.time()
            end_disk = _dir_size_bytes(resolved_disk_path) if resolved_disk_path else None

            elapsed = end_time - start_time
            peak_mem_gb = peak_mem / 1e9

            disk_change = None
            if resolved_disk_path and start_disk is not None and end_disk is not None:
                disk_change = (end_disk - start_disk) / 1e9

            metrics = {
                "time": elapsed,
                "peak_mem_gb": peak_mem_gb,
                "disk_written_gb": disk_change,
            }
            wrapper.last_metrics = metrics
            
            # logger.info("----- Resource usage -----")
            # logger.info(f"Wall time: {elapsed:.2f} sec")
            # logger.info(f"Peak memory (process tree): {peak_mem_gb:.2f} GB")
            # if disk_change is not None:
            #     logger.info(f"Disk written: {disk_change:.2f} GB")
            # logger.info("--------------------------")

            return result

        return wrapper

    return decorator

def add_metrics(metrics, total=None):
    if total is None:
        total = {"time": 0, "peak_mem": 0, "disk": 0}
    total["time"] += metrics["time"]
    total["peak_mem"] = max(total["peak_mem"], metrics["peak_mem_gb"])
    if metrics["disk_written_gb"] is not None:
        total["disk"] += metrics["disk_written_gb"]
    return total

def define_default_out_nifti(image_file, suffix="_masked"):
    if not isinstance(image_file, str):
        raise ValueError(f"Expected a file path for image_file, got {type(image_file)}")
    out = image_file.replace(".nii", f"{suffix}.nii")
    if not out.endswith(".gz"):
        out += ".gz"
    return out

def mask_dcm_to_nii(image_dcm_dir, seg_file, image_nifti_file = None, seg_dir_out = "."):
    # Load CT reference
    ct_files = sorted(glob.glob(os.path.join(image_dcm_dir, "*.dcm")))
    ct_datasets = [pydicom.dcmread(f) for f in ct_files]

    # Build SOPInstanceUID → slice index lookup dict
    uid_to_index = {ct.SOPInstanceUID: idx for idx, ct in enumerate(ct_datasets)}

    # Load segmentation
    seg_ds = pydicom.dcmread(seg_file)
    seg = hd.seg.Segmentation.from_dataset(seg_ds)

    if image_nifti_file is not None:
        ct_img = nib.load(image_nifti_file)
        affine = ct_img.affine
        ct_nifti_shape = ct_img.shape  # (rows, cols, slices)
    else:
        px_spacing = [float(x) for x in ct_datasets[0].PixelSpacing]
        slice_thickness = float(ct_datasets[0].SliceThickness)
        affine = np.diag([px_spacing[0], px_spacing[1], slice_thickness, 1.0])

        # Get volume shape from CT
        rows, cols = int(ct_datasets[0].Rows), int(ct_datasets[0].Columns)
        num_slices = len(ct_datasets)
        ct_nifti_shape = (rows, cols, num_slices)

    # Allocate mask volume
    mask_type_index_to_mask_name = {}
    mask_type_index_to_mask_array = {}
    for seg_type in seg_ds.SegmentSequence:
        mask_type_index_to_mask_name[seg_type.SegmentNumber] = seg_type.SegmentLabel
        mask_type_index_to_mask_array[seg_type.SegmentNumber] = np.zeros(ct_nifti_shape, dtype=np.uint8)

    # Map segmentation frames back into CT space
    for i, seg_slice in enumerate(seg_ds.PerFrameFunctionalGroupsSequence):
        ref_sop_uid = seg_slice.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID
        mask_type_index = seg_slice.SegmentIdentificationSequence[0].ReferencedSegmentNumber
        slice_idx = uid_to_index.get(ref_sop_uid)
        if slice_idx is not None:
            # mask_type_index_to_mask_array[mask_type_index][slice_idx, :, :] = seg.pixel_array[i]
            mask_type_index_to_mask_array[mask_type_index][:, :, slice_idx] = seg.pixel_array[i]

    for mask_index, mask_name in mask_type_index_to_mask_name.items():
        mask = mask_type_index_to_mask_array[mask_index]
        mask = np.rot90(mask, k=1, axes=(0,1))  # Rotate 90 degrees clockwise
        mask = np.flip(mask, axis=0)  # Flip left-right
        mask = np.flip(mask, axis=2)  # Ensure 1st slice of mask matches 1st slice of CT
        nii = nib.Nifti1Image(mask, affine)  # (rows, cols, slices)
        nib.save(nii, os.path.join(seg_dir_out, f"seg_{mask_name}.nii.gz"))
        logger.info(f"Saved segmentation mask for '{mask_name}' to {os.path.join(seg_dir_out, f'seg_{mask_name}.nii.gz')}")

@measure_time_memory_storage(enabled=PROFILE_PIPELINE, disk_path=lambda: PROFILE_PIPELINE_DATA_DIR)
def set_canonical_orientation(image_path, out=True, overwrite=False):
    if image_path is None:
        return None

    if out is True:
        out = define_default_out_nifti(image_path, suffix="_oriented")

    if out is not None and os.path.exists(out) and not overwrite:
        logger.debug(f"Resampled image already exists at {out} and overwrite=False, skipping resampling.")
        return out

    if isinstance(image_path, str):
        img_nib = nib.load(image_path)
    elif isinstance(image_path, nib.Nifti1Image):
        img_nib = image_path
    else:
        raise ValueError(f"Expected a Nifti1Image or file path, got {type(image_path)}")

    canonical_orient = nib.orientations.aff2axcodes(img_nib.affine)
    if canonical_orient != ('R', 'A', 'S'):
        logger.debug(f"Reorienting image from {canonical_orient} to ('R', 'A', 'S')")
        img_nib = nib.as_closest_canonical(img_nib)
    else:
        logger.debug(f"Image already in canonical orientation ('R', 'A', 'S'), no reorientation needed.")
    
    if out is None:
        return img_nib
    
    nib.save(img_nib, out)
    return out

@measure_time_memory_storage(enabled=PROFILE_PIPELINE, disk_path=lambda: PROFILE_PIPELINE_DATA_DIR)
def resample_image(image_path, target_spacing=(0.8, 0.8, 3.0), is_label=False, out=True, overwrite=False):
    if image_path is None:
        return None

    if out is True:
        out = define_default_out_nifti(image_path, suffix="_resampled")

    if out is not None and os.path.exists(out) and not overwrite:
        logger.debug(f"Resampled image already exists at {out} and overwrite=False, skipping resampling.")
        return out

    if isinstance(image_path, str):
        img_sitk = sitk.ReadImage(image_path)
    elif isinstance(image_path, sitk.Image):
        img_sitk = image_path
    else:
        raise ValueError(f"Expected a sitk.Image or file path, got {type(image_path)}")

    original_spacing = img_sitk.GetSpacing()
    original_size = img_sitk.GetSize()

    target_spacing_list = list(target_spacing)
    for idx, dim in enumerate(target_spacing):
        if dim is None:
            target_spacing_list[idx] = original_spacing[idx]
    target_spacing = target_spacing_list
    
    if original_spacing == target_spacing:  # No resampling needed
        logger.debug(f"Original spacing {original_spacing} matches target spacing {target_spacing}, skipping resampling.")
        if out is not None:
            sitk.WriteImage(img_sitk, out)
        return out

    # Compute new size to preserve physical dimensions
    new_size = [
        int(np.round(original_size[i] * (original_spacing[i] / target_spacing[i])))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img_sitk.GetDirection())
    resampler.SetOutputOrigin(img_sitk.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    
    if is_label:
        logger.debug(f"Using nearest neighbor interpolation for label image resampling to target spacing of {target_spacing}.")
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        logger.debug(f"Using linear interpolation for image resampling to target spacing of {target_spacing}.")
        resampler.SetInterpolator(sitk.sitkLinear)

    img_sitk_resampled = resampler.Execute(img_sitk)
    
    if out is None:
        return img_sitk_resampled

    sitk.WriteImage(img_sitk_resampled, out)
    
    return out

@measure_time_memory_storage(enabled=PROFILE_PIPELINE, disk_path=lambda: PROFILE_PIPELINE_DATA_DIR)
def clip_intensity_range(image_path, clip_min=-200, clip_max=300, out=True, overwrite=False):
    if image_path is None:
        return None

    if out is True:
        out = define_default_out_nifti(image_path, suffix="_clipped")

    if out is not None and os.path.exists(out) and not overwrite:
        logger.debug(f"Clipped image already exists at {out} and overwrite=False, skipping clipping.")
        return out

    if isinstance(image_path, str):
        img_sitk = sitk.ReadImage(image_path)
    elif isinstance(image_path, sitk.Image):
        img_sitk = image_path
    elif isinstance(image_path, np.ndarray):
        assert out is None, "Cannot return clipped array if out file path is provided"
        return np.clip(image_path, clip_min, clip_max)
    else:
        raise ValueError(f"Expected a sitk.Image or file path, got {type(image_path)}")

    logger.debug(f"Clipping image intensities to range [{clip_min}, {clip_max}] for image at {image_path}.")
    img_sitk_clipped = sitk.Clamp(
        img_sitk,
        lowerBound=clip_min,
        upperBound=clip_max
    )

    if out is None:
        return img_sitk_clipped

    sitk.WriteImage(img_sitk_clipped, out)

    return out


def pad_image_and_mask(
    image_file,
    mask_file=None,
    target_xy=(512, 512),
    out=True,
    overwrite=False
):
    """
    Pad image and optional mask in X/Y dimensions to target size.

    Parameters
    ----------
    image_file : str
        Path to NIfTI image
    mask_file : str or None
        Path to mask NIfTI
    target_xy : tuple
        Target (X,Y) size
    out : bool
        Write output file
    overwrite : bool
        Overwrite existing padded file

    Returns
    -------
    padded_image_file, padded_mask_file
    """

    def _pad_array(arr, target_xy):
        x, y, z = arr.shape
        tx, ty = target_xy

        if x > tx or y > ty:
            raise ValueError(
                f"Image larger than target size: {arr.shape} vs {target_xy}"
            )

        pad_x = tx - x
        pad_y = ty - y

        pad_x_before = pad_x // 2
        pad_x_after = pad_x - pad_x_before

        pad_y_before = pad_y // 2
        pad_y_after = pad_y - pad_y_before

        padded = np.pad(
            arr,
            (
                (pad_x_before, pad_x_after),
                (pad_y_before, pad_y_after),
                (0, 0),
            ),
            mode="constant",
            constant_values=0,
        )

        return padded

    # Load image
    img = nib.load(image_file)
    img_data = img.get_fdata()

    padded_img = _pad_array(img_data, target_xy)

    if out:
        padded_image_file = define_default_out_nifti(image_file, suffix="_padded")

        if not os.path.exists(padded_image_file) or overwrite:
            nib.save(
                nib.Nifti1Image(padded_img, img.affine, img.header.copy()),
                padded_image_file,
            )
    else:
        padded_image_file = padded_img

    padded_mask_file = None

    if mask_file is not None:
        mask = nib.load(mask_file)
        mask_data = mask.get_fdata()

        padded_mask = _pad_array(mask_data, target_xy)

        if out:
            padded_mask_file = define_default_out_nifti(mask_file, suffix="_padded")

            if not os.path.exists(padded_mask_file) or overwrite:
                nib.save(
                    nib.Nifti1Image(padded_mask, mask.affine, mask.header.copy()),
                    padded_mask_file,
                )
        else:
            padded_mask_file = padded_mask

    return padded_image_file, padded_mask_file

@measure_time_memory_storage(enabled=PROFILE_PIPELINE, disk_path=lambda: PROFILE_PIPELINE_DATA_DIR)
def normalize_intensity(
    image_paths,
    normalization_method="volume",  # "volume" or "dataset"
    out=True,
    overwrite=False,
):
    """
    image_paths: str, sitk.Image, or list of str/sitk.Image
    normalization_method:
        - "volume"  → normalize each volume independently
        - "dataset" → compute global mean/std across all volumes
    """
    if normalization_method not in ["volume", "dataset"]:
        raise ValueError("normalization_method must be 'volume' or 'dataset'")

    # Ensure list behavior
    if not isinstance(image_paths, list):
        image_paths = [image_paths]
        single_input = True
    else:
        single_input = False

    # ---------- Compute dataset statistics if needed ----------
    if normalization_method == "dataset":
        all_voxels_sum = 0.0
        all_voxels_sq_sum = 0.0
        total_voxels = 0

        for img_input in image_paths:
            img = sitk.ReadImage(img_input) if isinstance(img_input, str) else img_input
            arr = sitk.GetArrayFromImage(img).astype(np.float64)

            all_voxels_sum += arr.sum()
            all_voxels_sq_sum += (arr ** 2).sum()
            total_voxels += arr.size

        global_mean = all_voxels_sum / total_voxels
        global_var = (all_voxels_sq_sum / total_voxels) - global_mean ** 2
        global_std = np.sqrt(global_var)

    outputs = []

    # ---------- Normalize images ----------
    for img_input in image_paths:
        # Load
        if isinstance(img_input, str):
            img = sitk.ReadImage(img_input)
            image_path = img_input
        elif isinstance(img_input, sitk.Image):
            img = img_input
            image_path = None
        else:
            raise ValueError(f"Unsupported type {type(img_input)}")
        
        if out is True:
            out = define_default_out_nifti(image_path, suffix="_normalized") if image_path else None

        arr = sitk.GetArrayFromImage(img).astype(np.float32)

        if normalization_method == "volume":
            mean = arr.mean()
            std = arr.std()
        elif normalization_method == "dataset":
            mean = global_mean
            std = global_std
        else:
            raise ValueError("Invalid normalization method")

        if std == 0:
            raise ValueError("Standard deviation is zero — cannot normalize.")

        arr_norm = (arr - mean) / std

        img_norm = sitk.GetImageFromArray(arr_norm)
        img_norm.CopyInformation(img)

        # Handle output
        if out:
            if not os.path.exists(out) or overwrite:
                sitk.WriteImage(img_norm, out)

            outputs.append(out)
        else:
            outputs.append(img_norm)

    if single_input:
        return outputs[0]

    return outputs


def dcm2nii_manual(dcm_dir, nii_path, gzip=True):
    files = sorted(os.listdir(dcm_dir))
    slices = []
    for fname in files:
        if fname.endswith(".dcm"):
            ds = pydicom.dcmread(os.path.join(dcm_dir, fname))
            if hasattr(ds, "ImagePositionPatient"):
                slices.append(ds)

    # Orientation vectors
    ref = slices[0]
    row_cosines = np.array(ref.ImageOrientationPatient[0:3], dtype=float)
    col_cosines = np.array(ref.ImageOrientationPatient[3:6], dtype=float)
    slice_cosines = np.cross(row_cosines, col_cosines)

    # Sort by position along slice normal
    slices.sort(key=lambda s: np.dot(slice_cosines, np.array(s.ImagePositionPatient, dtype=float)))

    # Build volume (transpose so columns=dim0, rows=dim1)
    volume = np.stack([s.pixel_array.T for s in slices], axis=-1)

    # Spacings
    row_spacing, col_spacing = [float(x) for x in ref.PixelSpacing]
    positions = [np.array(s.ImagePositionPatient, dtype=float) for s in slices]
    slice_positions = [np.dot(slice_cosines, p) for p in positions]
    slice_spacing = np.mean(np.diff(sorted(slice_positions)))

    # Affine
    affine = np.eye(4)
    affine[0:3, 0] = col_cosines * col_spacing
    affine[0:3, 1] = row_cosines * row_spacing
    affine[0:3, 2] = slice_cosines * slice_spacing
    affine[0:3, 3] = np.array(ref.ImagePositionPatient, dtype=float)

    nii = nib.Nifti1Image(volume, affine)

    if gzip and not nii_path.endswith(".gz"):
        nii_path += ".gz"
    elif not gzip and nii_path.endswith(".gz"):
        nii_path = nii_path[:-3]

    nib.save(nii, nii_path)


def check_bad_series_description(json_path):  # return True if bad, False if good
    if not os.path.exists(json_path):
        logger.warning(f"JSON file {json_path} does not exist.")
        return True
    with open(json_path, "r") as jf:
        json_content = json.load(jf)
    desc = json_content.get("SeriesDescription", "").lower()
    if any(k in desc for k in BAD_SERIES_KEYWORD):
        return True
    return False

def is_viable_dicom_series(dicom_folder, min_files=5, max_thickness_mm=10, include_kernel_keywords=True):
    """
    Determine whether a DICOM folder contains a usable CT volume.

    Checks:
    - Any bad keyword in ImageType
    - Too few files in folder
    - Excessive slice thickness
    """

    # Collect DICOM files
    dcm_files = [
        os.path.join(dicom_folder, f)
        for f in os.listdir(dicom_folder)
        if f.lower().endswith(".dcm")
    ]

    if len(dcm_files) < min_files:
        return False, f"Too few DICOM files ({len(dcm_files)})"

    # Read first DICOM header only
    try:
        dcm = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
    except Exception as e:
        return False, f"Failed to read DICOM: {e}"

    # --- Check ImageType keywords ---
    bad_series_keywords = BAD_SERIES_KEYWORD
    if include_kernel_keywords:
        bad_series_keywords = bad_series_keywords.union(set(SERIES_DESCRIPTION_KEYWORDS_EXCLUDE_RADIOMICS))
    for dicom_header in ["ImageType", "SeriesDescription", "ProtocolName"]:
        tokens = []
        if hasattr(dcm, dicom_header):
            value = getattr(dcm, dicom_header, None)
            if isinstance(value, str):
                tokens.extend(value.lower().split("\\"))
            else:  # list
                for token in getattr(dcm, dicom_header):
                    tokens.append(token.lower())

        for token in tokens:
            for bad in BAD_SERIES_KEYWORD:
                if bad in token:
                    return False, f"Bad keyword detected in {dicom_header}: {token}"

    # --- Check slice thickness ---
    if hasattr(dcm, "SliceThickness"):
        try:
            thickness = float(dcm.SliceThickness)
            if thickness > max_thickness_mm:
                return False, f"Slice thickness too large: {thickness} mm"
        except:
            pass

    return True, "Series appears viable"

def make_series_to_folder_mapping(dcm_dir):
    series_to_folder = {}

    for root, _, files in os.walk(dcm_dir):
        dcm_files = [f for f in files if f.endswith(".dcm")]
        if not dcm_files:
            continue
        # Read just one file in this folder to get SeriesInstanceUID
        dcm_path = os.path.join(root, dcm_files[0])
        try:
            dcm = pydicom.dcmread(dcm_path, stop_before_pixels=True)
            uid = dcm.SeriesInstanceUID
            series_to_folder[uid] = root
        except Exception as e:
            logger.warning(f"Skipping {dcm_path}: {e}")
    
    return series_to_folder

def add_viable_info(dcm_dir, metadata_csv, min_files=5, max_thickness_mm=10, out=None, overwrite=False, include_kernel_keywords=True):
    if isinstance(metadata_csv, str):
        metadata_df = pd.read_csv(metadata_csv)
    elif isinstance(metadata_csv, pd.DataFrame):
        metadata_df = metadata_csv
    else:
        raise ValueError("Invalid metadata_csv input")
    
    if "is_viable" in metadata_df.columns or "viable_reason" in metadata_df.columns:
        if overwrite:
            logger.info("Overwriting existing is_viable and viable_reason columns with new viability check results")
            metadata_df = metadata_df.drop(columns=["is_viable", "viable_reason"], errors="ignore")
        else:
            logger.warning("is_viable column already exists, skipping viability check")
            return metadata_df

    series_to_folder = make_series_to_folder_mapping(dcm_dir)

    case_viable, viable_reason = [], []
    for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Processing series"):
        case_id = row["series_id"]
        series_uid = row["Series UID"]

        if series_uid not in series_to_folder:
            logger.debug(f"Series UID {series_uid} imaging not found in DICOM directory tree")
            case_viable.append(False)
            viable_reason.append("Series UID not found in DICOM directory")
            continue
        
        dicom_folder = series_to_folder[series_uid]
        is_viable, reason = is_viable_dicom_series(dicom_folder, min_files=min_files, max_thickness_mm=max_thickness_mm, include_kernel_keywords=include_kernel_keywords)
        if not is_viable:
            logger.debug(f"Series UID {series_uid} for case {case_id} is not viable: {reason}")
        case_viable.append(is_viable)
        viable_reason.append(reason)

    metadata_df["is_viable"] = case_viable
    metadata_df["viable_reason"] = viable_reason

    if out is not None:
        metadata_df.to_csv(out, index=False)
    
    print(f"Viability check complete: {sum(case_viable)}/{len(case_viable)} ({sum(case_viable)/len(case_viable)*100:.2f}%) series appear viable.")

    return metadata_df

def pad_mask_to_image(mask_path, image_path, out_path):
    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)

    # mask_array = sitk.GetArrayFromImage(mask)
    # nonzero_voxel_count = np.count_nonzero(mask_array)

    # resample mask to match image geometry
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # preserve labels
    resampled_mask = resampler.Execute(mask)

    sitk.WriteImage(resampled_mask, out_path)

@measure_time_memory_storage(enabled=PROFILE_PIPELINE, disk_path=lambda: PROFILE_PIPELINE_DATA_DIR)
def convert_dcm_to_nii_and_organize(imaging_dcm_dir, imaging_metadata_df, nifti_dir, segmentation_dcm_dir=None, segmentation_metadata_df=None, segimage2itkimage_conda=False, min_files=5, max_thickness_mm=10):
    series_to_folder = make_series_to_folder_mapping(imaging_dcm_dir)
    
    manually_created_niftis = []
    no_niftis = []
    for _, row in tqdm(imaging_metadata_df.iterrows(), total=len(imaging_metadata_df), desc="Processing series"):
        case_id = row["series_id"]
        series_uid = row["Series UID"]

        case_outdir = os.path.join(nifti_dir, case_id)
        if os.path.exists(case_outdir) and len(os.listdir(case_outdir)) > 0:
            logger.debug(f"Output directory {case_outdir} already exists and is not empty, skipping conversion for case {case_id}")
            continue

        os.makedirs(case_outdir, exist_ok=True)

        # -------------------
        # Imaging conversion
        # -------------------
        # dcm2niix can take the whole images dir, but we only want this Series UID
        # So let it output everything, then rename/move the one we want
        if series_uid not in series_to_folder:
            logger.warning(f"Series UID {series_uid} imaging not found in DICOM directory tree")
            continue
        
        dicom_folder = series_to_folder[series_uid]
        
        if "is_viable" in imaging_metadata_df.columns:
            is_viable = row["is_viable"]
            reason = row.get("viable_reason", "N/A")
        else:
            is_viable, reason = is_viable_dicom_series(dicom_folder, min_files=min_files, max_thickness_mm=max_thickness_mm)
        
        if not is_viable:
            logger.warning(f"Series UID {series_uid} for case {case_id} is not viable: {reason}")
            no_niftis.append(case_id)
            continue

        if len(os.listdir(case_outdir)) == 0:
            try:
                logger.info(f"Converting imaging for {case_id}, Series UID {series_uid}")
                dcm2niix_cmd = f"dcm2niix -z y -f %j -o '{case_outdir}' '{dicom_folder}'"
                subprocess.run(dcm2niix_cmd, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                logger.warning(f"dcm2niix failed for {case_id}, Series UID {series_uid}: {e}")
                # continue
                try:
                    logger.info(f"Attempting manual DICOM to NIfTI conversion for {case_id}, Series UID {series_uid}")
                    dcm2nii_manual(dicom_folder, os.path.join(case_outdir, f"{series_uid}.nii.gz"), gzip=True)  #!!! check this
                    breakpoint()
                    manually_created_niftis.append(case_id)
                except Exception as e2:
                    logger.error(f"Manual conversion failed for {case_id}, Series UID {series_uid}: {e2}")
                    no_niftis.append(case_id)
                    breakpoint()
                    continue

        # Rename to imaging.nii.gz
        nii_src = os.path.join(case_outdir, f"{series_uid}.nii.gz")
        nii_dst = os.path.join(case_outdir, "imaging.nii.gz")
        
        if not os.path.exists(nii_dst):
            if os.path.exists(nii_src):
                os.replace(nii_src, nii_dst)
            elif os.path.exists(os.path.join(case_outdir, f"{series_uid}_i00001.nii.gz")):  # split view ie axial, coronal, sagittal in 3 files - pick the one with >= min_files slices (required) and axial orientation if possible
                # loop through niftis in case_outdir
                nii_backups = []
                for f in os.listdir(case_outdir):
                    if f.startswith(series_uid) and f.endswith(".nii.gz"):
                        json_path = os.path.join(case_outdir, f[:-7] + ".json")  # replace .nii.gz with .json
                        if os.path.exists(json_path):
                            with open(json_path, "r") as jf:
                                json_content = json.load(jf)
                            nii_src = os.path.join(case_outdir, f)
                            if json_content["ImageOrientationPatientDICOM"] == [1, 0, 0, 0, 1, 0]:  # axial
                                nii = nib.load(nii_src)
                                z_dim = nii.shape[2]
                                if z_dim >= min_files:  # if this axial view has at least min_files slices, use it
                                    os.replace(nii_src, nii_dst)
                                    break
                            else:
                                nii_backups.append(nii_src)
                # backup
                z_best = 0
                nii_src = None
                for nii_path in nii_backups:
                    nii = nib.load(nii_path)
                    z_dim = nii.shape[2]
                    if z_dim > z_best:
                        z_best = z_dim
                        nii_src = nii_path
                if z_best >= min_files:  # if we found a backup with at least min_files slices, use it
                    logger.warning(f"Using backup nifti {nii_src} with {z_best} slices for {case_id} since original split view files did not have axial orientation")
                    os.replace(nii_src, nii_dst)

            elif os.path.exists(os.path.join(case_outdir, f"{series_uid}_e1.nii.gz")):  # identical copies, but displayed window settings differ (but the voxel data is the same) - just pick one
                os.replace(os.path.join(case_outdir, f"{series_uid}_e1.nii.gz"), nii_dst)
            elif len([f for f in os.listdir(case_outdir) if f.endswith(".nii.gz")]) == 1:  # a single nifti with a suffix - might be poor quality; but if not, then just rename
                nii_src = os.path.join(case_outdir, [f for f in os.listdir(case_outdir) if f.endswith(".nii.gz")][0])  # get the single nifti file
                json_path = nii_src[:-7] + ".json"  # replace .nii.gz with .json
                bad_series = check_bad_series_description(json_path)  # log warning if json missing
                if bad_series:
                    logger.info(f"Skipping localizer/survey/asset/calibration/MIP series for {case_id}")
                    no_niftis.append(case_id)
                    continue
                os.replace(nii_src, nii_dst)
            else:
                json_files = sorted(glob.glob(os.path.join(case_outdir, "*.json")))
                if len(json_files) > 0:
                    first_json_path = json_files[0]
                    bad_series = check_bad_series_description(first_json_path)  # log warning if json missing
                    if bad_series:
                        logger.info(f"Skipping localizer/survey/asset/calibration/MIP series for {case_id}")
                        no_niftis.append(case_id)
                        continue
                logger.warning(f"Series UID {series_uid} imaging not converted to nifti for {case_id}")
                breakpoint()

        # -------------------
        # Segmentation conversion
        # -------------------
        if segmentation_dcm_dir is not None and segmentation_metadata_df is not None:
            combined_seg_path = os.path.join(case_outdir, "segmentation.nii.gz")
            if not os.path.exists(combined_seg_path):
                seg_match = segmentation_metadata_df[segmentation_metadata_df["series_id"] == case_id]

                if len(seg_match) == 0:
                    logger.info(f"no segmentation metadata found for {case_id}")
                    continue
                
                patient_id = seg_match["PatientID"].values[0]
                study_date = int(seg_match["StudyDate"].values[0])
                date_suffix = int(seg_match["StudyDate_suffix"].values[0])

                seg_filename = f"{patient_id}_{study_date}_{date_suffix}.seg.dcm"
                seg_dcm = os.path.join(segmentation_dcm_dir, "ai-segmentations-dcm", f"ai_{seg_filename}")

                # use quality-adjusted segmentation if available, otherwise fallback to original (AI-generated)
                for qa_prefixes in ["rad1", "ne1"]:
                    seg_dcm_qa = os.path.join(segmentation_dcm_dir, "qa-segmentations-dcm", f"{qa_prefixes}_{seg_filename}")
                    if os.path.exists(seg_dcm_qa):
                        seg_dcm = seg_dcm_qa
                        break

                # Load segmentation
                seg_ds = pydicom.dcmread(seg_dcm)
                masks_existed_before = True
                for seg_type in seg_ds.SegmentSequence:
                    seg_number = seg_type.SegmentNumber
                    seg_label = seg_type.SegmentLabel

                    seg_out_tmp = os.path.join(case_outdir, f"{seg_number}.nii.gz")
                    if not os.path.exists(seg_out_tmp):
                        masks_existed_before = False
                        if os.path.exists(seg_dcm):
                            logger.info(f"Converting segmentation for {case_id} from {seg_dcm} to {seg_out_tmp}")
                            segimage2itkimage_cmd = [
                                "segimage2itkimage",
                                "--inputDICOM", seg_dcm,
                                "--outputDirectory", case_outdir,
                                "-t", "nii",
                            ]
                            if segimage2itkimage_conda:
                                conda_prefix = os.environ.get("CONDA_PREFIX")
                                os.environ["LD_LIBRARY_PATH"] = f"{conda_prefix}/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
                                segimage2itkimage_cmd[0] = os.path.join(conda_prefix, "bin", "segimage2itkimage")
                            subprocess.run(segimage2itkimage_cmd, check=True)
                            # pad_mask_to_image(seg_out_tmp, nii_dst, seg_out_tmp)  # ensure mask matches image geometry
                            # mask_dcm_to_nii(image_dcm_dir=dicom_folder, seg_file=seg_dcm, image_nifti_file=nii_dst, seg_dir_out=case_outdir, logger=logger)  # my manual function
                        else:
                            logger.warning(f"Segmentation DICOM {seg_filename} not found for {case_id}")
                            continue
                    if not masks_existed_before:
                        pad_mask_to_image(seg_out_tmp, nii_dst, seg_out_tmp)  # ensure mask matches image geometry
                
                if not os.path.exists(combined_seg_path) or not masks_existed_before:
                    combine_masks(case_outdir, combined_seg_path=combined_seg_path, include_cyst=False)
                
                logger.info(f"Segmentation conversion completed for {case_id}")
    
    logger.info(f"Manually created NIfTIs for {len(manually_created_niftis)} series: {', '.join(manually_created_niftis)}")
    logger.info(f"No NIfTI could be created for {len(no_niftis)} series: {', '.join(no_niftis)}")

def add_orientation_column(base_dir):

    def orientation_from_iop(iop):
        """
        iop: list of 6 floats (ImageOrientationPatientDICOM)
        returns: 'axial', 'coronal', or 'sagittal'
        """
        row = np.array(iop[:3], dtype=float)
        col = np.array(iop[3:], dtype=float)
        normal = np.cross(row, col)

        axis = np.argmax(np.abs(normal))
        return ["sagittal", "coronal", "axial"][axis]

    records = []

    for case_id in sorted(os.listdir(base_dir)):
        case_path = os.path.join(base_dir, case_id)
        if not os.path.isdir(case_path):
            continue

        # find json sidecars
        json_files = glob.glob(os.path.join(case_path, "*.json"))

        if not json_files:
            records.append({
                "series_id": case_id,
                "Imaging Plane": None,
                "json_file": None
            })
            continue

        # choose first JSON (or refine selection if needed)
        json_path = json_files[0]

        with open(json_path) as f:
            meta = json.load(f)

        iop = meta.get("ImageOrientationPatientDICOM")

        if iop is None:
            orientation = None
        else:
            orientation = orientation_from_iop(iop)

        records.append({
            "series_id": case_id,
            "Imaging Plane": orientation,
            "json_file": os.path.basename(json_path)
        })

    df_orientations = pd.DataFrame(records)
    return df_orientations

def combine_masks(case_outdir, combined_seg_path=None, include_cyst=False):
    if combined_seg_path is None:
        combined_seg_path = os.path.join(case_outdir, "segmentation.nii.gz")

    combined_arr = None
    combined_affine = None

    for seg_number, seg_label in seg_mask_number_to_label.items():
        if seg_label == "Background":
            continue
        if not include_cyst and seg_label == "Cyst":
            continue
        seg_path = os.path.join(case_outdir, f"{seg_number}.nii.gz")
        if not os.path.exists(seg_path):
            continue

        img = nib.load(seg_path)
        mask_arr = img.get_fdata()
        if combined_arr is None:
            combined_arr = np.zeros_like(mask_arr)
            combined_affine = img.affine

        combined_arr[mask_arr > 0] = seg_number  # overwrite with current label

    if combined_arr is not None:
        combined_img = nib.Nifti1Image(combined_arr.astype(np.uint8), combined_affine)
        nib.save(combined_img, combined_seg_path)
        logger.info(f"Combined segmentation saved to {combined_seg_path}")
    else:
        logger.warning(f"No segmentation files found in {case_outdir} to combine.")

def download_tcga_kirc_imaging_data(tcga_kirc_imaging_data_dir, nbia_data_retriever="nbia-data-retriever", yes=False, segimage2itkimage_conda=False, min_files=5, max_thickness_mm=10):
    """
    Contains path to directory with the manifest file
    """
    # download TCIA KIRC imaging raw data
    manifest_file_name = tcia_kirc_manifest_url.split("/")[-1]
    manifest_file_path = os.path.join(tcga_kirc_imaging_data_dir, manifest_file_name)

    imaging_dcm_dir = os.path.join(tcga_kirc_imaging_data_dir, manifest_file_name.split(".")[0])
    if not os.path.exists(imaging_dcm_dir) or len(os.listdir(imaging_dcm_dir)) == 0:
        # Check if nbia-data-retriever is installed and on PATH
        if shutil.which(nbia_data_retriever) is None:
            sys.exit(f"Error: {nbia_data_retriever} not found in PATH. Please install or add it to PATH.")
        os.makedirs(imaging_dcm_dir, exist_ok=True)
        if not os.path.exists(manifest_file_path):
            subprocess.run(f"wget {tcia_kirc_manifest_url} -P {tcga_kirc_imaging_data_dir}", shell=True, check=True)
        if not yes:
            resp = input("TCIA imaging data for TCGA-KIRC project not found. Download now? [y/N]: ").strip().lower()
            if resp != "y":
                logger.warning("User declined TCIA imaging data download.")
                return
        logger.info("Downloading TCIA imaging data for TCGA-KIRC project...")
        nbia_command = f"{nbia_data_retriever} --cli {manifest_file_path} -d {tcga_kirc_imaging_data_dir} -v -f"
        if yes:
            nbia_command = "echo Y | " + nbia_command
        logger.info(nbia_command)
        subprocess.run(nbia_command, shell=True, check=True)
    
    imaging_metadata_csv = os.path.join(tcga_kirc_imaging_data_dir, "metadata.csv")
    if not os.path.exists(imaging_metadata_csv):
        download_tcga_kirc_imaging_metadata(imaging_metadata_csv)
    imaging_metadata_df = pd.read_csv(imaging_metadata_csv)
    
    segmentation_dcm_dir = os.path.join(tcga_kirc_imaging_data_dir, "segmentation_dcm")
    if not os.path.exists(segmentation_dcm_dir) or len(os.listdir(segmentation_dcm_dir)) == 0:
        os.makedirs(segmentation_dcm_dir, exist_ok=True)
        logger.info("Downloading segmentation DICOM files...")
        subprocess.run(["wget", "-O", f"{segmentation_dcm_dir}.zip", "https://zenodo.org/records/13244892/files/kidney-ct.zip?download=1"], check=True)
        subprocess.run(["unzip", f"{segmentation_dcm_dir}.zip", "-d", segmentation_dcm_dir], check=True)
    
    segmentation_metadata_csv = os.path.join(segmentation_dcm_dir, "qa-results.csv")
    segmentation_metadata_df = pd.read_csv(segmentation_metadata_csv)
    if "series_id" not in segmentation_metadata_df.columns:
        segmentation_metadata_df.rename(columns={"SeriesInstanceUID": "Series UID"}, inplace=True)
        segmentation_metadata_df = segmentation_metadata_df.merge(imaging_metadata_df[["series_id", "Series UID"]], on="Series UID", how="left")
        segmentation_metadata_df = segmentation_metadata_df[["series_id"] + [col for col in segmentation_metadata_df.columns if col != "series_id"]]
        segmentation_metadata_df.to_csv(segmentation_metadata_csv, index=False)
    
    nifti_dir = os.path.join(tcga_kirc_imaging_data_dir, "nifti")

    imaging_metadata_df = add_viable_info(imaging_dcm_dir, imaging_metadata_df, out=imaging_metadata_csv, min_files=min_files, max_thickness_mm=max_thickness_mm)
    imaging_metadata_df = imaging_metadata_df[imaging_metadata_df["is_viable"]]
    
    convert_dcm_to_nii_and_organize(imaging_dcm_dir=imaging_dcm_dir, segmentation_dcm_dir=segmentation_dcm_dir, imaging_metadata_df=imaging_metadata_df, segmentation_metadata_df=segmentation_metadata_df, nifti_dir=nifti_dir, segimage2itkimage_conda=segimage2itkimage_conda, min_files=min_files, max_thickness_mm=max_thickness_mm)

    # if "Imaging Plane" not in imaging_metadata_df.columns:
    #     orientation_df = add_orientation_column(nifti_dir)
    #     imaging_metadata_df = imaging_metadata_df.merge(orientation_df[["series_id", "Imaging Plane"]], on="series_id", how="left")
    #     imaging_metadata_df.to_csv(imaging_metadata_csv, index=False)

    return nifti_dir

def download_tcga_kirc_imaging_metadata(imaging_metadata_csv, overwrite=False):
    if os.path.exists(imaging_metadata_csv) and not overwrite:
        logger.info(f"Imaging metadata CSV already exists at {imaging_metadata_csv} and overwrite=False, skipping download.")
        return

    imaging_metadata_csv_dir = os.path.dirname(imaging_metadata_csv) if os.path.dirname(imaging_metadata_csv) != "" else "."
    additional_metadata_file_name = tcga_kirc_metadata_url.split("/")[-1]
    additional_metadata_xlsx = os.path.join(imaging_metadata_csv_dir, additional_metadata_file_name)

    os.makedirs(imaging_metadata_csv_dir, exist_ok=True)
    if not os.path.exists(additional_metadata_xlsx):
        subprocess.run(["wget", "-O", additional_metadata_xlsx, tcga_kirc_metadata_url], check=True)
    
    # add short patient ID
    imaging_metadata_df = pd.read_excel(additional_metadata_xlsx)
    if "series_id" not in imaging_metadata_df.columns:
        imaging_metadata_df.insert(0, "series_id", [f"series_{i:05d}" for i in range(len(imaging_metadata_df))])
    
    # change column names to match old format
    col_renames = {
        "Series Instance UID": "Series UID",
        "Study Instance UID": "study_id",
        "Patient ID": "patient_id",
        "Image Count": "Number of Images Original",
    }
    imaging_metadata_df.rename(columns=col_renames, inplace=True)
    imaging_metadata_df.to_csv(imaging_metadata_csv, index=False)
    
    # # save text file of patient_ids
    # subject_ids_txt = os.path.join(imaging_metadata_csv_dir, "subject_ids.txt")
    # with open(subject_ids_txt, "w") as sf:
    #     for sid in imaging_metadata_df["patient_id"].unique():
    #         sf.write(f"{sid}\n")

def get_seriesid_from_dicom_zip(zip_path, return_val=None):
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Zip file {zip_path} does not exist.")

    with tempfile.TemporaryDirectory() as tmpdir:
        # unzip
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(tmpdir)

        # grab first dicom file recursively
        dcm_files = glob.glob(os.path.join(tmpdir, "**/*.dcm"), recursive=True)

        if not dcm_files:
            # sometimes TCIA files don't end in .dcm
            dcm_files = glob.glob(os.path.join(tmpdir, "**/*"), recursive=True)

        first_dcm = dcm_files[0]

        # load header only (fast)
        dcm = pydicom.dcmread(first_dcm, stop_before_pixels=True)

        # print("File:", first_dcm)
        # print("SeriesInstanceUID:", dcm.SeriesInstanceUID)
        # print("StudyInstanceUID:", dcm.StudyInstanceUID)
        # print("SeriesDescription:", getattr(dcm, "SeriesDescription", None))
        # print("Modality:", dcm.Modality)
        # print("SliceThickness:", getattr(dcm, "SliceThickness", None))
        # print("ImageType:", getattr(dcm, "ImageType", None))

    if return_val == "SeriesInstanceUID":
        return dcm.SeriesInstanceUID
    elif return_val == "StudyInstanceUID":
        return dcm.StudyInstanceUID
    else:
        return dcm

def download_usc_tcga_kirc_data(usc_tcga_kirc_data_dir, imaging_metadata_csv=None, src_dir_name="TCIA KIRC N190", dst_dir_name="nifti", num_series=None):
    # rsync -avz "/Volumes/radiology/RenalSeg/TCIA KIRC N190" <usc_tcga_kirc_data_dir>/TCIA KIRC N190
    if not os.path.exists(usc_tcga_kirc_data_dir) or len(os.listdir(usc_tcga_kirc_data_dir)) == 0:
        raise FileNotFoundError(f"TCGA-KIRC-USC directory {usc_tcga_kirc_data_dir} does not exist or is empty. Because it is a private dataset, it must be initially downloaded manually. Please provide the path to the directory containing the USC-segmented TCGA-KIRC data.")
        # logger.error(f"TCGA-KIRC-USC directory {usc_tcga_kirc_data_dir} does not exist or is empty. Because it is a private dataset, it must be initially downloaded manually. Please provide the path to the directory containing the USC-segmented TCGA-KIRC data.")
        # return

    src_dir = os.path.join(usc_tcga_kirc_data_dir, src_dir_name)  # TCIA KIRC N190
    dst_root = os.path.join(usc_tcga_kirc_data_dir, dst_dir_name)  # nifti

    target_files = {"0502_VENOUS.nii": "Image", "ROI_602_Tumor_a.nii": "Mask"}
    target_file_types = {v: k for k, v in target_files.items()}

    project = "tcga"
    subproject = "tcga-kirc"

    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"Source directory {src_dir} does not exist. Please check the path to the USC-segmented TCGA-KIRC data.")

    rows = []
    for root, dirs, files in tqdm(os.walk(src_dir), desc="Processing USC TCGA-KIRC data"):
        row_dict = {}
        series_uid_to_series_id = {}
        for fname in files:
            if fname in target_files:
                # full source path
                src_path = os.path.join(root, fname)

                # series_id = first folder after src_dir
                rel_path = os.path.relpath(src_path, src_dir)
                patient_id = rel_path.split(os.sep)[0]
                dcm_zip_path = os.path.join(os.path.dirname(root), "DICOM.zip")
                dcm = get_seriesid_from_dicom_zip(dcm_zip_path)
                
                series_id = patient_id
                series_uid = dcm.SeriesInstanceUID
                study_id = dcm.StudyInstanceUID
                series_description = getattr(dcm, "SeriesDescription", "")
                modality = dcm.Modality
                series_uid_to_series_id[series_uid] = series_id

                # build destination path
                dst_dir = os.path.join(dst_root, series_id)
                dst_path = os.path.join(dst_dir, fname)

                # ensure destination exists
                os.makedirs(dst_dir, exist_ok=True)

                # copy
                if not os.path.exists(dst_path):
                    logger.info(f"Copying {src_path} -> {dst_path}")
                    shutil.copy2(src_path, dst_path)

                if len(row_dict) == 0:  # only add metadata once per series
                    row_dict["series_id"] = series_id
                    row_dict["project"] = project
                    row_dict["subproject"] = subproject
                    row_dict["patient_id"] = patient_id  # f"{project}_{patient_id}"
                    row_dict["study_id"] = study_id
                    row_dict["Series UID"] = series_uid
                    row_dict["Series Description"] = series_description
                    row_dict["Modality"] = modality  # all are CT
                    row_dict["Number of Images Original"] = nib.load(dst_path).get_fdata().shape[2]  # get number of slices from nifti header
                
                # colname = target_files[fname]
                # row_dict[colname] = dst_path
        
        if len(row_dict) > 0:
            rows.append(row_dict)
        
        if num_series is not None and len(rows) >= num_series:
            logger.info(f"Reached specified number of series ({num_series}), stopping further processing.")
            break
    
    def convert_tumor_mask_kirc_usc(dst_root):
        mask_file_name = target_file_types["Mask"]
        tumor_value = seg_mask_label_to_number["Tumor"]
        background_value = seg_mask_label_to_number["Background"]

        for root, dirs, files in tqdm(os.walk(dst_root), desc="Converting USC TCGA-KIRC tumor masks"):
            for fname in files:
                if fname == mask_file_name:
                    full_path = os.path.join(root, fname)
                    out_path = os.path.join(root, "segmentation_tumor.nii.gz")
                    if os.path.exists(out_path):
                        logger.debug(f"Converted tumor mask already exists at {out_path}, skipping conversion.")
                        continue

                    # Load image
                    img = nib.load(full_path)
                    data = img.get_fdata()

                    # Convert to binary mask: ==min_val → 0, >min_val → 2
                    min_val = np.min(data)
                    binary_mask = np.where(data != min_val, tumor_value, background_value).astype(np.uint8)

                    # Create new image with same affine/header
                    new_img = nib.Nifti1Image(binary_mask, img.affine, img.header.copy())

                    # Save as segmentation.nii.gz in same directory
                    nib.save(new_img, out_path)

                    logger.info(f"Converted tumor mask saved to {out_path}")
    
    convert_tumor_mask_kirc_usc(dst_root)

    #? uncomment if I can match series to original TCGA metadata
    # if imaging_metadata_csv:
    #     if isinstance(imaging_metadata_csv, str):
    #         if not os.path.exists(imaging_metadata_csv):
    #             download_tcga_kirc_imaging_metadata(imaging_metadata_csv)
    #         imaging_metadata_df = pd.read_csv(imaging_metadata_csv)
    #     elif isinstance(imaging_metadata_csv, pd.DataFrame):
    #         imaging_metadata_df = imaging_metadata_csv
    #     else:
    #         raise ValueError(f"Expected a file path or DataFrame for imaging_metadata_csv, got {type(imaging_metadata_csv)}")
    #     # make imaging_metadata_df["series_id_usc"] by merging key of series_uid_to_series_id with imaging_metadata_df["Series UID"]
    #     if "series_id_usc" in imaging_metadata_df.columns:
    #         imaging_metadata_df.drop(columns=["series_id_usc"], inplace=True)  # drop it
    #     imaging_metadata_df["series_id_usc"] = imaging_metadata_df["Series UID"].map(series_uid_to_series_id)
    # else:  # creates it new
    #     imaging_metadata_df = pd.DataFrame(rows)
    
    imaging_metadata_df = pd.DataFrame(rows)  #? erase if I can match series to original TCGA metadata

    if "patient_id" not in imaging_metadata_df.columns:
        imaging_metadata_df.insert(1, "patient_id", imaging_metadata_df["series_id"])  # copy series_id

    if not isinstance(imaging_metadata_csv, str):
        imaging_metadata_csv = os.path.join(dst_root, "metadata_usc.csv")
    imaging_metadata_df.to_csv(imaging_metadata_csv, index=False)
    logger.info(f"Saved USC TCGA-KIRC imaging metadata for {len(imaging_metadata_df)} series to {imaging_metadata_csv}")

    return dst_root

def fill_hole_and_morphological_close(left_nii, fill_holes=True, morphological_closing=True):
    if not fill_holes and not morphological_closing:
        return left_nii

    if isinstance(left_nii, nib.Nifti1Image):
        pass
    elif isinstance(left_nii, str):
        left_nii = nib.load(left_nii)
    else:
        raise ValueError("left_nii must be a file path or a Nifti1Image object")
    
    left_data = left_nii.get_fdata()
    left_mask = left_data > 0

    for z in range(left_mask.shape[2]):
        if fill_holes:
            left_mask[:, :, z] = ndi.binary_fill_holes(left_mask[:, :, z])

        if morphological_closing:
            left_mask[:, :, z] = ndi.binary_closing(left_mask[:, :, z], structure=np.ones((3,3)))

    left_mask = left_mask.astype(np.uint8)
    left_nii = nib.Nifti1Image(left_mask, affine=left_nii.affine, header=left_nii.header.copy())
    return left_nii

@measure_time_memory_storage(enabled=PROFILE_PIPELINE, disk_path=lambda: PROFILE_PIPELINE_DATA_DIR)
def run_totalsegmentator(nifti_dir, selected_segmentations, metadata_csv=None, metadata_csv_out=None, remove_small_blobs=True, fill_holes=True, morphological_closing=True, image_filename="0502_VENOUS.nii", tumor_mask_filename="segmentation_tumor.nii.gz", combined_organ_mask_filename="segmentation_organs_combined.nii.gz", mask_filename_out="segmentation.nii.gz", task="total", overwrite=False, visualize=True, orient=True, device=None):
    logger.info(f"run_totalsegmentator(nifti_dir={nifti_dir}, selected_segmentations={selected_segmentations}, metadata_csv={metadata_csv}, metadata_csv_out={metadata_csv_out}, remove_small_blobs={remove_small_blobs}, fill_holes={fill_holes}, morphological_closing={morphological_closing}, image_filename={image_filename}, tumor_mask_filename={tumor_mask_filename}, combined_organ_mask_filename={combined_organ_mask_filename}, mask_filename_out={mask_filename_out}, task={task}, overwrite={overwrite}, visualize={visualize}, orient={orient})")
    if selected_segmentations is None or len(selected_segmentations) == 0:
        raise ValueError("selected_segmentations must be a non-empty list of segmentation names to include in the combined mask.")

    if metadata_csv is not None:
        if isinstance(metadata_csv, str) and os.path.exists(metadata_csv):
            metadata_df = pd.read_csv(metadata_csv)
        elif isinstance(metadata_csv, pd.DataFrame):
            metadata_df = metadata_csv
        else:
            raise ValueError(f"Expected a file path or DataFrame for metadata_csv, got {type(metadata_csv)}")
        if "Modality" not in metadata_df.columns:
            logger.warning(f"Metadata CSV {metadata_csv} does not contain 'Modality' column. Assuming all series are CT.")
    else:
        metadata_df = None

    for essential_filenames in [image_filename, combined_organ_mask_filename, mask_filename_out]:
        if essential_filenames is None:
            raise ValueError(f"Essential filename parameter is None. Please provide a valid filename for {essential_filenames}.")
        

    if tumor_mask_filename is None:
        tumor_mask_filename = ""
    
    # iterate through subdirs of nifti_dir
    case_id_to_organ_overlap = {}
    for series_id in sorted(os.listdir(nifti_dir)):
        print(series_id)
        nifti_case_dir = os.path.join(nifti_dir, series_id)
        if not os.path.isdir(nifti_case_dir):
            continue

        if not os.path.exists(nifti_case_dir):
            logger.warning(f"NIfTI directory not found for series_id {series_id} at {nifti_case_dir}")
            continue

        if task is None:
            task = "total"
        
        modality = "CT"
        if metadata_df is not None:
            print(series_id)
            row = metadata_df[metadata_df['series_id'] == series_id]
            if not row.empty:
                modality = row['Modality'].values[0]

        #* run TotalSegmentator
        image_file = os.path.join(nifti_case_dir, image_filename)
        totalsegmentator_dir = os.path.join(nifti_case_dir, "totalsegmentator")
        totalsegmentator_command = ["TotalSegmentator", "-i", image_file, "-o", totalsegmentator_dir]
        if modality == "MRI" and task == "total":
            totalsegmentator_command += ["--task", "total_mr"]
        else:
            totalsegmentator_command += ["--task", task]
        
        if remove_small_blobs:
            totalsegmentator_command += ["--remove_small_blobs"]
        if device is not None:
            totalsegmentator_command += ["--device", device]
        
        if all(os.path.exists(os.path.join(totalsegmentator_dir, f"{seg_name}.nii.gz")) for seg_name in selected_segmentations) and not overwrite:
            logger.info(f"TotalSegmentator has already been run for series_id {series_id}.")
        else:
            logger.info(f"Running TotalSegmentator for series_id {series_id} with command: {' '.join(totalsegmentator_command)}")
            try:
                subprocess.run(totalsegmentator_command, check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Error occurred while running TotalSegmentator for series_id {series_id}: {e}")
                continue

        if not all(os.path.exists(os.path.join(totalsegmentator_dir, f"{seg_name}.nii.gz")) for seg_name in selected_segmentations):
            logger.error(f"Predicted segmentation files not found for series_id {series_id}. Skipping.")
            continue

        #* combine all organ segmentations
        tumor_segmentation_file = os.path.join(nifti_case_dir, tumor_mask_filename) if tumor_mask_filename else None  # segmentation_tumor.nii.gz
        predicted_organ_segmentation_file_combined = os.path.join(nifti_case_dir, combined_organ_mask_filename)  # segmentation_organs.nii.gz
        combined_organ_tumor_segmentation_file = os.path.join(nifti_case_dir, mask_filename_out)  # segmentation.nii.gz
        niis = {}
        if not os.path.exists(predicted_organ_segmentation_file_combined) or not os.path.exists(combined_organ_tumor_segmentation_file) or overwrite:
            for seg_name in selected_segmentations:
                seg_path = os.path.join(totalsegmentator_dir, f"{seg_name}.nii.gz")
                niis[seg_name] = nib.load(seg_path)
            logger.info(f"Combining full organ segmentations for series_id {series_id}...")

            # Sanity checks
            shapes = [nii.shape for nii in niis.values()]
            assert len(set(shapes)) == 1, f"Shape mismatch detected: {shapes}"
            affines = [nii.affine for nii in niis.values()]
            assert all(np.allclose(affines[0], a) for a in affines[1:]), "Affines do not match"

            #* manual updates
            if fill_holes or morphological_closing:
                for seg_name, seg_nii in niis.items():
                    logger.info(f"Applying hole filling and morphological closing to {seg_name} segmentation for series_id {series_id}...")
                    niis[seg_name] = fill_hole_and_morphological_close(seg_nii, fill_holes=fill_holes, morphological_closing=morphological_closing)

            # Union
            masks = [(nii.get_fdata() > 0) for nii in niis.values()]
            union = np.logical_or.reduce(masks).astype(np.uint8)

            # Save
            selected_nii = niis[selected_segmentations[0]]  # use the first selected segmentation as reference for affine/header
            union_nii = nib.Nifti1Image(union, selected_nii.affine, selected_nii.header.copy())

            # # orient - already done automatically
            # if orient:
            #     logger.info(f"Reorienting combined segmentation to RAS for series_id {series_id}...")
            #     union_nii = nib.as_closest_canonical(union_nii)

            nib.save(union_nii, predicted_organ_segmentation_file_combined)

        
        #* Combined organ and tumor segmentations
        if not os.path.exists(combined_organ_tumor_segmentation_file) or overwrite:
            if tumor_segmentation_file is None or not os.path.exists(tumor_segmentation_file):
                shutil.copy2(predicted_organ_segmentation_file_combined, combined_organ_tumor_segmentation_file)  # copy
                # os.rename(predicted_organ_segmentation_file_combined, combined_organ_tumor_segmentation_file)  # rename
            else:
                logger.info(f"Combining organ and tumor segmentations for series_id {series_id}...")
                tumor_nii = nib.load(tumor_segmentation_file)

                union_orient = nib.orientations.aff2axcodes(union_nii.affine)
                tumor_orient = nib.orientations.aff2axcodes(tumor_nii.affine)
                if tumor_orient != union_orient:
                    logger.debug(f"Reorienting tumor mask from {tumor_orient} to match organ masks orientation {union_orient}")
                    # use tumor as reference
                    if tumor_orient == ('R', 'A', 'S'):
                        union_nii = nib.as_closest_canonical(union_nii)
                        for seg_name, seg_nii in niis.items():
                            niis[seg_name] = nib.as_closest_canonical(seg_nii)
                    else:
                        pass  # not implemented
                        # transform = nib.orientations.ornt_transform(union_orient, tumor_orient)
                        # union_data = union_nii.get_fdata()
                        # union_reoriented = nib.orientations.apply_orientation(union_data, transform)
                        # new_affine = union_nii.affine @ nib.orientations.inv_ornt_aff(transform, union_nii.shape)
                        # union_nii = nib.Nifti1Image(union_reoriented, new_affine, union_nii.header.copy())
                        # for seg_name, seg_nii in niis.items():
                        #     seg_orient = nib.orientations.aff2axcodes(seg_nii.affine)
                        #     if seg_orient != tumor_orient:
                        #         logger.debug(f"Reorienting {seg_name} segmentation from {seg_orient} to match tumor mask orientation {tumor_orient}")
                        #         niis[seg_name] = nib.as_closest_canonical(seg_nii)
                        #     else:
                        #         niis[seg_name] = seg_nii

                # Sanity checks
                assert tumor_nii.shape == union_nii.shape, "Shapes do not match"
                assert np.allclose(tumor_nii.affine, union_nii.affine), "Affines do not match"

                # Combine: 0=background, 1=organ, 2=tumor (if tumor overlaps organ, tumor label takes precedence)
                tumor = tumor_nii.get_fdata() > 0
                combined = np.zeros_like(tumor, dtype=np.uint8)
                
                overlapping_organs = []
                for seg_name, seg_nii in niis.items():
                    organ = seg_nii.get_fdata() > 0
                    organ_overlap = np.sum(tumor & organ)
                    if organ_overlap > 0:  # keeps only the organs with any tumor overlap
                        overlapping_organs.append(seg_name)
                        combined[organ] = 1
                if len(overlapping_organs) == 0:
                    logger.warning(f"No overlap found between tumor and any organ for series_id {series_id}. Will keep all organs.")
                    for seg_name, seg_nii in niis.items():
                        organ = seg_nii.get_fdata() > 0
                        combined[organ] = 1

                combined[tumor] = 2  # tumor + organ gets labeled 2, organ alone gets labeled 1 (if there is a tumor in some part of this organ), background gets labeled 0
                case_id_to_organ_overlap[series_id] = ",".join(overlapping_organs) if overlapping_organs else "none"

                # Save
                combined_nii = nib.Nifti1Image(combined, tumor_nii.affine, tumor_nii.header.copy())
                nib.save(combined_nii, combined_organ_tumor_segmentation_file)

        #* visualize
        if visualize:
            logger.info(f"Visualizing series_id {series_id}...")
            
            #* loading
            # image
            img_nii = nib.load(image_file)
            img = img_nii.get_fdata()
            img = np.rot90(img, axes=(0, 1))

            # totalsegmentator mask
            visualization_mask_file = combined_organ_tumor_segmentation_file if os.path.exists(combined_organ_tumor_segmentation_file) else predicted_organ_segmentation_file_combined
            mask_totalsegmentator_nii = nib.load(visualization_mask_file)
            mask_totalsegmentator = mask_totalsegmentator_nii.get_fdata()
            mask_totalsegmentator = np.rot90(mask_totalsegmentator, axes=(0, 1))
            
            num_plots = 2
            vmin, vmax = None, None  # -200, 300  # soft tissue window

            for z in tqdm(range(img.shape[2]), desc=f"Visualizing slices for series_id {series_id}"):
                logger.debug(f"Slice {z}: Image min={img[:, :, z].min()}, max={img[:, :, z].max()}, mean={img[:, :, z].mean()}; Mask unique values={np.unique(mask_totalsegmentator[:, :, z])}")

                title = f"{series_id}_slice{z:03d}"
                out_path = os.path.join(totalsegmentator_dir, "visualization", f"{title}.png")
                out_path_with_organ = out_path.replace(".png", "_K.png")  # add suffix for slices with organ pixels

                n_organ_pixels = np.sum(mask_totalsegmentator[:, :, z] > 0)
                if n_organ_pixels > 0:
                    out_path = out_path_with_organ
                # else:  # skip visualizations of slices without any organ pixels to save space, since there are many such slices
                #     continue

                if os.path.exists(out_path) and not overwrite:
                    logger.debug(f"Visualization already exists at {out_path}. Skipping.")
                    continue

                fig, axes = plt.subplots(1, num_plots, figsize=(12, 6))

                # Left: image only
                axes[0].imshow(img[:, :, z], cmap="gray", vmin=vmin, vmax=vmax)  # rotate 90 degrees for correct orientation
                axes[0].set_title("Image only")
                axes[0].axis("off")
                
                # Right: image + mask overlay
                axes[1].imshow(img[:, :, z], cmap="gray", vmin=vmin, vmax=vmax)  # rotate 90 degrees for correct orientation
                axes[1].imshow(mask_totalsegmentator[:, :, z] > 0, cmap="Reds", alpha=0.2)
                axes[1].set_title(f"Image + organ mask totalsegmentator ({n_organ_pixels} organ pixels)")
                axes[1].axis("off")

                plt.suptitle(title)
                # plt.tight_layout()

                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                plt.savefig(out_path)
                plt.close(fig)
    
    #* save metadata CSV
    if metadata_csv is not None:
        if "tumor_side" in metadata_df.columns and overwrite:
            logger.debug(f"Metadata CSV {metadata_csv} already contains 'tumor_side' column. It will be overwritten with new values based on TotalSegmentator results.")
            metadata_df.drop(columns=["tumor_side"], inplace=True)
        if "tumor_side" not in metadata_df.columns:
            tumor_side_df = pd.DataFrame(list(case_id_to_organ_overlap.items()), columns=["series_id", "tumor_side"])
            metadata_df = metadata_df.merge(tumor_side_df, on="series_id", how="left")
        if metadata_csv_out is None and isinstance(metadata_csv, str):
            metadata_csv_out = metadata_csv
        metadata_df.to_csv(metadata_csv_out, index=False)

def get_label_value_from_mask(mask_path):
    # get label value from mask - unique non-zero values
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata()
    labels = np.unique(mask_data)
    labels = labels[labels != 0]  # exclude background
    if len(labels) == 0:
        raise ValueError(f"No non-zero labels found in mask {mask_path}.")
    if len(labels) > 1:
        raise ValueError(f"Multiple non-zero labels found in mask {mask_path}: {labels}. Please ensure only one label is present, or provide the desired label value.")
    label = labels[0]  # get the first one
    return int(label)

def get_number_of_voxels_and_number_of_slices(mask_path):
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata()
    num_voxels = np.count_nonzero(mask_data)
    num_slices = mask_data.shape[2]  # assuming slices are along the third dimension
    return num_voxels, num_slices

@measure_time_memory_storage(enabled=PROFILE_PIPELINE, disk_path=lambda: PROFILE_PIPELINE_DATA_DIR)
def prepare_csv_for_pyradiomics(raw_image_data_dir, output_csv_path = "radiogenomics_imaging_data.csv", imaging_file_name="imaging.nii.gz", mask_file_name="segmentation.nii.gz", metadata_df=None, metadata_df_columns_to_merge=None, series_description_keywords_exclude="default", overwrite=False):
    """
    Expected structure of raw_image_data_dir:
    raw_image_data_dir/SERIES_ID/
        {imaging_file_name}
        {mask_file_name}
    metadata_df: optional DataFrame with metadata to merge (must contain 'series_id' column)
    """
    if os.path.exists(output_csv_path) and os.path.getsize(output_csv_path) > 0:
        if not overwrite:
            logger.info(f"Output CSV {output_csv_path} already exists. Skipping CSV preparation.")
            return
        else:
            logger.warning(f"Overwriting existing CSV {output_csv_path}.")
    
    input_data = []
    # Validate input directory structure
    if not os.path.exists(raw_image_data_dir):
        raise FileNotFoundError(f"Raw image data directory {raw_image_data_dir} does not exist.")
    required_files = [imaging_file_name]
    for case_dir in sorted(os.listdir(raw_image_data_dir)):
        case_path = os.path.join(raw_image_data_dir, case_dir)
        if not os.path.isdir(case_path):
            continue
        if any(not os.path.exists(os.path.join(case_path, req_file)) for req_file in required_files):
            continue

        image_path = os.path.join(case_path, imaging_file_name)
        mask_path = os.path.join(case_path, mask_file_name)
        # num_voxels, num_slices = get_number_of_voxels_and_number_of_slices(mask_path)

        if not os.path.exists(mask_path):
            mask_path = None

        input_data.append({
            "series_id": case_dir,
            "Image": image_path,
            "Mask": mask_path,
        })
    
    if len(input_data) == 0:
        raise ValueError(f"No valid series found in {raw_image_data_dir} with required files: {', '.join(required_files)}")
    
    input_df = pd.DataFrame(input_data)
    
    if metadata_df is not None:
        if isinstance(metadata_df, str):
            if not os.path.exists(metadata_df):
                raise FileNotFoundError(f"Metadata CSV file {metadata_df} does not exist.")
            metadata_df = pd.read_csv(metadata_df)
        if not isinstance(metadata_df, pd.DataFrame):
            raise ValueError("metadata_df must be a pandas DataFrame or a valid CSV file path.")
        if "series_id" not in metadata_df.columns:
            raise ValueError("metadata_df must contain 'series_id' column for merging.")
        if metadata_df_columns_to_merge is not None:
            if "series_id" in metadata_df_columns_to_merge:
                metadata_df_columns_to_merge.remove("series_id")  # avoid duplication
            metadata_df_columns_to_merge = [col for col in metadata_df_columns_to_merge if col in metadata_df.columns]
            metadata_df = metadata_df[["series_id"] + metadata_df_columns_to_merge]
        input_df = input_df.merge(metadata_df, on="series_id", how="left")
    
    if series_description_keywords_exclude == "default":
        logger.info(f"Using default series description keywords to exclude for radiomics: {SERIES_DESCRIPTION_KEYWORDS_EXCLUDE_RADIOMICS}. To disable this filtering, set series_description_keywords_exclude to None.")
        series_description_keywords_exclude = SERIES_DESCRIPTION_KEYWORDS_EXCLUDE_RADIOMICS

    if "Series Description" in input_df.columns and series_description_keywords_exclude is not None:
        pattern = "|".join(series_description_keywords_exclude)
        exclude_mask = input_df["Series Description"].str.contains(pattern, case=False, na=False)
        input_df = input_df[~exclude_mask]

    input_df.to_csv(output_csv_path, index=False)
    logger.info(f"Saved radiogenomics imaging DataFrame for {len(input_df)} series to {output_csv_path}")

def perform_pyradiomics_on_single_image_and_mask(image_file, segmentation_file, params=None, label=[1,2]):
    def label_is_int(label):
        try:
            _ = int(label)
            return True
        except ValueError:
            return False

    if isinstance(label, list):
        ma = sitk.ReadImage(segmentation_file)
        ma_arr = sitk.GetArrayFromImage(ma)
        for l in label:
            if not label_is_int(l):
                raise ValueError(f"Label '{l}' is not an integer. All labels must be integers.")
            ma_arr[ma_arr == int(l)] = 1

        ma_merged = sitk.GetImageFromArray(ma_arr)
        ma_merged.CopyInformation(ma)  # geometric information
        label = 1
    else:
        ma_merged = segmentation_file
        if not label_is_int(label):
            raise ValueError(f"Label '{label}' is not an integer. Label must be an integer.")
    label = int(label)

    if params is not None and (params.endswith(".yaml") or params.endswith(".yml")):
        extractor = featureextractor.RadiomicsFeatureExtractor(params)
    else:
        extractor = featureextractor.RadiomicsFeatureExtractor()
    
    features = extractor.execute(image_file, ma_merged, label=label)

    return dict(features)

@measure_time_memory_storage(enabled=PROFILE_PIPELINE, disk_path=lambda: PROFILE_PIPELINE_DATA_DIR)
def perform_radiomics_pipeline(input_csv_path, output_csv_path, threads=1, param=None, image_column="Image", mask_column="Mask", label=[1,2], overwrite=False):
    """
    Expected input_csv_path to have 2 columns: {image_column} and {mask_column}
    """
    if os.path.exists(output_csv_path) and os.path.getsize(output_csv_path) > 0:
        if not overwrite:
            logger.info(f"Output CSV {output_csv_path} already exists. Skipping radiomics feature extraction.")
            return

    radiomics_df = pd.read_csv(input_csv_path)

    if image_column not in radiomics_df.columns or mask_column not in radiomics_df.columns or "series_id" not in radiomics_df.columns:
        raise ValueError(f"Input CSV {input_csv_path} must contain '{image_column}' and '{mask_column}' columns.")
    
    logger.info(f"Starting radiomics feature extraction for {input_csv_path}...")
    radiomic_features = []
    for idx, row in tqdm(radiomics_df.iterrows(), total=radiomics_df.shape[0]):  # drop rows with missing files
        image_path = row[image_column]
        mask_path = row[mask_column]
        if not os.path.exists(image_path):
            logger.warning(f"Image file {image_path} does not exist. Skipping.")
            radiomics_df = radiomics_df.drop(idx)
            continue
        if not os.path.exists(mask_path):
            logger.warning(f"Mask file {mask_path} does not exist. Skipping.")
            radiomics_df = radiomics_df.drop(idx)
            continue
        radiomic_features_individual = perform_pyradiomics_on_single_image_and_mask(image_path, mask_path, params=param, label=label)  #? investigate multithreading
        radiomic_features_individual["series_id"] = row["series_id"]
        radiomic_features.append(radiomic_features_individual)

    # merge all individual feature dictionaries into a single DataFrame
    radiomic_features_df = pd.DataFrame(radiomic_features)
    radiomics_df = radiomics_df.merge(radiomic_features_df, on="series_id", how="left")
    radiomics_df.to_csv(output_csv_path, index=False)

    logger.info(f"Radiomics feature extraction completed. Features saved to {output_csv_path}")

def plot_histogram(data, bins=20, vertical_line=None, vertical_line_label=None, xlabel="Value", ylabel="Frequency", ylog=False, title=None, output_path=None):
    # -------- Plot histogram --------
    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=bins)
    if vertical_line is not None:
        plt.axvline(vertical_line, color="red", linestyle="--", label=vertical_line_label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ylog:
        plt.yscale("log")
    if title is not None:
        plt.title(title)
    plt.legend()
    if output_path is not None:
        plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved histogram to {output_path}")

def load_nifti_file(nifti_file):
    if isinstance(nifti_file, nib.Nifti1Image):
        return nifti_file
    elif isinstance(nifti_file, str):
        if not os.path.exists(nifti_file):
            raise FileNotFoundError(f"NIfTI file {nifti_file} does not exist.")
        return nib.load(nifti_file)
    else:
        raise ValueError("nifti_file must be a file path or a Nifti1Image object")

def crop_to_nonempty(image, threshold=None, pad=5):
    """
    Crop an image to the bounding box of voxels > threshold.
    Works with nib.Nifti1Image or numpy arrays.

    Parameters
    ----------
    image : nib.Nifti1Image or np.ndarray
    threshold : float
    pad : int

    Returns
    -------
    cropped_image : same type as input
    bbox : tuple
        (min0, max0, min1, max1, ...)
    """

    if isinstance(image, nib.Nifti1Image):
        data = image.get_fdata()
        affine = image.affine
        header = image.header.copy()
        return_nifti = True
    elif isinstance(image, np.ndarray):
        data = image
        return_nifti = False
    else:
        raise ValueError(f"Expected nib.Nifti1Image or np.ndarray, got {type(image)}")

    # foreground detection
    if threshold is None:  # get min value and add small epsilon to avoid numerical issues with exact zeros
        threshold = data.min() + 1e-5
    nonempty = np.argwhere(data > threshold)

    if nonempty.size == 0:
        raise ValueError(f"No voxels above threshold {threshold}")

    mins = nonempty.min(axis=0)
    maxs = nonempty.max(axis=0) + 1

    shape = np.array(data.shape)

    if pad:
        mins = np.maximum(mins - pad, 0)
        maxs = np.minimum(maxs + pad, shape)

    slices = tuple(slice(mn, mx) for mn, mx in zip(mins, maxs))
    cropped = data[slices]

    bbox = tuple(v for pair in zip(mins, maxs) for v in pair)

    if not return_nifti:
        return cropped, bbox

    # update affine
    new_affine = affine.copy()
    dims = min(3, data.ndim)

    new_affine[:dims, 3] = affine[:dims, :dims] @ mins[:dims] + affine[:dims, 3]

    cropped_nii = nib.Nifti1Image(cropped, new_affine, header)

    return cropped_nii, bbox

def crop_with_bbox(nii, bbox):
    """
    Crop a NIfTI image using a bounding box.
    Works for 2D or 3D images.

    Parameters
    ----------
    nii : nib.Nifti1Image
    bbox : tuple
        (min0, max0, min1, max1, ...)

    Returns
    -------
    nib.Nifti1Image
    """

    data = nii.get_fdata()
    affine = nii.affine

    # convert bbox -> mins/maxs
    mins = np.array(bbox[0::2])
    maxs = np.array(bbox[1::2])

    # build slice tuple dynamically
    slices = tuple(slice(mn, mx) for mn, mx in zip(mins, maxs))
    cropped = data[slices]

    # update affine origin shift
    new_affine = affine.copy()
    shift = affine[:len(mins), :len(mins)] @ mins + affine[:len(mins), 3]
    new_affine[:len(mins), 3] = shift

    return nib.Nifti1Image(cropped, new_affine, nii.header.copy())

@measure_time_memory_storage(enabled=PROFILE_PIPELINE, disk_path=lambda: PROFILE_PIPELINE_DATA_DIR)
def apply_mask(image_file, mask_file, label=None, crop=True, min_value=None, pad_after_crop=5, out_image=True, out_mask=True, overwrite=False):
    if out_image is True:
        out_image = define_default_out_nifti(image_file, suffix="_masked")
    if out_mask is True:
        out_mask = define_default_out_nifti(mask_file, suffix="_masked")
    
    if out_image is not None and os.path.exists(out_image) and out_mask is not None and os.path.exists(out_mask) and not overwrite:
        logger.debug(f"Masked image and mask already exist at {out_image} and {out_mask}. Skipping masking.")
        return out_image, out_mask
    
    image_nii = load_nifti_file(image_file)
    mask_nii = load_nifti_file(mask_file)

    image_data = image_nii.get_fdata()
    mask_data = mask_nii.get_fdata().astype(np.int16)

    if image_data.shape != mask_data.shape:
        raise ValueError(f"Image and mask shapes do not match: {image_data.shape} vs {mask_data.shape}")

    if min_value is None:
        min_value = image_data.min()  # default to minimum value in the image if not provided

    if label is None:
        masked_image_data = np.where(mask_data > 0, image_data, min_value)
    else:
        if isinstance(label, int) or isinstance(label, float):
            masked_image_data = np.where(mask_data == label, image_data, min_value)
        elif isinstance(label, list):
            masked_image_data = np.where(np.isin(mask_data, label), image_data, min_value)
        else:
            raise ValueError(f"label must be an int, float, list of ints/floats, or None. Got {type(label)}")

    masked_image_nii = nib.Nifti1Image(masked_image_data, affine=image_nii.affine, header=image_nii.header.copy())

    if crop:
        masked_image_nii, bbox = crop_to_nonempty(masked_image_nii, threshold=min_value, pad=pad_after_crop)
        mask_nii = crop_with_bbox(mask_nii, bbox)
    
    if out_image:
        nib.save(masked_image_nii, out_image)
        logger.info(f"Saved masked image to {out_image}")

    if out_mask:
        nib.save(mask_nii, out_mask)
        logger.info(f"Saved masked mask to {out_mask}")
    
    # convert False/"" to None for consistent return type
    if not out_image:
        out_image = None
    if not out_mask:
        out_mask = None

    if out_image is None and out_mask is None:
        return masked_image_nii, mask_nii

    return out_image, out_mask

@measure_time_memory_storage(enabled=PROFILE_PIPELINE, disk_path=lambda: PROFILE_PIPELINE_DATA_DIR)
def compute_shape_histogram(nifti_dir, image_filename):
    x_extents, y_extents, z_extents = [], [], []

    # -------- First Pass: compute tumor z-extent for all series --------
    for series_id in sorted(os.listdir(nifti_dir)):
        case_dir = os.path.join(nifti_dir, series_id)
        image_path = os.path.join(case_dir, image_filename)

        if not os.path.exists(image_path):
            continue

        image_nii = nib.load(image_path)
        x_extents.append(image_nii.shape[0])
        y_extents.append(image_nii.shape[1])
        if len(image_nii.shape) > 2:
            z_extents.append(image_nii.shape[2])
    
    visualization_dir = os.path.join(os.path.dirname(nifti_dir), "visualization")
    os.makedirs(visualization_dir, exist_ok=True)

    extents_95th = {}
    for axis, extents in zip(["x", "y", "z"], [x_extents, y_extents, z_extents]):
        if len(extents) == 0:
            logger.warning(f"No extents found for axis {axis}. Skipping histogram and 95th percentile calculation.")
            extents_95th[axis] = None
            continue

        extents = np.array(extents)
        extent_max = extents.max()
        extent_95th = int(np.percentile(extents, 95))

        logger.info(f"{axis}-extent: max={extent_max}, 95th percentile={extent_95th}")
        plot_histogram(extents, bins=20, xlabel=f"{axis}-Extent (voxels)", vertical_line_label=f"95th percentile ({extent_95th})", title=f"{axis}-extent Distribution", output_path=os.path.join(os.path.dirname(nifti_dir), "visualization", f"{axis.lower()}_extent_histogram.png"))
        extents_95th[axis] = extent_95th
    
    return extents_95th

@measure_time_memory_storage(enabled=PROFILE_PIPELINE, disk_path=lambda: PROFILE_PIPELINE_DATA_DIR)
def choose_slice_with_most_mask_single_image(image, mask, mask_value=2, out_image=True, out_mask=True, overwrite=False):
    # --------------------------------------------------
    # Determine input types
    # --------------------------------------------------
    if isinstance(image, str):
        img_nii = nib.load(image)
        img = img_nii.get_fdata()
        affine = img_nii.affine
        header = img_nii.header
        image_path = image
    elif isinstance(image, nib.Nifti1Image):
        img_nii = image
        img = img_nii.get_fdata()
        affine = img_nii.affine
        header = img_nii.header
        image_path = None
    elif isinstance(image, np.ndarray):
        img = image
        img_nii = None
        affine = None
        header = None
        image_path = None
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    if isinstance(mask, str):
        mask_nii = nib.load(mask)
        mask_arr = mask_nii.get_fdata()
        mask_path = mask
    elif isinstance(mask, nib.Nifti1Image):
        mask_nii = mask
        mask_arr = mask_nii.get_fdata()
        mask_path = None
    elif isinstance(mask, np.ndarray):
        mask_arr = mask
        mask_nii = None
        mask_path = None
    else:
        raise ValueError(f"Unsupported mask type: {type(mask)}")

    # --------------------------------------------------
    # Output paths (only if input paths exist)
    # --------------------------------------------------
    out_img_path = None
    out_mask_path = None

    if image_path and out_image is True:
        out_img_path = define_default_out_nifti(image_path, suffix="_best_slice")

    if mask_path and out_mask is True:
        out_mask_path = define_default_out_nifti(mask_path, suffix="_best_slice")

    if (
        out_img_path
        and out_mask_path
        and os.path.exists(out_img_path)
        and os.path.exists(out_mask_path)
        and not overwrite
    ):
        logger.debug("Best slice already exists, skipping.")
        return out_img_path, out_mask_path, {}

    # sanity check for matching shapes
    if img.shape != mask_arr.shape:
        raise ValueError(f"Image and mask shapes do not match: {img.shape} vs {mask_arr.shape}")

    # check if mask_value exists in mask
    if (isinstance(mask_value, (list, tuple, set)) and not np.isin(mask_value, mask_arr).any()) or (not isinstance(mask_value, (list, tuple, set)) and mask_value not in mask_arr):
            logger.warning(f"None of the specified mask values {mask_value} found in mask. Returning None.")
            return None, 0, {}

    # check if image only has one slice (2D image)
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        logger.info("Image has only one slice. Returning original image and mask.")
        slice_info = {
            f"slice_with_most_mask_{mask_value}": 0,
            f"number_of_{mask_value}_mask_pixels_in_best_slice": int(np.sum(mask_arr == mask_value)),
        }
        if out_image is None and out_mask is None:
            return img, mask_arr, slice_info

        if out_image and not os.path.exists(out_img_path):
            nib.save(nib.Nifti1Image(img, affine, header), out_img_path)
            logger.info(f"Saved best slice image to {out_img_path}")

        if out_mask and not os.path.exists(out_mask_path):
            nib.save(nib.Nifti1Image(mask_arr, affine, header), out_mask_path)
            logger.info(f"Saved best slice mask to {out_mask_path}")

        return out_img_path, out_mask_path, slice_info
    

    
    # --------------------------------------------------
    # Identify mask voxels
    # --------------------------------------------------
    if isinstance(mask_value, (list, tuple, set)):
        selected_mask = np.isin(mask_arr, list(mask_value))
        mask_value_str = ",".join(map(str, mask_value))
    else:
        selected_mask = (mask_arr == mask_value)
        mask_value_str = str(mask_value)

    # --------------------------------------------------
    # Compute mask area per slice
    # --------------------------------------------------
    mask_area_per_slice = selected_mask.sum(axis=(0, 1))

    if mask_area_per_slice.max() == 0:
        return None, 0

    best_slice_idx = int(np.argmax(mask_area_per_slice))
    tumor_pixels_in_best_slice = int(mask_area_per_slice[best_slice_idx])

    slice_info = {
        f"slice_with_most_mask_{mask_value_str}": best_slice_idx,
        f"number_of_{mask_value_str}_mask_pixels_in_best_slice": tumor_pixels_in_best_slice,
    }

    # --------------------------------------------------
    # Extract slices
    # --------------------------------------------------
    img_slice = img[:, :, best_slice_idx]
    mask_slice = mask_arr[:, :, best_slice_idx]

    # --------------------------------------------------
    # If arrays were passed → return arrays
    # --------------------------------------------------
    if img_nii is None and mask_nii is None:
        return img_slice, mask_slice, slice_info

    # --------------------------------------------------
    # Otherwise construct NIfTI outputs
    # --------------------------------------------------
    new_affine = affine.copy()

    out_img = nib.Nifti1Image(img_slice, new_affine, header)
    out_mask = nib.Nifti1Image(mask_slice, new_affine, header)

    if out_image is None and out_mask is None:
        return out_img, out_mask, slice_info

    if out_img_path:
        nib.save(out_img, out_img_path)

    if out_mask_path:
        nib.save(out_mask, out_mask_path)

    return out_img_path, out_mask_path, slice_info
    
@measure_time_memory_storage(enabled=PROFILE_PIPELINE, disk_path=lambda: PROFILE_PIPELINE_DATA_DIR)
def crop_and_pad(image_path, xdim=None, ydim=None, zdim=None, min_value=None, out=True, overwrite=False):
    if image_path is None:
        return None

    # -------------------------
    # Determine input type
    # -------------------------
    if isinstance(image_path, str):
        if out is True:
            out = define_default_out_nifti(image_path, suffix="_sized")

        if out is not None and os.path.exists(out) and not overwrite:
            logger.debug(f"Cropped/padded image already exists at {out} and overwrite=False, skipping.")
            return out

        nii = nib.load(image_path)
        img = nii.get_fdata()
        affine = nii.affine
        header = nii.header.copy()
        return_nifti = True

    elif isinstance(image_path, nib.Nifti1Image):
        nii = image_path
        img = nii.get_fdata()
        affine = nii.affine
        header = nii.header.copy()
        return_nifti = True
        out = None

    elif isinstance(image_path, np.ndarray):
        img = image_path
        return_nifti = False
        out = None

    else:
        raise ValueError(f"Unsupported input type: {type(image_path)}")

    # -------------------------
    # Prepare shapes
    # -------------------------
    current_shape = img.shape
    ndim = img.ndim

    if min_value is None:
        min_value = img.min()

    dims = [xdim, ydim, zdim]
    target_shape = tuple(
        dims[i] if dims[i] is not None else current_shape[i]
        for i in range(ndim)
    )

    new_img = np.full(target_shape, min_value, dtype=img.dtype)

    src_slices = []
    dst_slices = []

    for i in range(ndim):

        src = current_shape[i]
        dst = target_shape[i]

        if src >= dst:
            start_src = (src - dst) // 2
            end_src = start_src + dst
            start_dst = 0
            end_dst = dst
        else:
            start_src = 0
            end_src = src
            start_dst = (dst - src) // 2
            end_dst = start_dst + src

        src_slices.append(slice(start_src, end_src))
        dst_slices.append(slice(start_dst, end_dst))

    new_img[tuple(dst_slices)] = img[tuple(src_slices)]

    # -------------------------
    # Array case (simple)
    # -------------------------
    if not return_nifti:
        return new_img

    # -------------------------
    # Update affine for NIfTI
    # -------------------------
    crop_offset = np.array([s.start for s in src_slices])
    new_affine = affine.copy()

    if ndim >= 3:
        R = new_affine[:3, :3]
        t = new_affine[:3, 3]
        new_affine[:3, 3] = R @ crop_offset[:3] + t
    else:
        R = new_affine[:2, :2]
        t = new_affine[:2, 3]
        new_affine[:2, 3] = R @ crop_offset[:2] + t

    new_nii = nib.Nifti1Image(new_img, new_affine, header)

    if out is None:
        return new_nii

    nib.save(new_nii, out)
    return out

def process_images(nifti_dir, orient=False, resample=False, target_spacing=(0.8, 0.8, 3.0), clip_min=None, clip_max=None, normalize=False, normalization_method="volume", image_filename="0502_VENOUS.nii", mask_filename="segmentation.nii.gz", overwrite=False):
    # walk through all subdirs
    image_filename_set, mask_filename_set = set(), set()
    image_files = []
    for series_id in tqdm(sorted(os.listdir(nifti_dir)), desc="Processing images"):
        case_dir = os.path.join(nifti_dir, series_id)
        image_file = os.path.join(case_dir, image_filename)
        mask_file = os.path.join(case_dir, mask_filename)
        if not os.path.exists(image_file):
            logger.warning(f"Image file not found for series_id {series_id} at {image_file}. Skipping.")
            continue

        if orient:
            image_file = set_canonical_orientation(image_file, out=True, overwrite=overwrite)
            if os.path.exists(mask_file):
                mask_file = set_canonical_orientation(mask_file, out=True, overwrite=overwrite)
        
        if resample:
            image_file = resample_image(image_file, target_spacing=target_spacing, is_label=False, out=True, overwrite=overwrite)
            if os.path.exists(mask_file):
                mask_file = resample_image(mask_file, target_spacing=target_spacing, is_label=True, out=True, overwrite=overwrite)
        
        if clip_min is not None or clip_max is not None:  # eg (-200, 300) for soft tissue window - done in training loop
            image_file = clip_intensity_range(image_file, clip_min=clip_min, clip_max=clip_max, out=True, overwrite=overwrite)

        if normalize and normalization_method == "volume":  # done in training loop
            image_file = normalize_intensity(image_file, normalization_method=normalization_method, out=True, overwrite=overwrite)

        image_filename_updated, mask_filename_updated = os.path.basename(image_file), os.path.basename(mask_file)
        image_filename_set.add(image_filename_updated)
        mask_filename_set.add(mask_filename_updated)
        image_files.append(image_file)
    
    if len(image_filename_set) > 1 or len(mask_filename_set) > 1:
        raise ValueError(f"Multiple different image or mask filenames found after processing: {image_filename_set}, {mask_filename_set}. Please ensure consistent naming.")

    check_dataset_intensity_consistency(image_files)

    if normalize and normalization_method == "dataset":  # done in training loop
        normalize_intensity(image_files, normalization_method=normalization_method, out=True, overwrite=overwrite)

    return image_filename_updated, mask_filename_updated

def compute_volume_stats(path):
    img = nib.load(path)
    data = img.get_fdata()

    # Remove extreme padding
    data = np.clip(data, a_min=-1024, a_max=None)

    return {
        "path": path,
        "min": np.min(data),
        "max": np.max(data),
        "mean": np.mean(data),
        "std": np.std(data)
    }

def check_dataset_intensity_consistency(image_files, mean_tol=100, std_frac_tol=0.15, min_floor=-1024, max_ceiling=3000):
    stats = [compute_volume_stats(f) for f in image_files]

    means = np.array([s["mean"] for s in stats])
    stds  = np.array([s["std"] for s in stats])
    # mins  = np.array([s["min"] for s in stats])
    # maxs  = np.array([s["max"] for s in stats])

    median_mean = np.median(means)
    median_std  = np.median(stds)

    errors = []

    for s in stats:
        if abs(s["mean"] - median_mean) > mean_tol:
            errors.append(f"{s['path']} mean out of range: {s['mean']:.1f}")

        if abs(s["std"] - median_std) > std_frac_tol * median_std:
            errors.append(f"{s['path']} std out of range: {s['std']:.1f}")

        if s["min"] < min_floor:
            errors.append(f"{s['path']} min below floor: {s['min']:.1f}")

        if s["max"] > max_ceiling:
            errors.append(f"{s['path']} max above ceiling: {s['max']:.1f}")

    if errors:
        logger.error("Intensity QC FAILED with the following issues:")
        for error in errors:
            logger.error(error)
    else:
        logger.info("Intensity QC PASSED.")

    return stats

BASE_PATTERNS = {
    "chest": r"(chest|thorax|thor|lung|breast|mammo|mammary|axilla|ch\b|pa\b)",
    "abdomen": r"(abdomen|abdom|abdo|abd\b|ab\b|kub)",
    "pelvis": r"(pelvis|pelv|bladder|pel\b|hip\b)",
    "head_neck": r"(skull|head|neck|brain|c spine)",
    "whole_body": r"(pet ct|skull base to mid thigh|whole body)",
    "renal": r"(renal|kidney|kidneys|neph|ureter|urogram|uro\b|pyelo)",
}

PHASE_PATTERNS = {
    "Scout": r"(scout|topogram|surview|locator|scanogram)",
    "Non-contrast": r"(non[_\-\s]?contrast|without contrast|w/o|w o\b|unenhanced|native|c-|i-|no contrast|renal colic|stone)",
    "Arterial": r"(arterial|art\b|45 ?sec|60 ?sec|70 ?sec)",
    "Nephrographic": r"(neph|paren|90 ?sec|100 ?sec|100s|120 ?sec)",
    "Delayed": r"(delay|delayed|excret|urogram|3 ?min|5 ?min|8 ?min|10 ?min|12 ?min|15 ?min|180 ?sec)",
    "Post-contrast (unspecified phase)": r"(post|with contrast|i\+|c\+|contrast\b)",
}

PROJECT_OVERRIDES = {
    "BRCA": {
        "extra_patterns": {
            "chest": r"\b(breast|mammo|mammary|axilla)\b",
        },
        "rename": {
            "chest": "Chest/Breast",
        },
    },
    "KIRC": {
        "extra_patterns": {
            "renal": r"\b(renal|kidney|neph|ureter|urogram|stone)\b",
        },
        "special_rules": "renal",
    },
    "OV": {
        "extra_patterns": {
            "vascular": r"\b(vascular|aorta)\b",
            "cardiac": r"\b(cardiac)\b",
        },
    },
    "BLCA": {
        "extra_patterns": {
            "renal": r"\b(urogram|pyelo|renal|kidney|triphasic|uro)\b",
            "pelvis": r"\b(bladder)\b",
        },
    },
}

def normalize(s):
    s = s.lower()
    s = re.sub(r"[^\w]+", " ", s)   # replace / - ^ etc with space
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def has_tokens(s, *tokens):
    toks = s.split()
    return all(t in toks for t in tokens)



def categorize_region_tcga(s, project=None):
    if not isinstance(s, str):
        return "Unknown"

    s = normalize(s)
    flags = {}

    for region, pattern in BASE_PATTERNS.items():
        flags[region] = bool(re.search(pattern, s))
    
    if project in PROJECT_OVERRIDES:
        extra = PROJECT_OVERRIDES[project].get("extra_patterns", {})
        for region, pattern in extra.items():
            if re.search(pattern, s):
                flags[region] = True
    
    # Explicit CAP shorthand (single token)
    if re.search(r"\bcap\b", s):
        flags["chest"] = flags["abdomen"] = flags["pelvis"] = True

    if has_tokens(s, "c", "a", "p"):
        flags["chest"] = flags["abdomen"] = flags["pelvis"] = True
    elif has_tokens(s, "a", "p"):
        flags["abdomen"] = flags["pelvis"] = True
    elif has_tokens(s, "c", "a"):
        flags["chest"] = flags["abdomen"] = True

    if flags.get("whole_body"):
        return "Whole Body"

    if flags.get("renal"):
        if flags.get("abdomen") or flags.get("pelvis"):
            return "Abdomen/Pelvis (Renal)"
        return "Renal"

    if flags.get("chest") and flags.get("abdomen") and flags.get("pelvis"):
        return "Chest/Abdomen/Pelvis"
    if flags.get("abdomen") and flags.get("pelvis"):
        return "Abdomen/Pelvis"
    if flags.get("chest") and flags.get("abdomen"):
        return "Chest/Abdomen"

    for region in ["chest", "abdomen", "pelvis", "head_neck"]:
        if flags.get(region):
            label = region.capitalize()
            if project in PROJECT_OVERRIDES:
                label = PROJECT_OVERRIDES[project].get("rename", {}).get(region, label)
            return label

    return "Other"

def categorize_phase(series_desc):
    if not isinstance(series_desc, str):
        return "Other"
    
    s = normalize(series_desc)
    
    for phase, pattern in PHASE_PATTERNS.items():
        if re.search(pattern, s):
            return phase
    
    return "Other"

def print_tcia_info(df, project=None, series_uid_col=None, study_uid_col=None, patient_id_col=None):
    if series_uid_col is None:
        for col in ["Series Instance UID", "Series UID"]:
            if col in df.columns:
                series_uid_col = col
                break
    if study_uid_col is None:
        for col in ["Study Instance UID", "study_id", "Study UID"]:
            if col in df.columns:
                study_uid_col = col
                break
    if patient_id_col is None:
        for col in ["Patient ID", "patient_id", "Subject ID"]:
            if col in df.columns:
                patient_id_col = col
                break

    total_series = len(df)
    total_studies = df[study_uid_col].nunique()
    total_patients = df[patient_id_col].nunique()

    print("=== TOTALS ===")
    print(f"Total series:   {total_series}")
    print(f"Total studies:  {total_studies}")
    print(f"Total patients: {total_patients}")
    print()

    print("=== BY MODALITY ===")

    modality_summary = (
        df.groupby("Modality")
        .agg(
            num_series=(series_uid_col, "nunique"),
            num_studies=(study_uid_col, "nunique"),
            num_patients=(patient_id_col, "nunique")
        )
        .sort_values("num_series", ascending=False)
    )

    print(modality_summary)
    print()

    print("=== BY MODALITY + REGION ===")

    modality_region_summary = (
        df.groupby(["Modality", "ParsedRegion"])
        .agg(
            num_series=(series_uid_col, "nunique"),
            num_studies=(study_uid_col, "nunique"),
            num_patients=(patient_id_col, "nunique")
        )
        .sort_values(["Modality", "num_series"], ascending=[True, False])
    )

    print(modality_region_summary)

    if "kirc" in project.lower():
        print("=== KIRC PHASE SUMMARY (CT only) ===")

        # Restrict to CT abdomen/pelvis-ish regions
        df_ct = df[df["Modality"] == "CT"].copy()

        # Create phase column from Series Description
        if "Phase" not in df_ct.columns:
            df_ct["Phase"] = df_ct["Series Description"].apply(categorize_phase)

        phase_summary = (
            df_ct.groupby("Phase")
            .agg(
                num_series=(series_uid_col, "nunique"),
                num_studies=(study_uid_col, "nunique"),
                num_patients=(patient_id_col, "nunique")
            )
            .sort_values("num_series", ascending=False)
        )

        print(phase_summary)
        print()

    print()

def get_series_uids_from_manifest(manifest_file_path):
    # read in manifest_file_name as text file
    series_instance_uids = []
    with open(manifest_file_path, "r") as f:
        collect = False
        for line in f:
            # print(f"Reading line: {line}")
            line = line.strip()
            if line == "ListOfSeriesToDownload=":
                collect = True
                continue
            if collect and line:
                series_instance_uids.append(line)
    return series_instance_uids

def load_mask_data(mask_path):
    if ".nii" in mask_path:
        mask_nii = nib.load(mask_path)
        mask_data = mask_nii.get_fdata()
    elif ".dcm" in mask_path:
        mask_data = pydicom.dcmread(mask_path).pixel_array
    else:
        raise ValueError(f"Unsupported mask file format: {mask_path}")
    return mask_data

def dice_score(mask1, mask2, val=None):
    # if val is None, set all >0 to 1
    if val is not None:
        mask1 = (mask1 == val)
        mask2 = (mask2 == val)
    else:
        mask1 = (mask1 > 0)
        mask2 = (mask2 > 0)
    
    # check if either mask is empty
    if mask1.size == 0 or mask2.size == 0:
        logger.warning("One or both masks are empty. Returning Dice score of 1.0 if both empty, else 0.0.")
        return 1.0 if np.array_equal(mask1, mask2) else 0.0

    intersection = np.sum(mask1 & mask2)
    denom = np.sum(mask1) + np.sum(mask2)
    if denom == 0:
        return 1.0  # both empty
    return 2.0 * intersection / denom

def create_totalseg_scirep_dice_histograms(totalseg_dir, scirep_dir, segmentation_metadata_df, visualization_dir=".", mask_filename="segmentation.nii.gz", tumor_mask_filename=None, image_filename=None, visualize=False):
    labels_dict = {"organ": None, "tumor": 2}  # None means all
    totalseg_scirepai_dice_scores_dict = {label: {} for label in labels_dict.keys()}
    totalseg_scirepcorr_dice_scores_dict = {label: {} for label in labels_dict.keys()}

    for idx, row in segmentation_metadata_df.iterrows():
        case_id = row["series_id"]
        patient_id = row["PatientID"]
        study_uid = row["StudyInstanceUID"]
        series_uid = row["SeriesInstanceUID"]
        ai_segmentation_filename = row["AISegmentation"]
        corrected_segmentation_filename = row["CorrectedSegmentation"]

        logger.debug(f"Patient ID: {patient_id}, study_id: {study_uid}, Series UID: {series_uid}")

        totalsegmentator_mask_path = os.path.join(totalseg_dir, case_id, mask_filename)
        scirep_mask_ai_path = os.path.join(scirep_dir, "ai-segmentations-dcm", ai_segmentation_filename)
        scirep_mask_corrected_path = os.path.join(scirep_dir, "corrected-segmentations-dcm", corrected_segmentation_filename) if pd.notna(corrected_segmentation_filename) else None

        totalsegmentator_mask = load_mask_data(totalsegmentator_mask_path)
        scirep_mask_ai = load_mask_data(scirep_mask_ai_path)
        scirep_mask_corrected = load_mask_data(scirep_mask_corrected_path) if scirep_mask_corrected_path else None

        for label_name, label in labels_dict.items():
            if label_name == "tumor" and tumor_mask_filename is None:
                continue  # skip tumor evaluation if no tumor mask is available

            totalseg_scirepai_label_dice = dice_score(totalsegmentator_mask, scirep_mask_ai, val=label)
            totalseg_scirepcorr_label_dice = dice_score(totalsegmentator_mask, scirep_mask_corrected, val=label) if scirep_mask_corrected is not None else None
            totalseg_scirepai_dice_scores_dict[label_name][series_uid] = totalseg_scirepai_label_dice
            totalseg_scirepcorr_dice_scores_dict[label_name][series_uid] = totalseg_scirepcorr_label_dice
        
        if visualize:
            if not image_filename:
                logger.warning("Image filename not provided, skipping visualization.")
                continue
            
            image_path = os.path.join(totalseg_dir, case_id, image_filename)
            if not os.path.exists(image_path):
                logger.warning(f"Image file {image_path} not found, skipping visualization for case {case_id}.")
                continue
        
            img = nib.load(image_path).get_fdata()
            vmin, vmax = None, None  # -200, 300  # soft tissue window
            for z in range(img.shape[2]):
                # print(f"Slice {z}: Image min={img[:, :, z].min()}, max={img[:, :, z].max()}, mean={img[:, :, z].mean()}; Mask unique values={np.unique(mask_totalsegmentator[:, :, z])}")

                out_path = os.path.join(totalseg_dir, case_id, "totalseg_and_scirep_visualizations", f"visualization_{case_id}_slice{z:03d}.png")

                if os.path.exists(out_path):
                    logger.debug(f"Visualization already exists at {out_path}. Skipping.")
                    continue

                fig, axes = plt.subplots(1, 3, figsize=(12, 6))

                # Left: image only
                axes[0].imshow(np.rot90(img[:, :, z]), cmap="gray", vmin=vmin, vmax=vmax)  # rotate 90 degrees for correct orientation
                axes[0].set_title("Image only")
                axes[0].axis("off")
                
                # Middle: image + mask overlay
                n_organ_pixels = np.sum(totalsegmentator_mask[:, :, z] > 0)
                axes[1].imshow(np.rot90(img[:, :, z]), cmap="gray", vmin=vmin, vmax=vmax)  # rotate 90 degrees for correct orientation
                axes[1].imshow(np.rot90(totalsegmentator_mask[:, :, z] > 0), cmap="Reds", alpha=0.2)
                axes[1].set_title(f"Image + organ mask totalsegmentator ({n_organ_pixels} organ pixels)")
                axes[1].axis("off")

                # Right: image + TCGA mask overlay
                n_organ_pixels = np.sum(scirep_mask_ai[:, :, z] > 0)
                axes[2].imshow(np.rot90(img[:, :, z]), cmap="gray", vmin=vmin, vmax=vmax)  # rotate 90 degrees for correct orientation
                axes[2].imshow(np.rot90(scirep_mask_ai[:, :, z] > 0), cmap="Reds", alpha=0.2)
                axes[2].set_title(f"Image + organ mask SciRep ({n_organ_pixels} organ pixels)")
                axes[2].axis("off")

                plt.suptitle(f"Axial slice {z}")
                plt.tight_layout()
                
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                plt.savefig(out_path)
                plt.close(fig)

    for label_name in labels_dict:
        if label_name == "tumor" and tumor_mask_filename is None:
            continue  # skip tumor evaluation if no tumor mask is available

        totalseg_scirepai_dice_scores = [score for score in totalseg_scirepai_dice_scores_dict[label_name].values() if score is not None]
        totalseg_scirepcorr_dice_scores = [score for score in totalseg_scirepcorr_dice_scores_dict[label_name].values() if score is not None]
        plot_histogram(totalseg_scirepai_dice_scores, xlabel="Dice Score", title=f"TotalSegmentator vs SciRep AI Segmentation Dice Scores for {label_name}", output_path=os.path.join(visualization_dir, f"totalseg_scirepai_dice_histogram_{label_name}.png"))
        if totalseg_scirepcorr_dice_scores:
            plot_histogram(totalseg_scirepcorr_dice_scores, xlabel="Dice Score", title=f"TotalSegmentator vs SciRep Corrected Segmentation Dice Scores for {label_name}", output_path=os.path.join(visualization_dir, f"totalseg_scirepcorr_dice_histogram_{label_name}.png"))

def add_acquisition_time(metadata_df, dcm_dir):
    metadata_df = metadata_df.copy()
    if "Acquisition Time" in metadata_df.columns:
        logger.warning("'Acquisition Time' column already exists in metadata_df. It will be overwritten with new values extracted from DICOM files.")
        return metadata_df  # already has acquisition time, skip processing
        # metadata_df.drop(columns=["Acquisition Time"], inplace=True)

    series_to_folder = make_series_to_folder_mapping(dcm_dir)

    acquisition_times = []
    for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Processing series"):
        case_id = row["series_id"]
        series_uid = row["Series UID"]

        if series_uid not in series_to_folder:
            logger.debug(f"Series UID {series_uid} imaging not found in DICOM directory tree")
            continue
        
        dicom_folder = series_to_folder[series_uid]
        dcm_files = glob.glob(os.path.join(dicom_folder, "**/*.dcm"), recursive=True)
        first_dcm = dcm_files[0]
        dcm = pydicom.dcmread(first_dcm, stop_before_pixels=True)
        acquisition_time = int(dcm.AcquisitionTime) if "AcquisitionTime" in dcm else None
        if acquisition_time is None:
            logger.debug(f"Acquisition time not found for Series UID {series_uid} in case {case_id}")
        acquisition_times.append(acquisition_time)

    metadata_df["Acquisition Time"] = acquisition_times
    return metadata_df

phase_to_time_range = {
    # "Non-contrast": (0, 10),
    "Arterial": (15, 50),  # commonly 30-35 sec
    "Nephrographic": (55, 100),  # commonly 65-80 sec
    "Delayed": (250, 700),  # commonly 300-375 sec
}

def update_phase_column_with_acquisition_time(metadata_df, dcm_dir=None):
    metadata_df = metadata_df.copy()

    phase_update_targets = {
        "Post-contrast (unspecified phase)",
        "Other"
    }

    if "Acquisition Time" not in metadata_df.columns:
        metadata_df = add_acquisition_time(metadata_df, dcm_dir)
    
    study_ids = metadata_df["study_id"].unique()
    for study_id in tqdm(study_ids, desc="Updating phase labels based on acquisition time"):
        study_mask = metadata_df["study_id"] == study_id
        study_df = metadata_df[study_mask].copy()

        # Restrict to viable CT only
        study_df = study_df[
            (study_df["Modality"] == "CT") &
            (study_df["is_viable"])
        ].copy()

        if len(study_df) <= 1:
            continue

        # Convert Acquisition Time to sortable numeric if needed
        study_df["acq_seconds"] = study_df["Acquisition Time"].astype(str).str.zfill(6)
        study_df["acq_seconds"] = (
            study_df["acq_seconds"].str[0:2].astype(int) * 3600 +
            study_df["acq_seconds"].str[2:4].astype(int) * 60 +
            study_df["acq_seconds"].str[4:6].astype(int)
        )

        study_df.sort_values("acq_seconds", inplace=True)

        # -----------------------------
        # Find non-contrast baseline
        # -----------------------------
        noncontrast_rows = study_df[study_df["phase"] == "Non-contrast"]

        if len(noncontrast_rows) == 0:
            continue  # cannot compute time since noncontrast

        baseline_time = noncontrast_rows["acq_seconds"].min()

        # Compute delta from non-contrast
        study_df["delta_from_noncontrast"] = (
            study_df["acq_seconds"] - baseline_time
        )

        # -----------------------------
        # Update ambiguous phases only
        # -----------------------------
        update_mask = study_df["phase"].isin(phase_update_targets)

        for idx in study_df[update_mask].index:
            delta = study_df.loc[idx, "delta_from_noncontrast"]

            for phase_name, (low, high) in phase_to_time_range.items():
                if low <= delta <= high:
                    study_df.loc[idx, "phase"] = phase_name
                    break  # stop once matched

        # Write back
        metadata_df.loc[study_df.index, "Phase"] = study_df["phase"]
    
    return metadata_df

def check_and_delete_bad_niftis(
    metadata_df,
    nifti_dir,
    is_4d=True,
    min_z=10,
    max_zoom_maximum=20,
    filter_from_metadata=True,
    image_filename="imaging.nii.gz",
    filter_if_max_zoom_not_in_si_position=False,
    out=None
):
    num_cases_original = len(metadata_df)
    series_ids_original = set(metadata_df["series_id"].unique())

    status_rows = []
    max_zoom_dict = {}

    for case_id in tqdm(metadata_df["series_id"].unique(), desc="Checking NIfTI files for quality control"):
        case_status = {
            "series_id": case_id,
            "is_4d": False,
            "is_thin": False,
            "is_missing": False,
            "max_zoom": np.nan,
            "orientation_original": np.nan,
            "sampling_original": np.nan,
            "max_zoom_not_in_si_position": False,
        }
        case_dir = os.path.join(nifti_dir, case_id)
        nifti_path = os.path.join(case_dir, image_filename)
        if not os.path.exists(nifti_path):
            case_status["is_missing"] = True
            if os.path.exists(case_dir):
                logger.warning(f"Image file {nifti_path} not found for case {case_id}. This case will be removed from the dataset.")
                shutil.rmtree(case_dir, ignore_errors=True)
            status_rows.append(case_status)
            continue
        img_nii = nib.load(nifti_path)
        try:
            img = img_nii.get_fdata()
        except np.exceptions.DTypePromotionError as e:  # weird corrupted file
            logger.warning(f"Error loading NIfTI file {nifti_path} for case {case_id}: {e}. This may indicate a corrupted or non-standard NIfTI file. This case will be removed from the dataset.")
            case_status["is_missing"] = True
            if os.path.exists(case_dir):
                shutil.rmtree(case_dir, ignore_errors=True)
            status_rows.append(case_status)
            continue
        
        if min_z is not None and img.shape[2] < min_z:
            logger.warning(f"Case {case_id} has only {img.shape[2]} slices, which is below the minimum threshold of {min_z}. This case will be removed from the dataset.")
            case_status["is_thin"] = True
            if os.path.exists(case_dir):
                shutil.rmtree(case_dir, ignore_errors=True)
            status_rows.append(case_status)
            continue
        
        is_4d_case = (img.ndim == 4)
        case_status["is_4d"] = is_4d_case
        orientation_original = nib.orientations.aff2axcodes(img_nii.affine)
        sampling_original = img_nii.header.get_zooms()
        case_status["orientation_original"] = orientation_original
        case_status["sampling_original"] = sampling_original
        spatial_zooms = sampling_original[:len(orientation_original)]
        if len(spatial_zooms) > 0:
            max_zoom_axis = int(np.argmax(spatial_zooms))
            case_status["max_zoom_not_in_si_position"] = orientation_original[max_zoom_axis] not in ("S", "I")
        if is_4d and is_4d_case:
            # erase folder
            if os.path.exists(case_dir):
                logger.warning(f"Case {case_id} appears to have 4D NIfTI image. This may indicate a DICOM series with multiple time points or phases that was incorrectly converted to a single 4D NIfTI file. Please review the original DICOM data for this case to determine if it should be split into separate series for each phase/time point. For now, this case will be marked as 4D in the metadata and may need special handling in downstream processing.")
                shutil.rmtree(case_dir, ignore_errors=True)
        if max_zoom_maximum is not None:
            zooms = img_nii.header.get_zooms()
            case_status["max_zoom"] = max(zooms)
            if max(zooms) > max_zoom_maximum:
                logger.warning(f"Case {case_id} has at least one zoom value above the maximum threshold of {max_zoom_maximum}. This case will be removed from the dataset.")
                if os.path.exists(case_dir):
                    shutil.rmtree(case_dir, ignore_errors=True)

        status_rows.append(case_status)

    status_df = pd.DataFrame(status_rows)
    status_cols = [
        "is_missing",
        "is_thin",
        "is_4d",
        "max_zoom",
        "orientation_original",
        "sampling_original",
        "max_zoom_not_in_si_position",
    ]
    drop_cols = [c for c in status_cols if c in metadata_df.columns]
    if drop_cols:
        metadata_df = metadata_df.drop(columns=drop_cols)
    metadata_df = metadata_df.merge(status_df, on="series_id", how="left")

    if filter_from_metadata:
        metadata_df = metadata_df[~metadata_df["is_missing"].fillna(False)].copy()  # filter out missing series from metadata if they were deleted from dataset
        metadata_df = metadata_df[~metadata_df["is_thin"].fillna(False)].copy()  # filter out thin series from metadata if they were deleted from dataset
        if is_4d:
            metadata_df = metadata_df[~metadata_df["is_4d"].fillna(False)].copy()  # filter out 4D series from metadata if they were deleted from dataset
        if max_zoom_maximum is not None:
            metadata_df["max_zoom"] = metadata_df["max_zoom"].fillna(0)  # if missing zoom info, assume it's 0 which is below any reasonable threshold
            max_zoom_dict = dict(zip(metadata_df["series_id"], metadata_df["max_zoom"]))
            metadata_df = metadata_df[metadata_df["max_zoom"] <= max_zoom_maximum].copy()  # filter out series with zoom values exceeding the maximum threshold from metadata if they were deleted from dataset
        if filter_if_max_zoom_not_in_si_position:
            metadata_df = metadata_df[~metadata_df["max_zoom_not_in_si_position"].fillna(False)].copy()

    if filter_from_metadata:
        num_cases_after = len(metadata_df)
        logger.info(f"Filtered out {num_cases_original - num_cases_after} / {num_cases_original} cases from metadata based on missing files, 4D images, or excessive zoom values. Remaining cases: {num_cases_after}.")
        removed_cases = series_ids_original - set(metadata_df["series_id"].unique())
        if removed_cases:
            logger.info(f"Removed cases: {', '.join(removed_cases)}")
            logger.info(
                f"Flags: '4D': {int(status_df['is_4d'].fillna(False).sum())}, "
                f"'thin': {int(status_df['is_thin'].fillna(False).sum())}, "
                f"'missing': {int(status_df['is_missing'].fillna(False).sum())}, "
                f"'zooms': {len(max_zoom_dict)}"
            )
    if out:
        metadata_df.to_csv(out, index=False)
    return metadata_df


def check_few_slices(metadata_df, nifti_dir, image_filename="imaging.nii.gz"):
    is_4d = {}
    for case_id in metadata_df["series_id"].unique():
        case_dir = os.path.join(nifti_dir, case_id)
        nifti_path = os.path.join(case_dir, image_filename)
        if not os.path.exists(nifti_path):
            is_4d[case_id] = False
            continue
        img_nii = nib.load(nifti_path)
        img = img_nii.get_fdata()
        is_4d_case = (img.ndim == 4)
        is_4d[case_id] = is_4d_case
        if is_4d_case:
            # erase folder
            if os.path.exists(case_dir):
                logger.warning(f"Case {case_id} appears to have 4D NIfTI image. This may indicate a DICOM series with multiple time points or phases that was incorrectly converted to a single 4D NIfTI file. Please review the original DICOM data for this case to determine if it should be split into separate series for each phase/time point. For now, this case will be marked as 4D in the metadata and removed from the dataset.")
                shutil.rmtree(case_dir, ignore_errors=True)
    
    # merge is_4d info back to metadata_df by series_id
    is_4d_df = pd.DataFrame(list(is_4d.items()), columns=["series_id", "is_4d"])
    metadata_df = metadata_df.merge(is_4d_df, on="series_id", how="left")
    return metadata_df

def get_slice_position(d):
    if hasattr(d, "ImagePositionPatient"):
        return float(d.ImagePositionPatient[2])
    elif hasattr(d, "SliceLocation"):
        return float(d.SliceLocation)
    elif hasattr(d, "InstanceNumber"):
        return float(d.InstanceNumber)
    else:
        return 0  # last fallback


def view_dicom_file(dicom_file, title="default", vmin=-200, vmax=300, show_colorbar=True, out_path=None):
    print(f"Viewing DICOM file: {dicom_file}")
    if not dicom_file.lower().endswith(".dcm"):
        print(f"Warning: {dicom_file} does not have a .dcm extension, but will attempt to read as DICOM.")
    dcm = pydicom.dcmread(dicom_file)
    plt.imshow(dcm.pixel_array, cmap="gray", vmin=vmin, vmax=vmax)
    if show_colorbar:
        plt.colorbar()
    if title == "default":
        title = f'DICOM {os.path.basename(dicom_file).split(".")[0]}'
    if title:
        plt.title(title)
    plt.axis("off")
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.show()

def view_dicom_directory(dicom_dir, vmin=None, vmax=None):
    def get_dicom_volume(dicom_dir):
        print(f"Viewing DICOM series in directory: {dicom_dir}")
        files = [pydicom.dcmread(os.path.join(dicom_dir, f))
                for f in os.listdir(dicom_dir)]

        if not files:
            print(f"No DICOM files found in directory: {dicom_dir}")
            return None
        
        files.sort(key=get_slice_position)
        
        volume = np.stack([f.pixel_array for f in files])
        return volume
    volume = get_dicom_volume(dicom_dir)
    if volume is not None:
        def show_slice(i):
            plt.imshow(volume[i], cmap="gray", vmin=vmin, vmax=vmax)
            plt.axis("off")
            plt.show()
        interact(show_slice, i=(0, volume.shape[0]-1));

def view_dicom(dicom_path, vmin=None, vmax=None, title="default", show_colorbar=True, out_path=None):
    logger.info(f"Viewing DICOM path: {dicom_path}")
    if os.path.isfile(dicom_path):
        view_dicom_file(dicom_path, title=title, vmin=vmin, vmax=vmax, show_colorbar=show_colorbar, out_path=out_path)
    elif os.path.isdir(dicom_path):
        view_dicom_directory(dicom_path, vmin=vmin, vmax=vmax)
    else:
        raise ValueError(f"Path {dicom_path} is neither a file nor a directory.")

def view_nifti(nifti_file, z=None, title="default", vmin=None, vmax=None, overlay_mask=None, show_colorbar=True, out_path=None, _out_dir=None):
    if isinstance(nifti_file, str):
        if not os.path.exists(nifti_file):
            print(f"NIfTI file not found: {nifti_file}")
            return

        logger.info(f"Viewing NIfTI file: {nifti_file}")
        
        nii = nib.load(nifti_file)
    elif isinstance(nifti_file, nib.Nifti1Image):
        nii = nifti_file
    else:
        raise ValueError(f"Expected nifti_file to be a file path or Nifti1Image object, got {type(nifti_file)}")

    volume = nii.get_fdata()

    # If 2D, make it behave like a single-slice 3D volume
    if volume.ndim == 2:
        volume = volume[..., np.newaxis]
        z = 0
    nz = volume.shape[2]

    volume = np.rot90(volume)
    if overlay_mask is not None:
        cmap = ListedColormap([
            (0, 0, 0, 0),      # 0 = transparent
            (1, 0, 0, 0.7),    # 1 = light red (RGBA)
            (1, 0, 0, 0.9)     # 2 = darker red
        ])
        mask = nib.load(overlay_mask).get_fdata()
        if mask.ndim == 2:
            mask = mask[..., np.newaxis]
        mask = np.rot90(mask)
        assert mask.shape == volume.shape, "Mask and image shapes do not match"
    if z is None:
        # view volume
        def show_slice(z):
            plt.figure(figsize=(6,6))
            plt.imshow(volume[:, :, z], cmap="gray", vmin=vmin, vmax=vmax)
            if overlay_mask is not None:
                plt.imshow(mask[:, :, z], cmap=cmap, alpha=0.3)
            plt.axis("off")
            if title:
                plt.title(title)
            plt.show()
        interact(show_slice, z=(0, nz-1))
    else:
        # view slice
        if z < 0 or z >= nz:
            raise ValueError(f"z must be between 0 and {nz-1}")

        plt.imshow(volume[:, :, z], cmap="gray", vmin=vmin, vmax=vmax)
        if show_colorbar:
            plt.colorbar()
        if overlay_mask is not None:
            plt.imshow(mask[:, :, z], cmap=cmap, alpha=0.3)
        if title == "default":
            title = f"NIfTI slice {z}"
        if title:
            plt.title(title)
        plt.axis("off")
        if out_path is True:
            # save at nifti_file, but replace .nii/.nii.gz with _slice{z}.png
            out_path = nifti_file.replace(".nii.gz", f"_slice{z:03d}.png").replace(".nii", f"_slice{z:03d}.png")
            if overlay_mask is not None:
                out_path = out_path.replace(".png", "_with_mask.png")
            if _out_dir is not None:
                out_path = os.path.join(_out_dir, os.path.basename(out_path))
        if out_path:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            plt.savefig(out_path, bbox_inches='tight', dpi=300)
        plt.show()

@measure_time_memory_storage(enabled=PROFILE_PIPELINE, disk_path=lambda: PROFILE_PIPELINE_DATA_DIR)
def nii_to_npy(nifti_file, out=True, overwrite=False):
    if not isinstance(nifti_file, str):
        raise ValueError(f"Expected a file path for nifti_file, got {type(nifti_file)}")
    if not os.path.exists(nifti_file):
        raise FileNotFoundError(f"NIfTI file not found: {nifti_file}")

    if out is True:
        if nifti_file.endswith(".nii.gz"):
            out = nifti_file[:-7] + ".npy"
        elif nifti_file.endswith(".nii"):
            out = nifti_file[:-4] + ".npy"
        else:
            out = nifti_file + ".npy"
    
    if out is not None and os.path.exists(out) and not overwrite:
        return out
    
    nii = nib.load(nifti_file)
    volume = np.asanyarray(nii.dataobj)
    
    if out is None:
        return volume

    np.save(out, volume)
    logger.info(f"Saved NumPy volume to {out}")
    return out


def generate_all_orientations(img):
    if isinstance(img, str):
        if not os.path.exists(img):
            raise FileNotFoundError(f"NIfTI file not found: {img}")
        img = nib.load(img)
    elif not isinstance(img, nib.Nifti1Image):
        raise ValueError(f"Expected img to be a file path or Nifti1Image object, got {type(img)}")

    data = img.get_fdata()
    affine = img.affine

    current_axcodes = nib.orientations.aff2axcodes(affine)
    current_ornt = nib.orientations.axcodes2ornt(current_axcodes)

    orientations = [
        ('R','A','S'),
        ('L','A','S'),
        ('R','P','S'),
        ('L','P','S'),
        ('R','A','I'),
        ('L','A','I'),
        ('R','P','I'),
        ('L','P','I'),
    ]

    out = {}

    for target_axcodes in orientations:
        target_ornt = nib.orientations.axcodes2ornt(target_axcodes)
        transform = nib.orientations.ornt_transform(current_ornt, target_ornt)

        new_data = nib.orientations.apply_orientation(data, transform)

        # ✅ fix affine properly
        new_affine = affine @ nib.orientations.inv_ornt_aff(transform, data.shape)

        new_img = nib.Nifti1Image(new_data, new_affine)
        out["".join(target_axcodes)] = new_img

    return out
