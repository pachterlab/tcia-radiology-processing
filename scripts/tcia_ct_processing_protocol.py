import os
import sys
import shutil
import subprocess
import argparse
import pandas as pd
from tqdm import tqdm
import yaml
from tcia_radiology_processing import utils
from tcia_radiology_processing.constants import tcia_dataset_to_info

base_directory = os.path.dirname(os.path.abspath(""))


def parse_args():
    parser = argparse.ArgumentParser(description="TCIA CT processing pipeline")
    parser.add_argument("--dataset", default="tcga-kirc")
    parser.add_argument("--data_dir_base", default="/home/jrich/data/radiogenomics_apr26", help="Base directory for radiogenomics data.")
    parser.add_argument("--nbia_data_retriever", default="nbia-data-retriever", help="Path or executable name for NBIA Data Retriever.")
    parser.add_argument("--num_series", default=None, help="Number of series to keep. Use 'none' for all series.")
    parser.add_argument("--using_usc_data", action="store_true", help="Whether to use USC TCGA-KIRC data instead of downloading from TCIA. Only applicable for tcga-kirc dataset, and will be set to False for other datasets.")
    parser.add_argument("--image_dimensionality", default="3D", choices=["2D", "3D"], help="Whether to process images as 2D (selecting single slice with most mask) or 3D (keeping all slices).")
    parser.add_argument("--do_radiomics", action="store_true", help="Whether to perform radiomics analysis.")
    parser.add_argument("--do_masking", action="store_true", help="Whether to apply masking to images (using tumor mask if available, otherwise using organ mask). Masking will be applied before standardizing dimensions and normalization.")
    parser.add_argument("--disable_orient", action="store_false", help="Whether to disable reorienting images to canonical orientation. By default, images will be reoriented to canonical orientation.")
    parser.add_argument("--disable_clip", action="store_false", help="Whether to disable clipping intensity values. By default, intensity values will be clipped to the range specified in tcia_dataset_to_info for each dataset.")
    parser.add_argument("--disable_resample", action="store_false", help="Whether to disable resampling images to a common spacing. By default, images will be resampled to a common spacing specified in the code.")
    parser.add_argument("--mask_value_for_best_slice_selection", default=2, choices=[1,2], type=int, help="Whether to use tumor mask (2) or organ mask (1) for best slice selection when image_dimensionality is 2D. Ignored if image_dimensionality is 3D. If tumor mask is not available and this is set to 2, will fall back to using organ mask.")
    parser.add_argument("--mask_values", default=None, choices=[None, 1, 2], type=int, help="Mask label to apply (1/2) or 'none' for all labels > 0.")
    parser.add_argument("--normalize", action="store_true", help="Whether to normalize intensity values. By default, intensity values will be normalized using the method specified in --normalization-method.")
    parser.add_argument("--normalization-method", default="volume")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = args.dataset
    data_dir_base = args.data_dir_base  #!!! os.path.join(base_directory, "data", "radiogenomics")
    nbia_data_retriever = args.nbia_data_retriever  # path to nbia-data-retriever executable
    num_series = args.num_series  # number of series to keep - set to None for all series
    using_usc_data = args.using_usc_data
    image_dimensionality = args.image_dimensionality  # "2D" or "3D"
    do_radiomics = args.do_radiomics
    do_masking = args.do_masking

    # processing settings
    orient = args.disable_orient
    clip = args.disable_clip
    resample = args.disable_resample  # True if not do_radiomics else False  # handled inside params yaml file for radiomics
    # do_masking = do_masking if not do_radiomics else False
    mask_value_for_best_slice_selection = args.mask_value_for_best_slice_selection  # use tumor if available, otherwise use organs (will switch later if needed)
    mask_values = args.mask_values  # 1 for organ, 2 for tumor, None for all > 0
    standardize_dimensions = True
    normalize = args.normalize  # True if not do_radiomics else False  # handled inside params yaml file for radiomics
    normalization_method = args.normalization_method

    # radiomics settings
    resampledPixelSpacing = [1, 1, 1] if image_dimensionality == "3D" else [1, 1]
    pyradiomics_param = {
        "imageType": {
            "Original": {}
        },
        "setting": {
            "binWidth": 25,
            "resampledPixelSpacing": resampledPixelSpacing,
            "interpolator": "sitkBSpline",
            "normalize": False,
            "padDistance": 5
        }
    }

    # leave as-is
    utils.PROFILE_PIPELINE_DATA_DIR = None  # None to skip measuring storage, data_dir to measure storage (takes 1-3s per measurement, so only set if you want to measure storage)


    if dataset not in tcia_dataset_to_info:
        raise ValueError(f"Dataset {dataset} not recognized. Please add it to tcia_dataset_to_info.")
    if dataset != "tcga-kirc":
        using_usc_data = False  # only tcga-kirc has USC data available, so set to False for other datasets
    project = "other"
    if "tcga" in dataset:
        project = "tcga"
    elif "cptac" in dataset:
        project = "cptac"

    create_organ_masks = False
    if do_masking or do_radiomics or image_dimensionality == "2D":
        create_organ_masks = True
    if create_organ_masks and tcia_dataset_to_info[dataset].get("totalsegmentator_organs") is None:
        raise ValueError(f"Selected segmentations not specified for dataset {dataset}. Please specify the segmentations to use for this dataset in tcia_dataset_to_info.")

    data_dir = os.path.join(data_dir_base, project, dataset, "imaging")

    clip_min, clip_max = tcia_dataset_to_info[dataset].get("clip_min,clip_max", (None, None))
    xdim, ydim, zdim = tcia_dataset_to_info[dataset].get("xdim,ydim,zdim_masked", (None, None, None)) if do_masking else tcia_dataset_to_info[dataset].get("xdim,ydim,zdim_unmasked", (None, None, None))

    if do_radiomics:
        resample = False
        do_masking = False
        standardize_dimensions = False
        normalize = False

    if project == "tcga" and tcia_dataset_to_info[dataset].get("manifest_url") is None:
        tcia_dataset_to_info[dataset]["manifest_url"] = f"https://www.cancerimagingarchive.net/wp-content/uploads/TCIA_{dataset.upper()}_09-16-2015.tcia"


    metadata_name = f"metadata_{num_series}.csv" if num_series is not None else "metadata.csv"
    imaging_metadata_csv = os.path.join(data_dir, metadata_name)


    if not using_usc_data:
        if not os.path.exists(imaging_metadata_csv):
            imaging_metadata_csv_dir = os.path.dirname(imaging_metadata_csv) if os.path.dirname(imaging_metadata_csv) != "" else "."
            metadata_url = tcia_dataset_to_info[dataset].get("metadata_url") or tcia_dataset_to_info[dataset]["manifest_url"].replace(".tcia", "-nbia-digest.xlsx")
            additional_metadata_file_name = metadata_url.split("/")[-1]
            additional_metadata_xlsx = os.path.join(imaging_metadata_csv_dir, additional_metadata_file_name)

            os.makedirs(imaging_metadata_csv_dir, exist_ok=True)
            if not os.path.exists(additional_metadata_xlsx):
                subprocess.run(["wget", "-O", additional_metadata_xlsx, metadata_url], check=True)
            
            # add short patient ID
            imaging_metadata_df = pd.read_excel(additional_metadata_xlsx)
            imaging_metadata_df.insert(0, "series_id", [f"series_{i:05d}" for i in range(len(imaging_metadata_df))])
            imaging_metadata_df["project"] = project
            imaging_metadata_df["subproject"] = dataset
            imaging_metadata_df["cancer_organ"] = tcia_dataset_to_info[dataset]["cancer_organ"]
            imaging_metadata_df["cancer_type"] = tcia_dataset_to_info[dataset]["cancer_type"]
            
            # change column names to match old format
            col_renames = {
                "Series Instance UID": "Series UID",
                "Study Instance UID": "study_id",
                "Patient ID": "patient_id",
                "Image Count": "Number of Images Original",
            }
            imaging_metadata_df.rename(columns=col_renames, inplace=True)

            if "study_id" not in imaging_metadata_df.columns:
                imaging_metadata_df = imaging_metadata_df.rename(columns={"Study UID": "study_id"})
            if "patient_id" not in imaging_metadata_df.columns:
                imaging_metadata_df = imaging_metadata_df.rename(columns={"Subject ID": "patient_id"})
            if "Modality" not in imaging_metadata_df.columns:
                imaging_metadata_df["Modality"] = (
                    imaging_metadata_df["Study Description"]
                    .str.upper()
                    .str.extract(r"(MR|MRI|CT|PT|XR|X-RAY|X RAY|US|ULTRASOUND|NM)", expand=False)
                    .map({
                        "MR": "MRI",
                        "MRI": "MRI",
                        "XR": "X-ray",
                        "X-RAY": "X-ray",
                        "X RAY": "X-ray",
                        "US": "Ultrasound",
                        "ULTRASOUND": "Ultrasound",
                        "CT": "CT",
                        "PT": "PET",
                        "NM": "NM"
                    })
                    .fillna("CT")
                )

            imaging_metadata_df.to_csv(imaging_metadata_csv, index=False)

        metadata_df = pd.read_csv(imaging_metadata_csv)

        metadata_df["ParsedRegion"] = metadata_df["Study Description"].apply(utils.categorize_region_tcga)
        
        if dataset == "tcga-kirc":
            metadata_df["Phase"] = metadata_df["Series Description"].apply(utils.categorize_phase)
        utils.print_tcia_info(metadata_df, project=dataset)

        manifest_url = tcia_dataset_to_info[dataset]["manifest_url"]
        manifest_file_name = manifest_url.split("/")[-1]
        manifest_file_path = os.path.join(data_dir, manifest_file_name)

        if not os.path.exists(manifest_file_path):
            subprocess.run(f"wget {manifest_url} -P {data_dir}", shell=True, check=True)
        if num_series is not None:
            manifest_file_path_subset_series = manifest_file_path.replace(".tcia", f"_subset_{num_series}.tcia")
            if not os.path.exists(manifest_file_path_subset_series):
                with open(manifest_file_path, "r") as f_in, open(manifest_file_path_subset_series, "w") as f_out:
                    num_lines = num_series + 6  # 6 header lines in manifest file
                    for i, line in enumerate(f_in):
                        if i >= num_lines:
                            break
                        f_out.write(line)
            manifest_file_path = manifest_file_path_subset_series
            manifest_file_name = manifest_file_path.split("/")[-1]
            series_uids = utils.get_series_uids_from_manifest(manifest_file_path)
            metadata_df = metadata_df[metadata_df["Series UID"].isin(series_uids)]
        
        dicom_dir = os.path.join(data_dir, manifest_file_name.split(".")[0], dataset.upper())
        
        if not os.path.exists(dicom_dir) or len(os.listdir(dicom_dir)) == 0:   #!!! comment out
            if shutil.which(nbia_data_retriever) is None:
                sys.exit(f"Error: {nbia_data_retriever} not found in PATH. Please install or add it to PATH.")

            nbia_command = f"yes 'Y\nM' | {nbia_data_retriever} --cli {manifest_file_path} -d {data_dir} -v -f"

            print(f"Running NBIA Data Retriever with command:\n{nbia_command}")
            subprocess.run(nbia_command, shell=True, check=True)

            print(f"Downloaded images to: {dicom_dir}")

            metadata_df = utils.add_viable_info(dicom_dir, metadata_df, min_files=10, max_thickness_mm=10, include_kernel_keywords=True, out=imaging_metadata_csv, overwrite=True)

            metadata_df = metadata_df[metadata_df["is_viable"]]
            metadata_df = metadata_df[metadata_df["Modality"] == "CT"]
            utils.print_tcia_info(metadata_df, project=dataset)


        image_filename = "imaging.nii.gz"
        tumor_mask_filename = None
        nifti_dir_name = f"nifti_{num_series}" if num_series is not None else "nifti"
        nifti_dir = os.path.join(data_dir, nifti_dir_name)

        if not os.path.exists(nifti_dir) or len(os.listdir(nifti_dir)) == 0:   #!!! comment out
            utils.convert_dcm_to_nii_and_organize(dicom_dir, metadata_df, nifti_dir, segimage2itkimage_conda=False)
            print(f"convert_dcm_to_nii_and_organize metrics: {utils.convert_dcm_to_nii_and_organize.last_metrics}")

            # filter out 4D volumes and niis with big max zoom (sometimes some series will have an axial localizer but an otherwise coronal/sagittal series - we want to exclude these)
            metadata_df = utils.check_and_delete_bad_niftis(metadata_df, nifti_dir, image_filename=image_filename, is_4d=True, min_z=10, max_zoom_maximum=20, filter_if_max_zoom_not_in_si_position=False, out=imaging_metadata_csv)
            utils.print_tcia_info(metadata_df, project=dataset)
    else:
        metadata_name = f"metadata_usc_{num_series}.csv" if num_series is not None else "metadata_usc.csv"
        imaging_metadata_csv = os.path.join(data_dir, metadata_name)
        
        nifti_dir_name = f"nifti_usc_{num_series}" if num_series is not None else "nifti_usc"
        nifti_dir = os.path.join(data_dir, nifti_dir_name)
        
        image_filename = "0502_VENOUS.nii"
        tumor_mask_filename = "segmentation_tumor.nii.gz"

        if not os.path.exists(nifti_dir) or len(os.listdir(nifti_dir)) == 0:
            _ = utils.download_usc_tcga_kirc_data(data_dir, imaging_metadata_csv=imaging_metadata_csv, num_series=num_series, dst_dir_name=nifti_dir_name)

        metadata_df = pd.read_csv(imaging_metadata_csv)

    if orient:
        oriented_image_files, oriented_mask_files, final_image_files, final_mask_files = [], [], [], []
        orient_metrics = None
        for series_id in tqdm(sorted(os.listdir(nifti_dir)), desc="Processing images"):
            series_dir = os.path.join(nifti_dir, series_id)
            image_file = os.path.join(series_dir, image_filename)
            mask_file = os.path.join(series_dir, tumor_mask_filename) if tumor_mask_filename else ""
            if not os.path.exists(image_file):
                print(f"Image file not found for series_id {series_id} at {image_file}. Skipping.")
                continue

            if orient:
                image_file = utils.set_canonical_orientation(image_file, out=True)
                oriented_image_files.append(image_file)
                orient_metrics = utils.add_metrics(total=orient_metrics, metrics=utils.set_canonical_orientation.last_metrics)
                print(f"Set canonical orientation for image file for series_id {series_id} at {image_file}.")
                if os.path.exists(mask_file):
                    mask_file = utils.set_canonical_orientation(mask_file, out=True)
                    oriented_mask_files.append(mask_file)
                    orient_metrics = utils.add_metrics(total=orient_metrics, metrics=utils.set_canonical_orientation.last_metrics)
            
            final_image_files.append(image_file)
            final_mask_files.append(mask_file)

        image_filename_set = set([os.path.basename(f) for f in final_image_files])
        assert len(image_filename_set) == 1, f"Expected all image files to have the same filename, but found: {image_filename_set}"
        image_filename = list(image_filename_set)[0]
        tumor_mask_filename = os.path.basename(final_mask_files[0]) if final_mask_files else None

        print(f"Orientation metrics: {orient_metrics}")

    if tumor_mask_filename is None and mask_value_for_best_slice_selection == 2:
        mask_value_for_best_slice_selection = 1

    mask_filename = None
    if create_organ_masks:
        combined_organ_mask_filename = "segmentation_organs.nii.gz"
        mask_filename = "segmentation.nii.gz"  # tumor + organs

        metadata_df.to_csv(imaging_metadata_csv, index=False)  # save before running totalsegmentator in case it modifies metadata_df
        utils.run_totalsegmentator(nifti_dir, selected_segmentations=tcia_dataset_to_info[dataset]["totalsegmentator_organs"], metadata_csv=imaging_metadata_csv, metadata_csv_out=imaging_metadata_csv, remove_small_blobs=True, fill_holes=True, morphological_closing=True, image_filename=image_filename, tumor_mask_filename=tumor_mask_filename, combined_organ_mask_filename=combined_organ_mask_filename, mask_filename_out=mask_filename, visualize=False, task=tcia_dataset_to_info[dataset].get("totalsegmentator_task", "total"))
        metadata_df = pd.read_csv(imaging_metadata_csv)
        print(f"run_totalsegmentator metrics: {utils.run_totalsegmentator.last_metrics}")

    # clip = True
    # resample = resample if not do_radiomics else False  # handled inside params yaml file for radiomics
    # mask_value_for_best_slice_selection = mask_value_for_best_slice_selection if tumor_mask_filename else 1  # use tumor if available, otherwise use organs
    # mask_values = None  # 1 for organ, 2 for tumor, None for all > 0

    if clip or resample or image_dimensionality == "2D" or do_masking:
        slice_info_list = []
        clipped_image_files, resampled_image_files, resampled_mask_files, slice_image_files, slice_mask_files, masked_image_files, masked_mask_files, final_image_files, final_mask_files = [], [], [], [], [], [], [], [], []
        clip_metrics, resample_metrics, slice_selection_metrics, masking_metrics = None, None, None, None
        for series_id in tqdm(sorted(os.listdir(nifti_dir)), desc="Processing images"):
            series_dir = os.path.join(nifti_dir, series_id)
            image_file = os.path.join(series_dir, image_filename)
            mask_file = os.path.join(series_dir, mask_filename) if mask_filename else ""
            if not os.path.exists(image_file):
                print(f"Image file not found for series_id {series_id} at {image_file}. Skipping.")
                continue
            
            if clip:
                if clip_min is None and clip_max is None:  # eg (-200, 300) for soft tissue window - done in training loop
                    raise ValueError(f"clip_min and clip_max cannot both be None if clip is True. Got clip_min={clip_min}, clip_max={clip_max}.")
                print(f"Clipping intensity range for image file for series_id {series_id} at {image_file} with clip_min={clip_min}, clip_max={clip_max}.")
                image_file = utils.clip_intensity_range(image_file, clip_min=clip_min, clip_max=clip_max, out=True)
                clipped_image_files.append(image_file)
                print(f"Clipped intensity range for image file for series_id {series_id} at {image_file} with clip_min={clip_min}, clip_max={clip_max}.")
                clip_metrics = utils.add_metrics(total=clip_metrics, metrics=utils.clip_intensity_range.last_metrics)

            if resample:
                image_file = utils.resample_image(image_file, target_spacing=(0.8, 0.8, 3.0), is_label=False, out=True)
                resampled_image_files.append(image_file)
                print(f"Resampled image file for series_id {series_id} at {image_file}.")
                resample_metrics = utils.add_metrics(total=resample_metrics, metrics=utils.resample_image.last_metrics)
                if os.path.exists(mask_file):
                    mask_file = utils.resample_image(mask_file, target_spacing=(0.8, 0.8, 3.0), is_label=True, out=True)
                    resampled_mask_files.append(mask_file)
                    resample_metrics = utils.add_metrics(total=resample_metrics, metrics=utils.resample_image.last_metrics)
            
            if image_dimensionality == "2D":
                if not os.path.exists(mask_file):
                    raise ValueError(f"Mask file not found for series_id {series_id} at {mask_file}. Cannot select slice with most mask without mask file.")

                image_file, mask_file, slice_info = utils.choose_slice_with_most_mask_single_image(image=image_file, mask=mask_file, mask_value=mask_value_for_best_slice_selection, out_image=True, out_mask=True)
                slice_image_files.append(image_file)
                slice_mask_files.append(mask_file)
                slice_info["series_id"] = series_id
                slice_info_list.append(slice_info)
                slice_selection_metrics = utils.add_metrics(total=slice_selection_metrics, metrics=utils.choose_slice_with_most_mask_single_image.last_metrics)
            
            if do_masking and os.path.exists(mask_file):
                image_file, mask_file = utils.apply_mask(image_file, mask_file, label=mask_values, min_value=clip_min, crop=True, pad_after_crop=5, out_image=True, out_mask=True)
                masked_image_files.append(image_file)
                masked_mask_files.append(mask_file)
                masking_metrics = utils.add_metrics(total=masking_metrics, metrics=utils.apply_mask.last_metrics)
                print(f"Applied masking to image file for series_id {series_id} at {image_file} using mask file at {mask_file} with mask values {mask_values}.")

            final_image_files.append(image_file)
            if mask_filename:
                final_mask_files.append(mask_file)

        image_filename_set = set([os.path.basename(f) for f in final_image_files])
        assert len(image_filename_set) == 1, f"Expected all image files to have the same filename, but found: {image_filename_set}"
        image_filename = list(image_filename_set)[0]

        if mask_filename:
            mask_filename_set = set([os.path.basename(f) for f in final_mask_files if f])  # filter out empty mask files
            assert len(mask_filename_set) <= 1, f"Expected all mask files to have the same filename, but found: {mask_filename_set}"
            mask_filename = list(mask_filename_set)[0] if mask_filename_set else ""
        
        if slice_info_list:
            slice_info_df = pd.DataFrame(slice_info_list)
            if len(slice_info_df.columns) > 1:  # ie has a column other than series_id
                slice_info_df_columns = list(slice_info_df.columns)
                for col in slice_info_df_columns:
                    if col in metadata_df.columns and col != "series_id":
                        # metadata_df.drop(columns=[col], inplace=True)  # replace with new value
                        slice_info_df.drop(columns=[col], inplace=True)  # keep old value
                
                metadata_df = metadata_df.merge(slice_info_df, on="series_id", how="left")
        
        print(f"clip_metrics: {clip_metrics}")
        print(f"resample_metrics: {resample_metrics}")
        print(f"slice_selection_metrics: {slice_selection_metrics}")
        print(f"masking_metrics: {masking_metrics}")


    # standardize_dimensions = True if not do_radiomics else False

    extents_95th = {"x": None, "y": None, "z": None}
    if standardize_dimensions:
        extents_95th = utils.compute_shape_histogram(nifti_dir, image_filename=image_filename)
        print(extents_95th)

    xdim = extents_95th["x"] if xdim is None else xdim
    ydim = extents_95th["y"] if ydim is None else ydim
    zdim = extents_95th["z"] if zdim is None else zdim
    # normalize = False  # True if not do_radiomics else False  # handled inside params yaml file for radiomics
    # normalization_method = "volume"

    if standardize_dimensions or normalize:
        standardized_image_files, standardized_mask_files, normalized_image_files, final_image_files, final_mask_files = [], [], [], [], []
        standardized_metrics, normalize_metrics = None, None
        for series_id in tqdm(sorted(os.listdir(nifti_dir)), desc="Processing images"):
            series_dir = os.path.join(nifti_dir, series_id)
            image_file = os.path.join(series_dir, image_filename)
            mask_file = os.path.join(series_dir, mask_filename) if mask_filename else ""
            if not os.path.exists(image_file):
                print(f"Image file not found for series_id {series_id} at {image_file}. Skipping.")
                continue
            
            if standardize_dimensions:
                print(f"Standardizing dimensions for image file for series_id {series_id} at {image_file} to xdim={xdim}, ydim={ydim}, zdim={zdim}.")
                image_file = utils.crop_and_pad(image_file, xdim=xdim, ydim=ydim, zdim=zdim, min_value=clip_min, out=True)
                standardized_image_files.append(image_file)
                standardized_metrics = utils.add_metrics(total=standardized_metrics, metrics=utils.crop_and_pad.last_metrics)
                if os.path.exists(mask_file):
                    mask_file = utils.crop_and_pad(mask_file, xdim=xdim, ydim=ydim, zdim=zdim, min_value=0, out=True)
                    standardized_mask_files.append(mask_file)
                    standardized_metrics = utils.add_metrics(total=standardized_metrics, metrics=utils.crop_and_pad.last_metrics)

            if normalize:
                print(f"Normalizing intensity for image file for series_id {series_id} at {image_file} with method={normalization_method}.")
                image_file = utils.normalize_intensity(image_file, normalization_method=normalization_method, out=True)
                normalized_image_files.append(image_file)
                normalize_metrics = utils.add_metrics(total=normalize_metrics, metrics=utils.normalize_intensity.last_metrics)

            final_image_files.append(image_file)
            if mask_filename:
                final_mask_files.append(mask_file)
        
        image_filename_set = set([os.path.basename(f) for f in final_image_files])
        assert len(image_filename_set) == 1, f"Expected all image files to have the same filename, but found: {image_filename_set}"
        image_filename = list(image_filename_set)[0]
        if mask_filename:
            mask_filename_set = set([os.path.basename(f) for f in final_mask_files if f])  # filter out empty mask files
            assert len(mask_filename_set) <= 1, f"Expected all mask files to have the same filename, but found: {mask_filename_set}"
            mask_filename = list(mask_filename_set)[0] if mask_filename_set else ""
        
        print(f"standardized_metrics: {standardized_metrics}")
        print(f"normalize_metrics: {normalize_metrics}")



    convert_to_npy = True if (not do_radiomics and do_masking) else False  # convert to npy if we're not doing radiomics (radiomics wants nifti) AND we are doing masking (omitting masking creates massive files)

    # image_filename_nii, mask_filename_nii = image_filename, mask_filename
    if convert_to_npy:
        npy_image_files, npy_mask_files, final_image_files, final_mask_files = [], [], [], []
        npy_metrics = None
        for series_id in tqdm(sorted(os.listdir(nifti_dir)), desc="Converting to npy"):
            series_dir = os.path.join(nifti_dir, series_id)
            image_file = os.path.join(series_dir, image_filename)
            mask_file = os.path.join(series_dir, mask_filename) if mask_filename else ""
            if not os.path.exists(image_file):
                print(f"Image file not found for series_id {series_id} at {image_file}. Skipping.")
                continue
            
            image_npy_file = utils.nii_to_npy(image_file, out=True)
            npy_image_files.append(image_npy_file)
            npy_metrics = utils.add_metrics(total=npy_metrics, metrics=utils.nii_to_npy.last_metrics)
            print(f"Converted image file for series_id {series_id} at {image_file} to npy at {image_npy_file}.")
            
            if os.path.exists(mask_file):
                mask_npy_file = utils.nii_to_npy(mask_file, out=True)
                npy_mask_files.append(mask_npy_file)
                npy_metrics = utils.add_metrics(total=npy_metrics, metrics=utils.nii_to_npy.last_metrics)
            
            final_image_files.append(image_npy_file)
            if mask_filename:
                final_mask_files.append(mask_npy_file)

        image_filename_set = set([os.path.basename(f) for f in final_image_files])
        assert len(image_filename_set) == 1, f"Expected all image files to have the same filename, but found: {image_filename_set}"
        image_filename = list(image_filename_set)[0].replace(".nii.gz", ".npy")
        if mask_filename:
            mask_filename_set = set([os.path.basename(f) for f in final_mask_files if f])  # filter out empty mask files
            assert len(mask_filename_set) <= 1, f"Expected all mask files to have the same filename, but found: {mask_filename_set}"
            mask_filename = list(mask_filename_set)[0].replace(".nii.gz", ".npy") if mask_filename_set else ""
        
        # image_filename_nii, mask_filename_nii = image_filename, mask_filename
        
        print(f"npy_metrics: {npy_metrics}")

    dirs_up_for_relative_dst_path = 6  # None for absolute path, or number of directories up to make relative path for final CSV
    file_extension = "npy" if convert_to_npy else "nii.gz"
    suffix = f"{image_dimensionality}{'_masked' if do_masking else ''}{'_radiomics' if do_radiomics else ''}.{file_extension}"
    image_filename_final, mask_filename_final = f"imaging_final_{suffix}", f"segmentation_final_{suffix}"

    final_image_files_dict = {}
    for image_path in final_image_files:
        dst_abs_image_path = os.path.join(os.path.dirname(image_path), image_filename_final)
        dst_rel_image_path = "/".join(dst_abs_image_path.split("/")[-dirs_up_for_relative_dst_path:])
        if not os.path.exists(dst_abs_image_path):
            shutil.copy(image_path, dst_abs_image_path)
        series_id = os.path.basename(os.path.dirname(image_path))
        final_image_files_dict[series_id] = dst_rel_image_path
    assert len(set(final_image_files_dict.values())) == len(final_image_files_dict.values()), f"Expected no duplicate values in final_image_files_dict, but found duplicates: {final_image_files_dict}"
    metadata_df[image_filename_final.split(".")[0]] = metadata_df["series_id"].map(final_image_files_dict)

    if final_mask_files:
        final_mask_files_dict = {}
        for mask_path in final_mask_files:
            dst_abs_mask_path = os.path.join(os.path.dirname(mask_path), mask_filename_final)
            dst_rel_mask_path = "/".join(dst_abs_mask_path.split("/")[-dirs_up_for_relative_dst_path:])
            if not os.path.exists(dst_abs_mask_path):
                shutil.copy(mask_path, dst_abs_mask_path)
            series_id = os.path.basename(os.path.dirname(mask_path))
            final_mask_files_dict[series_id] = dst_rel_mask_path
        assert len(set(final_mask_files_dict.values())) == len(final_mask_files_dict.values()), f"Expected no duplicate values in final_mask_files_dict, but found duplicates: {final_mask_files_dict}"
        metadata_df[mask_filename_final.split(".")[0]] = metadata_df["series_id"].map(final_mask_files_dict)

    print(f"Final image filename: {image_filename_final}, Final mask filename: {mask_filename_final}")

    mask_values_radiomics = [1,2]  # 1 for organ, 2 for tumor
    if do_radiomics:
        pyradiomics_param_file = os.path.join(data_dir, "pyradiomics_param.yaml")
        with open(pyradiomics_param_file, "w") as f:
            yaml.dump(pyradiomics_param, f, sort_keys=False, default_flow_style=False)

        pyradiomics_input_csv_path = os.path.join(data_dir, f"metadata_{image_dimensionality}_preradiomics.csv")
        utils.prepare_csv_for_pyradiomics(nifti_dir, output_csv_path=pyradiomics_input_csv_path, imaging_file_name=image_filename, mask_file_name=mask_filename)  # image_filename_nii, mask_filename_nii
        print(f"prepare_csv_for_pyradiomics metrics: {utils.prepare_csv_for_pyradiomics.last_metrics}")

        output_csv_path = os.path.join(data_dir, f"data_{image_dimensionality}_radiomics.csv")
        utils.perform_radiomics_pipeline(pyradiomics_input_csv_path, output_csv_path=output_csv_path, label=mask_values_radiomics, param=pyradiomics_param_file)
        print(f"perform_radiomics_pipeline metrics: {utils.perform_radiomics_pipeline.last_metrics}")

        radiomics_df = pd.read_csv(output_csv_path)
        radiomics_df.head()

    metadata_df.to_csv(imaging_metadata_csv, index=False)

if __name__ == "__main__":
    main()
