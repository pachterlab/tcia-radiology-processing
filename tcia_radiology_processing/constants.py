# verify links by visiting TCIA - manifest URL is what I get by clicking big blue "download" button for imaging data, and metadata url is what I get by clicking small "view" link under Metadata column
tcia_dataset_to_info = {
    "tcga-kirc": {  # kidney renal clear cell carcinoma
        "manifest_url": None,  # None for default
        "metadata_url": None,  # None for default
        "totalsegmentator_organs": ["kidney_left", "kidney_right"],
        "clip_min,clip_max": (-200, 300),  # (min, max) to clip pixel values to before resampling and feature extraction - set to (None, None) for no clipping
        "xdim,ydim,zdim_masked": (185, 185, 75),  # dimensions to standardize to when do_masking is True - set to (None, None, None) to use 95th percentile of extents across all series
        "xdim,ydim,zdim_unmasked": (625, 625, 200)  # dimensions to standardize to when do_masking is False - set to (None, None, None) to use 95th percentile of extents across all series
    },
    "tcga-lihc": {  # liver hepatocellular carcinoma
        "manifest_url": "https://www.cancerimagingarchive.net/wp-content/uploads/doiJNLP-TCGA-LIHC-01-30-2017.tcia",  # None for default
        "metadata_url": None,  # None for default
        "totalsegmentator_organs": ["liver"],
        "clip_min,clip_max": (-200, 400),  # (min, max) to clip pixel values to before resampling and feature extraction - set to (None, None) for no clipping
    },
    "tcga-blca": {  # bladder urothelial carcinoma
        "manifest_url": "https://www.cancerimagingarchive.net/wp-content/uploads/TCGA-BLCA-August-30-2019-NBIA-manifest.tcia",  # None for default
        "metadata_url": "https://www.cancerimagingarchive.net/wp-content/uploads/TCGA-BLCA-August-30-2019-NBIA-manifes-nbia-digest.xlsx",  # None for default
        "totalsegmentator_organs": ["urinary_bladder"],
        "clip_min,clip_max": (-200, 300),  # (min, max) to clip pixel values to before resampling and feature extraction - set to (None, None) for no clipping
    },
    # "tcga-brca": {  # breast invasive carcinoma - all XRs
    #     "manifest_url": None,  # None for default
    #     "metadata_url": None,  # None for default
    #     "totalsegmentator_organs": ["breasts"],
    #     "totalsegmentator_task": "breasts",  # None/omit for total
    #     "clip_min,clip_max": (-200, 300),  # (min, max) to clip pixel values to before resampling and feature extraction - set to (None, None) for no clipping
    # },
    "tcga-cesc": {  # cervical squamous cell carcinoma and endocervical adenocarcinoma
        "manifest_url": None,  # None for default
        "metadata_url": None,  # None for default
        "totalsegmentator_organs": [],
        "clip_min,clip_max": (-200, 300),  # (min, max) to clip pixel values to before resampling and feature extraction - set to (None, None) for no clipping
    },
    "tcga-coad": {  # colon adenocarcinoma
        "manifest_url": None,  # None for default
        "metadata_url": None,  # None for default
        "totalsegmentator_organs": ["colon"],
        "clip_min,clip_max": (-200, 300),  # (min, max) to clip pixel values to before resampling and feature extraction - set to (None, None) for no clipping
    },
    "tcga-esca": {  # esophageal carcinoma
        "manifest_url": "https://www.cancerimagingarchive.net/wp-content/uploads/TCIA_TCGA-ESCA-09-16-2015.tcia",  # None for default
        "metadata_url": None,  # None for default
        "totalsegmentator_organs": ["esophagus"],
        "clip_min,clip_max": (-200, 300),  # (min, max) to clip pixel values to before resampling and feature extraction - set to (None, None) for no clipping
    },
    # "tcga-gbm": {  # glioblastoma multiforme - protected
    #     "manifest_url": None,  # None for default
    #     "metadata_url": None,  # None for default
    #     "totalsegmentator_organs": ["brain"],
    #     "clip_min,clip_max": (-200, 300),  # (min, max) to clip pixel values to before resampling and feature extraction - set to (None, None) for no clipping
    # },
    # "tcga-hnsc": {  # head and neck squamous cell carcinoma - protected
    #     "manifest_url": None,  # None for default
    #     "metadata_url": None,  # None for default
    #     "totalsegmentator_organs": ["skull"],
    #     "clip_min,clip_max": (-200, 300),  # (min, max) to clip pixel values to before resampling and feature extraction - set to (None, None) for no clipping
    # },
    "tcga-kich": {  # kidney renal papillary cell carcinoma
        "manifest_url": None,  # None for default
        "metadata_url": None,  # None for default
        "totalsegmentator_organs": ["kidney_left", "kidney_right"],
        "clip_min,clip_max": (-200, 300),  # (min, max) to clip pixel values to before resampling and feature extraction - set to (None, None) for no clipping
    },
    "tcga-kirp": {  # kidney renal clear cell carcinoma
        "manifest_url": "https://www.cancerimagingarchive.net/wp-content/uploads/doiJNLP-TCGA-KIRP-01-30-2017.tcia",  # None for default
        "metadata_url": None,  # None for default
        "totalsegmentator_organs": ["kidney_left", "kidney_right"],
        "clip_min,clip_max": (-200, 300),  # (min, max) to clip pixel values to before resampling and feature extraction - set to (None, None) for no clipping
    },
    "tcga-lgg": {  # brain lower grade glioma
        "manifest_url": None,  # None for default
        "metadata_url": None,  # None for default
        "totalsegmentator_organs": ["brain"],
        "clip_min,clip_max": (-200, 300),  # (min, max) to clip pixel values to before resampling and feature extraction - set to (None, None) for no clipping
    },
    "tcga-luad": {  # lung adenocarcinoma
        "manifest_url": "https://www.cancerimagingarchive.net/wp-content/uploads/doiJNLP-TCGA-LUAD-01-30-2017.tcia",  # None for default
        "metadata_url": None,  # None for default
        "totalsegmentator_organs": ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"],
        "clip_min,clip_max": (-1000, 400),  # (min, max) to clip pixel values to before resampling and feature extraction - set to (None, None) for no clipping
    },
    "tcga-lusc": {  # lung squamous cell carcinoma
        "manifest_url": "https://www.cancerimagingarchive.net/wp-content/uploads/doiJNLP-TCGA-LUSC-01-30-2017.tcia",  # None for default
        "metadata_url": None,  # None for default
        "totalsegmentator_organs": ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"],
        "clip_min,clip_max": (-1000, 400),  # (min, max) to clip pixel values to before resampling and feature extraction - set to (None, None) for no clipping
    },
    "tcga-ov": {  # ovarian serous cystadenocarcinoma
        "manifest_url": None,  # None for default
        "metadata_url": None,  # None for default
        "totalsegmentator_organs": [],
        "clip_min,clip_max": (-200, 300),  # (min, max) to clip pixel values to before resampling and feature extraction - set to (None, None) for no clipping
    },
    "tcga-prad": {  # prostate adenocarcinoma
        "manifest_url": None,  # None for default
        "metadata_url": None,  # None for default
        "totalsegmentator_organs": ["prostate"],
        "clip_min,clip_max": (-200, 300),  # (min, max) to clip pixel values to before resampling and feature extraction - set to (None, None) for no clipping
    },
    "tcga-read": {  # rectum adenocarcinoma
        "manifest_url": None,  # None for default
        "metadata_url": None,  # None for default
        "totalsegmentator_organs": ["colon"],
        "clip_min,clip_max": (-200, 300),  # (min, max) to clip pixel values to before resampling and feature extraction - set to (None, None) for no clipping
    },
    "tcga-stad": {  # stomach adenocarcinoma
        "manifest_url": None,  # None for default
        "metadata_url": None,  # None for default
        "totalsegmentator_organs": ["stomach"],
        "clip_min,clip_max": (-200, 300),  # (min, max) to clip pixel values to before resampling and feature extraction - set to (None, None) for no clipping
    },
    "tcga-ucec": {  # uterine corpus endometrial carcinoma
        "manifest_url": "https://www.cancerimagingarchive.net/wp-content/uploads/TCIA_TCGA-UCEC-2018-10-24.tcia",  # None for default
        "metadata_url": None,  # None for default
        "totalsegmentator_organs": [],
        "clip_min,clip_max": (-200, 300),  # (min, max) to clip pixel values to before resampling and feature extraction - set to (None, None) for no clipping
    },
    "cptac-ccrcc": {  # clear cell renal cell carcinoma
        "manifest_url": "https://www.cancerimagingarchive.net/wp-content/uploads/TCIA-CPTAC-CCRCC_v11_20230818.tcia",
        "metadata_url": None,
        "totalsegmentator_organs": ["kidney_left", "kidney_right"],
        "clip_min,clip_max": (-200, 300),
    },
    # "cptac-ucec": {  # uterine corpus endometrial carcinoma
    #     "manifest_url": "https://www.cancerimagingarchive.net/wp-content/uploads/TCIA-CPTAC-UCEC_v12_20240405b.tcia",
    #     "metadata_url": None,  #!!! truly doesn't exist - TODO: make a custom one after downloading DICOMs
    #     "totalsegmentator_organs": [],
    #     "clip_min,clip_max": (-200, 300),
    # },
    "cptac-luad": {  # lung adenocarcinoma
        "manifest_url": "https://www.cancerimagingarchive.net/wp-content/uploads/TCIA-CPTAC-LUAD_v13_20250801.tcia",
        "metadata_url": None,
        "totalsegmentator_organs": ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"],
        "clip_min,clip_max": (-1000, 400),
    },
    "cptac-lscc": {  # lung squamous cell carcinoma
        "manifest_url": "https://www.cancerimagingarchive.net/wp-content/uploads/TCIA-CPTAC-LSCC_v15.tcia",
        "metadata_url": None,
        "totalsegmentator_organs": ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"],
        "clip_min,clip_max": (-1000, 400),
    },
    # "cptac-hnscc": {  # head and neck squamous cell carcinoma - protected
    #     "manifest_url": "https://www.cancerimagingarchive.net/wp-content/uploads/TCIA-CPTAC-HNSCC_v19_20250226.tcia",
    #     "metadata_url": None,
    #     "totalsegmentator_organs": ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"],
    #     "clip_min,clip_max": (-1000, 400),
    # },
    # "cptac-gbm": {  # glioblastoma multiforme - protected
    #     "manifest_url": "https://www.cancerimagingarchive.net/wp-content/uploads/TCIA-CPTAC-GBM_v16_20240708.tcia",
    #     "metadata_url": None,
    #     "totalsegmentator_organs": ["brain"],
    #     "clip_min,clip_max": (-200, 300),
    # },
    "cptac-aml": {  # pancreatic ductal adenocarcinoma
        "manifest_url": "https://www.cancerimagingarchive.net/wp-content/uploads/CPTAC-AML_V5_20250415-1.tcia",
        "metadata_url": None,
        "totalsegmentator_organs": [],
        "clip_min,clip_max": (-200, 300),
    },
    # "cptac-stad": {  # stomach adenocarcinoma
    #     "manifest_url": "https://www.cancerimagingarchive.net/wp-content/uploads/CPTAC-STAD_v1_20260304.tcia",
    #     "metadata_url": None,  #!!! truly doesn't exist - TODO: make a custom one after downloading DICOMs
    #     "totalsegmentator_organs": ["stomach"],
    #     "clip_min,clip_max": (-200, 300),
    # },
    "cptac-pda": {  # pancreatic ductal adenocarcinoma
        "manifest_url": "https://www.cancerimagingarchive.net/wp-content/uploads/TCIA-CPTAC-PDA_v15_20250226.tcia",
        "metadata_url": None,
        "totalsegmentator_organs": ["pancreas"],
        "clip_min,clip_max": (-200, 300),
    },
    "cptac-cm": {  # cutaneous melanoma
        "manifest_url": "https://www.cancerimagingarchive.net/wp-content/uploads/TCIA-CPTAC-CM_v11_20240429.tcia",
        "metadata_url": None,
        "totalsegmentator_organs": [],
        "clip_min,clip_max": (-200, 300),
    },
    # add more datasets here as needed
}