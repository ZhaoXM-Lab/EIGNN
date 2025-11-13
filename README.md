# EIGNN: An explainable imaging-genetic neural network for augmented and robust risk prediction of Alzheimer's Disease
## Get started

To train the EIGNN model for CN vs. AD classification using SNP data
```shell
TRAINDATA_RATIO=1.0 
python main.py --no_log --cuda_index 0 --batch_size 16 --n_epoch 100 --trtype 5cv --method EIGNN --dataset ADNI_DX \ 
  --clfsetting CN-AD \ 
  --modals mwp1 array_snp.Brain_All.gwas_p_5-e16_200 \ 
  --method_para "{'useimg': False, 'usesnp': True, 'useeqtl': True, 'usegraph': True}"  

```

To train the EIGNN model for CN vs. AD classification using SNP and sMRI data
```shell
TRAINDATA_RATIO=1.0 
python main.py --no_log --cuda_index 0 --batch_size 16 --n_epoch 100 --trtype 5cv --method EIGNN --dataset ADNI_DX \ 
  --clfsetting CN-AD \ 
  --modals mwp1 array_snp.Brain_All.gwas_p_5-e16_200 \ 
  --method_para "{'useimg': True, 'usesnp': True, 'useeqtl': True, 'usegraph': True}"  

```

## Data Preparation

1. Data preprocessing.
   1. For genetic data you may use the scripts we provide in **scripts/preprocess/genetic_data**.
   2. For sMRI you may use the scripts (**scripts/preprocess/imaging_data**) with following command:
      ```shell
      cat_batch_cat.sh path_to_nii_file -d cat_config.m -p 1 -m path_to_matlab -l log_file_path
      ```
2. Specify the sMRI data path in utils/opts.py (e.g., ADNI_PATH). The sMRI data should be arranged like "ADNI/subject_id/viscode.nii".
3. Specify the SNP data path in utils/opts.py (e.g., ADNI_GENETIC_PATH). The SNP data should be stored in plink format, and named as ADNI_1.\*, ADNI_2.\*, etc
4. Download ADNIMERGE_28Jun2023.csv and PTDEMOG_28Jun2023.csv from the ADNI website and store
   them in data/.

##
More detailed description will be available soon.