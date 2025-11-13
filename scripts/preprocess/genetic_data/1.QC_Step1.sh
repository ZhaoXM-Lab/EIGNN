#!/bin/bash

# =============================================================================
#                           Software
# =============================================================================
PLINK2=/public/home/GENE_proc/tools/plink2/20230109/plink2
PLINK=/public/home/GENE_proc/tools/plink/20230116/plink
king=/public/home/GENE_proc/tools/king/king_2.3.0/king
bcftools=~/anaconda3/envs/tool_bcftools/bin/bcftools
tabix=~/anaconda3/envs/tool_bcftools/bin/tabix
RTools=/public/home/caojx/Project/NACC/02_PreProcess/code/RTool

# =============================================================================
#                           Set Work Path
# =============================================================================
work_path=/public/home/caojx/Project/NACC/02_PreProcess
batch=ADC1
rawPlink=/public/home/caojx/Project/NACC/01_RawData/02_DataUsed/Human660W/ADC1_full_062513.fwd.hg19.r
runDir=$work_path/01_QC_standard_GWAS/$batch

mkdir -p $runDir
mkdir -p $runDir/tmp
mkdir -p $runDir/log


logfile=$runDir/log/QC_StandardGWAS.log
exec >$logfile 2>&1

module load R
# =============================================================================
#                           Exec Pipeline
# =============================================================================
cd $runDir


cut -f 2 $rawPlink.bim | sort | uniq -d > HapMap_3_r3_1.dup
$PLINK --bfile $rawPlink --recode vcf --exclude HapMap_3_r3_1.dup --keep-allele-order --out HapMap_3_r3_1


### Step 1 ###  Missing
$PLINK --bfile $rawPlink --geno 0.2 --make-bed --out HapMap_3_r3_2
$PLINK --bfile HapMap_3_r3_2 --mind 0.2 --make-bed --out HapMap_3_r3_3
$PLINK --bfile HapMap_3_r3_3 --geno 0.05 --make-bed --out HapMap_3_r3_4
$PLINK --bfile HapMap_3_r3_4 --mind 0.05 --make-bed --out HapMap_3_r3_5


### Step2 #### SEX
$PLINK --bfile HapMap_3_r3_5 --check-sex 0.6 0.8
grep "PROBLEM" plink.sexcheck | awk '{print$1,$2}'> sex_discrepancy.txt
$PLINK --bfile HapMap_3_r3_5 --remove sex_discrepancy.txt --make-bed --out HapMap_3_r3_6_1
$PLINK --bfile HapMap_3_r3_6_1 --set-hh-missing --make-bed --out HapMap_3_r3_6


### Step 3 ### MAF
awk '{ if ($1 >= 1 && $1 <= 22) print $2 }' HapMap_3_r3_6.bim > snp_1_22.txt
$PLINK --bfile HapMap_3_r3_6 --extract snp_1_22.txt --make-bed --out HapMap_3_r3_7
$PLINK --bfile HapMap_3_r3_7 --maf 0.01 --make-bed --out HapMap_3_r3_8


### Step 4 ### HWE
$PLINK --bfile HapMap_3_r3_8 --hwe 1e-12 --make-bed --out HapMap_3_r3_9


### step 5 ### HET
$PLINK --bfile HapMap_3_r3_9 --exclude $RTool/inversion.txt --range --indep-pairwise 50 5 0.2 --out indepSNP
$PLINK --bfile HapMap_3_r3_9 --extract indepSNP.prune.in --het --out R_check
Rscript --no-save $RTool/heterozygosity_outliers_list.R
sed 's/"// g' fail-het-qc.txt | awk '{print$1, $2}'> het_fail_ind.txt
$PLINK --bfile HapMap_3_r3_9 --remove het_fail_ind.txt --make-bed --out HapMap_3_r3_10


### step 6 ### IBD
$king -b HapMap_3_r3_10.bed --related --degree 2


### step 7 ### write qc files
cut -f 2 HapMap_3_r3_10.bim | sort | uniq -d > HapMap_3_r3_10.dup
$PLINK --bfile HapMap_3_r3_10 --remove IBD_removed.txt --exclude HapMap_3_r3_10.dup --make-bed --out HapMap_3_r3_11
$PLINK --bfile HapMap_3_r3_11 --a2-allele HapMap_3_r3_1.vcf 4 3 '#' --make-bed --out HapMap_3_r3_12
