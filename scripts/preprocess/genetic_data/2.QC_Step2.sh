#!/bin/bash

# =============================================================================
#                           Resource
# =============================================================================
referenceFasta=/public/home/GENE_proc/resource/GRCH37/ReferenceGenome/human_g1k_v37/human_g1k_v37.fasta


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
runDir=$work_path/02_QC_for_imputation/$batch

mkdir -p $runDir
mkdir -p $runDir/log



# =============================================================================
#                           Exec Pipeline
# =============================================================================
# Step1: Flip Position
logfile=$runDir/log/01_FlipPos.log
exec >$logfile 2>&1

mkdir -p $runDir/01_FlipPos
python flippyr.py $referenceFasta $work_path/01_QC_standard_GWAS/$batch/HapMap_3_r3_12.bim -o $runDir/01_FlipPos/$batch
$runDir/01_FlipPos/$batch.runPlink


# Step2: SplitChr
logfile=$runDir/log/02_SplitChr.log
exec >$logfile 2>&1

mkdir -p $runDir/02_SplitChr
mkdir -p $runDir/02_SplitChr/tmp
for chr in `echo {1..22}`
do
	$PLINK --bfile $runDir/01_FlipPos/${batch}_flipped --chr $chr --keep-allele-order --make-bed --out $runDir/02_SplitChr/tmp/${batch}_chr${chr}
	$PLINK --bfile $runDir/02_SplitChr/tmp/${batch}_chr${chr} --keep-allele-order --recode vcf --out $runDir/02_SplitChr/tmp/${batch}_chr${chr}
	$bcftools sort $runDir/02_SplitChr/tmp/${batch}_chr${chr}.vcf -Oz -o $runDir/02_SplitChr/${batch}_chr${chr}.vcf.gz
	$tabix $runDir/02_SplitChr/${batch}_chr${chr}.vcf.gz
done
