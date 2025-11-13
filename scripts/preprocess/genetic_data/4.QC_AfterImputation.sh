#!/bin/bash
#SBATCH -n 6
#SBATCH -N 1
#SBATCH -t 100-5:00
#SBATCH -p DCU
#SBATCH --mem=60000
#SBATCH -J ip_b1

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
bcftools=/public/home/GENE_proc/anaconda3/envs/tool_bcftools/bin/bcftools
tabix=/public/home/GENE_proc/anaconda3/envs/tool_bcftools/bin/tabix
RTools=/public/home/caojx/Project/NACC/02_PreProcess/code/RTool
beagle=/public/home/GENE_proc/tools/beagle


# =============================================================================
#                           Set Work Path
# =============================================================================
work_path=/public/home/caojx/Project/NACC
batch=ADC1
inputPath=$work_path/03_Impute/$batch
runDir=$work_path/04_QC/$batch

mkdir -p $runDir
mkdir -p $runDir/tmp
mkdir -p $runDir/log

module load R
# =============================================================================
#                           Exec Pipeline
# =============================================================================
cd $runDir

for chr in {1..22}
do
{
	logfile=$runDir/log/01_QCVar_chr${chr}.log
	exec >$logfile 2>&1
	
	# High Imputation Quality
	$bcftools filter $inputPath/${batch}_imputation_chr${chr}.vcf.gz \
		-i 'INFO/DR2>=0.8' \
		-Oz -o $runDir/tmp/${batch}_imputation_chr${chr}.DR8.vcf.gz

	# Format Change
	$PLINK \
		--vcf $runDir/tmp/${batch}_imputation_chr${chr}.DR8.vcf.gz \
		--make-bed \
		--keep-allele-order \
		--out $runDir/tmp/${batch}_imputation_chr${chr}.DR8
	
	# Remove variants outlier
	$PLINK --bfile $runDir/tmp/${batch}_imputation_chr${chr}.DR8 \
		--geno 0.05 \
		--maf 0.01 \
		--hwe 1e-16 \
		--keep-allele-order \
		--make-bed \
		--out $runDir/tmp/${batch}_imputation_chr${chr}.VarQC1
	
	# Remove duplicated ID
	cut -f 2 $runDir/tmp/${batch}_imputation_chr${chr}.VarQC1.bim | sort | uniq -d > $runDir/tmp/${batch}_imputation_chr${chr}.VarQC1.dup
	
	$PLINK \
		--bfile $runDir/tmp/${batch}_imputation_chr${chr}.VarQC1 \
		--exclude $runDir/tmp/${batch}_imputation_chr${chr}.VarQC1.dup \
		--keep-allele-order \
		--make-bed \
		--out $runDir/tmp/${batch}_imputation_chr${chr}.VarQC2
	
	echo $runDir/tmp/${batch}_imputation_chr${chr}.VarQC2 >> $runDir/MergeList.txt
}
done

