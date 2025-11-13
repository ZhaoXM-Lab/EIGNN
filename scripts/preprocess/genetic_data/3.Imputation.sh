#!/bin/bash
#SBATCH -n 36
#SBATCH -N 1
#SBATCH -t 100-5:00
#SBATCH -p DCU
#SBATCH --mem=360000
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
bcftools=~/anaconda3/envs/tool_bcftools/bin/bcftools
tabix=~/anaconda3/envs/tool_bcftools/bin/tabix
RTools=/public/home/caojx/Project/NACC/02_PreProcess/code/RTool
beagle=/public/home/GENE_proc/tools/beagle

# =============================================================================
#                           Set Work Path
# =============================================================================
work_path=/public/home/caojx/Project/NACC
batch=ADC1
inputPath=$work_path/02_PreProcess/02_QC_for_imputation/$batch/02_SplitChr
runDir=$work_path/03_Impute/$batch

mkdir -p $runDir
mkdir -p $runDir/log

# =============================================================================
#                           Exec Pipeline
# =============================================================================
for chr in {1..22}
do
	logfile=$runDir/log/imputation_chr${chr}.log
	exec >$logfile 2>&1

	java -jar $beagle/beagle.22Jul22.46e.jar \
		ref=$beagle/panel/chr"${chr}".1kg.phase3.v5a.vcf.gz \
		gt=$inputPath/${batch}_chr${chr}.vcf.gz \
		out=$runDir/${batch}_imputation_chr"${chr}" \
		map=$beagle/genetic_maps/plink.chr"${chr}".GRCh37.map \
		impute=true \
		seed=998877 \
		nthreads=30
done

