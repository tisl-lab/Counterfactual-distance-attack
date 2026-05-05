#!/bin/bash
#SBATCH --time=02:00:00
##SBATCH --cpus-per-task=8
#SBATCH --ntasks=5
#SBATCH --mem-per-cpu=8G
#SBATCH --array=0,1,2,3,4
#SBATCH --mail-user=maryam.babaei.1@ens.etsmtl.ca
#SBATCH --mail-type=ALL
#SBATCH --account=def-umaivodj

datasets=(compas adult acs_income heloc)

module purge
module load python/3.11
module load mpi4py

source ~/envs/cf_dist_env/bin/activate
echo "#!/bin/bash"
echo "#SBATCH --time=12:00:00"
echo "#SBATCH --cpus-per-task=2"
echo "#SBATCH --mem-per-cpu=12G"
echo "export TMPDIR=/tmp"
echo "cd ../.."



for dataset in "${datasets[@]}"
        do
                echo "--dataset" $dataset "--rseed" $SLURM_ARRAY_TASK_ID  
                srun python draw_plots_allinone.py --dataset $dataset --rseed $SLURM_ARRAY_TASK_ID 
        done	
