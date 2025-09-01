#!/bin/bash
#SBATCH --ntasks=1           ### How many CPU cores do you need?
#SBATCH --mem=100G            ### How much RAM memory do you need?
#SBATCH -p hm           ### The queue to submit to: express, short, long, interactive
#SBATCH --gres=gpu:1         ### How many GPUs do you need?
#SBATCH --nodelist=gpu-hm-001         ### How many GPUs do you need?
#SBATCH -t 0-40:00:00        ### The time limit in D-hh:mm:ss format
#SBATCH -o /trinity/home/jyu/out/out_%j.log        ### Where to store the console output (%j is the job number)
#SBATCH -e /trinity/home/jyu/error/error_%j.log      ### Where to store the error output



module purge
module load TensorFlow/2.2.0-fosscuda-2019b-Python-3.7.4

while getopts ":n:i:" opt; do
  case $opt in
    n) DATA=$OPTARG;;
    i) SPLIT=$OPTARG ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

if [ -z "$DATA" ]; then
	echo " need -n data file name"
	exit 2
fi
if [ -z "$SPLIT" ]; then
	echo "need -i split number"
	exit 2
fi


python /trinity/home/jyu/DeepSurvival/batch_scripts/MRI_Gender_Age_Apoe/Train_MRI_Gender_Age_Apoe_TFS_FS.py -n ${DATA} -i ${SPLIT}



