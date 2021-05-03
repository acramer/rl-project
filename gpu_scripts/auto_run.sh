. ~/miniconda3/etc/profile.d/conda.sh
conda activate rlearn

cd ../
python main.py -PM 10000 -A deep-central-q -E 10 --huber
