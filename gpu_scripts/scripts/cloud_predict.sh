. ~/miniconda3/etc/profile.d/conda.sh
conda activate rlearn

cd ~/rl-project
git pull
cd code
python main.py --load best_model --mode predict
git add .
git commit -m 'predictions'
git push
sudo shutdown now
