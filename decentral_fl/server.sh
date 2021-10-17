!/usr/bin/env bash
chmod +x node.sh 
#source ~/miniconda3/etc/profile.d/conda.sh
#eval "$(conda shell.bash hook)"
#conda active my_env

conda activate testmonai
# server.sh
#cd /home/habib/monaifl/aggregator/coordinator/src
#cd /home/mhr21/monaifl/aggregator/coordinator/src
#cd /hubnspoke
pwd
echo $$
python decentral_fl/aggregator/coordinator/src/server.py
