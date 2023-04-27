echo "[Check for Requirements]"

# declare flags
d_flag=false;
gpu_flag=false;
cnt_flag=false;

# Docker
d=$(docker --version | awk '{print $3}' | awk -F'.' '{print $1}');
if [ "${d}" -ge 19 ]
then
  echo "[ Docker     ] PASS"
  d_flag=true;
else
  echo "[ Docker     ] NOT PASS"
fi

# gpu
gpu=$(nvidia-smi);
gpu=$?;
if [ ${gpu} -eq 0 ]
then
  echo "[ GPU        ] PASS"
  gpu_flag=true;
else
  echo "[ GPU        ] NOT PASS"
fi

# container
cnt=$(which nvidia-container-toolkit);
cnt=$?;
if [ ${cnt} -eq 0 ]
then
  echo "[ nvidia-cnt ] PASS"
  cnt_flag=true;
else
  echo "[ nvidia-cnt ] NOT PASS"
fi

# final check
if [ $d_flag ] && $gpu_flag && $cnt_flag
then
  echo -e "\e[32m ==> Sure! You can install this repo :) \e[0m"
else
  echo -e "\e[31m ==> Sorry, but you cannot install this repo :( \e[0m"
fi