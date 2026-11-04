#!/bin/bash
# Starts m2ds_chain.sh unless it is already running or finished; run it by
# hand (also after a reboot: the chain resumes where it stopped).
cd ~/m2ds_eval || exit 1
[ -f ALL_DONE ] && exit 0
exec 9> .chain.lock
flock -n 9 || exit 0           # another instance holds the lock
echo "$(date '+%F %T') chain (re)started" >> chain.log
./m2ds_chain.sh >> chain.log 2>&1
