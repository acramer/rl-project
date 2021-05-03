gpu: setup-gpu
	echo [MAKE - gpu]
	for n in 1 2 3 4 5 ; do \
	gcloud compute ssh alan --command 'nohup ./rl-project/scripts/cloud_script.sh &> log.txt < /dev/null &' && echo 'SSH successful' && break; \
	sleep 5; \
	done

gpu-next: check-gpu
	echo [MAKE - gpu-soft-kill]
	for n in 1 2 3 4 5 ; do \
	gcloud compute ssh alan --command 'nohup ./rl-project/scripts/cloud_next.sh &> log.txt < /dev/null &' && echo 'SSH successful' && break; \
	sleep 5; \
	done

over-gpu: setup-gpu
	echo [MAKE - over-gpu]
	for n in 1 2 3 4 5 ; do \
	gcloud compute ssh alan --command 'nohup ./rl-project/scripts/cloud_script.sh &> log.txt < /dev/null &' && echo 'SSH successful' && break; \
	sleep 5; \
	done

under-gpu:
	echo [MAKE - under-gpu]
	for n in 1 2 3 4 5 ; do \
	gcloud compute ssh alan --command 'nohup ./rl-project/scripts/cloud_script.sh &> log.txt < /dev/null &' && echo 'SSH successful' && break; \
	sleep 5; \
	done

setup-gpu:
	echo [MAKE - gpu]
	git diff-index --quiet HEAD code || (git add ../ && git commit -m 'gpu commit' && git push)
	make gpu-soft-kill || echo "CPU already off"
	gcloud compute instances start alan

check-gpu:
	echo [MAKE - check-gpu]
	gcloud compute instances list | python scripts/parse_instances.py -m alan

gpu-kill: check-gpu
	echo [MAKE - gpu-kill]
	gcloud compute instances stop alan

gpu-soft-kill: check-gpu
	echo [MAKE - gpu-soft-kill]
	for n in 1 2 3 4 5 ; do \
	gcloud compute ssh alan --command 'nohup ./rl-project/scripts/cloud_kill.sh &> log.txt < /dev/null &' && echo 'SSH successful' && break; \
	sleep 5; \
	done
