#!/bin/bash

set -euo pipefail



run_local() {
    echo "Starting server"
    python3 runner.py --server --rounds 60 --epochs 2 &
    sleep 5  # Sleep for 3s to give the server enough time to start

    num_clients=10
    poison_list="1"
    for i in `seq 0 $((num_clients-1))`; do
        echo "Starting client $i"
        if exists_in_list "$poison_list" " " $i; then
            python3 runner.py --client --total-clients=$num_clients --client-index $i --server-address 127.0.0.1 --is-poisoned &
        else
            python3 runner.py --client --total-clients=$num_clients --client-index $i --server-address 127.0.0.1 &
        fi
    done


    # This will allow you to use CTRL+C to stop all background processes
    trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
    # Wait for all background processes to complete
    wait
}

function exists_in_list() {
    LIST=$1
    DELIMITER=$2
    VALUE=$3
    LIST_WHITESPACES=`echo $LIST | tr "$DELIMITER" " "`
    for x in $LIST_WHITESPACES; do
        if [ "$x" = "$VALUE" ]; then
            return 0
        fi
    done
    return 1
}

run_remote() {
    update="$1"

    if [[ ! -f config.sh ]]; then
        echo 'no config file found'
        exit 1
    fi

    source config.sh
    server_ssh="${server%:*}"
    server_ip="${server#*@}"
    server_port="${server_ip#*:}"
    server_ip="${server_ip%:*}"

    if ! ping -c1 -W1 "${server_ip#*@}" &>/dev/null; then
        echo 'CONNECT TO THE VPN!'
        exit 1
    fi


    echo "=== SERVER $server ==="
    if [[ "$update" == true ]]; then
        zip fl.zip *.py
        scp fl.zip "$server_ssh":~
        ssh "$server_ssh" "
            cd ~ && mkdir -p fl &&
            unzip -o fl.zip -d fl && cd fl &&
	    if [[ ! -d data ]]; then
                echo '--- DOWNLOADING DATASET ON SERVER $server_ssh'
	        wget -q https://datashare.ed.ac.uk/download/DS_10283_3192.zip &&
	        unzip DS_10283_3192.zip && rm DS_10283_3192.zip &&
	        mkdir -p data &&
	        tar -C data -x -z -f CINIC-10.tar.gz && rm CINIC-10.tar.gz;
	    fi &&

	    if ! command -v docker; then
	        echo 'install docker on server $server_ssh' &&
	        exit 1 ;
            fi
        "
    fi
    ssh "$server_ssh" -- "
        if pgrep -f ^docker\ .*\ role=server; then
          pkill --signal 9 -f ^docker\ .*\ role=server;
        fi;
        sleep 2;
        tmux new-session -d -- 'docker run -v ~/fl/:/app/ -p $server_port:$server_port -e role=server --rm $docker_image; bash'
    "

    sleep 10

    i=0
    for client in "${clients[@]}" "${poisoned_clients[@]}"; do
	echo "=== CLIENT $i $client ==="

	if [[ "$update" == true ]]; then
	    echo "--- PREPARING CLIENT $i $client ---"
	    scp fl.zip "$client":~
	    ssh "$client" "
	        cd ~ && mkdir -p fl &&
	        unzip -o fl.zip -d fl && cd fl &&
                if [[ ! -d data ]]; then
                    echo '--- DOWNLOADING DATASET ON CLIENT $i $client'
                    wget -q https://datashare.ed.ac.uk/download/DS_10283_3192.zip &&
                    unzip DS_10283_3192.zip && rm DS_10283_3192.zip &&
                    mkdir -p data &&
                    tar -C data -x -z -f CINIC-10.tar.gz && rm CINIC-10.tar.gz;
                fi &&

	        if ! command -v docker; then
	            echo 'install docker on client $i $client' &&
                    exit 1;
	        fi
	    " &
	fi
        
        wait

	is_poisoned=false

	if [[ " ${poisoned_clients[@] } " == *" $client "* ]]; then
	    is_poisoned=true
	fi

	echo "--- RUNNING FL CLIENT ON $i $client ---"
	ssh "$client" "
            if pgrep -f ^docker\ .*\ role=client; then
              pkill --signal 9 -f ^docker\ .*\ role=client;
            fi;
            sleep 2;
	    tmux new-session -d -- 'docker run -v ~/fl/:/app $([[ $is_poisoned == true ]] && echo -- '-e is_poisoned true') -e role=client -e server_address=$server_ip -e client_index=$i -e total_clients=${#clients[@]} --rm -it $docker_image; bash'
	"
	(( i++ )) || :
    done
}

local=true
update=false
debug=false
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --local)
            local=true
            ;;
	--update)
	    update=true
	    ;;
        --debug)
	    debug=true
	    ;;
    esac
    shift
done


if [[ "$debug" == true ]]; then
    set -x
fi

if [[ "$local" == true ]]; then
    run_local
else
    run_remote "$update"
fi
