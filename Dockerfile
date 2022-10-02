FROM ubuntu

ENV role=client
ENV rounds=2
ENV server_address='0.0.0.0'
ENV total_clients=2
ENV client_index=0
#ENV is_poisoned=false

WORKDIR /app

RUN apt-get update && apt-get install -y python3 python3-pip git curl
RUN pip3 install flwr numpy tensorflow-cpu tensorflow-datasets

COPY . .

CMD python3 runner.py --$role --rounds $rounds ${is_poisoned+--is_poisoned} --server-address $server_address --total-clients $total_clients --client-index $client_index

