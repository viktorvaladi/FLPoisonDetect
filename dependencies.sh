curl -sSL https://install.python-poetry.org | python3 -
sleep 10
export PATH="/root/.local/bin:$PATH"
sleep 3
poetry install
sleep 25
poetry shell
sleep 3
pip install pandas

