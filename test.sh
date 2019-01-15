# python main.py --mode test \
#                --config sessions/11/config.json \
#                --noisy_input_path ./wsj0-mix/2speakers/wav8k/min/tt/mix/ \
# 			--clean_input_path ./wsj0-mix/2speakers/wav8k/min/tt/ \
#                --load_checkpoint sessions/11/checkpoints/checkpoint.00200-49.337.hdf5 \
# 			--use_pit \
#                --use_pad > log/11_pit_pad

python main.py --mode test \
               --config sessions/11/config.json \
               --noisy_input_path ./wsj0-mix/2speakers/wav8k/min/tt/mix/ \
               --clean_input_path ./wsj0-mix/2speakers/wav8k/min/tt/ \
               --load_checkpoint sessions/11/checkpoints/checkpoint.00200-49.337.hdf5 \
               --no_pit \
               --use_pad > log/11_pad

python main.py --mode test \
               --config sessions/11/config.json \
               --noisy_input_path ./wsj0-mix/2speakers/wav8k/min/tt/mix/ \
               --clean_input_path ./wsj0-mix/2speakers/wav8k/min/tt/ \
               --load_checkpoint sessions/11/checkpoints/checkpoint.00200-49.337.hdf5 \
               --use_pit \
               --no_pad > log/11_pit

python main.py --mode test \
               --config sessions/11/config.json \
               --noisy_input_path ./wsj0-mix/2speakers/wav8k/min/tt/mix/ \
               --clean_input_path ./wsj0-mix/2speakers/wav8k/min/tt/ \
               --load_checkpoint sessions/11/checkpoints/checkpoint.00200-49.337.hdf5 \
               --no_pit \
               --no_pad > log/11_no
