#!/bin/bash
# All training runs of the paper, one after the other on GPU ${GPU:-0} (one line per row of configs/paper_runs.csv), followed by the
# forward runs of the parameter-space ensemble (configs/paper_forward.csv). Inputs are read from data/ (see README.md and DATA.md).
# Wall time per run on an NVIDIA L40S: CaRT 0.3-1 h, CaRT+ family 1-2.8 h, HBV 0.3-1 h, dHBV 0.1-0.3 h, LSTM 0.2-0.5 h, EA-LSTM 3-3.7 h.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/src
G=${GPU:-0}
python -m cart.train --model cart --dataset gb --seed 0 --out runs/gb_cart_s0 --gpu $G
python -m cart.train --model cart --dataset gb --seed 1 --out runs/gb_cart_s1 --gpu $G
python -m cart.train --model cart --dataset gb --seed 2 --out runs/gb_cart_s2 --gpu $G
python -m cart.train --model cartplus --dataset gb --seed 0 --out runs/gb_cartplus_s0 --gpu $G
python -m cart.train --model cartplus --dataset gb --seed 1 --out runs/gb_cartplus_s1 --gpu $G
python -m cart.train --model cartplus --dataset gb --seed 2 --out runs/gb_cartplus_s2 --gpu $G
python -m cart.train --model cartplus_no_p1 --dataset gb --seed 0 --out runs/gb_cartplus_no_p1_s0 --gpu $G
python -m cart.train --model cartplus_no_p1 --dataset gb --seed 1 --out runs/gb_cartplus_no_p1_s1 --gpu $G
python -m cart.train --model cartplus_no_p1 --dataset gb --seed 2 --out runs/gb_cartplus_no_p1_s2 --gpu $G
python -m cart.train --model cartplus_no_p2 --dataset gb --seed 0 --out runs/gb_cartplus_no_p2_s0 --gpu $G
python -m cart.train --model cartplus_no_p2 --dataset gb --seed 1 --out runs/gb_cartplus_no_p2_s1 --gpu $G
python -m cart.train --model cartplus_no_p2 --dataset gb --seed 2 --out runs/gb_cartplus_no_p2_s2 --gpu $G
python -m cart.train --model cartplus_no_p3 --dataset gb --seed 0 --out runs/gb_cartplus_no_p3_s0 --gpu $G
python -m cart.train --model cartplus_no_p3 --dataset gb --seed 1 --out runs/gb_cartplus_no_p3_s1 --gpu $G
python -m cart.train --model cartplus_no_p3 --dataset gb --seed 2 --out runs/gb_cartplus_no_p3_s2 --gpu $G
python -m cart.train --model cartplus_no_p4 --dataset gb --seed 0 --out runs/gb_cartplus_no_p4_s0 --gpu $G
python -m cart.train --model cartplus_no_p4 --dataset gb --seed 1 --out runs/gb_cartplus_no_p4_s1 --gpu $G
python -m cart.train --model cartplus_no_p4 --dataset gb --seed 2 --out runs/gb_cartplus_no_p4_s2 --gpu $G
python -m cart.train --model cartplus_no_p5 --dataset gb --seed 0 --out runs/gb_cartplus_no_p5_s0 --gpu $G
python -m cart.train --model cartplus_no_p5 --dataset gb --seed 1 --out runs/gb_cartplus_no_p5_s1 --gpu $G
python -m cart.train --model cartplus_no_p5 --dataset gb --seed 2 --out runs/gb_cartplus_no_p5_s2 --gpu $G
python -m baselines.hbv_regional --dataset gb --seed 0 --out runs/gb_hbv_s0 --gpu $G
python -m baselines.hbv_regional --dataset gb --seed 1 --out runs/gb_hbv_s1 --gpu $G
python -m baselines.hbv_regional --dataset gb --seed 2 --out runs/gb_hbv_s2 --gpu $G
python -m baselines.dhbv --dataset gb --seed 0 --out runs/gb_dhbv_s0 --gpu $G
python -m baselines.dhbv --dataset gb --seed 1 --out runs/gb_dhbv_s1 --gpu $G
python -m baselines.dhbv --dataset gb --seed 2 --out runs/gb_dhbv_s2 --gpu $G
python -m baselines.lstm_stateful --dataset gb --seed 0 --out runs/gb_lstm_s0 --gpu $G
python -m baselines.lstm_stateful --dataset gb --seed 1 --out runs/gb_lstm_s1 --gpu $G
python -m baselines.lstm_stateful --dataset gb --seed 2 --out runs/gb_lstm_s2 --gpu $G
python -m baselines.ealstm --dataset gb --seed 0 --out runs/gb_ealstm_s0 --gpu $G
python -m baselines.ealstm --dataset gb --seed 1 --out runs/gb_ealstm_s1 --gpu $G
python -m baselines.ealstm --dataset gb --seed 2 --out runs/gb_ealstm_s2 --gpu $G
python -m cart.train --model cart --dataset us --seed 0 --out runs/us_cart_s0 --gpu $G
python -m cart.train --model cart --dataset us --seed 1 --out runs/us_cart_s1 --gpu $G
python -m cart.train --model cart --dataset us --seed 2 --out runs/us_cart_s2 --gpu $G
python -m cart.train --model cartplus --dataset us --seed 0 --out runs/us_cartplus_s0 --gpu $G
python -m cart.train --model cartplus --dataset us --seed 1 --out runs/us_cartplus_s1 --gpu $G
python -m cart.train --model cartplus --dataset us --seed 2 --out runs/us_cartplus_s2 --gpu $G
python -m cart.train --model cartplus_no_p1 --dataset us --seed 0 --out runs/us_cartplus_no_p1_s0 --gpu $G
python -m cart.train --model cartplus_no_p1 --dataset us --seed 1 --out runs/us_cartplus_no_p1_s1 --gpu $G
python -m cart.train --model cartplus_no_p1 --dataset us --seed 2 --out runs/us_cartplus_no_p1_s2 --gpu $G
python -m cart.train --model cartplus_no_p2 --dataset us --seed 0 --out runs/us_cartplus_no_p2_s0 --gpu $G
python -m cart.train --model cartplus_no_p2 --dataset us --seed 1 --out runs/us_cartplus_no_p2_s1 --gpu $G
python -m cart.train --model cartplus_no_p2 --dataset us --seed 2 --out runs/us_cartplus_no_p2_s2 --gpu $G
python -m cart.train --model cartplus_no_p3 --dataset us --seed 0 --out runs/us_cartplus_no_p3_s0 --gpu $G
python -m cart.train --model cartplus_no_p3 --dataset us --seed 1 --out runs/us_cartplus_no_p3_s1 --gpu $G
python -m cart.train --model cartplus_no_p3 --dataset us --seed 2 --out runs/us_cartplus_no_p3_s2 --gpu $G
python -m cart.train --model cartplus_no_p4 --dataset us --seed 0 --out runs/us_cartplus_no_p4_s0 --gpu $G
python -m cart.train --model cartplus_no_p4 --dataset us --seed 1 --out runs/us_cartplus_no_p4_s1 --gpu $G
python -m cart.train --model cartplus_no_p4 --dataset us --seed 2 --out runs/us_cartplus_no_p4_s2 --gpu $G
python -m cart.train --model cartplus_no_p5 --dataset us --seed 0 --out runs/us_cartplus_no_p5_s0 --gpu $G
python -m cart.train --model cartplus_no_p5 --dataset us --seed 1 --out runs/us_cartplus_no_p5_s1 --gpu $G
python -m cart.train --model cartplus_no_p5 --dataset us --seed 2 --out runs/us_cartplus_no_p5_s2 --gpu $G
python -m baselines.hbv_regional --dataset us --seed 0 --out runs/us_hbv_s0 --gpu $G
python -m baselines.hbv_regional --dataset us --seed 1 --out runs/us_hbv_s1 --gpu $G
python -m baselines.hbv_regional --dataset us --seed 2 --out runs/us_hbv_s2 --gpu $G
python -m baselines.dhbv --dataset us --seed 0 --out runs/us_dhbv_s0 --gpu $G
python -m baselines.dhbv --dataset us --seed 1 --out runs/us_dhbv_s1 --gpu $G
python -m baselines.dhbv --dataset us --seed 2 --out runs/us_dhbv_s2 --gpu $G
python -m baselines.lstm_stateful --dataset us --seed 0 --out runs/us_lstm_s0 --gpu $G
python -m baselines.lstm_stateful --dataset us --seed 1 --out runs/us_lstm_s1 --gpu $G
python -m baselines.lstm_stateful --dataset us --seed 2 --out runs/us_lstm_s2 --gpu $G
python -m baselines.ealstm --dataset us --seed 0 --out runs/us_ealstm_s0 --gpu $G
python -m baselines.ealstm --dataset us --seed 1 --out runs/us_ealstm_s1 --gpu $G
python -m baselines.ealstm --dataset us --seed 2 --out runs/us_ealstm_s2 --gpu $G
python -m cart.train --model cart --dataset aus --seed 0 --out runs/aus_cart_s0 --gpu $G
python -m cart.train --model cart --dataset aus --seed 1 --out runs/aus_cart_s1 --gpu $G
python -m cart.train --model cart --dataset aus --seed 2 --out runs/aus_cart_s2 --gpu $G
python -m cart.train --model cartplus --dataset aus --seed 0 --out runs/aus_cartplus_s0 --gpu $G
python -m cart.train --model cartplus --dataset aus --seed 1 --out runs/aus_cartplus_s1 --gpu $G
python -m cart.train --model cartplus --dataset aus --seed 2 --out runs/aus_cartplus_s2 --gpu $G
python -m baselines.hbv_regional --dataset aus --seed 0 --out runs/aus_hbv_s0 --gpu $G
python -m baselines.hbv_regional --dataset aus --seed 1 --out runs/aus_hbv_s1 --gpu $G
python -m baselines.hbv_regional --dataset aus --seed 2 --out runs/aus_hbv_s2 --gpu $G
python -m baselines.dhbv --dataset aus --seed 0 --out runs/aus_dhbv_s0 --gpu $G
python -m baselines.dhbv --dataset aus --seed 1 --out runs/aus_dhbv_s1 --gpu $G
python -m baselines.dhbv --dataset aus --seed 2 --out runs/aus_dhbv_s2 --gpu $G
python -m baselines.lstm_stateful --dataset aus --seed 0 --out runs/aus_lstm_s0 --gpu $G
python -m baselines.lstm_stateful --dataset aus --seed 1 --out runs/aus_lstm_s1 --gpu $G
python -m baselines.lstm_stateful --dataset aus --seed 2 --out runs/aus_lstm_s2 --gpu $G
python -m baselines.ealstm --dataset aus --seed 0 --out runs/aus_ealstm_s0 --gpu $G
python -m baselines.ealstm --dataset aus --seed 1 --out runs/aus_ealstm_s1 --gpu $G
python -m baselines.ealstm --dataset aus --seed 2 --out runs/aus_ealstm_s2 --gpu $G
# forward runs of the parameter-space ensemble (configs/paper_forward.csv): sim_noise<sigma>.npy in the CaRT+ run folders
python -m cart.predict --run runs/gb_cartplus_s0 --noise 0.1 --draws 32 --gpu $G
python -m cart.predict --run runs/gb_cartplus_s0 --noise 0.2 --draws 32 --gpu $G
python -m cart.predict --run runs/gb_cartplus_s0 --noise 0.3 --draws 32 --gpu $G
python -m cart.predict --run runs/gb_cartplus_s0 --noise 0.5 --draws 32 --gpu $G
python -m cart.predict --run runs/gb_cartplus_s0 --noise 0.7 --draws 32 --gpu $G
python -m cart.predict --run runs/gb_cartplus_s0 --noise 1.0 --draws 32 --gpu $G
python -m cart.predict --run runs/gb_cartplus_s1 --noise 0.1 --draws 32 --gpu $G
python -m cart.predict --run runs/gb_cartplus_s1 --noise 0.2 --draws 32 --gpu $G
python -m cart.predict --run runs/gb_cartplus_s1 --noise 0.3 --draws 32 --gpu $G
python -m cart.predict --run runs/gb_cartplus_s1 --noise 0.5 --draws 32 --gpu $G
python -m cart.predict --run runs/gb_cartplus_s1 --noise 0.7 --draws 32 --gpu $G
python -m cart.predict --run runs/gb_cartplus_s1 --noise 1.0 --draws 32 --gpu $G
python -m cart.predict --run runs/gb_cartplus_s2 --noise 0.1 --draws 32 --gpu $G
python -m cart.predict --run runs/gb_cartplus_s2 --noise 0.2 --draws 32 --gpu $G
python -m cart.predict --run runs/gb_cartplus_s2 --noise 0.3 --draws 32 --gpu $G
python -m cart.predict --run runs/gb_cartplus_s2 --noise 0.5 --draws 32 --gpu $G
python -m cart.predict --run runs/gb_cartplus_s2 --noise 0.7 --draws 32 --gpu $G
python -m cart.predict --run runs/gb_cartplus_s2 --noise 1.0 --draws 32 --gpu $G
python -m cart.predict --run runs/us_cartplus_s0 --noise 0.1 --draws 32 --gpu $G
python -m cart.predict --run runs/us_cartplus_s0 --noise 0.2 --draws 32 --gpu $G
python -m cart.predict --run runs/us_cartplus_s0 --noise 0.3 --draws 32 --gpu $G
python -m cart.predict --run runs/us_cartplus_s0 --noise 0.5 --draws 32 --gpu $G
python -m cart.predict --run runs/us_cartplus_s0 --noise 0.7 --draws 32 --gpu $G
python -m cart.predict --run runs/us_cartplus_s0 --noise 1.0 --draws 32 --gpu $G
python -m cart.predict --run runs/us_cartplus_s1 --noise 0.1 --draws 32 --gpu $G
python -m cart.predict --run runs/us_cartplus_s1 --noise 0.2 --draws 32 --gpu $G
python -m cart.predict --run runs/us_cartplus_s1 --noise 0.3 --draws 32 --gpu $G
python -m cart.predict --run runs/us_cartplus_s1 --noise 0.5 --draws 32 --gpu $G
python -m cart.predict --run runs/us_cartplus_s1 --noise 0.7 --draws 32 --gpu $G
python -m cart.predict --run runs/us_cartplus_s1 --noise 1.0 --draws 32 --gpu $G
python -m cart.predict --run runs/us_cartplus_s2 --noise 0.1 --draws 32 --gpu $G
python -m cart.predict --run runs/us_cartplus_s2 --noise 0.2 --draws 32 --gpu $G
python -m cart.predict --run runs/us_cartplus_s2 --noise 0.3 --draws 32 --gpu $G
python -m cart.predict --run runs/us_cartplus_s2 --noise 0.5 --draws 32 --gpu $G
python -m cart.predict --run runs/us_cartplus_s2 --noise 0.7 --draws 32 --gpu $G
python -m cart.predict --run runs/us_cartplus_s2 --noise 1.0 --draws 32 --gpu $G
python -m cart.predict --run runs/aus_cartplus_s0 --noise 0.3 --draws 32 --gpu $G
python -m cart.predict --run runs/aus_cartplus_s0 --noise 0.5 --draws 32 --gpu $G
python -m cart.predict --run runs/aus_cartplus_s1 --noise 0.3 --draws 32 --gpu $G
python -m cart.predict --run runs/aus_cartplus_s1 --noise 0.5 --draws 32 --gpu $G
python -m cart.predict --run runs/aus_cartplus_s2 --noise 0.3 --draws 32 --gpu $G
python -m cart.predict --run runs/aus_cartplus_s2 --noise 0.5 --draws 32 --gpu $G
