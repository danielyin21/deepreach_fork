python run_experiment.py \
  --mode train \
  --experiment_class DeepReach \
  --dynamics_class Quad2DReachAvoid \
  --experiment_name quad2d_reachavoid_v0 \
  --loss_type HJ \
  --set_mode avoid \
  --minWith target \
  --m 1.0 --Iyy 0.1 --g -9.81 \
  --u1_min 0.0 --u1_max 19.62 --u2_min -0.05 --u2_max 0.05 \
  --obs_cx 2.5 --obs_cz 2.5 --obs_r 1.0 \
  --goal_x 4.5 --goal_z 2.5 \
  --goal_pos_tol 0.05 --goal_th_tol 0.05 --goal_vel_tol 0.05 --goal_om_tol 0.05 \
  --x_lo 0 --x_hi 5 --z_lo 0 --z_hi 5 \
  --th_lo -3.1415926536 --th_hi 3.1415926536 \
  --vx_lo -3 --vx_hi 3 --vz_lo -3 --vz_hi 3 --om_lo -1 --om_hi 1 \
  --kwargs '{}'



python run_experiment.py --mode train --experiment_class DeepReach --dynamics_class Quad2DReachAvoid --experiment_name quad2d_reachavoid_v0 --minWidth target --obs_cx 2.5 --obs_cz 2.5 --obs_r 1.0 --set_mode avoid

python run_experiment.py -c quad2d_reachavoid.yaml