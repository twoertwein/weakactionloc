#!/usr/bin/env bash
python launcher.py --res_dir './data/UCF101-24/res/' --datasetname 'UCF101' --path_tracks './data/UCF101-24/tracks/onlinelink_detectron_tracks_1key/' --path_log_eval './log/' --n_iterations '30000' --cache_dir './data/UCF101-24/cache' --cstrs_name 'at_least_one_per_temporal_point_unit_time_with_keyframes' --track_class_agnostic --write_eval 'True' --path_list './data/UCF101-24/list/' --path_info './data/UCF101-24/infos/onlinelink_detectron_tracks_1key' --use_calibration --prepend_name 'UCF101_detectron_onlinelink_1key' --n_actions '25'
