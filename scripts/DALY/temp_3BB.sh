#!/usr/bin/env bash
python launcher.py --res_dir './data/DALY/res/' --datasetname 'DALY' --path_tracks './data/DALY/tracks/onlinelink_detectron_tracks_allAnn/' --path_log_eval './log/' --n_iterations '30000' --cache_dir './data/DALY/cache' --cstrs_name 'at_least_one_per_instance_unit_time_with_keyframes' --write_eval 'True' --path_list './data/DALY/list/' --path_info './data/DALY/infos/onlinelink_detectron_tracks_allAnn' --use_calibration --prepend_name 'DALY_detectron_onlinelink_allAnn' --n_actions '11'
