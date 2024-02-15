
y_global_column = ["skill"]


x_episode_columns = [

"ec_fs_ball_racket_dir",
"ec_fs_ball_racket_ratio",
"ec_fs_ball_rball_dist",
"ec_start_fs",
"im_ball_force",
"im_rack_wrist_dist",
"im_racket_force",
"im_rb_ang_collision",
"im_rb_dist",
"pr_p1_al",
"pr_p1_al_gM",
"pr_p1_al_miDo",
"pr_p1_al_on",
"pr_p1_sf",
"pr_p2_al",
"pr_p2_al_gM",
"pr_p2_al_miDo",
"pr_p2_al_on",
"pr_p2_sf",
"pr_p3_fx",
"pr_p3_fx_du",
"pr_p3_fx_on",
"pr_p3_phaseDA",
# # bouncing point
"bouncing_point_to_cent",

]

x_perception = [
"pr_p1_al_gM",
"pr_p1_al_miDo",
"pr_p1_al_on",
"pr_p1_sf",
"pr_p2_al_gM",
"pr_p2_al_miDo",
"pr_p2_al_on",
"pr_p2_sf",
"pr_p3_fx_du",
"pr_p3_fx_on",
"pr_p3_phaseDA",

]


x_important = [
"ec_fs_ball_racket_dir",
"ec_fs_ball_racket_ratio",
"ec_fs_ball_rball_dist",
"ec_start_fs",
"im_ball_force",
"im_rack_wrist_dist",
"im_racket_force",
"im_rb_ang_collision",
"im_rb_dist",
"pr_p1_al_gM",
"pr_p1_al_miDo",
"pr_p1_al_on",
"pr_p1_sf",
"pr_p2_al_gM",
"pr_p2_al_miDo",
"pr_p2_al_on",
"pr_p2_sf",
"pr_p3_fx_du",
"pr_p3_fx_on",
"pr_p3_phaseDA",
# bouncing point
"bouncing_point_to_cent",
]

normalize_x_episode_columns =  [
    # "pr_p1_al",
    # "pr_p2_al",
    # "pr_p3_fx",
    # "pr_p1_sf",
    # "pr_p2_sf",
    "pr_p1_al_on",
    # "pr_p1_al_miDo",
    "pr_p1_al_gM",
    "pr_p2_al_on",
    # "pr_p2_al_miDo",
    "pr_p2_al_gM",
    "pr_p3_fx_on",
    "pr_p3_fx_du",
    "pr_p3_phaseDA",
    "ec_start_fs",
    "ec_fs_ball_racket_ratio",
    "ec_fs_ball_racket_dir",
    "ec_fs_ball_rball_dist",
    "im_racket_force",
    "im_ball_force",
    "im_rb_ang_collision",
    "im_rb_dist",
    "im_rack_wrist_dist",
    # bouncing point
    "bouncing_point_to_cent",
]

x_double_features_all_column = [
    "session_id",
    "team_skill",
    "team_max_seq",
    "team_avg_seq",

    # gaze event of hitter
    "hitter_pr_p1_al",
    "hitter_pr_p2_al",
    "hitter_pr_p3_fx",
    "hitter_pr_p1_sf",
    "hitter_pr_p2_sf",
    "hitter_pr_p1_al_on",
    "hitter_pr_p1_al_miDo",
    "hitter_pr_p1_al_gM",
    "hitter_pr_p2_al_on",
    "hitter_pr_p2_al_miDo",
    "hitter_pr_p2_al_gM",
    "hitter_pr_p3_fx_on",
    "hitter_pr_p3_fx_du",
    "hitter_pr_p3_phaseDA",

    # gaze event of observer
    "observer_pr_p1_al",
    "observer_pr_p2_al",
    "observer_pr_p3_fx",
    "observer_pr_p1_sf",
    "observer_pr_p2_sf",
    "observer_pr_p1_al_on",
    "observer_pr_p1_al_miDo",
    "observer_pr_p1_al_gM",
    "observer_pr_p2_al_on",
    "observer_pr_p2_al_miDo",
    "observer_pr_p2_al_gM",
    "observer_pr_p3_fx_on",
    "observer_pr_p3_fx_du",
    "observer_pr_p3_phaseDA",

    # Joint attention
    "ja_p1_minDu",
    "ja_p1_maxDu",
    "ja_p1_avgDu",
    "ja_p1_per",

    "ja_p23_minDu",
    "ja_p23_maxDu",
    "ja_p23_avgDu",
    "ja_p23_per",

    # Gaze coorientation
    "gc_p1_crossCorr",
    "gc_p1_crossCorrLag",
    # "gc_p1_crossCorrAVG",
    "gc_p1_crossCorrVel",
    "gc_p1_crossCorrVelLag",
    # "gc_p1_crossCorrVelAVG",
    "gc_p1_phaseSync",
    # "gc_p1_mi",
    # "gc_p1_te",
    # "gc_p1_freq_lw",
    # "gc_p1_freq_mw",
    # "gc_p1_freq_hw",
    # "gc_p1_freq_lhw",

    "gc_p23_crossCorr",
    "gc_p23_crossCorrLag",
    # "gc_p23_crossCorrAVG",
    "gc_p23_crossCorrVel",
    "gc_p23_crossCorrVelLag",
    # "gc_p23_crossCorrVelAVG",
    "gc_p23_phaseSync",
    # "gc_p23_mi",
    # "gc_p23_te",
    # "gc_p23_freq_lw",
    # "gc_p23_freq_mw",
    # "gc_p23_freq_hw",
    # "gc_p23_freq_lhw",


    # Forward swing and impact
    "hitter_ec_start_fs",
    "hitter_ec_fs_ball_racket_ratio",
    "hitter_ec_fs_ball_racket_dir",
    "hitter_ec_fs_ball_rball_dist",
    "hitter_im_racket_force",
    "hitter_im_ball_force",
    "hitter_im_rb_ang_collision",
    "hitter_im_rb_dist",
    "hitter_im_rack_wrist_dist",
    "hitter_spatial_position",  # only for double

    # Pose sim swing and impact
    # "dtw_p23_sim",
    "lcc_p23_sim",

    # Features relative diff
    "rdiff_spatial_position",  # only for double
    "rdiff_bounce_dist",
    "rdiff_ec_start_fs",
    "rdiff_ec_fs_ball_racket_ratio",
    "rdiff_im_rack_wrist_dist",

]


x_double_features_column = [
    # gaze event of hitter
    "hitter_pr_p1_al",
    "hitter_pr_p2_al",
    "hitter_pr_p3_fx",
    "hitter_pr_p1_sf",
    "hitter_pr_p2_sf",
    "hitter_pr_p1_al_on",
    "hitter_pr_p1_al_miDo",
    "hitter_pr_p1_al_gM",
    "hitter_pr_p2_al_on",
    "hitter_pr_p2_al_miDo",
    "hitter_pr_p2_al_gM",
    "hitter_pr_p3_fx_on",
    "hitter_pr_p3_fx_du",
    "hitter_pr_p3_phaseDA",

    # gaze event of observer
    "observer_pr_p1_al",
    "observer_pr_p2_al",
    "observer_pr_p3_fx",
    "observer_pr_p1_sf",
    "observer_pr_p2_sf",
    "observer_pr_p1_al_on",
    "observer_pr_p1_al_miDo",
    "observer_pr_p1_al_gM",
    "observer_pr_p2_al_on",
    "observer_pr_p2_al_miDo",
    "observer_pr_p2_al_gM",
    "observer_pr_p3_fx_on",
    "observer_pr_p3_fx_du",
    "observer_pr_p3_phaseDA",

    # Joint attention
    "ja_p1_minDu",
    "ja_p1_maxDu",
    "ja_p1_avgDu",
    "ja_p1_per",

    "ja_p23_minDu",
    "ja_p23_maxDu",
    "ja_p23_avgDu",
    "ja_p23_per",

    # Gaze coorientation
    "gc_p1_crossCorr",
    "gc_p1_crossCorrLag",
    # "gc_p1_crossCorrAVG",
    "gc_p1_crossCorrVel",
    "gc_p1_crossCorrVelLag",
    # "gc_p1_crossCorrVelAVG",
    "gc_p1_phaseSync",
    # "gc_p1_mi",
    # "gc_p1_te",
    "gc_p1_freq_lw",
    "gc_p1_freq_mw",
    "gc_p1_freq_hw",
    # "gc_p1_freq_lhw",

    "gc_p23_crossCorr",
    "gc_p23_crossCorrLag",
    # "gc_p23_crossCorrAVG",
    "gc_p23_crossCorrVel",
    "gc_p23_crossCorrVelLag",
    # "gc_p23_crossCorrVelAVG",
    "gc_p23_phaseSync",
    # "gc_p23_mi",
    # "gc_p23_te",
    "gc_p23_freq_lw",
    "gc_p23_freq_mw",
    "gc_p23_freq_hw",
    # "gc_p23_freq_lhw",


    # Forward swing and impact
    "hitter_ec_start_fs",
    "hitter_ec_fs_ball_racket_ratio",
    "hitter_ec_fs_ball_racket_dir",
    "hitter_ec_fs_ball_rball_dist",
    "hitter_im_racket_force",
    "hitter_im_ball_force",
    "hitter_im_rb_ang_collision",
    "hitter_im_rb_dist",
    "hitter_im_rack_wrist_dist",
    "hitter_spatial_position",  # only for double

    # Pose sim swing and impact
    # "dtw_p23_sim",
    "lcc_p23_sim",

    # Features relative diff
    "rdiff_spatial_position",  # only for double
    "rdiff_bounce_dist",
    "rdiff_ec_start_fs",
    "rdiff_ec_fs_ball_racket_ratio",
    "rdiff_im_rack_wrist_dist",
]
normalize_x_double_episode_columns = [
    # gaze event of hitter
    "hitter_pr_p1_al_on",
    "hitter_pr_p1_al_miDo",
    "hitter_pr_p1_al_gM",
    "hitter_pr_p2_al_on",
    "hitter_pr_p2_al_miDo",
    "hitter_pr_p2_al_gM",
    "hitter_pr_p3_fx_on",
    "hitter_pr_p3_fx_du",
    "hitter_pr_p3_phaseDA",

    # gaze event of observer
    "observer_pr_p1_al_on",
    "observer_pr_p1_al_miDo",
    "observer_pr_p1_al_gM",
    "observer_pr_p2_al_on",
    "observer_pr_p2_al_miDo",
    "observer_pr_p2_al_gM",
    "observer_pr_p3_fx_on",
    "observer_pr_p3_fx_du",
    "observer_pr_p3_phaseDA",

    # Joint attention
    "ja_p1_minDu",
    "ja_p1_maxDu",
    "ja_p1_avgDu",
    "ja_p1_per",

    "ja_p23_minDu",
    "ja_p23_maxDu",
    "ja_p23_avgDu",
    "ja_p23_per",

    # Gaze coorientation
    "gc_p1_crossCorr",
    "gc_p1_crossCorrLag",
    # "gc_p1_crossCorrAVG",
    "gc_p1_crossCorrVel",
    "gc_p1_crossCorrVelLag",
    # "gc_p1_crossCorrVelAVG",
    "gc_p1_phaseSync",
    # "gc_p1_mi",
    # "gc_p1_te",
    "gc_p1_freq_lw",
    "gc_p1_freq_mw",
    "gc_p1_freq_hw",
    # "gc_p1_freq_lhw",

    "gc_p23_crossCorr",
    "gc_p23_crossCorrLag",
    # "gc_p23_crossCorrAVG",
    "gc_p23_crossCorrVel",
    "gc_p23_crossCorrVelLag",
    # "gc_p23_crossCorrVelAVG",
    "gc_p23_phaseSync",
    # "gc_p23_mi",
    # "gc_p23_te",
    "gc_p23_freq_lw",
    "gc_p23_freq_mw",
    "gc_p23_freq_hw",
    # "gc_p23_freq_lhw",



    # Forward swing and impact
    "hitter_ec_start_fs",
    "hitter_ec_fs_ball_racket_ratio",
    "hitter_ec_fs_ball_racket_dir",
    "hitter_ec_fs_ball_rball_dist",
    "hitter_im_racket_force",
    "hitter_im_ball_force",
    "hitter_im_rb_ang_collision",
    "hitter_im_rb_dist",
    "hitter_im_rack_wrist_dist",
    "hitter_spatial_position",  # only for double

    # Pose sim swing and impact
    # "dtw_p23_sim",
    "lcc_p23_sim",

    # Features relative diff
    "rdiff_spatial_position",  # only for double
    "rdiff_bounce_dist",
    "rdiff_ec_start_fs",
    "rdiff_ec_fs_ball_racket_ratio",
    "rdiff_im_rack_wrist_dist",

]


y_episode_column = ["success"]
y_regression_column = ["im_rb_ang_collision"]

excluded_subject = ["SE010C", "SE011A", "SE029C", "SE029A", "SE030B", "SE033A", "SE014B", "SE020B", "SE014A", "SE011C",
                    "SE018A", "SE017C"]

results_path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\summary\\"
single_results_path = results_path + "single\\"
double_results_path = results_path + "double\\"
double_results_avg_path = results_path + "double\\avg_features\\"
single_summary_path = results_path + "single_summary.csv"
double_summary_path = results_path + "double_summary.csv"
