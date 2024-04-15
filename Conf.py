
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
"pr_p3_stability",
"hit_to_bouncing_point",
# "pr_p3_phaseDA",
# # # bouncing point
# "bouncing_point_to_cent",

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
"pr_p3_stability",

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
# # bouncing point
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
    # # bouncing point
    "bouncing_point_to_cent",
]

x_double_features_all_column = [
    "session_id",
    "team_skill",
    "team_max_seq",
    "team_avg_seq",

    # gaze event of receiver
    "receiver_pr_p1_al",
    "receiver_pr_p2_al",
    "receiver_pr_p3_fx",
    "receiver_pr_p1_cs",
    "receiver_pr_p2_cs",
    "receiver_pr_p3_cs",
    "receiver_pr_p1_al_onset",
    "receiver_pr_p1_al_prec",
    "receiver_pr_p1_al_mag",
    "receiver_pr_p2_al_onset",
    "receiver_pr_p2_al_prec",
    "receiver_pr_p2_al_mag",
    "receiver_pr_p3_fx_onset",
    "receiver_pr_p3_fx_duration",
    "receiver_pr_p3_stability",

    # gaze event of hitter
    "hitter_pr_p1_al",
    "hitter_pr_p2_al",
    "hitter_pr_p3_fx",
    "hitter_pr_p1_cs",
    "hitter_pr_p2_cs",
    "hitter_pr_p3_cs",
    "hitter_pr_p1_al_onset",
    "hitter_pr_p1_al_prec",
    "hitter_pr_p1_al_mag",
    "hitter_pr_p2_al_onset",
    "hitter_pr_p2_al_prec",
    "hitter_pr_p2_al_mag",
    "hitter_pr_p3_fx_onset",
    "hitter_pr_p3_fx_duration",
    "hitter_pr_p3_stability",

    # Joint attention
    # "ja_p23_minDu",
    # "ja_p23_maxDu",
    # "ja_p23_avgDu",
    "ja_percentage",



    # Forward swing and impact
    "receiver_ec_start_fs",
    "receiver_ec_fs_ball_racket_ratio",
    "receiver_ec_fs_ball_racket_dir",
    "receiver_ec_fs_ball_rball_dist",
    "receiver_im_racket_force",
    "receiver_im_ball_force",
    "receiver_im_rb_ang_collision",
    "receiver_im_rb_dist",
    "receiver_im_rack_wrist_dist",


    # bouncing point
    "hitter_hit_to_bouncing_point",
    "receiver_racket_to_root",
    "team_spatial_position",  # only for double


]


x_double_features_column = [
    # gaze event of receiver
    "receiver_pr_p1_al",
    "receiver_pr_p2_al",
    "receiver_pr_p3_fx",
    "receiver_pr_p1_cs",
    "receiver_pr_p2_cs",
    "receiver_pr_p3_cs",
    "receiver_pr_p1_al_onset",
    "receiver_pr_p1_al_prec",
    "receiver_pr_p1_al_mag",
    "receiver_pr_p2_al_onset",
    "receiver_pr_p2_al_prec",
    "receiver_pr_p2_al_mag",
    "receiver_pr_p3_fx_onset",
    "receiver_pr_p3_fx_duration",
    "receiver_pr_p3_stability",

    # gaze event of hitter
    "hitter_pr_p1_al",
    "hitter_pr_p2_al",
    "hitter_pr_p3_fx",
    "hitter_pr_p1_cs",
    "hitter_pr_p2_cs",
    "hitter_pr_p3_cs",
    "hitter_pr_p1_al_onset",
    "hitter_pr_p1_al_prec",
    "hitter_pr_p1_al_mag",
    "hitter_pr_p2_al_onset",
    "hitter_pr_p2_al_prec",
    "hitter_pr_p2_al_mag",
    "hitter_pr_p3_fx_onset",
    "hitter_pr_p3_fx_duration",
    "hitter_pr_p3_stability",

    # Joint attention
    # "ja_p23_minDu",
    # "ja_p23_maxDu",
    # "ja_p23_avgDu",
    "ja_percentage",



    # Forward swing and impact
    "receiver_ec_start_fs",
    "receiver_ec_fs_ball_racket_ratio",
    "receiver_ec_fs_ball_racket_dir",
    "receiver_ec_fs_ball_rball_dist",
    "receiver_im_racket_force",
    "receiver_im_ball_force",
    "receiver_im_rb_ang_collision",
    "receiver_im_rb_dist",
    "receiver_im_rack_wrist_dist",


    # bouncing point
    "hitter_hit_to_bouncing_point",
    "receiver_racket_to_root",
    "team_spatial_position",  # only for double

]
normalize_x_double_episode_columns = [
    # gaze event of receiver
    # "receiver_pr_p1_al",
    # "receiver_pr_p2_al",
    # "receiver_pr_p3_fx",
    # "receiver_pr_p1_cs",
    # "receiver_pr_p2_cs",
    # "receiver_pr_p3_cs",
    # "receiver_pr_p1_al_onset",
    # "receiver_pr_p1_al_prec",
    # "receiver_pr_p1_al_mag",
    # "receiver_pr_p2_al_onset",
    # "receiver_pr_p2_al_prec",
    # "receiver_pr_p2_al_mag",
    # "receiver_pr_p3_fx_onset",
    # "receiver_pr_p3_fx_duration",
    # "receiver_pr_p3_stability",

    # gaze event of hitter
    # "hitter_pr_p1_al",
    # "hitter_pr_p2_al",
    # "hitter_pr_p3_fx",
    # "hitter_pr_p1_cs",
    # "hitter_pr_p2_cs",
    # "hitter_pr_p3_cs",
    # "hitter_pr_p1_al_onset",
    # "hitter_pr_p1_al_prec",
    # "hitter_pr_p1_al_mag",
    # "hitter_pr_p2_al_onset",
    # "hitter_pr_p2_al_prec",
    # "hitter_pr_p2_al_mag",
    # "hitter_pr_p3_fx_onset",
    # "hitter_pr_p3_fx_duration",
    # "hitter_pr_p3_stability",

    # Joint attention
    # "ja_p23_minDu",
    # "ja_p23_maxDu",
    # "ja_p23_avgDu",
    # "ja_percentage",



    # Forward swing and impact
    # "receiver_ec_start_fs",
    # "receiver_ec_fs_ball_racket_ratio",
    "receiver_ec_fs_ball_racket_dir",
    # "receiver_ec_fs_ball_rball_dist",
    "receiver_im_racket_force",
    "receiver_im_ball_force",
    # "receiver_im_rb_ang_collision",
    # "receiver_im_rb_dist",
    # "receiver_im_rack_wrist_dist",


    # bouncing point
    # "hitter_hit_to_bouncing_point",
    # "receiver_racket_to_root",
    # "team_spatial_position",  # only for double

]


y_episode_column = ["success"]
y_regression_column = ["im_rb_ang_collision"]

excluded_subject = ["SE010C", "SE011A", "SE029C", "SE029A", "SE030B", "SE033A", "SE014B", "SE020B", "SE014A", "SE011C",
                    "SE018A", "SE017C"]

results_path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\summary\\"
single_results_path = results_path + "single_combined\\"
double_results_path = results_path + "double\\"
double_results_avg_path = results_path + "double\\avg_features\\"
single_summary_path = results_path + "single_summary.csv"
double_summary_path = results_path + "double_summary.csv"
