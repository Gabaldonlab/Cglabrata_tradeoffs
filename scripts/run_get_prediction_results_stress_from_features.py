#!/usr/bin/env python

# Runs the function get_prediction_results_stress_from_features for some situations

import os, sys, itertools
import numpy as np
import pandas as pd

ParentDir = "%s/samba"%(os.getenv("HOME"))
if not os.path.isdir(ParentDir): ParentDir = "/gpfs/projects/bsc40/mschikora"

# define dirs related to this work
CurDir = "%s/Cglabrata_tradeoffs"%ParentDir

# imports
sys.path.insert(0, CurDir)
import Cglabrata_tradeoffs_functions as fun

# parse arguments
idx, reshuffle_y, tryID_general, prediction_results_file, df_data_all_file, df_feature_info_all_file, only_predictors_univariate_corr, consider_interactions, model_name, stress_condition, fitness_estimate, type_feature_selection, tol_r2, try_ID = sys.argv[1:]

str_to_bool = {"True":True, "False":False}
only_predictors_univariate_corr = str_to_bool[only_predictors_univariate_corr]
consider_interactions = str_to_bool[consider_interactions]
reshuffle_y = str_to_bool[reshuffle_y]

if tol_r2=="None": tol_r2 = None
else: tol_r2 = float(tol_r2)

try_ID = int(try_ID)
tryID_general = int(tryID_general)

# define field
yfield = "%s_%s"%(fitness_estimate, stress_condition)

# exit if already done
if not fun.file_is_empty(prediction_results_file):
	print("Already generated, exiting")
	sys.exit(0)

# load dfs 
print("load dfs...")
df_data_all = fun.load_object(df_data_all_file)
df_feature_info_all = fun.load_object(df_feature_info_all_file)

# change yfield if necessary
if reshuffle_y is True: 

    initial_y_values = list(df_data_all[yfield].copy())
    df_data_all[yfield] = df_data_all[yfield].sample(frac=1, random_state=tryID_general+1).copy().values
    if initial_y_values==list(df_data_all[yfield].copy()): raise ValueError("y values should have changed")

# define WT-related info predictors that could be related to the resistance (fitness in the WT stress, afecting normalizations)
potential_WT_predictors = {"M_rAUC_norm": {"WT_M_AUC", "WT_M_fAUC_norm"},
                           "M_rAUC_diff": {"WT_M_AUC", "WT_M_fAUC_diff"},
                           "M_AUC_norm_WT": {"WT_M_AUC"},
                           "M_AUC_diff_WT": {"WT_M_AUC"}}[fitness_estimate]

potential_WT_predictors = {"%s_%s"%(p, stress_condition) for p in potential_WT_predictors}
potential_WT_predictors.add("WT_M_AUC_YPD") # add the WT fitness with no drug

# filter predictors
def get_is_interaction(x): return x.startswith("pairwise_interaction")
df_feature_info = df_feature_info_all[(df_feature_info_all.data_type=="continuous") | (df_feature_info_all.n_lineages_with_minor_cathegory_binary_features>=2)]
df_feature_info = df_feature_info[(df_feature_info.type_feature.isin({"broad_resistance_genes_profile_hot_encoded", "WT_info_hot_encoded", "single_variants", "domain_info", "pathway_info"})) | (df_feature_info.feature.isin(potential_WT_predictors)) | (df_feature_info.type_feature.apply(get_is_interaction))]

if only_predictors_univariate_corr is True: 
    print("keeping univariate predictors")

    # calculate pvals if not provided
    bp_to_samples = dict(df_data_all[[f for f in df_feature_info[df_feature_info.data_type=="binary"].feature]].apply(lambda c: set(c[c==1].index)))
    df_feature_info["pval_association"] = df_feature_info.apply(fun.get_pval_association_one_feature, yfield=yfield, df_data=df_data_all[[yfield, "sampleID"] + sorted(set(df_feature_info.feature))].set_index("sampleID"), all_samples=set(df_data_all.sampleID), bp_to_samples=bp_to_samples, axis=1)
    
    # define predictors that are correlated to resistance
    predictors_w_univariate_correlation = set(df_feature_info[(df_feature_info.pval_association<0.05)].feature)
    if len(predictors_w_univariate_correlation)==0: raise ValueError("There can't be 0 univariate predictors")
    df_feature_info = df_feature_info[(df_feature_info.feature.isin(predictors_w_univariate_correlation))]

# handle interactions. Only consider for linear regression and when only_predictors_univariate_corr is True. This is so because for all predictors there are too many tried interactions, perhaps leading to overfitting 
if model_name=="linear_regression" and only_predictors_univariate_corr is True and consider_interactions is True:
    print("adding combinations for continuous predictors...")

    # init the predictors
    all_continuous_predictors = list(df_feature_info[df_feature_info.data_type=="continuous"].feature)
    initial_all_continuous_predictors = set(all_continuous_predictors)
    all_binary_predictors =  list(df_feature_info[(df_feature_info.data_type=="binary") & ~(df_feature_info.type_feature.apply(get_is_interaction))].feature)

    all_combinations_bin = sorted({tuple(sorted([cp, bp])) for cp in all_continuous_predictors for bp in all_binary_predictors})
    all_combinations = list(itertools.combinations(all_continuous_predictors, 2)) + all_combinations_bin

    # add to df
    for Ip, (p1, p2) in enumerate(all_combinations):
    
        interaction_predictor_values = df_data_all[p1] * df_data_all[p2]
        if not any(df_data_all[all_continuous_predictors].apply(lambda p: np.corrcoef(p, interaction_predictor_values)[0, 1] , axis=0).apply(abs)==1):
            interaction_predictor = "%s * %s"%(p1, p2)
            all_continuous_predictors.append(interaction_predictor)
            df_data_all = df_data_all.assign(**{interaction_predictor: interaction_predictor_values})

    # add to df features
    df_feature_info_interactions = pd.DataFrame({"feature" : sorted(set(all_continuous_predictors).difference(initial_all_continuous_predictors)), "data_type":"continuous"})
    df_feature_info = pd.concat([df_feature_info[["feature", "data_type"]], df_feature_info_interactions])

# remove them if it is not linear regresssion
else: df_feature_info = df_feature_info[~(df_feature_info.type_feature.apply(get_is_interaction))][["feature", "data_type"]]

# check
if any(~df_feature_info.data_type.isin({"binary", "continuous"})): raise ValueError("data should be binary or continuous")

# get predictors
continuous_predictors = sorted(df_feature_info[df_feature_info.data_type=="continuous"].feature)
binary_predictors = sorted(df_feature_info[df_feature_info.data_type=="binary"].feature)
all_predictors = continuous_predictors + binary_predictors
# for data_type in ["binary"]: print(yfield, data_type, "predictors", list(df_feature_info[df_feature_info.data_type==data_type].feature))
# if len(all_predictors)==0: raise ValueError("predictors are 0")

# Standardize the continuous predictors (optional).  this does not transform the linearity, but it allows to pick the most important predictors
#if len(continuous_predictors)>0: df_data[continuous_predictors] = StandardScaler().fit_transform(df_data[continuous_predictors])

# keep some fields
df_data = df_data_all[[yfield, "exp_evol_lineage", "sampleID"] + all_predictors]

# get the prediction results
print("getting predicted features...")
prediction_results = fun.get_prediction_results_stress_from_features(idx, df_data, binary_predictors, continuous_predictors, fitness_estimate, stress_condition, model_name, yfield, try_ID, only_predictors_univariate_corr, type_feature_selection, tol_r2, consider_interactions, reshuffle_y, tryID_general)

# save
fun.save_object(prediction_results, prediction_results_file)
