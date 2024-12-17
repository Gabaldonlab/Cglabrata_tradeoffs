#!/usr/bin/env python

"""
Analysis of fitness tradeoffs in C. glabrata
"""
#%% SET ENV

import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from Bio import SeqIO
import sys
import pandas as pd
import seaborn as sns

# define dirs
ParentDir = "%s/samba"%(os.getenv("HOME"))
if not os.path.isdir(ParentDir): ParentDir = "/gpfs/projects/bsc40/mschikora"
CurDir = "%s/Cglabrata_tradeoffs"%ParentDir

# redefine for MAC
#CurDir = "/Users/mschikora/Desktop/Cglabrata_tradeoffs_4Feb"

# define dirs related to this work
ManualDataDir = "%s/manual_data"%CurDir
ProcessedDataDir = "%s/processed_data"%CurDir
PlotsDir = "%s/plots"%CurDir
PlotsDir_paper = "%s/plots_paper"%CurDir
Ksiezopolska2021_dir = "%s/Cglabrata_antifungals"%ParentDir

# imports
sys.path.insert(0, CurDir)
import Cglabrata_tradeoffs_functions as fun

# make some folder
fun.make_folder(PlotsDir_paper)

#%% LOAD DATASETS

# define samples
print("def samples")
#all_samples_wgs = (set(pd.read_csv("%s/VarCall/integrated_CNV_genes_and_regions.tab"%Ksiezopolska2021_dir, sep="\t").sampleID).difference(fun.wrong_samples_Ksiezopolska2021))
all_samples_wgs = set(pd.read_csv("%s/Ksiezopolska2021_integrated_CNV_genes_and_regions.tab"%ManualDataDir, sep="\t").sampleID).difference(fun.wrong_samples_Ksiezopolska2021)
sampleNames_with_wgs = {"_".join(s.split("_")[1:]) for s in all_samples_wgs}
iv_evol_conditions = {'AinF', 'FLZ', 'YPD', 'FinA', 'ANIFLZ', 'ANI', 'WT'}
drug_conditions = {'AinF', 'FLZ', 'FinA', 'ANIFLZ', 'ANI'}
non_wt_conditions =  {'AinF', 'FLZ', 'YPD', 'FinA', 'ANIFLZ', 'ANI'}
sampleNames_with_wgs = {s for s in sampleNames_with_wgs if s.split("_")[-1] in iv_evol_conditions}

# load datasets from Ksiezopolska 2021 (vars appearing in vitro and phenotypes). These are generated with the script paper_inVitroEvol_figures_and_tables.py
print("load Ksiezopolska data")
df_new_vars = pd.read_csv("%s/Ksiezopolska2021_inVitroEvol_vars_df_with_SVsCNVs.tab"%ManualDataDir, sep="\t") # all variants appearing during evolution (small, FKS, ERG3, aneuploidies, SVs)
df_phenotypes_Ksiezopolska2021 = pd.read_csv("%s/Ksiezopolska2021_fitness_df_MICs_and_AUCs.tab"%ManualDataDir, sep="\t") # this has the profiles used to generate figure S6 from Ksiezopolska2021
df_all_haploid_vars_Ksiezopolska2021 = pd.read_csv("%s/Ksiezopolska2021_df_vars_haploid_not_in_parents.tab"%ManualDataDir, sep="\t")

# add fields
df_new_vars["condition"] = df_new_vars.sampleID.apply(lambda x: x.split("_")[2])
df_new_vars["with_wgs"] = df_new_vars.sampleID.isin(sampleNames_with_wgs)

# filter vars
genes_mutated_YPD = set(df_new_vars[df_new_vars.condition=="YPD"].final_name) # I had a few vars in CST78_4G_YPD and F15_8G_YPD which could be errors
wrong_genes = {"CAGL0G05522g", "Scer_IRA1"}.union(genes_mutated_YPD) # the first are some that do not pass manual curation
df_new_vars = df_new_vars[df_new_vars.condition.isin(drug_conditions) & ~(df_new_vars.final_name.isin(wrong_genes))]

# load stress fitness data, provided by JC
print("load stress phenos")
df_stress_phenotypes, df_stress_phenotypes_all = fun.get_df_stress_phenotypes(ManualDataDir, df_new_vars, sampleNames_with_wgs, ProcessedDataDir)

# check correct samples
samples_evol_wgs_muts = set(df_new_vars[(df_new_vars.with_wgs) & (df_new_vars.condition.isin(drug_conditions))].sampleID)
if {s for s in sampleNames_with_wgs if s.split("_")[-1] in drug_conditions}!=samples_evol_wgs_muts: raise ValueError("some missing samples")
if len(set(df_new_vars.sampleID).difference(set(df_phenotypes_Ksiezopolska2021.new_sample_name)))>0: raise ValueError("strange samples in df_new_vars")

# load gene and pathway annotation data, from Schikora-Tamarit 2024 (used the same reference genome)
print("load gene feats...")
gene_features_df, df_pathway_annots = fun.get_df_gene_annotations(ManualDataDir, ProcessedDataDir)

# get df with all data and predictive features processed_data
print("load df_data")
df_data_all, df_feature_info, df_data_all_file, df_feature_info_file, df_data_no_pruning_file, df_feature_info_no_pruning_file = fun.get_df_all_predictive_features(df_new_vars, df_stress_phenotypes, df_phenotypes_Ksiezopolska2021, sampleNames_with_wgs, ProcessedDataDir, PlotsDir, gene_features_df, df_pathway_annots)

df_stress_phenotypes["relevant_row"] = (df_stress_phenotypes.Treatment.isin(drug_conditions)) & ((df_stress_phenotypes.type_measurement=="Q-PHAST_round1") | (df_stress_phenotypes.condition=="CycA"))

# get univariate associations for each variant
df_univariate_associations = fun.check_univariate_associations(df_data_all, df_feature_info, ProcessedDataDir)

# filter to keep one tradeoff per stress condition
df_univariate_associations["tradeoff"] = df_univariate_associations.yfield.apply(lambda x: "_".join(x.split("_")[0:-1]))
df_univariate_associations["relevant_tradeoff"] = (df_univariate_associations.tradeoff)==(df_univariate_associations.stress_condition.apply(lambda x: fun.condition_to_tradeoff_f[x]))
df_univariate_associations = df_univariate_associations[df_univariate_associations.relevant_tradeoff].copy()
if len(df_univariate_associations)!=len(df_univariate_associations[["stress_condition", "feature"]].drop_duplicates()):
    raise ValueError("non unique datasets")
    
# get df with the results of different predictors
print("load df_models")
df_models_all = fun.get_results_different_models(df_data_all_file, df_feature_info_file, ProcessedDataDir, PlotsDir)
df_models = fun.get_grouped_df_models_across_tries(df_models_all, ProcessedDataDir)
    
# get the df models without reshuffling but with p values
df_models_real, df_models_1st_reshuffle = fun.get_df_models_with_pvalues(df_models, ProcessedDataDir)

# define sig model
df_models_real["sig_model_all"] = (df_models_real.pval_resampling<0.05) & (df_models_real.pval_maxT<0.05) & (df_models_real.ntries_predictors>=3)

condition_to_min_pval = dict(df_models_real[df_models_real.sig_model_all].groupby("stress_condition").apply(lambda df: max([0.1, max(df.mean_mean_r2) - 0.05])))
for c in set(df_models_real.stress_condition).difference(set(condition_to_min_pval.keys())): condition_to_min_pval[c] = 0.1
df_models_real["sig_model"] = (df_models_real.sig_model_all) & (df_models_real.apply(lambda r: r.mean_mean_r2>=condition_to_min_pval[r.stress_condition], axis=1))


#%% PLOTS & TABLES FOR PAPER

#%% find sample with expected mechanisms 

# keep samples with low susceptibility towards NaCl and CycA
df = df_data_all[(df_data_all.M_rAUC_norm_NaCl<0.5) & (df_data_all.M_rAUC_norm_CycA<0.5)].copy()

# print
df_w = df[["M_rAUC_norm_NaCl", "M_rAUC_norm_CycA", "WT_M_fAUC_norm_NaCl", "presence_variant-ChrE-DUP", "GO_BP_GO:0046394_truncation",   "FKS2_miss_659-663", "GO_CC_GO:0016020_truncation"]]

#%% mutations and WT values for all strains

# load
df_data_no_pruning = fun.load_object(df_data_no_pruning_file)
df_feature_info_no_pruning = fun.load_object(df_feature_info_no_pruning_file)

# get_table 
df_info_vars = fun.generate_table_mutations_and_phenos(df_data_no_pruning, df_feature_info_no_pruning, PlotsDir_paper)

#print certain samples
for sampleID, conditions in [("CBS138_9F_AinF", ["CycA", "YPD", "NaCl"])]:
    print(sampleID)
    sample_series = df_info_vars.loc[sampleID]
    
    # print vars
    var_fields = [k.replace("presence_variant-", "") for k in df_feature_info_no_pruning[df_feature_info_no_pruning.type_feature=="single_variants"].feature]
    print("variants: ", ", ".join([k for k in var_fields if sample_series[k]==1]))
    print("susceptibilities: ", ", ".join(["%s=%.2f"%(f, sample_series[f])  for f in sample_series.keys() if not f in var_fields and f.split("_")[-1] in conditions]))
    

#%% Pvalue tables
df_pvals = fun.get_pvalue_tables_per_condition(df_stress_phenotypes, PlotsDir_paper)

#%% Distribution of tradeoff (i.e. M_rAUC_norm)

df_plot = df_stress_phenotypes[df_stress_phenotypes.relevant_row].copy()
df_plot["tradeoff"] = df_plot.apply(lambda r: r[fun.condition_to_tradeoff_f[r.condition]], axis=1)    

for conds, plt_w, ylab in  [({'CFW', 'CR', 'CycA', 'DTT', 'H2O2', 'NaCl', 'SDS'}, 3, "$rAUC$"), ({"YPD"}, 0.5, "$AUC\ /\ AUC_{WT}$")]:
    
    df_p = df_plot[df_plot.condition.isin(conds)]
    
    fig = plt.figure(figsize=(plt_w, 1.5))
    order_x = sorted(set(df_p.condition))
    ax = sns.swarmplot(data=df_p, x="condition", hue="condition", y="tradeoff", s=2, order=order_x, palette=fun.stress_condition_to_color) # dodge=True
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
    ax.set_ylabel(ylab)
    
    ax.set_ylim([-0.05, 1.65])
    
    ax.set_xlabel("")
    ax.axhline(1.0, color="k", linestyle="--", linewidth=.7)
    ax.set_xticklabels(order_x, rotation=90)

    fig.savefig("%s/distribution_tradeoff_all_%s.pdf"%(PlotsDir_paper, conds), bbox_inches="tight")

#%% Pval vs r2 univariate analysis

# get the plot for univariate analysis
df_plot = df_univariate_associations[(df_univariate_associations.field_for_pred==True)].copy()
df_plot["pval_fdr"] = fun.multitest.fdrcorrection(df_plot.pval)[1]
df_plot["pval_bonferroni"] = fun.multitest.multipletests(df_plot.pval, alpha=0.05, method="bonferroni")[1]
df_plot["r2"] = df_plot.mean_r2_lin_model.apply(lambda x: max([0, x]))
df_plot["minus_logp_bonf"] = df_plot.pval_bonferroni.apply(lambda x: -np.log10(x))

fig = plt.figure(figsize=(1.5, 1.5))
plt.scatter(df_plot.r2, df_plot.minus_logp_bonf, edgecolor=[fun.stress_condition_to_color[c] for c in df_plot.stress_condition], facecolor="none", s=9)
ax = fig.axes[0]
ax.set_xlabel("explained var. ($r^2$)")
ax.set_ylabel("$-log\ p_{\ BONFERRONI}$\nKS / spearman r test")
plt.axhline(-np.log10(0.05), linewidth=.7, color="k", linestyle="--")
plt.axvline(0.1, linewidth=.7, color="k", linestyle="--")

ax.set_xlim([-0.05, 0.7])
ax.set_ylim([-0.15, 6])

fig.savefig("%s/univariate_pval_vs_r2.pdf"%(PlotsDir_paper), bbox_inches='tight')

#%% Broad results multivariant analysis

# get # of sig models per condition
print(df_models_real.groupby("stress_condition").apply(lambda df: sum(df.sig_model)))
print(df_models_real[df_models_real.sig_model].groupby("stress_condition").apply(lambda df: max(df[df.sig_model].mean_mean_r2) ))

# plot
fun.plot_broad_results_multivariate_analysis(df_models_real, PlotsDir_paper)

#%% Heatmap most relevant features

fun.plot_heatmap_sig_features(df_models_real, PlotsDir_paper, df_feature_info)

#%% Plot sig models

df_models_real_sig = df_models_real[df_models_real.sig_model]
for cond in sorted(set(df_models_real_sig.stress_condition)):
        
    #if not cond in {"DTT", "NaCl", "CycA", "CR", "CFW", "YPD", "H2O2"}: continue
    #if not cond in {"H2O2", "YPD", "CFW"}: continue

    ax = fun.plot_best_multivariate_model(df_models_real_sig, cond, PlotsDir_paper, df_feature_info, df_data_all)

