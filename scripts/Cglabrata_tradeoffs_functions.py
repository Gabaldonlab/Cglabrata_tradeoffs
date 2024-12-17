#!/usr/bin/env python

# Functions


######## ENV ##########
print("loading functions...")

import pickle, sys, random, re, os, shutil, subprocess
import pandas as pd
import copy as cp
import numpy as np
from collections import Counter
import scipy.stats as stats
from statsmodels.stats import multitest
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import time
import matplotlib.patches as mpatches

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


import statsmodels.api as sm_api
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import multiprocessing as multiproc
from sklearn.feature_selection import SelectFromModel
from matplotlib.lines import Line2D


ParentDir = "%s/samba"%(os.getenv("HOME"))
if not os.path.isdir(ParentDir): ParentDir = "/gpfs/projects/bsc40/mschikora"

#######################


######### VARIABLES #########

# define samples that have various types of duplications in chromosomes
chrE_total_dup = {"RUN1_CST34_2A_FLZ", "RUN4_CST34_2C_FLZ", "RUN4_CST34_2C_FinA", "RUN1_EB0911_3H_FLZ", "RUN1_EB0911_3H_FinA", "RUN1_EB0911_3H_AinF", "RUN2_CST78_4G_FLZ",  "RUN2_CST78_4G_FinA", "RUN4_EF1620_7B_FLZ", "RUN4_EF1620_7B_FinA", "RUN3_EF1620_7B_AinF", "RUN1_EF1620_7D_FLZ", "RUN1_EF1620_7D_FinA",  "RUN1_EF1620_7F_AinF", "RUN1_EF1620_7F_ANIFLZ", "RUN2_F15_8G_FLZ", "RUN2_F15_8G_FinA", "RUN1_CBS138_9H_FLZ", "RUN1_CBS138_9H_FinA",  "RUN1_P35_10G_ANIFLZ", "RUN1_BG2_11B_FLZ", "RUN1_BG2_11B_FinA", "RUN1_BG2_11H_FLZ", "RUN1_BG2_11H_FinA"}
chrE_partial_dup = {"RUN1_CBS138_9F_FLZ", "RUN1_CBS138_9F_FinA", "RUN1_CST34_2A_AinF"}
chrA_dup = {"RUN1_CST34_2G_ANIFLZ"}
chrI_dup = {"RUN1_EB0911_3H_FLZ", "RUN1_EB0911_3H_AinF"}
chrL_dup = {"RUN3_EF1620_7B_AinF"}

chr_to_samplesDuplicated = {"ChrA_C_glabrata_CBS138": chrA_dup, 
                            "ChrE_C_glabrata_CBS138": chrE_total_dup.union(chrE_partial_dup), 
                            "ChrI_C_glabrata_CBS138": chrI_dup, 
                            "ChrL_C_glabrata_CBS138": chrL_dup, 
                            }

wrong_samples_Ksiezopolska2021 = {"RUN3_BG2_11B_YPD", "RUN3_BG2_SRA_WT", "RUN1_EF1620_7B_ANI", "RUN1_EF1620_7B_AinF", "SRAdownload_F15_SRA_WT", "RUN1_EF1620_7F_FLZ", "RUN1_EF1620_7F_FinA", "RUN1_CST34_2G_FLZ", "RUN1_CST34_2G_FinA", "RUN5_CST78_4A_YPD", "RUN5_F15_8A_YPD"}

condition_to_color = {"WT":"gray", "WTerg3ko":"gray", "ANI":"magenta", "FLZ":"c", "AinF":"darkred", "FinA":"navy", "ANIFLZ":"green", "YPD":"black", "WTfks2Reintroduced":"dimgray", "WTfks1Reintroduced":"dimgray", "ANIerg3Reintroduced":"red"}


condition_to_tradeoff_f = {'CFW': 'M_rAUC_norm', 'CR': 'M_rAUC_norm', 'CycA': 'M_rAUC_norm', 'DTT': 'M_rAUC_norm', 'H2O2': 'M_rAUC_norm', 'NaCl': 'M_rAUC_norm', 'SDS': 'M_rAUC_norm', 'YPD': 'M_AUC_norm_WT'}

stress_condition_to_color = {'CFW': 'tab:blue', 'CR': 'tab:red', 'CycA': 'tab:olive', 'DTT': 'tab:green', 'H2O2': 'tab:orange', 'NaCl': 'tab:pink', 'SDS': 'tab:brown', 'YPD': 'black'}



#############################



####### FUNCTIONS ##########

def save_df_as_tab(df, file):

    """Takes a df and saves it as tab"""

    file_tmp = "%s.tmp"%file
    df.to_csv(file_tmp, sep="\t", index=False, header=True)
    os.rename(file_tmp, file)

def save_object(obj, filename):
    
    """ This is for saving python objects """
    
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    
    """ This is for loading python objects """

    return pickle.load(open(filename,"rb"))

    """
    dfs to load
    gene_features_df.py
    df_pathway_annots.py
    df_data_all_predictive_features.py
    df_feature_info_predictive_features.py
    df_univariate_associations.py

    # for certain dfs, save as tab and reload
    if get_file(filename) in {}:

        filename_tab = "%s.tab"%filename
        if file_is_empty(filename_tab): save_df_as_tab(pickle.load(open(filename,"rb")), filename_tab)
        return pd.read_csv(filename_tab, sep="\t")

    else: 
        # print(get_file(filename))
        return pickle.load(open(filename,"rb"))


    """

def get_interesting_name_FeatsDF(row):
    
    """Takes a features dataframe row that has gene_name and Scerevisiae_orthologs and returns the interesting name"""

    if not pd.isna(row["gene_name"]): return row["gene_name"]
    elif not pd.isna(row["Scerevisiae_orthologs"]): return "Scer_%s"%(row["Scerevisiae_orthologs"])
    else: return row["gff_upmost_parent"]

def get_df_gene_annotations(ManualDataDir, ProcessedDataDir):

    """Gets per gene annotations, mostly based on the data of Schikora-Tamarit 2024, which used the same genomes and annotations"""

    # define files
    gene_features_df_file = "%s/gene_features_df.py"%ProcessedDataDir 
    df_pathway_annots_file = "%s/df_pathway_annots.py"%ProcessedDataDir 

    if file_is_empty(gene_features_df_file) or file_is_empty(df_pathway_annots_file):
    

        # load gene features df for c. glabrata
        gene_features_df = pd.read_csv("%s/SchikoraTamarit2024_all_species_gene_features_df.tab"%ManualDataDir, sep="\t")[['species', 'gff_upmost_parent', 'gene_name', 'aliases', 'feature_type', 'description', 'Scerevisiae_orthologs', 'start', 'end', 'gene_length', 'chromosome', 'orthofinder_orthocluster']]
        gene_features_df = gene_features_df[gene_features_df.species=="Candida_glabrata"]
        gene_features_df["final_name"] = gene_features_df.apply(get_interesting_name_FeatsDF, axis=1)

        # add pathway annotations
        df_pathway_annots = pd.DataFrame()
        for type_pathway in ["MetaCyc", "GO_BP", "GO_CC", "GO_MF", "Reactome", "IP_domains"]:
            df = pd.read_csv("%s/SchikoraTamarit2024_annotation_Candida_glabrata_%s.tab"%(ManualDataDir, type_pathway), sep="\t")

            # add to gene_features_df, with checks
            geneID_to_annots = dict(df.groupby("Gene").apply(lambda df_g: set(df_g.ID)))
            geneIDs_not_in_gene_features = set(geneID_to_annots).difference(set(gene_features_df.gff_upmost_parent))
            if len(geneIDs_not_in_gene_features)>0: raise ValueError("There are some genes not in geneIDs_not_in_gene_features: %s"%geneIDs_not_in_gene_features)

            geneIDs_no_annotation = set(gene_features_df.gff_upmost_parent).difference(set(geneID_to_annots))
            print("%s. There are %i genes with no annotations"%(type_pathway, len(geneIDs_no_annotation)))

            if len(geneIDs_no_annotation)>0: geneID_to_annots = {**geneID_to_annots, **{g : set() for g in geneIDs_no_annotation}}
            gene_features_df[type_pathway] = gene_features_df.gff_upmost_parent.apply(lambda g: geneID_to_annots[g])

            # keep df
            df["type_pathway"] = type_pathway
            df_pathway_annots = pd.concat([df_pathway_annots, df]).reset_index(drop=True)

        # add gene name
        geneID_to_final_name = dict(gene_features_df.set_index("gff_upmost_parent").final_name)
        df_pathway_annots["final_name"] = df_pathway_annots.Gene.apply(lambda g: geneID_to_final_name[g])
        if len(df_pathway_annots[["Gene", "final_name"]].drop_duplicates())!=len(set(df_pathway_annots.Gene)): raise ValueError("there is no one to one correspondance")

        save_object(gene_features_df, gene_features_df_file)
        save_object(df_pathway_annots, df_pathway_annots_file)

    # load
    gene_features_df = load_object(gene_features_df_file)
    df_pathway_annots = load_object(df_pathway_annots_file)


    return gene_features_df, df_pathway_annots


def analyze_overlaps_pathways_samples_new_data(df_new_vars, gene_features_df, df_pathway_annots):


    """Analyzes whether different samples acquired mutations in dfferent genes with the same condition"""

    df_new_vars = cp.deepcopy(df_new_vars)
    gene_features_df = cp.deepcopy(gene_features_df)
    df_pathway_annots = cp.deepcopy(df_pathway_annots)

    # filter
    df_new_vars = df_new_vars[~df_new_vars.final_name.isin({"ChrA", "ChrE", "ChrI", "ChrL"})]


    df_pathway_annots= df_pathway_annots[df_pathway_annots.final_name.isin(set(df_new_vars.final_name))][["final_name", "ID", "group_name", "type_pathway"]]

    # analyze genes that are mutated in >1 sample, and print them
    known_recurrent_genes = {'FKS2', 'PDR1', 'FKS1', 'ERG11', 'ERG3', 'CDR1', 'Scer_CNE1', 'ERG4', "EPA13"}
    gene_to_samples = df_new_vars[~df_new_vars.isin(known_recurrent_genes)].groupby("final_name").apply(lambda df_g: sorted(set(df_g.sampleID)))
    print("Genes mutated in >1", gene_to_samples[(gene_to_samples.apply(len)>1)])
    print(df_new_vars[df_new_vars.final_name.isin(set(gene_to_samples[gene_to_samples.apply(len)==1].index))])

    for similar_conditions in [{"ANI", "FinA", "ANIFLZ"}, {"AinF", "FLZ", "ANIFLZ"}]:
        
        # get the df with the genes that are in the same condition 
        #df_new_vars_cond  = df_new_vars[(df_new_vars.condition.isin(similar_conditions)) & ~(df_new_vars.final_name.isin({"ChrA", "ChrE", "ChrI", "ChrL"}))]
        df_new_vars_cond  = df_new_vars[(df_new_vars.condition.isin(similar_conditions)) & ~(df_new_vars.final_name.isin({"ChrA", "ChrE", "ChrI", "ChrL"})) & ~(df_new_vars.final_name.isin(known_recurrent_genes))]

        # get genes
        interesting_genes = set(df_new_vars_cond.final_name)
        df_genes = gene_features_df[gene_features_df.final_name.isin(interesting_genes)]
        if set(df_genes.final_name)!=interesting_genes: raise ValueError("not all genes in gene feats")

        # for each pathway, print overlaps
        for type_pathway in ["GO_BP"]: #["MetaCyc", "GO_BP", "GO_CC", "GO_MF", "Reactome", "IP_domains"]: # IP_domains
             

            # all_pathways = set.union(*df_genes[type_pathway])
            pathway_to_genes = df_pathway_annots[(df_pathway_annots.type_pathway==type_pathway) & (df_pathway_annots.final_name.isin(interesting_genes))].groupby("ID").apply(lambda df_g: set(df_g.final_name))
            pathway_to_genes = pathway_to_genes[pathway_to_genes.apply(len)>1]


            if len(pathway_to_genes)>0:
                
                for p, genes in dict(pathway_to_genes).items():

                    #if "calc"  in str(df_pathway_annots[df_pathway_annots.ID==p].group_name.iloc[0]).lower(): print(similar_conditions, type_pathway, df_pathway_annots[df_pathway_annots.ID==p].group_name.iloc[0], genes)
                    print(similar_conditions, type_pathway, df_pathway_annots[df_pathway_annots.ID==p].group_name.iloc[0], genes)

def get_df_cyclosporinA_formatted_for_df_stress_phenotypes(ManualDataDir, sampleNames_with_wgs):

    """Gets cyclosporinA df formated to addd to df_stress_phenotypes"""

    # keep
    sampleNames_with_wgs = cp.deepcopy(sampleNames_with_wgs)

    print("loading cyclosporin data...")

    # load df
    df_cyclosporinA_all = pd.read_excel("%s/CsAScreenig_mailJC_22March2024.xlsx"%ManualDataDir)


    # discard reintroduced mutants (they have no effect)
    df_cyclosporinA_all = df_cyclosporinA_all[~(df_cyclosporinA_all.strain.isin({"3H_ANI+ERG3:(D122Y)NAT", "EB0911/3B:(D122Y)", "ERG3_KO/EB0911"}))]

    # add fields of the strain
    treatment_dict = {"FLZANI":"ANIFLZ", "FLZ":"FLZ", "ANI":"ANI", "FinA":"FinA", "AinF":"AinF"}
    def get_r_with_added_fields_strain(r):
        strain_split = r.strain.split(":")
        r["Origin"] = strain_split[0]
        r["real_strain"] = strain_split[0].split("_")[0]
        if r["Origin"]=="P35": r["Origin"] = "P35_2"

        if len(strain_split)==1: 
            r["Treatment"] = "WT"
            r["exp_evol_rep"] = "SRA"

        elif len(strain_split)==4:
            r["Treatment"] = treatment_dict[strain_split[2]]
            r["exp_evol_rep"] = strain_split[3]

        else: raise ValueError("invalid %s, %s"%(strain_split, r.strain))

        r["sampleID"] = f'{r.real_strain}_{r.exp_evol_rep}_{r.Treatment}'

        return r

    df_cyclosporinA_all = df_cyclosporinA_all.apply(get_r_with_added_fields_strain, axis=1)

    # modify
    df_cyclosporinA_all["YPD_OD"] = df_cyclosporinA_all["YA"].apply(float)
    df_cyclosporinA_all["cycA_OD"] = df_cyclosporinA_all["Ca"].apply(float)
    df_cyclosporinA_all["fauc"] = df_cyclosporinA_all["fauc"].apply(float)

    # checks of df
    if any( (df_cyclosporinA_all["fauc"].apply(lambda x: round(x, 2)))!=((df_cyclosporinA_all.cycA_OD / df_cyclosporinA_all.YPD_OD).apply(lambda x: round(x, 2))) ):
        raise ValueError("the ratio is not as ecpected")

    for f in ["YPD_OD", "cycA_OD"]:
        if any(df_cyclosporinA_all[f]<0): raise ValueError("non negatives allowed")
        if any(df_cyclosporinA_all[f]>2): raise ValueError("non >2 allowed")

    list_expected = [("Origin", {'BG2', 'CBS138', 'CST34', 'CST78', 'EB0911', 'EF1620', 'F15', 'M12', 'P35_2'}),
                     ("Treatment", {'ANI', 'ANIFLZ', 'AinF', 'FLZ', 'FinA', 'WT'})] # YPD-evolved strains not measured

    for f, expected_fields in list_expected:
        if set(df_cyclosporinA_all[f])!=expected_fields: raise ValueError("not as expected %s"%sorted(set(df_cyclosporinA_all[f])))

    sampleNames_with_wgs = {s for s in sampleNames_with_wgs if not s.endswith("_WT") and not s.endswith("_YPD")}
    samples_phenotypes = {s for s in set(df_cyclosporinA_all["sampleID"]) if not s.endswith("_WT")}
    if samples_phenotypes!=sampleNames_with_wgs: raise ValueError("samples should be the same")

    # get the df, also with YPD measurements
    fields_all = ["Origin", "Treatment", "sampleID", "Well"]
    df_cyclosporinA_all_cycA = df_cyclosporinA_all[fields_all + ["cycA_OD"]].rename(columns={"cycA_OD":"AUC"}).copy() 
    df_cyclosporinA_all_cycA["condition"] = "CycA"

    df_cyclosporinA_all_YPD = df_cyclosporinA_all[fields_all + ["YPD_OD"]].rename(columns={"YPD_OD":"AUC"}).copy() 
    df_cyclosporinA_all_YPD["condition"] = "YPD"
    df_cyclosporinA_all = pd.concat([df_cyclosporinA_all_cycA, df_cyclosporinA_all_YPD]).reset_index(drop=True).sort_values(by=["sampleID", "condition", "Well"]).set_index("sampleID", drop=False)

    # add the normalized values per sample
    def get_df_cycA_r_w_inhouse_vals(r):

        # get datasets for the sample and the WT
        data_sample = df_cyclosporinA_all.loc[r.sampleID]
        data_wt = df_cyclosporinA_all.loc["%s_SRA_WT"%r.sampleID.split("_")[0]]
        if len(data_sample)!=len(data_wt): raise ValueError("should be same len")

        # add values relative to YPD
        for df in [data_sample, data_wt]:
            spot_to_ypd_AUC = dict(df[df.condition=="YPD"].set_index("Well").AUC)
            df["AUC_diff_AUCypd"] = df.AUC  - df.Well.map(lambda x: spot_to_ypd_AUC[x])
            df["AUC_norm_AUCypd"] = df.AUC  / df.Well.map(lambda x: spot_to_ypd_AUC[x])

        # keep only the df of condition, which already has the ypd-relative info
        data_wt = data_wt[data_wt.condition==r.condition]
        data_sample = data_sample[data_sample.condition==r.condition]

        # add the rAUCs for each 
        data_sample["rAUC_norm"] = data_sample["AUC_norm_AUCypd"] / np.mean(data_wt["AUC_norm_AUCypd"]) 
        data_sample["rAUC_diff"] = data_sample["AUC_diff_AUCypd"] - np.mean(data_wt["AUC_diff_AUCypd"]) 

        # checks
        if len(data_wt)!=4: raise ValueError("data_wt should be 4")
        if len(data_sample)!=4: raise ValueError("data_sample should be 4")

        # add estimates to r
        r["M_AUC"] = np.median(data_sample["AUC"])
        r["M_fAUC_norm"] = np.median(data_sample["AUC_norm_AUCypd"]) # fitness normalized by the YPD
        r["M_fAUC_diff"] = np.median(data_sample["AUC_diff_AUCypd"]) # fitness difference to YPD

        r["M_rAUC_norm"] = np.median(data_sample["rAUC_norm"])
        r["M_rAUC_diff"] = np.median(data_sample["rAUC_diff"])

        r["M_AUC_norm_WT"] = np.median(data_sample["AUC"] / np.mean(data_wt["AUC"]))
        r["M_AUC_diff_WT"] = np.median(data_sample["AUC"] - np.mean(data_wt["AUC"]))


        # lists
        r["rAUC_norm_list"] = data_sample["rAUC_norm"].values
        r["AUC_norm_WT_list"] = (data_sample["AUC"] / np.mean(data_wt["AUC"])).values


        return r

    df_cyclosporinA = df_cyclosporinA_all[["sampleID", "condition", "Origin", "Treatment"]].drop_duplicates()
    df_cyclosporinA = df_cyclosporinA.apply(get_df_cycA_r_w_inhouse_vals, axis=1)

    # return 
    expected_fields = ["Origin", "condition", "Treatment", "M_AUC", "sampleID", "M_fAUC_norm", "M_fAUC_diff", "M_rAUC_norm", "M_rAUC_diff", "M_AUC_norm_WT", "M_AUC_diff_WT", "rAUC_norm_list", "AUC_norm_WT_list"]
    return df_cyclosporinA[expected_fields]

def get_df_stress_phenotypes(ManualDataDir, df_new_vars, sampleNames_with_wgs, ProcessedDataDir):

    """Gets the stress phenotypes measured by JC"""

    df_stress_phenotypes_file = "%s/df_stress_phenotypes_all.py"%ProcessedDataDir

    if file_is_empty(df_stress_phenotypes_file):

        # load df provided by JC
        df_stress_phenotypes_all = pd.read_excel("%s/V9_EGLA_Phenotypes_SentJC_09012024.xlsx"%ManualDataDir)

        # add sampleID to match the one in df_new_vars
        def get_sampleID_for_one_r(r):
            if r.Treatment=="WT": replicate_expevol = "SRA"
            else: replicate_expevol = r.Name.split("-")[0]
            return "%s_%s_%s"%(r.Origin.replace("P35_2", "P35"), replicate_expevol, r.Name.split("-")[1])

        df_stress_phenotypes_all["sampleID"] = df_stress_phenotypes_all.apply(get_sampleID_for_one_r, axis=1)

        # keep reduced dataset, one line per sampleID and condition
        df_stress_phenotypes_all["spotID"] =  df_stress_phenotypes_all.row.apply(str) + "_" + df_stress_phenotypes_all.column.apply(str) 
        df_stress_phenotypes = df_stress_phenotypes_all.drop_duplicates(subset=["sampleID", "condition"], keep="first")

        # add in-house relative values
        df_stress_phenotypes_all = df_stress_phenotypes_all.set_index(["sampleID"], drop=False)[["Origin", "Treatment", "condition", "spotID", "AUC", "Plate", "fAUC", "rAUC", "Gs"]]
        for f in ["AUC", "rAUC", "fAUC"]: df_stress_phenotypes_all[f] = df_stress_phenotypes_all[f].apply(float)


        # change data types
        plateID_to_YPD_plateID = {1:1, 2:1, 3:1, 4:1, 5:5, 6:5, 7:5, 8:5}    
        def get_df_stress_phenotypes_r_w_inhouse_vals(r):

            # get datasets for the sample and the WT
            data_sample = df_stress_phenotypes_all.loc[r.sampleID]
            data_wt = df_stress_phenotypes_all.loc["%s_SRA_WT"%r.sampleID.split("_")[0]]

            # filter
            if r.condition=="YPD": interesting_plates = {plateID_to_YPD_plateID[r.Plate]}
            else: interesting_plates = {plateID_to_YPD_plateID[r.Plate], r.Plate}
            data_sample = data_sample[data_sample.Plate.isin(interesting_plates)].set_index("spotID", drop=False)
            data_wt = data_wt[data_wt.Plate.isin(interesting_plates)].set_index("spotID", drop=False)
            if len(data_sample)!=len(data_wt): raise ValueError("should be same len")

            # add values relative to YPD
            for df in [data_sample, data_wt]:
                spot_to_ypd_AUC = dict(df[df.condition=="YPD"].AUC)
                df["AUC_diff_AUCypd"] = df.AUC  - df.spotID.map(lambda x: spot_to_ypd_AUC[x])
                df["AUC_norm_AUCypd"] = df.AUC  / df.spotID.map(lambda x: spot_to_ypd_AUC[x])

            # keep only the df of condition, which already has the ypd-relative info
            data_wt = data_wt[data_wt.condition==r.condition]
            data_sample = data_sample[data_sample.condition==r.condition]

            # add the rAUCs for each 
            data_sample["rAUC_norm"] = data_sample["AUC_norm_AUCypd"] / np.mean(data_wt["AUC_norm_AUCypd"]) 
            data_sample["rAUC_diff"] = data_sample["AUC_diff_AUCypd"] - np.mean(data_wt["AUC_diff_AUCypd"]) 

            # checks
            if any((data_sample.fAUC-data_sample.AUC_norm_AUCypd).apply(abs)>=0.01): raise ValueError("fAUC and AUC_norm_AUCypd should always be the same. %s"%(data_sample[["fAUC", "AUC_norm_AUCypd"]]))
            if any((data_wt.fAUC-data_wt.AUC_norm_AUCypd).apply(abs)>=0.01): raise ValueError("fAUC and AUC_norm_AUCypd  in wt")

            if any((data_sample.rAUC-data_sample.rAUC_norm).apply(abs)>=0.01): 
                values_mean_wt = data_sample["fAUC"] / np.mean(data_wt["fAUC"])
                if any((values_mean_wt-data_sample.rAUC_norm).apply(abs)>=0.01): raise ValueError("different numbers")
                #print("WARNING: for %s-%s.rAUC and rAUC_norm  are not exactly the same. The values differ by %s. These are the WT fAUCs: %s"%(r.sampleID, r.condition, (data_sample.rAUC-data_sample.rAUC_norm).apply(abs).apply(lambda x: round(x, 2)).values, data_wt.fAUC.values))

            if len(data_wt)!=4: raise ValueError("data_wt should be 4")
            if len(data_sample)!=4: raise ValueError("data_sample should be 4")

            # add estimates to r
            r["M_fAUC_norm"] = np.median(data_sample["AUC_norm_AUCypd"]) # fitness normalized by the YPD
            r["M_fAUC_diff"] = np.median(data_sample["AUC_diff_AUCypd"]) # fitness difference to YPD

            r["M_rAUC_norm"] = np.median(data_sample["rAUC_norm"])
            r["M_rAUC_diff"] = np.median(data_sample["rAUC_diff"])

            r["M_AUC_norm_WT"] = np.median(data_sample["AUC"] / np.mean(data_wt["AUC"]))
            r["M_AUC_diff_WT"] = np.median(data_sample["AUC"] - np.mean(data_wt["AUC"]))

            # add lists of values
            r["rAUC_norm_list"] = data_sample["rAUC_norm"].values
            r["AUC_norm_WT_list"] = (data_sample["AUC"] / np.mean(data_wt["AUC"])).values

            return r

        print("adding per r info...")
        df_stress_phenotypes = df_stress_phenotypes.apply(get_df_stress_phenotypes_r_w_inhouse_vals, axis=1)

        # keep some fields
        df_stress_phenotypes = df_stress_phenotypes[["Origin", "condition", "Treatment", "M_AUC", "sampleID", "M_fAUC_norm", "M_fAUC_diff", "M_rAUC_norm", "M_rAUC_diff", "M_AUC_norm_WT", "M_AUC_diff_WT", "rAUC_norm_list", "AUC_norm_WT_list"]].copy()
        df_stress_phenotypes["type_measurement"] = "Q-PHAST_round1" # define the run ID

        # add the cyclosporin A data, formatted as in df_stress_phenotypes (although, note that this is OD readings)
        df_cyclosporinA = get_df_cyclosporinA_formatted_for_df_stress_phenotypes(ManualDataDir, sampleNames_with_wgs)
        df_cyclosporinA["type_measurement"] = "OD_CycA_round1"
        df_stress_phenotypes = pd.concat([df_stress_phenotypes, df_cyclosporinA]).reset_index(drop=True)
        for k in df_stress_phenotypes.keys(): check_no_nans_series(df_stress_phenotypes[k])

        # add fields for the naming
        df_stress_phenotypes["M_fAUC"] = df_stress_phenotypes.M_fAUC_norm.copy()

        # add the difference between the fAUC in stress and the fAUC in the wt
        print("adding extra fields")
        df_stress_phenotypes_wt = df_stress_phenotypes[df_stress_phenotypes.Treatment=="WT"].set_index(["Origin", "condition", "type_measurement"], drop=False)
        if len(df_stress_phenotypes_wt)!=len(set(df_stress_phenotypes_wt.index)): raise ValueError("index should be unique")

        for f in ["M_fAUC", "M_AUC"]:
            df_stress_phenotypes["WT_%s"%f] = df_stress_phenotypes.apply(lambda r: df_stress_phenotypes_wt.loc[(r.Origin, r.condition, r.type_measurement)][f], axis=1)
            df_stress_phenotypes["%s_diff_WT_%s"%(f, f)] = df_stress_phenotypes[f] - df_stress_phenotypes["WT_%s"%f]
            df_stress_phenotypes["%s_norm_WT_%s"%(f, f)] = df_stress_phenotypes[f] / df_stress_phenotypes["WT_%s"%f]

        # check that we only have WGS-sequenced samples
        samples_with_wgs_no_wt = {s for s in sampleNames_with_wgs if not s.endswith("_WT")}
        samples_phenotypes_no_wt = {s for s in set(df_stress_phenotypes["sampleID"]) if not s.endswith("_WT")}

        strange_samples_wgs = samples_with_wgs_no_wt.difference(samples_phenotypes_no_wt)
        strange_samples_phenos = samples_phenotypes_no_wt.difference(samples_with_wgs_no_wt)
        print(len(samples_with_wgs_no_wt), "samples WGS")

        if samples_with_wgs_no_wt!=samples_phenotypes_no_wt:
            print("strange_samples_wgs", strange_samples_wgs)
            print("strange_samples_phenos", strange_samples_phenos)
            raise ValueError("samples are not the same")

        # checks of numbers
        if any(pd.isna(df_stress_phenotypes.M_rAUC_norm)): raise ValueError("M_rAUC_norm cannot be NAN")
        if any((df_stress_phenotypes.condition=="YPD") & (df_stress_phenotypes.M_rAUC_norm!=1.0)): raise ValueError("M_rAUC_norm for YPD should always be 1")

        save_object(df_stress_phenotypes, df_stress_phenotypes_file)
        save_object(df_stress_phenotypes_all, df_stress_phenotypes_file+".all.py")

    return load_object(df_stress_phenotypes_file), load_object(df_stress_phenotypes_file+".all.py")



def get_df_data_r_with_broad_resistance_genes_profile(r, profile, df_all_vars, all_genes, effect_to_order):

    """gets the profile valie for resistance genes."""

    # map each gene to the mutations
    gene_to_mutations = dict(df_all_vars.loc[[r.sampleID]].groupby("final_name").apply(lambda df_g: set(df_g.variant_effect)))
    if len(gene_to_mutations)==0: raise ValueError("can't be empty")

    # add empty genes
    for gene in all_genes.difference(set(gene_to_mutations)): gene_to_mutations[gene] = set()

    # map each gene to the sorted mutations (worts first)
    gene_to_sortedMutations = {gene : sorted(mutations, key=(lambda x: effect_to_order[x])) for gene, mutations in gene_to_mutations.items()}

    # profiles where the output is none, miss, truncation
    if profile in {"ERG3_profile", "CNE1_profile", "ERG4_profile", "PDR1_profile", "CDR1_profile", "EPA13_profile", "ChrA_profile", "ChrL_profile", "ChrI_profile"}:

        if profile=="CNE1_profile": gene_name = "Scer_CNE1"
        else: gene_name = profile.split("_")[0]

        mutations = gene_to_sortedMutations[gene_name]

        if len(mutations)>1: print("WARNING: In %s, %s has these mutations: %s"%(r.sampleID, gene_name, mutations))
        if len(mutations)==0: profile_value = "none"
        else: profile_value = mutations[0]

    # add the combination of FKS mutations
    elif profile=="FKS_profile":

        # define the mutations of each gene
        fks1_muts = gene_to_sortedMutations["FKS1"]
        fks2_muts = gene_to_sortedMutations["FKS2"]

        # define profiles
        if len(fks1_muts)==0 and len(fks2_muts)==0: profile_value = "none"
        elif len(fks1_muts)==0 and len(fks2_muts)>0: profile_value = "FKS2 %s"%(fks2_muts[0])
        elif len(fks1_muts)>0 and len(fks2_muts)==0: profile_value = "FKS1 %s"%(fks1_muts[0])
        elif len(fks1_muts)>0 and len(fks2_muts)>0: profile_value = "FKS1 %s & FKS2 %s"%(fks1_muts[0], fks2_muts[0])
        else: raise ValueError("not valid FKS")

        # validate that everything is correct
        if r.condition=="FLZ" and profile_value!="none": raise ValueError("There are FLZ samples with FKS mutations")
        if r.condition in {"ANI", "AinF", "FinA", "ANIFLZ"} and profile_value=="none": raise ValueError("There are non-FLZ samples without FKS mutations")

    # add the combinations of ERG11 and aneuploidies
    elif profile=="ERG11_chrE_profile":

        erg11_muts = gene_to_sortedMutations["ERG11"]
        chromE_muts = gene_to_sortedMutations["ChrE"]

        if len(erg11_muts)==0 and len(chromE_muts)==0: profile_value = "none"
        elif len(erg11_muts)>0 and len(chromE_muts)==0: profile_value = "ERG11 %s"%(erg11_muts[0])
        elif len(erg11_muts)==0 and len(chromE_muts)>0: profile_value = "ChrE %s"%(chromE_muts[0])
        elif len(erg11_muts)>0 and len(chromE_muts)>0: profile_value = "ERG11 %s & ChrE %s"%(erg11_muts[0], chromE_muts[0])

    else: raise ValueError("%s has not been considered"%profile) 

    #############################
    
    return profile_value


def check_no_nans_series(x):

    """Raise value error if nans"""

    if any(pd.isna(x)): raise ValueError("There can't be nans in series %s"%x)



def get_df_data_and_features_with_genetic_information_features(df_data, features_data, df_all_vars, gene_features_df, df_pathway_annots):

    """Gets a df and features information list with all the genetic information features"""

    ##### GENERAL PROCESSING #######
    print("adding genetic information...")

    # general vars
    resistance_genes = {"ERG3", "ERG4", "Scer_CNE1", "FKS1", "FKS2", "ERG11", "PDR1", "CDR1", "EPA13", "ChrA", "ChrE", "ChrI", "ChrL"}
    typeMut_to_effect = {"FS":"truncation", "PTC":"truncation", "5' DEL":"truncation", "mis":"miss", "lostATG":"truncation", "lostSTOP":"miss", "DEL":"truncation", "del":"miss", "DUP":"dup", "partial DUP":"dup", "TRA":"truncation", "ins":"miss"}
    effect_to_order = {"miss":2, "dup":1, "truncation":0}

    # add the effects of variants
    if any(pd.isna(df_all_vars.variant)): raise ValueError("nans in df_all_vars")
    df_all_vars["variant_effect"] = df_all_vars.variant.apply(lambda v: typeMut_to_effect[v.split("|")[0]])

    # get some dfs
    df_all_vars_resistance_genes = df_all_vars[df_all_vars.final_name.isin(resistance_genes)].set_index("sampleID", drop=False)
    if set(df_all_vars_resistance_genes.sampleID)!=set(df_all_vars.sampleID): raise ValueError("not all genes the same")

    ################################

    #### ADD VARIOUS TYPES OF GENETIC INFORMATION ###

    # resistance genes broad profiles, based on the main genes found in Ksiezopolska2021
    for profile in ["ChrA_profile", "ChrI_profile", "ChrL_profile", "CNE1_profile", "ERG11_chrE_profile", "FKS_profile", "ERG3_profile", "ERG4_profile", "PDR1_profile", "CDR1_profile", "EPA13_profile"]: 

        profile_f = "resistance_genes_broad_%s"%profile
        df_data[profile_f] = df_data.apply(get_df_data_r_with_broad_resistance_genes_profile, profile=profile, df_all_vars=df_all_vars_resistance_genes, all_genes=resistance_genes, effect_to_order=effect_to_order, axis=1)
        # for cond in ["FLZ", "FinA", "ANI", "AinF", "ANIFLZ"]: print(profile, cond, Counter(df_data[df_data.condition==cond][profile_f]))
        features_data.append((profile_f, "broad_resistance_genes_profile", "cathegoric", "Broad profile of resistance genes (%s)"%(profile)))

    # add the presence / absence of specific mutations
    print("adding unique mutations...")
    var_to_samples = dict(df_all_vars.groupby("unique_varID").apply(lambda df_v: set(df_v.sampleID)))
    df_vars_presence = pd.DataFrame({sID : {"presence_variant-%s"%v : sID in samples for v, samples in var_to_samples.items()} for sID in sorted(set(df_data.sampleID))}).transpose().map(lambda x: {True:1, False:0}[x])

    df_data = df_data.merge(df_vars_presence, how="left", validate="one_to_one", left_index=True, right_index=True)
    for v in df_vars_presence.keys(): features_data.append((v, "single_variants", "binary", "Presence / absence of variant %s"%("-".join(v.split("-")[1:]))))


    # add the distance to the hotspot in FKS mutations
    cglab_gene_to_hotspotPositions = {"FKS1": {625, 626, 627, 628, 629, 630, 631, 632, 633},
                                      "FKS2": {659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 1374, 1375, 1376, 1377, 1378, 1379, 1380, 1381, 1382, 1383},
                                      "ERG11": {141, 152}}

    fksProfile_to_HSgenes = {'FKS1 truncation & FKS2 miss': {'FKS2'}, 'FKS1 miss': {'FKS1'}, 'FKS1 miss & FKS2 truncation': {'FKS1'}, 'FKS1 miss & FKS2 miss': {'FKS1', 'FKS2'}, 'FKS2 miss': {'FKS2'}}



    #################################################

    ###### ADD DOMAIN INFO #######

    # add protein domain information (any possible combinations)
    print("adding domains...")
    df_all_vars_miss = df_all_vars[df_all_vars.variant_effect=="miss"]
    df_all_vars_miss["miss_position"] = df_all_vars_miss.variant.apply(lambda v: int(v.split("|")[1].split(".")[1].split("-")[0]))
    gene_to_positions = df_all_vars_miss.groupby("final_name").apply(lambda df_g: np.array(sorted(set(df_g.miss_position))))
    gene_to_positions = gene_to_positions[gene_to_positions.apply(len)>2]
    gene_to_positions_set = gene_to_positions.apply(set)
    sample_to_lineage = dict(df_data.set_index("sampleID").exp_evol_lineage)

    domain_fields_already_added = set()
    domain_f_to_samples_mut = {}
    for gene, all_positions in gene_to_positions.items():

        df_all_vars_miss_gene = df_all_vars_miss[df_all_vars_miss.final_name==gene]
        var_to_sampleIDs = df_all_vars_miss_gene.groupby("variant").apply(lambda df_v: set(df_v.sampleID))

        combinations_pos = [(p, p) for p in all_positions] + list(itertools.combinations(all_positions, 2)) # all positions and also unique positions
        for p1, p2 in combinations_pos:
            if p1>p2: raise ValueError("p1 should be always before p2")

            # get the samples that have variants in this domain
            positions_domain = set(all_positions[(all_positions>=p1) & (all_positions<=p2)])
            if len(positions_domain)<2 and p1!=p2: raise ValueError("cant be <2")
            if (positions_domain==gene_to_positions_set[gene]) and (p1!=p2): continue
            variants_in_domain = list(set(df_all_vars_miss_gene[df_all_vars_miss_gene.miss_position.isin(positions_domain)].variant))
            samples_domain_mutated = set.union(*var_to_sampleIDs.loc[variants_in_domain])
            if len(samples_domain_mutated)<2 and (p1!=p2): raise ValueError("there can't be <2 samples")

            # only consider if there are sufficient lineages affected
            lineages_domain_mutated = len(set(map(lambda s: sample_to_lineage[s], samples_domain_mutated)))
            if lineages_domain_mutated<2: continue

            # add to df
            domain_field = "%s_miss_%i-%i"%(gene, p1, p2)

            # check that it is not correlated
            if not any(map(lambda f: samples_domain_mutated==domain_f_to_samples_mut[f], domain_fields_already_added)):

                df_data = df_data.assign(**{domain_field: df_data.sampleID.isin(samples_domain_mutated).map({True:1, False:0})})
                check_no_nans_series(df_data[domain_field])
                features_data.append((domain_field, "domain_info", "binary", "Presence/absence of missense mutations in region %i-%i of gene %s"%(p1, p2, gene)))
                domain_fields_already_added.add(domain_field)
                domain_f_to_samples_mut[domain_field] = samples_domain_mutated

    ##############################

    return df_data, features_data, df_all_vars


def get_df_data_and_features_data_with_NR_pathway_info(df_data, features_data, df_all_vars, df_pathway_annots, gene_features_df):

    """Adds pathway info that is not redundant with previous info"""

    # I keep the presence / absence of variants in certain pathways, only for situations in which the pathway is not 100% correlated to some other 
    print("adding pathways...")

    # init misc information
    sample_to_lineage = dict(df_data.set_index("sampleID").exp_evol_lineage)
    initial_len_features_data = len(features_data)

    # init the binary features already considered, to not be redundant
    all_binary_features = {x[0] for x in features_data if x[2]=="binary"}
    f_to_samples = {f : set(df_data[df_data[f]==1].sampleID) for f in all_binary_features}

    # get some dfs
    df_pathways = df_pathway_annots[df_pathway_annots.final_name.isin(set(df_all_vars.final_name)) & (df_pathway_annots.type_pathway.isin({'GO_BP', 'GO_CC', 'GO_MF', 'MetaCyc', 'Reactome'}))]

    def get_gname_pathway(r):
        if r.ID=="All-Trans-Farnesyl-PP-Biosynthesis" and r.type_pathway=="MetaCyc": return r.ID
        else: return r.group_name

    df_pathways["group_name"] = df_pathways.apply(get_gname_pathway, axis=1)
    check_no_nans_series(df_pathways.group_name)

    df_pathways["gname_len"] = df_pathways.group_name.apply(len)
    df_all_vars = df_all_vars[df_all_vars.final_name.isin(set(df_pathways.final_name))]
    df_all_vars["exp_evol_lineage"] = df_all_vars.sampleID.apply(lambda s: sample_to_lineage[s])

    for type_pathway in ['GO_BP', 'GO_MF', 'GO_CC', 'MetaCyc', 'Reactome']: # sorted in reverse order
        for variant_effect in ['truncation', 'dup', 'miss']: # sorted in reverse order

            df_vars = df_all_vars[df_all_vars.variant_effect==variant_effect]
            df_p = df_pathways[(df_pathways.type_pathway==type_pathway) & (df_pathways.final_name.isin(set(df_vars.final_name)))]
            pID_to_name = dict(df_p[["ID", "group_name"]].drop_duplicates().set_index("ID").group_name)
            pID_to_genes = df_p[["ID", "final_name"]].drop_duplicates().groupby("ID").apply(lambda df_i: set(df_i.final_name))
            pathways_multiple_genes = set(pID_to_genes[pID_to_genes.apply(len)>=2].index)
            df_p = df_p[df_p.ID.isin(pathways_multiple_genes)]
            gene_to_samples_affected = dict(df_vars.groupby("final_name").apply(lambda df_g: set(df_g.sampleID)))

            # add iteratively pathways, so that longer names go first. Flag any overlaps
            sorted_pathwaysIDs = list(df_p[["ID", "group_name", "gname_len"]].drop_duplicates().sort_values(by="gname_len", ascending=False).ID)
            for pID in sorted_pathwaysIDs:

                # calculate the numer of samples that have this pathwat affected by variant type
                samples_pathway_affected = set.union(*[gene_to_samples_affected[g] for g in pID_to_genes[pID]])

                # filters
                if len(samples_pathway_affected)<2: continue
                lineages_pathway_affected = set(map(lambda s: sample_to_lineage[s], samples_pathway_affected))
                if len(lineages_pathway_affected)<2: continue
                if samples_pathway_affected==set(df_data.sampleID): continue
                if lineages_pathway_affected==set(df_data.exp_evol_lineage): continue

                # check if it is redundant with other fields, and add
                pathway_field = "%s_%s_%s"%(type_pathway, pID, variant_effect)

                redundant_other_fields = False
                for other_f, other_samples in f_to_samples.items():
                    if samples_pathway_affected==other_samples:
                        redundant_other_fields = True
                        break

                        """
                        # warn about some cases
                        if other_f.startswith(type_pathway):
                            other_f_ID = other_f.split("%s_"%type_pathway)[1].split("_%s"%variant_effect)[0]
                            print("%s (%s) is redundant with %s (%s)"%(pathway_field, pID_to_name[pID], other_f, pID_to_name[other_f_ID]))
                        """

                if redundant_other_fields is False:
                    df_data = df_data.assign(**{pathway_field: df_data.sampleID.isin(samples_pathway_affected).map({True:1, False:0})})
                    features_data.append((pathway_field, "pathway_info", "binary", "%s in pathway '%s' (%s, %s). Genes: %s"%(variant_effect, pID_to_name[pID], type_pathway, pID, ", ".join(sorted( pID_to_genes[pID])))))
                    f_to_samples[pathway_field] = samples_pathway_affected

    print("Added %i pathways"%(len(features_data)-initial_len_features_data))

    ###################################
    return df_data, features_data


def get_df_data_and_features_with_PCA_wt_information_features(df_data, features_data, PlotsDir, continuous_WT_features):

    """Adds to df_data the necessary PCs when considering all the WT measurements (fitness in different conditions)"""
    print("Adding WT fitness PC info...")

    # define df for PCA, with interesting predictors
    df_data_WTs = df_data.copy()[continuous_WT_features + ["strain"]].drop_duplicates(subset=["strain"], keep="first").set_index("strain").sort_index()[continuous_WT_features]

    # Standardize the data
    scaler = StandardScaler()
    df_data_standardized = scaler.fit_transform(df_data_WTs)

    # Perform PCA
    pca = PCA()
    pca_result = pca.fit_transform(df_data_standardized)

    # Create a DataFrame with PCA results
    pca_df = pd.DataFrame(pca_result, index=df_data_WTs.index, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])

    # Plot explained variance for each principal component
    fig = plt.figure(figsize=(5, 5))
    plt.plot(range(1, pca.n_components_ + 1), np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='-', color='b')
    plt.title('Explained Variance by Principal Components')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.ylim([0, 1.05])
    plt.grid(True)
    plt.show()
    fig.savefig("%s/PCA_WT_continuous_vars_cum_variance.pdf"%PlotsDir, bbox_inches='tight')

    # Plot PCA results for strains
    fig = plt.figure(figsize=(5, 5))
    sns.scatterplot(x='PC1', y='PC2', data=pca_df, hue=pca_df.index, style=pca_df.index, palette='tab10', legend='full', s=40, edgecolor="black")
    plt.title('PCA Results for Strains')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% Variance Explained)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% Variance Explained)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)    
    plt.show()
    fig.savefig("%s/PCA_WT_continuous_strains_position.pdf"%PlotsDir, bbox_inches='tight')


    # Plot PCA results for variables (columns)
    loadings_df = pd.DataFrame(pca.components_, index=["PC%i"%comp for comp in range(1, len(pca.components_)+1)], columns=continuous_WT_features).transpose()
    fig = plt.figure(figsize=(5, 5))
    sns.scatterplot(x='PC1', y='PC2', data=loadings_df, hue=loadings_df.index, style=loadings_df.index, palette='tab20', legend='full', s=40, edgecolor="black")
    plt.title('PCA loadings for Variables')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% Variance Explained)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% Variance Explained)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)    
    plt.show()
    fig.savefig("%s/PCA_WT_continuous_variables_contribution.pdf"%PlotsDir, bbox_inches='tight')

    # add the firts 6 principal components to the data (these explain ~99% of variability)
    for pc in range(1, 6+1):

        pc_field = "PC%i_WT_continuous_variables"%pc
        strain_to_wt_val = dict(pca_df["PC%i"%pc])
        df_data[pc_field] = df_data.strain.map(strain_to_wt_val)

        features_data.append((pc_field, "WT_info_PC_continuous_data", "continuous", "PC%i of the WT considering continuous fitness values"%pc))

    return df_data, features_data

def file_is_empty(path): 
    
    """ask if a file is empty or does not exist """

    if not os.path.isfile(path):
        return_val = True
    elif os.stat(path).st_size==0:
        return_val = True
    else:
        return_val = False
            
    return return_val


def get_df_all_predictive_features(df_new_vars, df_stress_phenotypes, df_phenotypes_Ksiezopolska2021, sampleNames_with_wgs, ProcessedDataDir, PlotsDir, gene_features_df, df_pathway_annots):

    """Gets a df where each row is one strain and one condition and it has all the possibly predictive features"""

    ##### FILTER INITIAL DATA ######

    df_data_file = "%s/df_data_all_predictive_features.py"%ProcessedDataDir
    df_feature_info_file = "%s/df_feature_info_predictive_features.py"%ProcessedDataDir
    df_data_all_file = "%s/df_data_all_predictive_features_all.py"%ProcessedDataDir
    df_feature_info_all_file = "%s/df_feature_info_predictive_features_all.py"%ProcessedDataDir

    if file_is_empty(df_data_file) or file_is_empty(df_feature_info_file) or file_is_empty(df_data_all_file) or file_is_empty(df_feature_info_all_file):

        # keep
        print("keeping")
        df_new_vars = cp.deepcopy(df_new_vars)
        df_stress_phenotypes = cp.deepcopy(df_stress_phenotypes)
        df_phenotypes_Ksiezopolska2021 = cp.deepcopy(df_phenotypes_Ksiezopolska2021).rename(columns={"new_sample_name":"sampleID"}) # profile fields based on 

        # for Ksiezopolska2021, only keep specific fields
        drug_conditions = {'AinF', 'FLZ', 'FinA', 'ANIFLZ', 'ANI'}
        df_phenotypes_Ksiezopolska2021 = df_phenotypes_Ksiezopolska2021[(df_phenotypes_Ksiezopolska2021.fitness_estimate=="nAUC_rel") & (df_phenotypes_Ksiezopolska2021.drug_eucast.isin({"FLZ", "ANI"})) & (df_phenotypes_Ksiezopolska2021.apply(lambda r: (r.condition=="WT") or (r.condition in drug_conditions and r.sampleID in sampleNames_with_wgs), axis=1))]

        # add fields
        df_stress_phenotypes["strain"] = df_stress_phenotypes.Origin.apply(lambda x: x.replace("P35_2", "P35"))

        # remove the ANI samples that have no ERG3 mutations but are in the df as a residual of the fact that we added independently the ANI ERG3 sequences
        df_new_vars = df_new_vars[~((df_new_vars.final_name=="ERG3") & (pd.isna(df_new_vars.variant)) & (df_new_vars.with_wgs))]

        # define interesting samples and check that they are in dfs
        print("checking")
        interesting_samples = {s for s in sampleNames_with_wgs if s.split("_")[-1] in drug_conditions}
        sorted_samples = sorted(interesting_samples)

        for df in [df_new_vars, df_stress_phenotypes, df_phenotypes_Ksiezopolska2021]:
            missing_samples = interesting_samples.difference(set(df.sampleID))
            if len(missing_samples)>0: raise ValueError("missing_samples: %s"%missing_samples)

        # for variants, only keep the interesting samples
        df_new_vars = df_new_vars[df_new_vars.sampleID.isin(interesting_samples)]
        if not all(df_new_vars.with_wgs): raise ValueError("all samples should have WGS")
        for f in df_new_vars.keys():
            if any(pd.isna(df_new_vars[f])): raise ValueError("nans in df")

        # add variants of the parentals
        print("adding parental vars")
        df_new_vars["unique_varID"] = df_new_vars.final_name + "-" + df_new_vars.variant
        chrom_dup_loss_vars = set(df_new_vars[df_new_vars.variant=="loss DUP"].unique_varID)
        chrom_dup_loss_vars_to_dup_var = {'ChrE-loss DUP':'ChrE-DUP', 'ChrI-loss DUP':'ChrI-partial DUP'}

        df_all_vars = cp.deepcopy(df_new_vars[df_new_vars.condition.isin({"FLZ", "ANI", "ANIFLZ"})])[['sampleID', 'final_name', 'variant', 'unique_varID']] # init with the parentals

        for condition, parent_condition in [("AinF", "ANI"), ("FinA", "FLZ")]:
            for sampleID in [s for s in sorted_samples if s.endswith("_%s"%condition)]:

                # get datasets of sample and parent
                parent_sampleID = "_".join(sampleID.split("_")[0:2]) + "_" + parent_condition
                if parent_sampleID not in interesting_samples: raise ValueError("%s not in samples"%parent_sampleID)
                df_vars_sID = cp.deepcopy(df_new_vars[df_new_vars.sampleID==sampleID])
                df_parental_vars_sID = cp.deepcopy(df_new_vars[df_new_vars.sampleID==parent_sampleID])
                if len(df_vars_sID)==0: raise ValueError("0 df %s"%df_vars_sID)
                if len(df_parental_vars_sID)==0: raise ValueError("0 df %s"%df_parental_vars_sID)

                # define the vars that are in this sample that are in this sample
                chrom_dup_vars_remove_parental = {chrom_dup_loss_vars_to_dup_var[v] for v in set(df_vars_sID.unique_varID).intersection(chrom_dup_loss_vars)}
                if not all({v in set(df_parental_vars_sID.unique_varID) for v in chrom_dup_vars_remove_parental}): raise ValueError("all vars should be in parental: %s - %s"%(chrom_dup_vars_remove_parental, set(df_parental_vars_sID.unique_varID)))

                vars_parental = set(df_parental_vars_sID.unique_varID).difference(chrom_dup_vars_remove_parental)
                vars_sample = set(df_vars_sID.unique_varID).difference(chrom_dup_loss_vars)

                df_final_sID = pd.DataFrame({"unique_varID": sorted(vars_parental.union(vars_sample))})
                df_final_sID["sampleID"] = sampleID
                df_final_sID["final_name"] = df_final_sID.unique_varID.apply(lambda x: x.split("-")[0])
                df_final_sID["variant"] = df_final_sID.unique_varID.apply(lambda x: x.split("-")[1])

                df_all_vars = pd.concat([df_all_vars, df_final_sID])

        df_all_vars = df_all_vars.drop_duplicates().sort_values(by=["sampleID", "final_name", "variant"]).reset_index(drop=True)
        if set(df_all_vars.sampleID)!=interesting_samples: raise ValueError("the df with vars has inconsistent samples")

        # write the df_all_vars (for future)
        df_all_vars.to_csv("%s/Ksiezopolska2021_all_new_variants_per_sample_WGS.tab"%ProcessedDataDir, sep="\t")
                  
        ################################

        ###### ADD DIFFERENT PREDICTIVE FEATURES ##########
        print("adding predictive features...")

        # init dfs
        df_data = pd.DataFrame(index=sorted_samples) # a df where each row is one sample and each column is any of the predictors or the response values of the models (i.e. fitness in different stressors)
        features_data = [] # a df with information about different predictors. It will have 'feature', 'type_feature', 'data_type', 'description'

        df_data["sampleID"] = df_data.index
        df_data["strain"] = df_data.sampleID.apply(lambda s: s.split("_")[0])
        df_data["exp_evol_replicate"] = df_data.sampleID.apply(lambda s: s.split("_")[1])
        df_data["condition"] = df_data.sampleID.apply(lambda s: s.split("_")[2])
        condition_to_lineage_condition = {"FLZ":"FLZ", "ANI":"ANI", "AinF":"ANI", "FinA":"FLZ", "ANIFLZ":"ANIFLZ"}
        df_data["exp_evol_lineage"] = df_data.apply(lambda r: "%s_%s_%s"%(r.strain, r.exp_evol_replicate, condition_to_lineage_condition[r.condition]), axis=1)

        # get strain as predictor
        features_data.append(("strain", "WT_info", "cathegoric", "Strain of origin"))

        # get a df with the mutation effects
        df_data, features_data, df_all_vars = get_df_data_and_features_with_genetic_information_features(df_data, features_data, df_all_vars, gene_features_df, df_pathway_annots)



        # for each stress condition, add several information lines
        all_dependent_vals_fields = set()
        for stress_condition in ['CFW', 'CR', 'DTT', 'H2O2', 'NaCl', 'SDS', 'YPD', 'CycA']:
            df_stress_phenotypes_c = df_stress_phenotypes[df_stress_phenotypes.condition==stress_condition]
            df_stress_phenotypes_c = df_stress_phenotypes_c[(df_stress_phenotypes_c.type_measurement==df_stress_phenotypes_c.type_measurement.iloc[0])].copy() # keep the type of measurement, for condition==YPD, this is the plate based, as the code is written

            # check that they are the expected condition        
            if stress_condition=="CycA": expected_type_measurement = "OD_CycA_round1"
            else: expected_type_measurement = "Q-PHAST_round1"
            if not all(df_stress_phenotypes_c.type_measurement==expected_type_measurement): raise ValueError("all should be %s"%expected_type_measurement)

            # add WT strain info
            df_w = df_stress_phenotypes_c[(df_stress_phenotypes_c.Treatment=="WT")]

            for f in ["M_AUC", "M_fAUC_norm", "M_fAUC_diff"]: 
                strain_to_wt_val = dict(df_w.set_index("strain")[f])
                field_wt = "WT_%s_%s"%(f, stress_condition)
                df_data[field_wt] = df_data.strain.map(strain_to_wt_val)
                set(df_data[field_wt]) # check hashable
                check_no_nans_series(df_data[field_wt])
                features_data.append((field_wt, "WT_info", "continuous", "Fitness (%s) of the WT in %s"%(f, stress_condition)))

            # add dependent variables
            for dependent_f in ["M_AUC", 'M_fAUC_norm', 'M_fAUC_diff', 'M_rAUC_norm', 'M_rAUC_diff', 'M_AUC_norm_WT', 'M_AUC_diff_WT']: # "M_AUC_diff_WT_M_AUC", "M_AUC_norm_WT_M_AUC", "M_fAUC_diff_WT_M_fAUC", "M_fAUC_norm_WT_M_fAUC", "M_fAUC", M_rAUC

                sample_to_val = dict(df_stress_phenotypes_c[df_stress_phenotypes_c.Treatment.isin(drug_conditions)].set_index("sampleID")[dependent_f])
                field_f = "%s_%s"%(dependent_f, stress_condition)
                df_data[field_f] = df_data.sampleID.map(sample_to_val)
                set(df_data[field_f]) # check hashable
                check_no_nans_series(df_data[field_f])
                features_data.append((field_f, "stress_fitness", "continuous", "%s in %s"%(dependent_f, stress_condition)))
                all_dependent_vals_fields.add(field_f)

        # fitnesss calculations (Ksiezopolska2021)
        fitness_field_to_sampleID_to_val = {field: dict(df_phenotypes_Ksiezopolska2021.groupby("sampleID").apply(lambda df_s: np.mean(df_s[field]))) for field in ["fitness_conc0_median", "fitness_conc0_rel_to_WT_median"]}
        for fitness_field, sampleID_to_val in fitness_field_to_sampleID_to_val.items():

            # add the field for each sample
            sample_field = "%s_Ksiezopolska2021"%fitness_field
            df_data[sample_field] = df_data.sampleID.map(sampleID_to_val)
            features_data.append((sample_field, "Ksiezopolska2021_fitness_conc0", "continuous", "Fitness no-drug (%s) in Ksiezopolska 2021"%(fitness_field)))

        # add fitness of the WT (Ksiezopolska2021)
        strain_to_wt_fitness_Ksiezopolska2021 = {s.split("_")[0]:f for s,f in fitness_field_to_sampleID_to_val["fitness_conc0_median"].items() if s.endswith("_WT")}
        df_data["fitness_conc0_median_WT_Ksiezopolska2021"] = df_data.strain.map(strain_to_wt_fitness_Ksiezopolska2021)
        features_data.append(("fitness_conc0_median_WT_Ksiezopolska2021", "Ksiezopolska2021_fitness_conc0_WT", "continuous", "Fitness no-drug (fitness_conc0_median) of the WT in Ksiezopolska 2021"))

        # add the resistance calculations of Ksiezopolska2021
        for drug_eucast in ["ANI", "FLZ"]:
            df_drug = df_phenotypes_Ksiezopolska2021[df_phenotypes_Ksiezopolska2021.drug_eucast==drug_eucast]
            for f in ["AUCs_log2_concentration_median", "AUCs_log2_concentration_rel_to_WT_median", "MICs_array_median", "MICs_array_rel_to_WT_median", "log2_MICs_median", "log2_MICs_rel_to_WT_median"]:

                # add the values to the strain
                sID_to_val = df_drug.set_index("sampleID")[f]
                field_susc = "%s_%s_Ksiezopolska2021"%(f, drug_eucast)
                df_data[field_susc] = df_data.sampleID.map(sID_to_val)
                features_data.append((field_susc, "Ksiezopolska2021_susceptibility", "continuous", "Susceptibility towards %s (%s) in Ksiezopolska 2021"%(drug_eucast, f)))

                # for some fields, add the WT values
                if not "_rel_to_WT_" in f:
                    strain_to_wt_val = {s.split("_")[0]:f for s,f in sID_to_val.items() if s.endswith("_WT")}

                    wt_f = "%s_%s_WT_Ksiezopolska2021"%(f, drug_eucast)
                    df_data[wt_f] = df_data.strain.map(strain_to_wt_val)
                    features_data.append((wt_f, "Ksiezopolska2021_susceptibility_WT", "continuous", "Susceptibility towards %s (%s) of the WT in Ksiezopolska 2021"%(drug_eucast, f)))

        # one-hot encoding of cathegoric predictors, and add as features
        print("encode cathegoric as dummy binaries")
        cathegoric_features_data = [x for x in features_data if x[2]=="cathegoric"]
        cathegoric_features = [x[0] for x in cathegoric_features_data]

        df_data_cathegoric = df_data[cathegoric_features].copy()
        for f in cathegoric_features: df_data_cathegoric[f] = df_data_cathegoric[f].apply(lambda x: {True:np.nan, False:x}[x=="none"])
        df_data_cathegoric = pd.get_dummies(df_data_cathegoric, columns=cathegoric_features, prefix=["%s|||"%f for f in cathegoric_features]).map(int)
        df_data_cathegoric = df_data_cathegoric.rename(columns={c : c.replace("|||_", "|||")  for c in  sorted(df_data_cathegoric.columns)})
        df_data = df_data.join(df_data_cathegoric)

        # add info for the one-hot-encoded predictors
        for f in df_data_cathegoric.keys(): 
            original_f, f_val = f.split("|||")
            original_f_type = [x[1] for x in cathegoric_features_data if x[0]==original_f][0]
            features_data.append((f, "%s_hot_encoded"%original_f_type, "binary", "Version of %s hot encoded value %s"%(original_f, f_val)))

        # add pathway info
        df_data, features_data = get_df_data_and_features_data_with_NR_pathway_info(df_data, features_data, df_all_vars, df_pathway_annots, gene_features_df)

        # add PCA information for WTs (and plot)
        continuous_WT_features = sorted([x[0] for x in features_data if (x[1] in {"WT_info", "Ksiezopolska2021_fitness_conc0_WT", "Ksiezopolska2021_susceptibility_WT"}) and (not x[0] in {"strain", "WT_M_fAUC_YPD"}) and not (x[0].startswith("MICs_") or x[0].startswith("log2_MICs_"))])
        # df_data, features_data = get_df_data_and_features_with_PCA_wt_information_features(df_data, features_data, PlotsDir, continuous_WT_features)

        # remove some specific predictors manually curated. These are either all the same or redundant with some other (manually curated)
        uninteresting_predictors = {"WT_M_fAUC_norm_YPD", "WT_M_fAUC_diff_YPD", "M_fAUC_norm_YPD", "M_rAUC_norm_YPD", "M_fAUC_diff_YPD", "M_rAUC_diff_YPD", "MICs_array_median_ANI_WT_Ksiezopolska2021", "resistance_genes_broad_ChrI_profile|||dup", "resistance_genes_broad_EPA13_profile|||truncation", "ERG3_miss_243-243", "FKS2_miss_657-657", "PDR1_miss_280-280", "PDR1_miss_935-935"} # "FKS2_miss_654-657" , "M_rAUC_DTT", "M_rAUC_H2O2",

        features_data  = [x for x in features_data if not x[0] in uninteresting_predictors]
        df_data = df_data.drop(list(uninteresting_predictors), axis=1)

        ###################################################

        ######## ADD INTERACTIONS ###########
        print("adding interactions. ")

        # define predictors with which to compute interactions (only the meaningful ones)
        binary_predictors = []
        for type_feature in ["broad_resistance_genes_profile_hot_encoded", "single_variants", "domain_info", "pathway_info", "WT_info_hot_encoded"]: binary_predictors += [x[0] for x in features_data if x[1]==type_feature]
        all_possible_binary_predictors = set([x[0] for x in features_data if x[2]=="binary"])
        if all_possible_binary_predictors!=set(binary_predictors): raise ValueError("not all binaries considered")
        if len(binary_predictors)!=len(set(binary_predictors)): raise ValueError("binary preds should be unique")

        # defone the sorted continuous predictors (only the WT related)
        f_to_type_feature = {x[0] : x[1] for x in features_data}
        continuous_predictors = []
        for type_feature in ["WT_info", "Ksiezopolska2021_susceptibility_WT", "Ksiezopolska2021_fitness_conc0_WT"]: continuous_predictors += [f for f in continuous_WT_features if not f in uninteresting_predictors and f_to_type_feature[f]==type_feature]
        if set(continuous_predictors)!=(set(continuous_WT_features).difference(uninteresting_predictors)): raise ValueError("all cont prdictors should be included")

        # define info for each predictors
        p_to_type = {x[0] : x[1] for x in features_data}
        p_to_data_type = {x[0] : x[2] for x in features_data}
        p_to_description = {x[0] : x[3] for x in features_data}
        bp_to_samples = dict(zip(binary_predictors, map(lambda p: set(df_data[df_data[p]==1].sampleID), binary_predictors)))
        sample_to_lineage = dict(df_data.set_index("sampleID").exp_evol_lineage)
        bp_to_nlineages = {bp : len(set(map(lambda s: sample_to_lineage[s], samples)))  for bp, samples in bp_to_samples.items()}
        binary_predictors = [bp for bp in binary_predictors if bp_to_nlineages[bp]>=2]
        bp_to_samples = {bp : bp_to_samples[bp] for bp in binary_predictors}

        # iterate through interactions so that I can work with them
        print("%i binary and %i cont predictors. Getting NR predictors..."%(len(binary_predictors), len(continuous_predictors)))

        #all_combinations = [(bp, cp) for bp in binary_predictors for cp in continuous_predictors] + list(itertools.combinations(binary_predictors, 2))
        #all_combinations = [(bp, cp) for bp in binary_predictors for cp in continuous_predictors]
        all_combinations = list(itertools.combinations(binary_predictors, 2))

        n_combinations = len(all_combinations)
        for Ip, (p1, p2) in enumerate(all_combinations):
            if (Ip%10000)==0: print(Ip, n_combinations, len(binary_predictors), "binary predictors")

            # define term
            interaction_predictor = "%s * %s"%(p1, p2)

            # init whether to keep
            interaction_predictor_NR = False

            # binary interactions
            if p_to_data_type[p1]=="binary" and p_to_data_type[p2]=="binary":

                # discard cases where there are not enough (or too much) shared samples or lineages
                shared_samples = bp_to_samples[p1].intersection(bp_to_samples[p2])
                if shared_samples==bp_to_samples[p1]: continue # cases where it is the same
                if len(shared_samples)<2: continue # there have to be multiple shared samples for it to be relevant
                shared_lineages = set(map(lambda s: sample_to_lineage[s], shared_samples))
                if len(shared_lineages)<2: continue # there have to be many lineages

                # if it does not overlap previous binary predictors, keep it
                if not any(map(lambda other_bp: shared_samples==bp_to_samples[other_bp], binary_predictors)):
                    binary_predictors.append(interaction_predictor)
                    bp_to_samples[interaction_predictor] = shared_samples
                    interaction_predictor_values = df_data[p1] * df_data[p2]
                    interaction_predictor_NR = True
                    data_type_interact = "binary"

                    # check
                    # if not set(df_data[interaction_predictor_values==1].index)==bp_to_samples[interaction_predictor]: raise ValueError("the samples should be the same for binary predictors")
                    
            # continuous interactions
            else:

                interaction_predictor_values = df_data[p1] * df_data[p2]
                if not any(df_data[continuous_predictors].apply(lambda p: np.corrcoef(p, interaction_predictor_values)[0, 1] , axis=0).apply(abs)==1):
                    continuous_predictors.append(interaction_predictor)
                    df_data[interaction_predictor] = interaction_predictor_values
                    interaction_predictor_NR = True
                    data_type_interact = "continuous"

            # append to features
            if interaction_predictor_NR is True: 
                df_data = df_data.assign(**{interaction_predictor: interaction_predictor_values})
                features_data.append((interaction_predictor, "pairwise_interaction %s & %s"%(p_to_type[p1], p_to_type[p2]), data_type_interact, "interaction between %s (%s) and %s (%s)"%(p1, p_to_description[p1], p2, p_to_description[p2])))

        #####################################


        ####### GET DF #####

        print("getting df. %i cols"%(len(df_data.columns)))


        # get the features df
        df_feature_info = pd.DataFrame(features_data, columns=["feature", "type_feature", "data_type", "description"])

        # save the single variants

        # final checks
        for f in df_data.keys():
            if any(pd.isna(df_data[f])): raise ValueError("nans in df, %s"%f)

        strange_columns_df_data = set(df_data.keys()).difference(set(df_feature_info.feature)).difference({'condition', 'sampleID', 'strain', 'exp_evol_lineage', 'exp_evol_replicate'})
        if len(strange_columns_df_data)>0: raise ValueError("strange cols: %s"%strange_columns_df_data)
        if len(set(df_feature_info.feature).difference(set(df_data.keys())))>0: raise ValueError("strange cols in feats df")
        strange_type_feat = set(df_feature_info.data_type).difference({"continuous", "cathegoric", "binary"})
        if len(strange_type_feat): raise ValueError("strange data_type : %s"%strange_type_feat)
        if len(df_feature_info)!=len(set(df_feature_info.feature)): raise ValueError("feats should be unique")
        if any([len(set(df_data[x]).difference({1, 0}))>0 for x in df_feature_info[df_feature_info.data_type=="binary"].feature]): raise ValueError("binary vals should be 1/0")
        if any([any(df_data[x].apply(type)!=str) for x in df_feature_info[df_feature_info.data_type=="cathegoric"].feature]): raise ValueError("cathegoric vals should be strings")

        # sort
        df_data = df_data.sort_index()
        df_data["sample_idx"] = list(range(len(df_data)))

        # define interesting continuous informations
        df_feature_info["feature_is_WT_continuous_info"] = df_feature_info.feature.isin(continuous_WT_features)

        # save all the variants, not only the uncommon ones
        save_object(df_feature_info, df_feature_info_all_file)
        save_object(df_data, df_data_all_file)

        # filter predictors that are not in sufficient lineages
        def get_n_expevevol_lineages_minor_cathegory(r):
            if r.data_type in {"continuous", "cathegoric"}: return -1
            elif r.data_type=="binary": return min(df_data[["exp_evol_lineage", r.feature]].groupby(r.feature).apply(lambda df_f: len(set(df_f.exp_evol_lineage))))
            else: raise ValueError("%s")

        df_feature_info["n_lineages_with_minor_cathegory_binary_features"] = df_feature_info.apply(get_n_expevevol_lineages_minor_cathegory, axis=1)

        uncommon_predictors = set(df_feature_info[(df_feature_info.n_lineages_with_minor_cathegory_binary_features<2) & (df_feature_info.data_type=="binary")].feature)
        df_feature_info = df_feature_info[~df_feature_info.feature.isin(uncommon_predictors)]
        df_data = df_data.drop(list(uncommon_predictors), axis=1)

        #####################

        ######### CHECK PREDICTORS ########

        # check that the predictors are not entirely correlated
        continuous_predictors = list(df_feature_info[df_feature_info.data_type=="continuous"].feature)
        binary_predictors = list(df_feature_info[df_feature_info.data_type=="binary"].feature)

        print("Checking %i predictors ..."%len(continuous_predictors + binary_predictors))
        for p in (continuous_predictors + binary_predictors):
            print(p)
            if len(set(df_data[p]))==1: raise ValueError("all vals the same for %s"%p)

        combinations_continuous = list(itertools.combinations(continuous_predictors, 2))
        df_correlations = pd.DataFrame({"correlation" : dict(zip(combinations_continuous, map(lambda x: np.corrcoef(df_data[x[0]], df_data[x[1]])[0, 1], combinations_continuous))), "combination":dict(zip(combinations_continuous, combinations_continuous))})
        df_corr1 = df_correlations[df_correlations.correlation.apply(abs)>=1]
        if len(df_corr1)>0: raise ValueError("correlations ==1\n%s"%(df_corr1))

        print("checking binary predictors...")
        p_to_samples = dict(zip(binary_predictors, map(lambda p: set(df_data[df_data[p]==1].sampleID), binary_predictors)))
        combinations_binary = list(itertools.combinations(binary_predictors, 2))
        series_equal_content = pd.Series(dict(zip(combinations_binary, map(lambda x: p_to_samples[x[0]]==p_to_samples[x[1]], combinations_binary))))
        if any(series_equal_content): raise ValueError("some predictors are the same\n%s"%series_equal_content[series_equal_content])

        #####################################

        # save
        print("saving")
        save_object(df_data, df_data_file)
        save_object(df_feature_info, df_feature_info_file)

    return load_object(df_data_file), load_object(df_feature_info_file), df_data_file, df_feature_info_file, df_data_all_file, df_feature_info_all_file

def generate_table_mutations_and_phenos(df_data_all, df_feature_info, PlotsDir_paper):

    """Generates a table with all mutations and phenotypes for the paper"""

    # keep
    df_feature_info = df_feature_info.copy()
    df_data_all = df_data_all.copy()

    # init with phenotypes
    fields_dict = {}
    all_fields = []
    for stress_condition in ['CFW', 'CR', 'DTT', 'H2O2', 'NaCl', 'SDS', 'CycA', 'YPD']:

        # add the strain phenotypes and WT info
        if stress_condition=="YPD":
            all_fields.append("M_AUC_norm_WT_%s"%stress_condition)
            fields_dict["M_AUC_norm_WT_%s"%stress_condition] = "AUC_R_%s"%stress_condition
            fields_wt = ["WT_M_AUC"]

        else:
            all_fields.append("M_rAUC_norm_%s"%stress_condition)
            fields_dict["M_rAUC_norm_%s"%stress_condition] = "fAUC_R_%s"%stress_condition
            fields_wt = ["WT_M_fAUC_norm"]

        # add the WT info
        for f in fields_wt:
            all_fields.append("%s_%s"%(f, stress_condition))
            fields_dict["%s_%s"%(f, stress_condition)] = "%s_%s"%(f.replace("WT_M_AUC", "AUC_WT").replace("WT_M_fAUC_norm", "fAUC_WT"), stress_condition)

    # add the mutations
    for f in sorted(set(df_feature_info[df_feature_info.type_feature=="single_variants"].feature)):
        all_fields.append(f)
        fields_dict[f] = f.replace("presence_variant-", "")

    # write
    if sorted(all_fields)!=sorted(fields_dict.keys()): raise ValueError("renaming should be the same")
    df_save = df_data_all[["sampleID"] + all_fields].copy().rename(columns=fields_dict)
    df_save.to_excel("%s/phenotypes_and_variants.xlsx"%PlotsDir_paper)

    return df_save

def get_correct_feature_for_formulas(f):

    """Formats feature so that it is valid for formulas (only letters)"""

    # change
    renamed_f = f.replace("-", "_").replace("|", "_").replace("*", "ptc").replace("/", "_").replace(".", "_").replace(" ", "_").replace("&", "_")

    # check
    if not re.compile("^[a-zA-Z0-9_]+$").match(renamed_f): raise ValueError("invalid renamed_f %s"%renamed_f)
    return renamed_f

def get_r2_two_series(series_A, series_B):

    """Gets r2"""

    # checks
    if any(pd.isna(series_A)): raise ValueError("nans in series_A")
    if any(pd.isna(series_B)): raise ValueError("nans in series_B")
    if list(series_A.index)!=list(series_B.index): raise ValueError("indices shuld be the same")

    slope, intercept, r_value, p_value, std_err = stats.linregress(series_A, series_B)
    r_squared = r_value**2
    return r_squared


def chunks(l, n):
    
    """Yield successive n-sized chunks from a list l"""
    
    for i in range(0, len(l), n):
        yield l[i:i + n]

def get_CVindices_per_lineage(X, Kfold=4):

    """Gets the CV indices in a Kfold CV strategy"""

    # get the lineages' df
    df_samples = pd.DataFrame({"sampleID":list(X.index)}).set_index("sampleID", drop=False)
    condition_to_lineage_condition = {"FLZ":"FLZ", "ANI":"ANI", "AinF":"ANI", "FinA":"FLZ", "ANIFLZ":"ANIFLZ"}
    df_samples["exp_evol_lineage"] = df_samples.sampleID.apply(lambda s: "%s_%s_%s"%(s.split("_")[0], s.split("_")[1], condition_to_lineage_condition[s.split("_")[2]]))

    # get shuffle
    all_lineages_set = set(df_samples["exp_evol_lineage"])
    all_lineages = list(all_lineages_set)
    random.shuffle(all_lineages)

    # Split the list into four sublists
    chunks_lineages = list(chunks(all_lineages, int(len(all_lineages) / Kfold)))

    # Distribute the remaining elements to the sublists
    remaining_lineages = len(all_lineages) % 4
    for i in range(remaining_lineages):
        chunks_lineages[i].append(chunks_lineages[-1][i])
        chunks_lineages[-1].pop(i)
    chunks_lineages.pop(-1)

    CV_indices = []
    for chunk_l in chunks_lineages:

        test_lineages = set(chunk_l)
        train_lineages = set(all_lineages_set).difference(test_lineages)

        test_samples = set(df_samples[df_samples.exp_evol_lineage.isin(test_lineages)].sampleID)
        train_samples = set(df_samples[df_samples.exp_evol_lineage.isin(train_lineages)].sampleID)

        test_idx = df_samples.index.isin(test_samples)
        train_idx = df_samples.index.isin(train_samples)
        if any(test_idx==train_idx): raise ValueError("All should be in different")
        if (sum(test_idx)+sum(train_idx)) != len(X): raise ValueError("all samples should be in some")

        CV_indices.append((train_idx, test_idx))

    return CV_indices


def get_jaccard_index_to_sets(set1, set2):

    """Jaccard index two sets"""

    np.seterr(divide='ignore', invalid='ignore')
    return np.divide(len(set1.intersection(set2)), len(set1.union(set2)))

def get_jaccard_index_samples(df_data, p1, p2):

    """Gets the jaccard index between samples"""


    samples_1 = set(df_data[df_data[p1]==1].sampleID)
    samples_2 = set(df_data[df_data[p2]==1].sampleID)
    jaccard_index = get_jaccard_index_to_sets(samples_1, samples_2)

    return jaccard_index

def get_DecisionTreeRegressor_old(ccp_alpha=0.0): return DecisionTreeRegressor(criterion="squared_error", splitter="best", min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, ccp_alpha=ccp_alpha)

def get_DecisionTreeRegressor_auto_ccp_alpha(X, y):

    """Gets a decision tree alpha that has an automatically set ccp_alpha based on CV"""

    # keep
    X = X.copy(); y = y.copy()

    # get all the possible alphas for the data at hand
    model = DecisionTreeRegressor()
    ccp_alphas = model.cost_complexity_pruning_path(X, y).ccp_alphas
    ccp_alphas = sorted(ccp_alphas[ccp_alphas>0])

    # get the best one
    if len(ccp_alphas)>0:

        # get for different ccp alphas, the r2 across different CVs
        all_cvs = list(range(0, 6))

        df_alphas = pd.DataFrame(dict(zip(all_cvs, 
                    map(lambda cv_idx: dict(zip(ccp_alphas, map(lambda ca: np.mean(cross_val_score(DecisionTreeRegressor(ccp_alpha=ca), X, y, cv=get_CVindices_per_lineage(X), scoring="r2")), ccp_alphas)))
                    , all_cvs)))).transpose()

        # get the alpha that is best across more cases, and the maximum one
        cv_to_best_alpha = df_alphas.apply(lambda r: max(r[r==max(r)].keys()), axis=1) # bigger alphas are better
        alpha_to_ncvs_best = pd.Series(Counter(cv_to_best_alpha))

        # if not any(alpha_to_ncvs_best>=2): raise ValueError("some alpha should be best at least twice")
        if len(df_alphas.columns)!=len(ccp_alphas): raise ValueError("change in the df cols")
        best_alpha = max(alpha_to_ncvs_best[alpha_to_ncvs_best==max(alpha_to_ncvs_best)].index)

    else: best_alpha = 0

    # get the tree with this alpha
    return DecisionTreeRegressor(ccp_alpha=best_alpha)

def get_predictor_model(model_name, X, y):

    """Gets the predictor model"""

    if model_name=="linear_regression": return LinearRegression()
    elif model_name=="regression_tree_auto_ccp": return get_DecisionTreeRegressor_auto_ccp_alpha(X, y)
    elif model_name=="regresssion_tree_custom": return DecisionTreeRegressor(min_samples_split=2, min_samples_leaf=2)
    elif model_name=="rf_regressor": return RandomForestRegressor(n_estimators=100, bootstrap=True, min_samples_split=2, min_samples_leaf=2)
    else: raise ValueError("invalid model")

def get_model_score_CV(X, y, model_name):

    """Gets the model score when using certain predictors"""

    model_obj = get_predictor_model(model_name, X, y)
    cv_scores = cross_val_score(model_obj, X, y, cv=get_CVindices_per_lineage(X), scoring="r2")
    return np.mean(cv_scores)

def get_important_features_inHouse_forward_selection(model_name, X, y, tol):

    """Runs a forward SequentialFeature selection for the provided model"""

    #print("Getting important features in house...")

    # keep
    X = X.copy(); y = y.copy()
    all_features = set(X.columns)

    # init features
    important_features = []
    features_to_discard = set() # these are features that are discarded when they are tried and add a score below tol
    previous_score = 0

    # keep adding features
    number_tries_zero_score = 0
    while True:

        # get the model score when adding each feature independently
        features_to_not_test = features_to_discard.union(set(important_features))
        features_to_test = sorted(all_features.difference(features_to_not_test))
        inputs_fn = [(X[important_features + [f]].copy(), y.copy(), model_name) for f in features_to_test]
        list_scores = list(map(lambda x: get_model_score_CV(x[0], x[1], x[2]), inputs_fn))

        # discard when there are not feats to test
        if len(features_to_test)==0: break

        # get the feature with the best score
        df_scores = pd.DataFrame({"new_feature":features_to_test, "score":list_scores}).sort_values(by=["score", "new_feature"], ascending=[False, True])
        df_scores["score_increase"] = df_scores.score - previous_score
        series_best_score = df_scores.iloc[0]

        # add it if necessary
        if series_best_score.score_increase >= tol:

            # keep feature as important
            important_features.append(series_best_score.new_feature)

            # update previous score
            previous_score = series_best_score.score

            # update features to discard
            features_to_discard.update(set(df_scores[df_scores.score_increase<tol].new_feature))

        # if it already had some good score break
        elif previous_score>0: break

        # break with too many attempts
        else:

            number_tries_zero_score+=1
            if number_tries_zero_score==2: break

    return important_features


def get_pval_association_one_feature(r, yfield, df_data, all_samples, bp_to_samples):

    """Takes a row of the feats df and returns the pval of the association with the yfield"""

    # spearman correlation for numeric data
    if r.data_type=="continuous":
        statistic_type = "r_spearman"
        statistic_value, pval = stats.spearmanr(df_data[r.feature], df_data[yfield], nan_policy="raise")

    # KS test for non-binary data
    elif r.data_type=="binary":
        statistic_type = "KS_statistic"

        # faster
        values_0s_samples = df_data.loc[list(all_samples.difference(bp_to_samples[r.feature])), yfield].values
        values_1s_samples = df_data.loc[list(bp_to_samples[r.feature]), yfield].values
        if len(values_0s_samples)<2 or len(values_1s_samples)<2: raise ValueError("invalid vals")
        statistic_value, pval = stats.ks_2samp(values_0s_samples, values_1s_samples, alternative="two-sided", method="auto") # auto for small arrays

    else: raise ValueError(r)

    return pval


def delete_folder(f):

    if os.path.isdir(f): shutil.rmtree(f)

def make_folder(f):

    if not os.path.isdir(f): os.mkdir(f)

def get_results_different_models_old(df_data_all, df_feature_info_all, df_univariate_associations_all, ProcessedDataDir, PlotsDir):

    """For different ways of building the models, save the model results"""

    # define file
    df_models_all_file = "%s/df_models_all_long.py"%ProcessedDataDir
    if file_is_empty(df_models_all_file):

        # keep
        df_data_all = cp.deepcopy(df_data_all)
        df_feature_info_all = cp.deepcopy(df_feature_info_all)
        df_univariate_associations_all = cp.deepcopy(df_univariate_associations_all)

        ######## PREPARE ##########

        # load inputs to a parallel function
        # inputs_fn = []; I=0
        all_cmds = []; I=0
        outdir = "%s/generating_models"%ProcessedDataDir; make_folder(outdir)

        for only_predictors_univariate_corr in [False, True]: # False
            for consider_interactions in [False, True]:
                for model_name in ["rf_regressor", "linear_regression", "regression_tree_auto_ccp", "regresssion_tree_custom"]:
                    for stress_condition in ['NaCl', 'CR', 'CFW', 'DTT', 'SDS', 'H2O2', 'YPD', 'CycA']:
                        for fitness_estimate in ["M_rAUC_norm", "M_rAUC_diff", "M_AUC_norm_WT", "M_AUC_diff_WT"]:
                            print(only_predictors_univariate_corr, model_name, stress_condition, fitness_estimate)

                            # discard some cases
                            if stress_condition=="YPD" and not fitness_estimate in {"M_AUC_norm_WT", "M_AUC_diff_WT"}: continue
                            if consider_interactions is True and model_name!="linear_regression": continue

                            # debug cases
                            #if stress_condition!="NaCl": continue

                            # define field
                            yfield = "%s_%s"%(fitness_estimate, stress_condition)

                        
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
                                if df_univariate_associations_all is None:

                                    bp_to_samples = dict(df_data_all[[f for f in df_feature_info[df_feature_info.data_type=="binary"].feature]].apply(lambda c: set(c[c==1].index)))
                                    df_feature_info["pval_association"] = df_feature_info.apply(get_pval_association_one_feature, yfield=yfield, df_data=df_data_all[[yfield, "sampleID"] + sorted(set(df_feature_info.feature))].set_index("sampleID"), all_samples=set(df_data_all.sampleID), bp_to_samples=bp_to_samples, axis=1)
                                    
                                else:

                                    # get df for this case
                                    df_univariate_associations = df_univariate_associations_all[df_univariate_associations_all.yfield==yfield]

                                    # check missing feats
                                    missing_feats = set(df_univariate_associations.feature).difference(set(df_feature_info_all.feature))
                                    missing_feats_feats_info = set(df_feature_info_all.feature).difference(set(df_univariate_associations.feature))
                                    if len(missing_feats)>0: raise ValueError("missing feats in df_feature_info")
                                    if len(missing_feats_feats_info)>0: raise ValueError("missing feats in df_univariate_associations. %s"%len(missing_feats_feats_info))
                                    if len(potential_WT_predictors.difference(set(df_univariate_associations.feature))): raise ValueError("some feats not in df_univariate_associations")

                                    # add to df 
                                    df_feature_info["pval_association"] = df_feature_info.feature.map(df_univariate_associations.set_index("feature").pval)
                                    check_no_nans_series(df_feature_info.pval_association)
                                   
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
                            if len(all_predictors)==0: raise ValueError("predictors are 0")

                            # Standardize the continuous predictors (optional).  this does not transform the linearity, but it allows to pick the most important predictors
                            #if len(continuous_predictors)>0: df_data[continuous_predictors] = StandardScaler().fit_transform(df_data[continuous_predictors])

                            # keep some fields
                            df_data = df_data_all[[yfield, "exp_evol_lineage", "sampleID"] + all_predictors].copy()

                            # different tries of the model
                            m_to_fs = {"linear_regression" : ["forward_sklearn", "forward_inHouse"], 
                                       "regression_tree_auto_ccp" : ["forward_inHouse", "regression_tree_feature_importance", "sel_from_model"],
                                       "regresssion_tree_custom" : ["forward_inHouse", "regression_tree_feature_importance", "sel_from_model", "RFECV", "forward_sklearn"],
                                       "rf_regressor" : ["forward_inHouse", "rf_regressor_feature_importance", "sel_from_model", "RFECV", "forward_sklearn"]
                                       }

                            default_tol_r2_list = [0.1, 0.15, 0.2]
                            feat_sel_to_tol_r2_list = {"forward_inHouse" : default_tol_r2_list, "forward_sklearn" : default_tol_r2_list, "RFECV" : [None], "regression_tree_feature_importance":[None], "rf_regressor_feature_importance":[None], "sel_from_model":[None]}


                            for type_feature_selection in m_to_fs[model_name]:
                                for tol_r2 in feat_sel_to_tol_r2_list[type_feature_selection]:
                                    for try_ID in range(1, 4):

                                        # keep
                                        inputs_fn.append((I+1, df_data.copy(), cp.deepcopy(binary_predictors), cp.deepcopy(continuous_predictors), fitness_estimate, stress_condition, PlotsDir, model_name, yfield, try_ID, only_predictors_univariate_corr, type_feature_selection, tol_r2, consider_interactions))
                                        I+=1

        ###########################

        ######### GET DF ############

        # run in parallel
        njobs = len(inputs_fn)
        inputs_fn = [tuple(list(x) + [njobs]) for x in inputs_fn]
        with multiproc.Pool(multiproc.cpu_count()) as pool:

            df_models_all = pd.DataFrame(pool.starmap(get_prediction_results_stress_from_features, inputs_fn, chunksize=1))
            pool.close()
            pool.terminate()

        if len(df_models_all)!=len(df_models_all[["idx", "try_ID"]].drop_duplicates()): raise ValueError("idx should be unique")

        for I, r in df_models_all.iterrows():
            print(r.idx, r.try_ID, r.mean_r2, r.selected_predictors)

        dakjgdadgh

        # save
        print("saving...")
        save_object(df_models_all, df_models_all_file)

        #############################

    return load_object(df_models_all_file)


    
def get_dir(filename): return "/".join(filename.split("/")[0:-1])

def get_file(filename): return filename.split("/")[-1]

def run_jobarray_file_MN4_greasy(jobs_filename, name, stddir, time="12:00:00", queue="bsc_ls", threads_per_job=4, nodes=1, submit=True):

    """
    This function takes a jobs filename and creates a jobscript with args (which is a list of lines to be written to the jobs cript). It works in MN4 for greasy    

    """

    # define dirs
    outdir = get_dir(jobs_filename)
    delete_folder(stddir); make_folder(stddir)
    os.chdir(outdir)

    # define the std files
    greasy_logfile = "%s/%s_greasy.log"%(stddir, name)
    stderr_file = "%s/%s_stderr.txt"%(stddir, name)
    stdout_file = "%s/%s_stdout.txt"%(stddir, name)

    # define the job script
    jobs_filename_run = "%s.run"%jobs_filename

    # define the number of jobs in the job array
    njobs = int("".join([x for x in str(subprocess.check_output("wc -l %s"%jobs_filename, shell=True)).split()[0] if x.isdigit()])) + 1

    # define the requested nodes. Each node has 48 threads
    total_nthreads = njobs*threads_per_job
    if total_nthreads%48==0: max_nodes = max([int((total_nthreads)/48), 1])
    else: max_nodes = max([int((total_nthreads)/48)+1, 1])
    requested_nodes = min([nodes, max_nodes])

    # define the number of tasks
    max_ntasks = int((requested_nodes*48)/threads_per_job)
    ntasks = min([njobs, max_ntasks])


    # define the arguments
    arguments = [ "#!/bin/sh",
                  "#SBATCH --error=%s"%stderr_file,
                  "#SBATCH --output=%s"%stdout_file,
                  "#SBATCH --job-name=%s"%name, 
                  "#SBATCH --get-user-env",
                  #"#SBATCH --workdir=%s"%outdir,
                  "#SBATCH --time=%s"%time,
                  "#SBATCH --qos=%s"%queue,
                  "#SBATCH --cpus-per-task=%i"%threads_per_job,
                  "#SBATCH --ntasks=%i"%ntasks,
                  "",
                  "rm %s/*"%stddir,
                  "module load greasy",
                  "export GREASY_LOGFILE=%s;"%(greasy_logfile),
                  "rm %s;"%greasy_logfile,
                  "echo 'running pipeline';",
                  "greasy %s"%jobs_filename
                ]

    # replace by the run in MN
    arguments = [a.replace(ParentDir, "/gpfs/projects/bsc40/mschikora") for a in arguments]

    # define and write the run filename
    with open(jobs_filename_run, "w") as fd: fd.write("\n".join(arguments))
    
    # run in cluster if specified
    if submit is True: os.system("sbatch %s"%jobs_filename_run)



def get_results_different_models(df_data_all_file, df_feature_info_all_file, ProcessedDataDir, PlotsDir):

    """For different ways of building the models, save the model results"""

    # define file
    df_models_all_file = "%s/df_models_all_long.py"%ProcessedDataDir
    if file_is_empty(df_models_all_file):

        # load jobs to a parallel function
        print("getting jobs...")
        all_cmds = []; Ijob=1
        outdir_all = "%s/generating_models"%ProcessedDataDir; make_folder(outdir_all)
        outdir = "%s/model_results"%outdir_all; make_folder(outdir)
        stddir = "%s/STDfiles"%outdir_all; make_folder(stddir)
        already_generated_results_files = set(os.listdir(outdir))

        print("getting cmds...")
        for reshuffle_y, ntries_general in {True:100, False:1}.items():
            for tryID_general in range(ntries_general):
                print(reshuffle_y, tryID_general)

                for only_predictors_univariate_corr in [False, True]: # False
                    for consider_interactions in [False, True]:
                        for model_name in ["rf_regressor", "linear_regression", "regression_tree_auto_ccp", "regresssion_tree_custom"]:
                            for stress_condition in ['NaCl', 'CR', 'CFW', 'DTT', 'SDS', 'H2O2', 'YPD', 'CycA']:
                                #for fitness_estimate in ["M_rAUC_norm", "M_rAUC_diff", "M_AUC_norm_WT", "M_AUC_diff_WT"]:
                                #if stress_condition=="YPD" and not fitness_estimate in {"M_AUC_norm_WT", "M_AUC_diff_WT"}: continue

                                if stress_condition=="YPD": list_fitness_estimates = ["M_AUC_norm_WT"]
                                else: list_fitness_estimates = ["M_rAUC_norm"]

                                for fitness_estimate in list_fitness_estimates:

                                    # discard some cases that do not make sense
                                    if consider_interactions is True and model_name!="linear_regression": continue

                                    # discard cases that are excessively costly (I actually checked them and saw that they were not adding much)
                                    if model_name=="rf_regressor" and only_predictors_univariate_corr is False: continue # I checked that this was not necessarily adding 

                                    # debug
                                    #if stress_condition!="NaCl" or fitness_estimate!="M_rAUC_norm" or only_predictors_univariate_corr!=False: continue
                                    #if stress_condition!="NaCl" or fitness_estimate!="M_rAUC_norm": continue
                                    # print(only_predictors_univariate_corr, consider_interactions, model_name, stress_condition, fitness_estimate, reshuffle_y, tryID_general)

                                    # different tries of the model
                                    m_to_fs = {"linear_regression" : ["forward_sklearn", "forward_inHouse"], 
                                               "regression_tree_auto_ccp" : ["forward_inHouse", "regression_tree_feature_importance", "sel_from_model"],
                                               "regresssion_tree_custom" : ["forward_inHouse", "regression_tree_feature_importance", "sel_from_model", "RFECV", "forward_sklearn"],
                                               "rf_regressor" : ["rf_regressor_feature_importance", "sel_from_model", "RFECV", "forward_sklearn"] # forward_inHouse was redundant
                                               }

                                    default_tol_r2_list = [0.1, 0.15, 0.2]
                                    feat_sel_to_tol_r2_list = {"forward_inHouse" : default_tol_r2_list, "forward_sklearn" : default_tol_r2_list, "RFECV" : [None], "regression_tree_feature_importance":[None], "rf_regressor_feature_importance":[None], "sel_from_model":[None]}


                                    for type_feature_selection in m_to_fs[model_name]:
                                        for tol_r2 in feat_sel_to_tol_r2_list[type_feature_selection]:
                                            for try_ID in range(1, 11):

                                                yfield = "%s_%s"%(fitness_estimate, stress_condition)
                                                idx = "---".join([str(x) for x in [reshuffle_y, tryID_general, fitness_estimate, stress_condition, model_name, yfield, only_predictors_univariate_corr, type_feature_selection, tol_r2, consider_interactions, try_ID]])
                                                prediction_results_file = "%s/prediction_results_%s.py"%(outdir, idx)

                                                # get cmd
                                                if not get_file(prediction_results_file) in already_generated_results_files:

                                                    args_list = [idx, reshuffle_y, tryID_general, prediction_results_file, df_data_all_file, df_feature_info_all_file, only_predictors_univariate_corr, consider_interactions, model_name, stress_condition, fitness_estimate, type_feature_selection, tol_r2, try_ID]
                                                    cmd = "%s/Cglabrata_tradeoffs/run_get_prediction_results_stress_from_features.py %s"%(ParentDir, " ".join(map(str, args_list)))
                                                    all_cmds.append("[ ! -e %s ] && source /gpfs/projects/bsc40/mschikora/anaconda3/etc/profile.d/conda.sh && conda activate Cglabrata_tradeoffs_env > /dev/null && %s > %s/job%i.std 2>&1"%(prediction_results_file, cmd, stddir, Ijob)); Ijob+=1

        # checks
        if len(all_cmds)!=len(set(all_cmds)): raise ValueError("not unique cmds")

        # run a few
        if len(all_cmds)>0 and len(all_cmds)<10:
            for cmd in all_cmds: 
                out_stat = os.system(cmd)
                if out_stat!=0: raise ValueError("invalid cmd")
                sys.exit(0)

        # run in the cluster in parallel
        elif len(all_cmds)>0:
            print("submitting to mn %i jobs"%(len(all_cmds)))

            # define the jobs filename
            jobs_filename = "%s/jobs.get_models"%(outdir_all)
            open(jobs_filename, "w").write("\n".join(all_cmds))

            # submit jobs
            os.chdir(outdir_all)
            run_jobarray_file_MN4_greasy(jobs_filename, "get_models", stddir, time="02:00:00", queue="debug", threads_per_job=1, nodes=16) # max 16 nodes
            sys.exit(0)

        # get the integrated file
        print("integrating...")
        chunk_size = 500
        chunks_results_files = list(chunks(sorted(already_generated_results_files), chunk_size))
        nchunks = len(chunks_results_files)
        outdir_chunks = "%s/model_results_chunks_dfs_cs=%i_nc=%i_njobs=%i"%(outdir_all, chunk_size, nchunks, len(already_generated_results_files)); make_folder(outdir_chunks)

        os.chdir(outdir)
        with multiproc.Pool(multiproc.cpu_count()) as pool:
            df_models_all = pd.concat(pool.starmap(get_df_models_one_chunk, list(map(lambda x: (x[0], x[1], nchunks, outdir_chunks), enumerate(chunks_results_files))), chunksize=1))
            pool.close()
            pool.join()
            pool.terminate()

        df_models_all = df_models_all.reset_index(drop=True)

        # check len
        if len(df_models_all)!=len(already_generated_results_files): raise ValueError("not the same len")

        # save
        print("saving...")
        save_object(df_models_all, df_models_all_file)

    print("get object...")
    return load_object(df_models_all_file)

def get_df_models_one_chunk(Ic, chunk_results_files, nchunks, outdir_chunks): 

    """Loads files as a df for several files"""

    # log
    if (Ic%2)==0: print("Chunk %i/%i"%(Ic+1, nchunks))

    # define file and generate
    df_file = "%s/chunk_%i.py"%(outdir_chunks, Ic)
    if len(chunk_results_files)==0: raise ValueError("no chunks")

    if file_is_empty(df_file):
        df = pd.DataFrame(map(load_object, map(lambda f: "./%s"%f, chunk_results_files)))
        save_object(df, df_file)

    # check
    # if file_is_empty(df_file): raise ValueError("empty file %s"%df_file)

    return load_object(df_file)

def get_grouped_df_models_across_tries(df_models_all, ProcessedDataDir):

    """Groups acrooss try"""

    df_models_file = "%s/df_models_grouped_by_modelID.py"%ProcessedDataDir
    if file_is_empty(df_models_file):
        print("getting file")

        if len(set(df_models_all.idx))!=len(df_models_all): raise ValueError("invalid idx")

        # add a numeric idx
        df_models_all["numeric_ID"] = list(range(len(df_models_all)))


        # add model ID
        print("adding model ID")
        df_models_all["modelID"] = ""
        model_ID_fields = ["reshuffle_y", "tryID_general", "fitness_estimate", "stress_condition", "model_name", "only_predictors_univariate_corr", "type_feature_selection", "tol_r2", "consider_interactions"]
        for f in model_ID_fields: 
            df_models_all["modelID"] = df_models_all.modelID + "|" + df_models_all[f].apply(str)
        if len(df_models_all["modelID"].drop_duplicates())!=len(df_models_all)/10: raise ValueError("invalid len")

        # get r for each ID
        print("getting grouped df")

        def get_series_info_one_tuple_predictors(df_p): return pd.Series({"mean_mean_r2":np.mean(df_p.mean_r2), "min_mean_r2":min(df_p.mean_r2), "std_mean_r2":np.std(df_p.mean_r2), "selected_predictors":df_p.name, "ntries_predictors":len(df_p)})

        field_to_zero_val = {"mean_mean_r2":0, "min_mean_r2":0, "std_mean_r2":0, "selected_predictors":(), "ntries_predictors":0, "len_selected_predictors":0}
        def get_model_info_one_model(df_m):

            first_n = df_m.numeric_ID.iloc[0]
            if (first_n%500)==0: print(first_n)

            # init with the general case
            data_dict = {"modelID":df_m.name}
            for f in model_ID_fields: data_dict[f] = df_m.iloc[0][f]

            # map each set of predictors (keep the most consistent one)
            predictors_info = df_m[["mean_r2", "std_r2", "selected_predictors"]].groupby("selected_predictors").apply(get_series_info_one_tuple_predictors).reset_index(drop=True)
            predictors_info["len_selected_predictors"] = predictors_info.selected_predictors.apply(len)
            predictors_info = predictors_info[(predictors_info.len_selected_predictors>0) & (predictors_info.mean_mean_r2>0) & (predictors_info.min_mean_r2>0)]

            # if there is something to map
            if len(predictors_info)>0: 
                
                predictors_info = predictors_info.sort_values(by=["ntries_predictors", "mean_mean_r2", "min_mean_r2", "len_selected_predictors", "std_mean_r2", "selected_predictors"], ascending=[False, False, False, True, True, True])
                for k,v in predictors_info.iloc[0].items(): data_dict[k] = v

            # else add all zeros
            else:
                for k in predictors_info.keys(): data_dict[k] = field_to_zero_val[k]

            return pd.Series(data_dict)

        df_models = df_models_all.groupby("modelID").apply(get_model_info_one_model).reset_index(drop=True)

        print("saving")
        save_object(df_models, df_models_file)

    print("loading mod results")
    return load_object(df_models_file)

def get_df_models_with_pvalues_one_fitness_estimate_and_stress_condition(df_models_all, criterion_df_models_real):

    """Gets the df models for one fitness estimate and stress condition and returns a df with all real model results and pvalues"""

    # define the df with the real data and reshuffled one
    if criterion_df_models_real=="real_df": 
        df_models_real = df_models_all[df_models_all.reshuffle_y==False]
        df_models_random = df_models_all[df_models_all.reshuffle_y==True][["model_idx", "ntries_predictors", "mean_mean_r2", "tryID_general"]].set_index("model_idx")

    elif criterion_df_models_real=="first_reshuffle": 
        df_models_real = df_models_all[(df_models_all.reshuffle_y==True) & (df_models_all.tryID_general==0)]
        df_models_random = df_models_all[(df_models_all.reshuffle_y==True) & (df_models_all.tryID_general!=0)][["model_idx", "ntries_predictors", "mean_mean_r2", "tryID_general"]].set_index("model_idx")

    # define the number of reshuffles
    nreshuffles = len(set(df_models_random.tryID_general))

    # define the number of tries that were found in the real data
    sorted_n_tries = sorted(set(df_models_real.ntries_predictors))

    # define all model IDXs
    sorted_model_IDXs = sorted(set(df_models_real.model_idx))

    # log
    print(df_models_all.name, nreshuffles)

    #### DEFINE THE NORMAL PVAL #####

    # for a given model with some r2 and ntries_predictors, the fraction of reshuffles with that same model that have the r2>=obs r2 & ntries_predictors>=obs ntries_predictors

    # map each n try to the random predictors with a given min_n
    min_tries_to_df_random = {min_n : df_models_random[df_models_random.ntries_predictors>=min_n] for min_n in sorted_n_tries}

    # map each combination of n_tries and modelIDs to the r2 distribution
    df_models_real["comb_model_idx_ntries_predictors"] = df_models_real[["model_idx", "ntries_predictors"]].apply(tuple, axis=1)
    combinations_idx_and_n_tries = list(df_models_real.comb_model_idx_ntries_predictors.drop_duplicates())

    def get_mean_mean_r2s_for_one_c_idx_and_min_ntries(c):
        df = min_tries_to_df_random[c[1]]
        if c[0] in df.index: return df.loc[[c[0]], "mean_mean_r2"].values
        else: return np.array([])

    c_to_r2_distribution = dict(zip(combinations_idx_and_n_tries, map(get_mean_mean_r2s_for_one_c_idx_and_min_ntries, combinations_idx_and_n_tries)))

    # calculate the pval
    df_models_real["pval_resampling"] = df_models_real.apply(lambda r: sum(c_to_r2_distribution[r.comb_model_idx_ntries_predictors]>=r.mean_mean_r2), axis=1) / nreshuffles

    #################################

    ##### DEFINE MAXT PVAL #####

    # for a given model with some r2 and ntries_predictors, the fraction of reshuffles in which the maximum r2 of random models is >= the observed r2

    # map each min_n_tries to the maximum r2 across models in a given random reshuffle
    all_tryIDs = set(df_models_random.tryID_general)
    def get_distribution_max_r2_one_min_ntries_df(df_m):
        max_r2_distribution = df_m.groupby("tryID_general").apply(lambda df: max(df.mean_mean_r2)).values
        missing_tryIDs = all_tryIDs.difference(set(df_m.tryID_general))
        return np.append(max_r2_distribution, np.zeros(len(missing_tryIDs)))

    min_tries_to_max_r2_distribution = {min_n : get_distribution_max_r2_one_min_ntries_df(df) for min_n,df in min_tries_to_df_random.items()}

    # get the p value
    df_models_real["pval_maxT"] = df_models_real.apply(lambda r: sum(min_tries_to_max_r2_distribution[r.ntries_predictors]>=r.mean_mean_r2), axis=1) / nreshuffles

    ############################

    return df_models_real

def get_df_models_with_pvalues(df_models_all, ProcessedDataDir):

    """Gets the df_models with the p values"""

    # keep
    df_models_all = df_models_all.copy()

    # add fields that identifiy each model
    df_models_all["model_idx"] = ""
    model_ID_fields = ["fitness_estimate", "stress_condition", "model_name", "only_predictors_univariate_corr", "type_feature_selection", "tol_r2", "consider_interactions"] # "reshuffle_y", "tryID_general"
    for f in model_ID_fields: 
        df_models_all["model_idx"] = df_models_all.model_idx + "|" + df_models_all[f].apply(str)

    # get df 
    print("getting df real with the real pvalues")
    df_models_real = df_models_all.groupby(["fitness_estimate", "stress_condition"]).apply(get_df_models_with_pvalues_one_fitness_estimate_and_stress_condition, criterion_df_models_real="real_df").reset_index(drop=True)

    # get df with reshuffled values
    print("getting df with 1st reshuffles")
    df_models_1st_reshuffle = df_models_all.groupby(["fitness_estimate", "stress_condition"]).apply(get_df_models_with_pvalues_one_fitness_estimate_and_stress_condition, criterion_df_models_real="first_reshuffle").reset_index(drop=True)

    return df_models_real, df_models_1st_reshuffle

def get_prediction_results_stress_from_features(idx, df_data, binary_predictors, continuous_predictors, fitness_estimate, stress_condition, model_name, yfield, try_ID, only_predictors_univariate_corr, type_feature_selection, tol_r2, consider_interactions, reshuffle_y, tryID_general):


    """For certain features predict stress, returning information about the model"""

    start_time = time.time()

    # select predictors
    if (len(binary_predictors) + len(continuous_predictors))>0:

        # keep
        random.shuffle(binary_predictors)
        random.shuffle(continuous_predictors)

        # define the index
        print("Working on %s. %i predictors"%(idx, len(binary_predictors) + len(continuous_predictors)))

        # keep data
        X = df_data[continuous_predictors + binary_predictors].copy()
        y = df_data[yfield].copy()

        # define the important features
        if type_feature_selection=="forward_inHouse": 

            # inhouse forward selection
            selected_predictors = get_important_features_inHouse_forward_selection(model_name, X, y, tol_r2)

        elif type_feature_selection=="forward_sklearn": 

            # check that there are sufficient data
            if len(X.columns)>1:

                # sequential feat selection by sklearn
                sfs = SequentialFeatureSelector(get_predictor_model(model_name, X, y), n_features_to_select="auto", direction="forward", tol=tol_r2, cv=get_CVindices_per_lineage(X), scoring="r2")
                sfs.fit(X, y)    
                selected_predictors = list(X.columns[sfs.get_support()])

            else: selected_predictors = list(X.columns)

        elif type_feature_selection=="regression_tree_feature_importance":

            # base on fthe feature importance of the tree
            if not model_name in {"regresssion_tree_custom", "regression_tree_auto_ccp"}: raise ValueError("invalid %s"%model_name)
            model = get_predictor_model(model_name, X, y)
            model.fit(X, y)
            df_features = pd.DataFrame({"predictor":model.feature_names_in_, "feature_importance":model.feature_importances_})
            df_features = df_features[df_features.feature_importance>0].sort_values(by=["feature_importance", "predictor"], ascending=[False, True])
            selected_predictors = list(df_features.iloc[0:10].predictor)

        elif type_feature_selection=="rf_regressor_feature_importance":

            # based on feat importance of rf
            if not model_name in {"rf_regressor"}: raise ValueError("invalid %s"%model_name)

            model = get_predictor_model(model_name, X, y)
            model.fit(X, y)
            df_features = pd.DataFrame({"predictor":model.feature_names_in_, "feature_importance":model.feature_importances_})
            df_features = df_features[df_features.feature_importance>0].sort_values(by=["feature_importance", "predictor"], ascending=[False, True])
            selected_predictors = list(df_features.iloc[0:10].predictor)

        elif type_feature_selection=="sel_from_model":

            # selects features that are more important than the default
            model = get_predictor_model(model_name, X, y)
            sel = SelectFromModel(model)
            sel.fit(X, y)
            selected_predictors = X.columns[(sel.get_support())]

        elif type_feature_selection=="RFECV":

            # check that there are sufficient data
            if len(X.columns)>1:

                # Use RFECV for feature selection
                selector = RFECV(estimator=get_predictor_model(model_name, X, y), step=1, cv=get_CVindices_per_lineage(X), scoring="r2")
                selector = selector.fit(X, y)
                selected_predictors = X.columns[selector.support_]

            else: selected_predictors = list(X.columns)

        else: raise ValueError("unconsidered %s"%model_name)

    else: selected_predictors = []

    # calculate the r2 on an independent dataset
    if len(selected_predictors)>0:

        model_obj = get_predictor_model(model_name, X[selected_predictors].copy(), y.copy())
        cv_scores = cross_val_score(model_obj, X[selected_predictors].copy(), y.copy(), cv=get_CVindices_per_lineage(X[selected_predictors].copy()), scoring="r2")

        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)

    else:

        mean_score = 0
        std_score = 0

    # calc time
    elapsed_time = time.time() - start_time

    # get final series
    series_info = pd.Series({"idx":idx, "reshuffle_y":reshuffle_y, "tryID_general":tryID_general, "fitness_estimate":fitness_estimate, "stress_condition":stress_condition, "yfield":yfield, "model_name":model_name, "only_predictors_univariate_corr":only_predictors_univariate_corr, "type_feature_selection":type_feature_selection, "tol_r2":tol_r2, "consider_interactions":consider_interactions, "try_ID":try_ID, "mean_r2":mean_score, "std_r2":std_score, "selected_predictors":tuple(sorted(selected_predictors)), "elapsed_time":elapsed_time})

    print("%s finished. Took %.2f"%(idx, elapsed_time))

    return series_info


def check_that_AinF_FinA_samples_did_not_lose_mutations(df_all_haploid_vars_Ksiezopolska2021, Ksiezopolska2021_dir, df_new_vars, sampleNames_with_wgs, gene_features_df):

    """Check function"""

    # keeps
    df_new_vars = cp.deepcopy(df_new_vars)
    df_new_vars["unique_varID"] = df_new_vars.final_name + "-" + df_new_vars.variant
    df_new_vars = df_new_vars[~df_new_vars.variant.isin({"DEL", "DUP", "5' DEL", "TRA", "partial DUP", "loss DUP"})]
    df_new_vars = df_new_vars[~((df_new_vars.final_name=="ERG3") & (pd.isna(df_new_vars.variant)) & (df_new_vars.with_wgs))]

    # adds
    gID_to_final_name = dict(gene_features_df.set_index("gff_upmost_parent").final_name)
    gID_to_final_name["-"] = "-"
    df_all_haploid_vars_Ksiezopolska2021["final_name"] = df_all_haploid_vars_Ksiezopolska2021.Gene.apply(lambda x: gID_to_final_name[x])
    df_all_haploid_vars_Ksiezopolska2021["unique_varID"] = df_all_haploid_vars_Ksiezopolska2021.final_name + "-" + df_all_haploid_vars_Ksiezopolska2021.short_variant
    df_all_haploid_vars_Ksiezopolska2021 = df_all_haploid_vars_Ksiezopolska2021[["sampleID", "unique_varID", "#Uploaded_variation"]].drop_duplicates()
    df_all_haploid_vars_Ksiezopolska2021["noRun_sampleID"] = df_all_haploid_vars_Ksiezopolska2021.sampleID.apply(lambda x: "_".join(x.split("_")[1:]))
    df_all_haploid_vars_Ksiezopolska2021 = df_all_haploid_vars_Ksiezopolska2021[(df_all_haploid_vars_Ksiezopolska2021.noRun_sampleID.isin(sampleNames_with_wgs)) & ~(df_all_haploid_vars_Ksiezopolska2021.sampleID.isin(wrong_samples_Ksiezopolska2021))]
    if len(df_all_haploid_vars_Ksiezopolska2021.drop_duplicates("sampleID"))!=len(df_all_haploid_vars_Ksiezopolska2021.drop_duplicates(subset=["sampleID", "noRun_sampleID"])): raise ValueError("there should be 1-1 correspondance")
    sID_to_full_sID = df_all_haploid_vars_Ksiezopolska2021[["sampleID", "noRun_sampleID"]].drop_duplicates().set_index("noRun_sampleID").sampleID

    # iterate through each sample
    for condition, parent_condition in [("AinF", "ANI"), ("FinA", "FLZ")]:
        for sampleID in [s for s in sampleNames_with_wgs if s.endswith("_%s"%condition)]:
            print(sampleID)

            # get datasets of sample and parent
            parent_sampleID = "_".join(sampleID.split("_")[0:2]) + "_" + parent_condition
            if parent_sampleID not in sampleNames_with_wgs: raise ValueError("%s not in samples"%parent_sampleID)

            # define the mutations in the parent
            new_short_vars_in_parent = set(df_new_vars[df_new_vars.sampleID==parent_sampleID].unique_varID)
            df_all_haploid_vars_Ksiezopolska2021_parent = df_all_haploid_vars_Ksiezopolska2021[(df_all_haploid_vars_Ksiezopolska2021.noRun_sampleID==parent_sampleID) & (df_all_haploid_vars_Ksiezopolska2021.unique_varID.isin(new_short_vars_in_parent))]
            new_uploaded_variation_parent = set(df_all_haploid_vars_Ksiezopolska2021_parent["#Uploaded_variation"])
            if new_short_vars_in_parent!=set(df_all_haploid_vars_Ksiezopolska2021_parent.unique_varID): raise ValueError("vars should be the same")

            # get the df of the sampleID
            df_vars_sID = pd.read_csv("%s/VarCall/VarCallOutdirs/%s_VarCallresults/integrated_variants_norm_vcflib_ploidy1.tab"%(Ksiezopolska2021_dir, sID_to_full_sID[sampleID]), sep="\t", low_memory=False)
            missing_vars = new_uploaded_variation_parent.difference(set(df_vars_sID["#Uploaded_variation"]))
            if len(missing_vars)>0: raise ValueError("missing vars: %s / %s"%(missing_vars, new_uploaded_variation_parent))


def get_pvalue_tables_per_condition(df_stress_phenotypes, PlotsDir_paper):

    """Gets csv tables with the comparisons per strain and per condition"""

    # keep only the relevant rows
    print("loading")
    df_stress_phenotypes = df_stress_phenotypes[(df_stress_phenotypes.Treatment!="WT") & ((df_stress_phenotypes.type_measurement=="Q-PHAST_round1") | (df_stress_phenotypes.condition=="CycA"))].copy()
    df_stress_phenotypes["strongest_tradeoff_condition"] = ""

    # keep some fields
    relevant_fields = ["Treatment", "condition", "M_rAUC_norm", "M_AUC_norm_WT", "rAUC_norm_list", "AUC_norm_WT_list", "sampleID", "strongest_tradeoff_condition"]
    df_stress_phenotypes = df_stress_phenotypes[relevant_fields].copy()
    if len(df_stress_phenotypes)!=len(df_stress_phenotypes[["sampleID", "Treatment", "condition"]].drop_duplicates()): raise ValueError("non unique vals")

    # create df with per sample info
    def get_pvals_one_row(r):

        # define the fitness estimate
        if r.condition=="YPD": fitness_estimate = "AUC_norm_WT"
        else: fitness_estimate = "rAUC_norm"

        # add the means and sds
        data_dict = {"condition":r.condition, "Treatment":r.Treatment, "sampleID":r.sampleID}
        tradeoff_estimate = {"AUC_norm_WT":"AUC_R", "rAUC_norm":"fAUC_R"}[fitness_estimate]
        data_dict["tradeoff_estimate"] = tradeoff_estimate
        data_dict["M_tradeoff"] = r["M_%s"%fitness_estimate]

        # add the fraction of samples that have a sig difference from 1
        for threshold_val in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95]:

            # define the values to be compared to 1
            x = r["%s_list"%fitness_estimate]
            bool_dict = {True:1, False:0}
            data_dict["ttest_p_diff_from_1<0.05_and_diff_to_1>=%.2f"%threshold_val] = bool_dict[stats.ttest_1samp(x, 1, nan_policy='raise', alternative='two-sided')[1]<0.05 and abs(np.median(x)-1)>=threshold_val]

            # >1 / <1
            above_1_threshold = 1 + threshold_val
            data_dict["ttest_p_above_1<0.05_and_M>=%.2f"%above_1_threshold] = bool_dict[stats.ttest_1samp(x, 1, nan_policy='raise', alternative='greater')[1]<0.05 and np.median(x)>=above_1_threshold]

            below_1_threshold = 1 - threshold_val
            data_dict["ttest_p_below_1<0.05_and_M<=%.2f"%below_1_threshold] = bool_dict[stats.ttest_1samp(x, 1, nan_policy='raise', alternative='less')[1]<0.05 and np.median(x)<=below_1_threshold]

        return pd.Series(data_dict)

    df_pvals_per_sample = df_stress_phenotypes.apply(get_pvals_one_row, axis=1)
    df_pvals_per_sample.to_csv("%s/comparisons_per_sample.tsv"%PlotsDir_paper, index=False, header=True, sep="\t")

    print("getting per condition...")
    # add values for the 'at least 1 tradeoff'. For a given condition and sample, I will put the values with the most extreme (different from 1) fitness estimates
    def get_tradeoff_val_depending_on_cond(r):
        if r.condition=="YPD": return r.M_AUC_norm_WT
        else: return r.M_rAUC_norm

    def get_series_most_extreme_tradeoff_vals_one_sampleID_and_Treatment(df):
        if len(set(df.condition))!=len(df): raise ValueError("")
        #df["tradeoff_diff_to_1"] = 1 - df.apply(get_tradeoff_val_depending_on_cond, axis=1)
        df["tradeoff_val"] = df.apply(get_tradeoff_val_depending_on_cond, axis=1)

        strongest_tradeoff_r = dict(df.sort_values(by=["tradeoff_val"], ascending=True).iloc[0])
        strongest_tradeoff_r["strongest_tradeoff_condition"] = strongest_tradeoff_r["condition"]

        return pd.Series(strongest_tradeoff_r)

    for condition_types in ["noYPD", "noCycA_and_YPD"]: # "all", "noCycA",

        # keep some conditions
        all_conds = set(df_stress_phenotypes.condition)
        if condition_types=="all": target_conds = all_conds
        elif condition_types=="noCycA": target_conds = all_conds.difference({"CycA"})
        elif condition_types=="noYPD": target_conds = all_conds.difference({"YPD"})
        elif condition_types=="noCycA_and_YPD": target_conds = all_conds.difference({"CycA", "YPD"})
        else: raise ValueError("not valid")

        df_stress_phenotypes_atLeast1_tradeoff = df_stress_phenotypes[df_stress_phenotypes.condition.isin(target_conds)].copy().groupby(["sampleID", "Treatment"]).apply(get_series_most_extreme_tradeoff_vals_one_sampleID_and_Treatment)
        df_stress_phenotypes_atLeast1_tradeoff["condition"] = "strongest_tradeoff_%s"%condition_types

        df_stress_phenotypes = pd.concat([df_stress_phenotypes, df_stress_phenotypes_atLeast1_tradeoff[relevant_fields]]).copy().reset_index(drop=True)

    # create a df where each row is one combination of condition and Treatment, and contains all conditions
    def get_pvals_row_one_t_and_c(df):

        # checks
        if len(df)!=len(set(df.sampleID)): raise ValueError("sampleID should be unique")

        # define the fitness estimate
        condition, Treatment = df.name
        if condition=="YPD": fitness_estimate = "AUC_norm_WT"
        else: fitness_estimate = "rAUC_norm"

        # add the means and sds
        data_dict = {"condition":df.condition.iloc[0], "Treatment":df.Treatment.iloc[0]}
        tradeoff_estimate = {"AUC_norm_WT":"AUC_R", "rAUC_norm":"fAUC_R"}[fitness_estimate]
        data_dict["tradeoff_estimate"] = tradeoff_estimate

        for fn_object, fun_name in [(np.median, "median"), (np.mean, "mean"), (stats.median_abs_deviation, "MAD"), (np.std, "SD")]:
            data_dict["%s_tradeoff"%(fun_name)] = fn_object(df["M_%s"%fitness_estimate])

        # add the strongest tradeoff condition
        data_dict["strongest_tradeoff_conditions"] = ",".join([x for x in df.strongest_tradeoff_condition if x!=""])

        # add the p values of the ks statistic, that the median is different than 1
        statistic_general, p_value_general = stats.wilcoxon(df["M_%s"%fitness_estimate].values - 1, alternative='two-sided', nan_policy="raise", zero_method="zsplit")
        data_dict["wilcoxon_stat_diff_from_1"] = statistic_general
        data_dict["wilcoxon_p_diff_from_1"] = p_value_general

        # add the fraction of samples that have a sig difference from 1
        data_dict["fraction_ttest_p_diff_from_1<0.05"] = sum(df["%s_list"%fitness_estimate].apply(lambda x: stats.ttest_1samp(x, 1, nan_policy='raise', alternative='two-sided')[1]<0.05)) / len(df)


        # add the fraction of samples that have a sig difference from 1
        for threshold_val in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95]:

            # changes with threshold
            data_dict["fraction_ttest_p_diff_from_1<0.05_and_diff_to_1>=%.2f"%threshold_val] = sum(df["%s_list"%fitness_estimate].apply(lambda x: (stats.ttest_1samp(x, 1, nan_policy='raise', alternative='two-sided')[1]<0.05) and abs(np.median(x)-1)>=threshold_val)) / len(df)

            # >1 / <1
            above_1_threshold = 1 + threshold_val
            data_dict["fraction_ttest_p_above_1<0.05_and_M>=%.2f"%above_1_threshold] = sum(df["%s_list"%fitness_estimate].apply(lambda x: (stats.ttest_1samp(x, 1, nan_policy='raise', alternative='greater')[1]<0.05) and np.median(x)>=above_1_threshold)) / len(df)

            below_1_threshold = 1 - threshold_val
            data_dict["fraction_ttest_p_below_1<0.05_and_M<=%.2f"%below_1_threshold] = sum(df["%s_list"%fitness_estimate].apply(lambda x: (stats.ttest_1samp(x, 1, nan_policy='raise', alternative='less')[1]<0.05) and np.median(x)<=below_1_threshold)) / len(df)

        return pd.Series(data_dict)

    df_pvals = df_stress_phenotypes.groupby(["condition", "Treatment"]).apply(get_pvals_row_one_t_and_c)
    df_pvals.to_csv("%s/comparisons_per_condition.tsv"%PlotsDir_paper, index=False, header=True, sep="\t")

    return df_pvals


def check_univariate_associations(df_data_all, df_feature_info, ProcessedDataDir):

    """Check features that could be associated to phenotypes in a univariant way"""

    df_univariate_associations_file = "%s/df_univariate_associations.py"%ProcessedDataDir
    if file_is_empty(df_univariate_associations_file):

        df_data_all = df_data_all.copy()
        df_feature_info = df_feature_info.copy()
        p_to_tf = dict(df_feature_info.set_index("feature").type_feature)

        # keep interesting predictors
        #df_feature_info = df_feature_info[~(df_feature_info.type_feature.isin({"stress_fitness", "Ksiezopolska2021_fitness_conc0", "Ksiezopolska2021_susceptibility"})) & ~(df_feature_info.type_feature.apply(lambda x: x.endswith("_hot_encoded"))) & ~(df_feature_info.feature.apply(lambda x: x.startswith("MICs_") or x.startswith("log2_MICs_")))]

        # remove interactions
        #df_data_all = df_data_all[[c for c in df_data_all.columns if not (c in p_to_tf and p_to_tf[c].startswith("pairwise_interaction "))]]

        # for each stress keep adding
        associations_info = {}; Ia=0
        for stress_condition in ['CFW', 'CR', 'DTT', 'H2O2', 'NaCl', 'SDS', 'YPD', 'CycA']:
            for dependent_f in ["M_rAUC_norm", "M_rAUC_diff", "M_AUC_norm_WT", "M_AUC_diff_WT"]: #["M_AUC", 'M_fAUC_norm', 'M_fAUC_diff', 'M_rAUC_norm', 'M_rAUC_diff', 'M_AUC_norm_WT', 'M_AUC_diff_WT']: # "M_AUC_diff_WT_M_AUC", "M_AUC_norm_WT_M_AUC", "M_fAUC_diff_WT_M_fAUC", "M_fAUC_norm_WT_M_fAUC", "M_fAUC"

                # discard some cases
                if stress_condition=="YPD" and not dependent_f in {"M_AUC_norm_WT", "M_AUC_diff_WT"}: continue

                # define field
                yfield = "%s_%s"%(dependent_f, stress_condition)
                print(stress_condition, yfield)

                # check each feature
                #for I,r in df_feature_info[~df_feature_info.type_feature.apply(lambda x: x.startswith("pairwise_interaction "))].iterrows():
                for I,r in df_feature_info.iterrows():

                    # spearman correlation for numeric data
                    if r.data_type=="continuous":
                        statistic_type = "r_spearman"
                        statistic_value, pval = stats.spearmanr(df_data_all[r.feature], df_data_all[yfield], nan_policy="raise")

                    # KS test for non-binary data
                    elif r.data_type=="binary":
                        statistic_type = "KS_statistic"

                        # check that there is sufficient data
                        val_to_n_lineages = df_data_all.groupby(r.feature).apply(lambda df_v: len(set(df_v.exp_evol_lineage)))

                        # discard features where there is not sufficient data (all values should have multiple lineages)
                        if not all(val_to_n_lineages>=2): 
                            statistic_value = -1
                            pval = 1.0

                        else:

                            # perform test
                            values_0s = df_data_all[df_data_all[r.feature]==0][yfield].values
                            values_1s = df_data_all[df_data_all[r.feature]==1][yfield].values
                            if (len(values_0s)+len(values_1s))!=len(df_data_all): raise ValueError("0 and 1 should add up")
                            statistic_value, pval = stats.ks_2samp(values_0s, values_1s, alternative="two-sided", method="auto") # auto for small arrays


                    # Kruskal wallis test for cathegoric data
                    elif r.data_type=="cathegoric":
                        statistic_type = "KruskalWallis_statistic"

                        # check that there is sufficient data
                        val_to_n_lineages = df_data_all.groupby(r.feature).apply(lambda df_v: len(set(df_v.exp_evol_lineage)))

                        # there should be at least two conditions with >=2 lineages
                        if sum(val_to_n_lineages>=2)<2: 
                            statistic_value = -1
                            pval = 1.0

                        else:
                        
                            # perform test
                            data_kw = list(df_data_all.groupby(r.feature).apply(lambda df_v: df_v[yfield].values))
                            statistic_value, pval = stats.kruskal(*data_kw, nan_policy='raise')

                    else: raise ValueError("%s"%r.data_type)

                    # indicate if this is a field used for preditions later
                    potential_WT_predictors = {"M_rAUC_norm": {"WT_M_AUC", "WT_M_fAUC_norm"},
                           "M_rAUC_diff": {"WT_M_AUC", "WT_M_fAUC_diff"},
                           "M_AUC_norm_WT": {"WT_M_AUC"},
                           "M_AUC_diff_WT": {"WT_M_AUC"}}[dependent_f]

                    potential_WT_predictors = {"%s_%s"%(p, stress_condition) for p in potential_WT_predictors}
                    potential_WT_predictors.add("WT_M_AUC_YPD") # add the WT fitness with no drug

                    if r.feature in potential_WT_predictors or r.type_feature in {"broad_resistance_genes_profile_hot_encoded", "WT_info_hot_encoded", "single_variants", "domain_info", "pathway_info"}: field_for_pred = True
                    else: field_for_pred = False

                    # get the r2 of a linear model using this
                    if r.data_type in {"continuous", "binary"} and field_for_pred is True:

                        X = df_data_all[[r.feature]].copy()
                        y = df_data_all[yfield].copy()
                        mean_r2_lin_model = np.mean(cross_val_score(LinearRegression(), X, y, cv=get_CVindices_per_lineage(X), scoring="r2"))

                    else: mean_r2_lin_model = 0.0

                    # keep
                    associations_info[Ia] = {"stress_condition":stress_condition, "yfield":yfield, "statistic_type":statistic_type, "statistic_value":statistic_value, "pval":pval, "field_for_pred":field_for_pred, "mean_r2_lin_model":mean_r2_lin_model}
                    for k in r.keys(): associations_info[Ia][k] = r[k]
                    Ia += 1

        df_associations = pd.DataFrame(associations_info).transpose()

        print("saving")
        save_object(df_associations, df_univariate_associations_file)

    # load
    df_associations = load_object(df_univariate_associations_file)

    # add corrected p vals
    df_associations["pval_fdr"] = multitest.fdrcorrection(df_associations.pval)[1]
    df_associations["pval_bonferroni"] = multitest.multipletests(df_associations.pval, alpha=0.05, method="bonferroni")[1]



    return df_associations

def plot_univariate_significant_associations(df_assoc, title_str, df_data, PlotsDir):

    """One plot for each association"""

    """
    ['Ksiezopolska2021_fitness_conc0_WT',
     'Ksiezopolska2021_susceptibility_WT',
     'WT_info',
     'broad_resistance_genes_profile',
     'single_variants']

    """
    df_data = df_data.copy()
    df_assoc = df_assoc.copy()


    # plot all WT continuous information
    df_assoc_WT_continuous = df_assoc[(df_assoc.type_feature.isin({"WT_info", "Ksiezopolska2021_fitness_conc0_WT", "Ksiezopolska2021_susceptibility_WT"})) & (df_assoc.feature!="strain")]

    if len(df_assoc_WT_continuous)>0:

        predictor_features  = sorted(set(df_assoc_WT_continuous.feature))
        y_fields = sorted(set(df_assoc_WT_continuous.yfield))

        nrows = len(predictor_features)
        ncols = len(y_fields)

        fig = plt.figure(figsize=(ncols*3, nrows*3)); Ip = 1

        for Ir,feature in enumerate(predictor_features):
            for Ic,yfield in enumerate(y_fields):

                ax = plt.subplot(nrows, ncols, Ip); Ip+=1

                ax = sns.scatterplot(data=df_data, x=feature, y=yfield, hue="strain", style="strain")


                # if it is significant add scatterplot
                df_as = df_assoc[(df_assoc.feature==feature) & (df_assoc.yfield==yfield)]
                if len(df_as)>0:

                    #sns.regplot(x=df_data[feature].values, y=df_data[yfield].values, ci=None, s=0)  # ci=None disables the confidence interval

                    x = df_data[feature].values
                    y = df_data[yfield].values
                    regression_line = np.polyfit(x, y, 1)
                    x_values = np.array([min(x), max(x)])
                    y_values = np.polyval(regression_line, x_values)
                    sns.lineplot(x=x_values, y=y_values, color='black')

                    # ax.set_title()


                ax.set_xlabel(feature.replace("_Ksiezopolska2021", ""))

                if (Ic+1)!=ncols or Ir!=0: ax.get_legend().remove()
                else: ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))

        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        fig.savefig("%s/WT_continuous_vars_associations_%s.pdf"%(PlotsDir, title_str),  bbox_inches='tight')

    # plot all binary / cathegoric info
    df_assoc_cathegoric = df_assoc[(df_assoc.data_type.isin({"binary", "cathegoric"})) & (df_assoc.feature!="strain")]

    if len(df_assoc_cathegoric)>0:


        predictor_features  = sorted(set(df_assoc_cathegoric.feature))
        y_fields = sorted(set(df_assoc_cathegoric.yfield))

        nrows = len(predictor_features)
        ncols = len(y_fields)

        fig = plt.figure(figsize=(ncols*3, nrows*3)); Ip = 1

        for Ir,feature in enumerate(predictor_features):
            for Ic,yfield in enumerate(y_fields):

                ax = plt.subplot(nrows, ncols, Ip); Ip+=1
                df_as = df_assoc[(df_assoc.feature==feature) & (df_assoc.yfield==yfield)]
                if len(df_as)==0: continue
                print(feature, yfield)

                ax = sns.boxplot(data=df_data, x=feature, y=yfield)
                ax = sns.swarmplot(data=df_data, x=feature, y=yfield, color="none", edgecolor="black", linewidth=0.9)

                ax.set_ylim([0, 1.7])

                # if it is significant add scatterplot
                if len(df_as)>0:

                    ax.set_title("p=%.3f, pFDR=%.3f"%(df_as.pval.iloc[0], df_as.pval_fdr.iloc[0]))

                ax.set_xlabel(feature.replace("resistance_genes_broad_", "rgenes_broad_"))
                
                if feature in {"resistance_genes_broad_FKS_profile", "resistance_genes_broad_ERG11_chrE_profile"}: ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=5)

        plt.subplots_adjust(wspace=0.6, hspace=0.6)
        fig.savefig("%s/cathegoric_vars_associations_%s.pdf"%(PlotsDir, title_str),  bbox_inches='tight')


def get_jaccard_index_two_arrays(array1, array2):

    """Jaccard index two indices"""


    series1 = pd.Series(array1)
    set1 = set(series1[series1==1].index)

    series2 = pd.Series(array2)
    set2 = set(series2[series2==1].index)

    np.seterr(divide='ignore', invalid='ignore')
    return np.divide(len(set1.intersection(set2)), len(set1.union(set2)))


def heatmap_correlations_univariate_binary_predictors(df_data_all, df_feature_info, PlotsDir, df_univariate_associations):

    """Plots a heatmap with the jaccard distances across different cases"""

    # keep
    df_data_all = df_data_all.copy()
    df_feature_info = df_feature_info.copy()
    df_univariate_associations = df_univariate_associations.copy()

    # filter
    def get_is_interaction(x): return x.startswith("pairwise_interaction")
    df_feature_info = df_feature_info[(df_feature_info.data_type=="binary") & (df_feature_info.n_lineages_with_minor_cathegory_binary_features>=2) & (df_feature_info.type_feature.isin({"broad_resistance_genes_profile_hot_encoded", "WT_info_hot_encoded", "single_variants", "domain_info", "pathway_info"}))]

    for yfield in sorted(set(df_univariate_associations.yfield)): 
        df_assoc = df_univariate_associations[(df_univariate_associations.yfield==yfield) & (df_univariate_associations.feature.isin(set(df_feature_info.feature))) & (df_univariate_associations.pval<0.05)]
        all_feats = sorted(df_assoc.feature)
        print(yfield, len(all_feats), "binary associations p_KS<0.05")

        # get matrix
        #print("get mat")
        correlation_matrix = df_data_all[all_feats].corr(method=get_jaccard_index_two_arrays)

        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

        """
        high_corr_pairs= []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i, j]) >= 0.75:
                    high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))

        # Print the pairs
        for pair in high_corr_pairs:
            print(f"{pair[0]} - {pair[1]} : {correlation_matrix.loc[pair[0], pair[1]]}")

        """

        # Set up the matplotlib figure
        #print("get heatmap")
        fig = plt.figure(figsize=(8, 6))

        # Draw the heatmap with the mask and correct aspect ratio
        ax = sns.heatmap(correlation_matrix, mask=mask, cmap="coolwarm", vmax=1, vmin=0, square=True, annot=False, fmt=".2f", yticklabels=False, xticklabels=False, cbar_kws={"shrink": 0.7}) # mt=".2f"
        ax.set_xlabel("Binary features associated with %s"%yfield)
        ax.set_ylabel("Binary features associated with %s"%yfield)

        # plt.title("Jaccard index heatmap. y=%s"%yfield)

        cbar = ax.collections[0].colorbar
        cbar.set_label("Jaccard index")
        plots_dir = "%s/jaccard_distances_binary_sig_predictors"%PlotsDir; make_folder(plots_dir)
        fig.savefig("%s/jaccard_heatmap_%s.pdf"%(plots_dir, yfield), bbox_inches='tight')

        plt.close(fig)

      


def heatmap_correlations_WT_continuous_vars(df_data_all, df_feature_info, PlotsDir):

    """Plots correlations between vars in WT data"""

    # keep
    df_data_all = df_data_all.copy()
    df_feature_info = df_feature_info.copy()

    # define 
    redundant_WT_predictors_max_corr_75 = {"WT_M_fAUC_NaCl", "WT_M_fAUC_H2O2", "WT_M_AUC_SDS", "WT_M_fAUC_CFW", "WT_M_AUC_DTT"}

    # define interesting features (WT related and continuous)
    #df_feature_info = df_feature_info[(df_feature_info.type_feature.isin({"WT_info", "Ksiezopolska2021_fitness_conc0_WT", "Ksiezopolska2021_susceptibility_WT"})) & (df_feature_info.feature!="strain") & ~(df_feature_info.feature.apply(lambda x: x.startswith("MICs_") or x.startswith("log2_MICs_"))) & (df_feature_info.feature!="WT_M_fAUC_YPD")]
    df_feature_info = df_feature_info[df_feature_info.feature_is_WT_continuous_info]
    df_feature_info = df_feature_info[~df_feature_info.feature.isin(redundant_WT_predictors_max_corr_75)] # try to remove redundant predictors
    interesting_features = list(df_feature_info.feature)

    df_data_all = df_data_all[interesting_features + ["strain"]].drop_duplicates(subset=["strain"], keep="first").set_index("strain")

    # plot pairplot
    # fig = sns.pairplot(df_data_all)

    # get matrucx
    correlation_matrix = df_data_all.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    high_corr_pairs= []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) >= 0.75:
                high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))

    # Print the pairs
    for pair in high_corr_pairs:
        print(f"{pair[0]} - {pair[1]} : {correlation_matrix.loc[pair[0], pair[1]]}")


    # Set up the matplotlib figure
    fig = plt.figure(figsize=(12, 10))

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(correlation_matrix, mask=mask, cmap="coolwarm", vmax=1, vmin=-1, square=True, annot=True, fmt=".2f") # mt=".2f"
    plt.title("Pearson Correlation Heatmap")
    plt.show()
    fig.savefig("%s/features_WT_continuous_info_correlation.pdf"%PlotsDir, bbox_inches='tight')


def plot_sig_results_stress_condition_old(df_sig_models, stress_condition, top_n, plots_dir, df_feature_info, df_data_all, df_univariate_associations):

    """Plots, for a given set of sig models"""

    # keep
    df_sig_models = df_sig_models.copy()
    df_feature_info = df_feature_info.copy().set_index("feature", drop=False)[["feature", "type_feature", "data_type", "description"]]
    df_data_all = df_data_all.copy()
    df_univariate_associations = df_univariate_associations.copy()

    # change
    df_sig_models["mean_mean_r2"] = df_sig_models.mean_mean_r2.apply(lambda x: round(x, 2))

    # define some feats
    all_binary_features = set(df_feature_info[df_feature_info.data_type=="binary"].feature)
    all_continuous_features = set(df_feature_info[df_feature_info.data_type=="continuous"].feature)
      
    # go through each y val
    for fitness_estimate in sorted(set(df_sig_models.fitness_estimate)):

        # get models
        df_all = df_sig_models[df_sig_models.fitness_estimate==fitness_estimate].sort_values(["fitness_estimate", "pval_maxT", "mean_mean_r2", "ntries_predictors"], ascending=[True, True, False, False]).iloc[0:top_n].drop_duplicates(subset=("selected_predictors"), keep="first")

        # define field
        fitness_y = "%s_%s"%(fitness_estimate, stress_condition)

        # one plot for each model
        for Iidx,r in df_all.iterrows():

            # define all the predictors of this model (summarizing interactions)
            all_ps = []
            for p in r.selected_predictors:
                if " * " in p: all_ps += p.split(" * ")
                else: all_ps.append(p)
            #all_ps = all_ps)

            # define the type data
            df_feats = df_feature_info.loc[all_ps]

            # add the correlations
            df_assoc = df_univariate_associations[(df_univariate_associations.yfield==fitness_y) & (df_univariate_associations.feature.isin(all_ps))][["statistic_value", "feature", "pval"]]
            df_feats = df_feats.reset_index(drop=True).merge(df_assoc, on="feature", validate="one_to_one", how="left").set_index("feature", drop=False)

            # get as tuple
            predictors_tuple = tuple(all_ps)
            type_feature_tuple = tuple(sorted(df_feats.type_feature))

            # initialize with empty fields, for debuging
            xfield = None 
            hue_field = None
            marker_field = None
            col_field = None
            row_field = None
            palette = None 

            # define some special cases
            if predictors_tuple==('GO_BP_GO:0044182_truncation', 'MetaCyc_PWY-6807_miss', 'WT_M_fAUC_norm_H2O2', 'WT_M_AUC_YPD'):

                xfield = "WT_M_AUC_YPD" 
                hue_field = "WT_M_fAUC_norm_H2O2"
                marker_field = 'GO_BP_GO:0044182_truncation'
                col_field = 'MetaCyc_PWY-6807_miss'
                row_field = None
                palette = "rocket_r"
                aspect = 1
                height = 3

            elif predictors_tuple==('GO_CC_GO:0031224_truncation', 'GO_CC_GO:0031227_miss', 'PDR1_miss_280-339', 'presence_variant-FKS2-mis|p.662|L/W'):

                xfield = "PDR1_miss_280-339" 
                hue_field = "presence_variant-FKS2-mis|p.662|L/W"
                marker_field = 'GO_CC_GO:0031224_truncation'
                col_field = 'GO_CC_GO:0031227_miss'
                row_field = None
                palette = None
                aspect = 1
                height = 3

            elif predictors_tuple==('resistance_genes_broad_ERG11_chrE_profile|||ChrE dup', 'MetaCyc_PWY-7921_miss', 'resistance_genes_broad_FKS_profile|||FKS1 truncation & FKS2 miss', 'strain|||CBS138'):

                xfield = "resistance_genes_broad_FKS_profile" 
                col_field = "resistance_genes_broad_ERG11_chrE_profile"
                row_field = None
                marker_field = "MetaCyc_PWY-7921_miss"
                hue_field = "strain"
                palette = "tab10"
                aspect = 1
                height = 3.5

            # cases by type of data
            elif type_feature_tuple==('domain_info', 'pathway_info'):

                xfield = df_feats[df_feats.type_feature=="domain_info"].feature.iloc[0]
                marker_field = None
                col_field = df_feats[df_feats.type_feature=="pathway_info"].feature.iloc[0]
                row_field = None
                hue_field = "condition"
                palette = condition_to_color               
                aspect = 1
                height = 3

            elif type_feature_tuple==("pathway_info",):

                xfield = predictors_tuple[0]
                marker_field = "condition"
                col_field = None
                row_field = None
                hue_field = "condition"
                palette = condition_to_color               
                aspect = 1
                height = 3

            elif type_feature_tuple==("WT_info",):

                xfield = predictors_tuple[0]
                marker_field = "strain"
                col_field = None
                row_field = None
                hue_field = "strain"
                palette = "tab10"               
                aspect = 1
                height = 3

            # cases with all binaries that are domains, pathways or single variants (auto)
            elif all([df_feats.loc[p, "data_type"]=="binary" for p in set(all_ps)]):

                df_feats = df_feats.reset_index(drop=True).sort_values(by=["pval", "statistic_value", "feature"], ascending=[True, False, True]).reset_index(drop=True)

                for I, row in df_feats.iterrows():

                    if I==0: xfield = row.feature
                    elif I==1: hue_field = row.feature
                    elif I==2: col_field = row.feature
                    elif I==3: marker_field = row.feature
                    elif I==4: row_field = row.feature
                    elif I>=5: print("WARNING: feature %s not included"%row.feature)

                palette = None  
                aspect = 1
                if row_field is None: height = 3
                else: height = 4

                df_feats = df_feats.set_index("feature", drop=False)

            # cases with one WT info and two binary predictors
            elif sum(df_feats.type_feature=="WT_info")==1 and all(df_feats[df_feats.type_feature!="WT_info"].data_type=="binary"):


                # the WT should be the case
                xfield = df_feats[df_feats.type_feature=="WT_info"].feature.iloc[0]

                df_f = df_feats[df_feats.type_feature!="WT_info"].reset_index(drop=True).sort_values(by=["pval", "statistic_value", "feature"], ascending=[True, False, True]).reset_index(drop=True)

                for I, row in df_f.iterrows():

                    if I==0: hue_field = row.feature
                    elif I==1: col_field = row.feature
                    elif I==2: marker_field = row.feature
                    elif I==3: row_field = row.feature
                    elif I>=4: print("WARNING: feature %s not included"%row.feature)

                palette = None  
                aspect = 1
                if row_field is None: height = 3
                else: row_field: height = 4

            else: raise ValueError("%s not considered. %s"%(predictors_tuple, df_feats.type_feature))

            if xfield is None: raise ValueError("xfield can't be none")        

            # keep
            df_plot = df_data_all.copy()

            # change x if it is cathegoric
            if df_feature_info.loc[xfield, "data_type"]=="cathegoric":
                all_xvals = sorted(set(df_plot[xfield]))
                xval_to_I = dict(zip(all_xvals, range(len(all_xvals))))
                df_plot[xfield] = df_plot[xfield].map(xval_to_I)
            
            # add jitter
            range_x = max(df_plot[xfield]) - min(df_plot[xfield])
            df_plot[xfield] = df_plot[xfield] + np.random.uniform(-range_x*0.05, range_x*0.05, len(df_plot))
            
            # make plot
            g = sns.relplot(kind="scatter", data=df_plot, hue=hue_field, x=xfield, y=fitness_y, row=row_field, col=col_field, aspect=aspect, height=height, palette=palette, edgecolor="black", s=25, style=marker_field)

            # define the pathway fields
            pathway_fields = list(df_feats[df_feats.type_feature=="pathway_info"].feature)

            # change axes
            nrows, ncols = g.axes.shape
            for row in range(g.axes.shape[0]):
                for col in range(g.axes.shape[1]):
                    ax = g.axes[row, col]

                    # change the text for the 1st row indicating the text
                    if row==0 and col==0:

                        if len(pathway_fields)>0: text_descriptions = "Pathway descriptions:\n" + "\n".join(["%s: %s"%(p, df_feats.loc[p, "description"]) for p in pathway_fields])
                        else: text_descriptions = ""

                        text_predictors = "%s ~ %s"%(fitness_y, " + ".join(["'%s'"%p for p in r.selected_predictors]))
                        text_model = ", ".join(["%s=%s"%(f, r[f]) for f in ["model_name", "only_predictors_univariate_corr", "type_feature_selection", 'tol_r2', 'consider_interactions', 'mean_mean_r2', 'ntries_predictors']])

                        ax.text(0, 1.5 + 0.2*len(pathway_fields), "%s\n%s\n\n%s"%(text_predictors, text_model, text_descriptions), transform=ax.transAxes, ha="left", va="top")

                    # change title in some cases
                    if not col_field is None and df_feature_info.loc[col_field, "type_feature"]=="broad_resistance_genes_profile":
                        tit = ax.get_title()
                        ax.set_title("%s=\n%s"%(tit.split("=")[0], tit.split("=")[1]), size=10)

                    # change x vals in some cases
                    if df_feature_info.loc[xfield, "type_feature"]=="broad_resistance_genes_profile":
                        ax.set_xticks([xval_to_I[x] for x in all_xvals])
                        ax.set_xticklabels(all_xvals, rotation=90)



            # reposition legend
            g.legend.set_bbox_to_anchor((1., 0.5))


            # show and save
            plt.show()
            g.savefig("%s/%s_from_%s.pdf"%(plots_dir, fitness_y, "|".join([f.replace("/", "-") for f in r.selected_predictors])))


def plot_sig_results_stress_condition(df_sig_models, stress_condition, top_n, plots_dir, df_feature_info, df_data_all, df_univariate_associations):

    """Plots, for a given set of sig models"""

    # keep
    make_folder(plots_dir)
    df_sig_models = df_sig_models.copy()
    df_feature_info = df_feature_info.copy().set_index("feature", drop=False)[["feature", "type_feature", "data_type", "description"]]
    df_data_all = df_data_all.copy()
    df_univariate_associations = df_univariate_associations.copy()

    # change
    df_sig_models["mean_mean_r2"] = df_sig_models.mean_mean_r2.apply(lambda x: round(x, 2))

    # define some feats
    all_binary_features = set(df_feature_info[df_feature_info.data_type=="binary"].feature)
    all_continuous_features = set(df_feature_info[df_feature_info.data_type=="continuous"].feature)

    # add missing predictors
    missing_feats = set.union(*df_sig_models.selected_predictors.apply(set)).difference(set(df_feature_info.feature))
    for f in missing_feats:
        if not " * " in f: raise ValueError("missing feats should only have *")
        f1, f2 = f.split(" * ")
        df_data_all[f] = df_data_all[f1] * df_data_all[f2]
        
    # go through each y val
    for fitness_estimate in sorted(set(df_sig_models.fitness_estimate)):

        #if not "rAUC_norm" in fitness_estimate: continue # debug

        # get models
        df_all = df_sig_models[df_sig_models.fitness_estimate==fitness_estimate].sort_values(["fitness_estimate", "mean_mean_r2", "ntries_predictors"], ascending=[True, False, False]).iloc[0:top_n].drop_duplicates(subset=("selected_predictors"), keep="first")

        # define field
        fitness_y = "%s_%s"%(fitness_estimate, stress_condition)

        # one plot for each model
        for Iidx,r in df_all.iterrows():
            print(fitness_y, r.selected_predictors)

            # define all the predictors of this model (summarizing interactions)
            all_ps = []
            for p in r.selected_predictors:
                if " * " in p: all_ps += p.split(" * ")
                else: all_ps.append(p)

            df_feats = df_feature_info.loc[all_ps]

            # get the predicted y according to the model
            X = df_data_all[list(r.selected_predictors)].copy()
            y = df_data_all[fitness_y].copy()
            model_obj = get_predictor_model(r.model_name, X, y)
            model_obj.fit(X, y)
            y_pred = model_obj.predict(X)


            # plot
            fig = plt.figure(figsize=(3,3))

            ax = plt.scatter(y_pred, y.values, color="none", edgecolor="blue")

            min_v = min([min(y.values), min(y_pred)]) - 0.1
            max_v = max([max(y.values), max(y_pred)]) + 0.1

            plt.plot([min_v, max_v], [min_v, max_v], color="gray", linewidth=.8, linestyle="--")
            plt.xlim([min_v, max_v])
            plt.ylim([min_v, max_v])

            plt.ylabel(fitness_y)
            plt.xlabel("predicted %s"%fitness_y)
            plt.title("r2=%.2f, tries=%i, %s"%(r.mean_mean_r2, r.ntries_predictors, r.model_name))

            plots_dir_pred = "%s_predictions"%plots_dir; make_folder(plots_dir_pred)
            fig.savefig("%s/model_%i_%s.pdf"%(plots_dir_pred, Iidx, fitness_y), bbox_inches="tight")
            plt.close(fig)
            
            # print tree
            if r.model_name in {"regresssion_tree_custom", "regression_tree_auto_ccp"}:

                from sklearn import tree

                fig = plt.figure(figsize=(25,20))
                tree.plot_tree(model_obj, feature_names=list(X.columns), filled=True)
                fig.savefig("%s/model_%i_%s_tree.pdf"%(plots_dir_pred, Iidx, fitness_y), bbox_inches="tight")
                plt.close(fig)

                text_representation = tree.export_text(model_obj, feature_names=list(X.columns))
                #print(text_representation)

            # define feature importance based on a regression tree (to prioritize)
            tree_obj = DecisionTreeRegressor()
            tree_obj.fit(df_data_all[all_ps].copy(), y)
            f_to_importance = dict(zip(tree_obj.feature_names_in_, tree_obj.feature_importances_))
            df_feats["feature_importance"] = df_feats.feature.map(f_to_importance)

            # sort by importance
            df_feats = df_feats.reset_index(drop=True).sort_values(by=["feature_importance", "feature"], ascending=[False, True]).set_index("feature", drop=False)

            # get as tuple
            predictors_tuple = tuple(all_ps)
            type_feature_tuple = tuple(sorted(df_feats.type_feature))

            # initialize with empty fields, for debuging
            xfield = None 
            hue_field = None
            marker_field = None
            col_field = None
            row_field = None
            palette = None 
            skiped_feats = []

            # define some special cases
            if predictors_tuple==('GO_BP_GO:0044182_truncation', 'MetaCyc_PWY-6807_miss', 'WT_M_fAUC_norm_H2O2', 'WT_M_AUC_YPD'):

                xfield = "WT_M_AUC_YPD" 
                hue_field = "WT_M_fAUC_norm_H2O2"
                marker_field = 'GO_BP_GO:0044182_truncation'
                col_field = 'MetaCyc_PWY-6807_miss'
                row_field = None


            elif predictors_tuple==('WT_M_fAUC_norm_H2O2', 'WT_M_AUC_YPD'):

                xfield = "WT_M_AUC_YPD" 
                hue_field = "WT_M_fAUC_norm_H2O2"
                marker_field = None
                col_field = None
                row_field = None

            # automatic, based on feature importance
            else:

                # set iteratively depending on the data type
                if sum(df_feats.data_type=="continuous")==1: 

                    xfield = df_feats[df_feats.data_type=="continuous"].feature.iloc[0]
                    for I, (feat, row) in enumerate(df_feats[df_feats.data_type!="continuous"].iterrows()):

                        if I==0: hue_field = row.feature
                        elif I==1: col_field = row.feature
                        elif I==2: marker_field = row.feature
                        elif I==3: row_field = row.feature
                        elif I>=4: skiped_feats.append(row.feature)

                elif sum(df_feats.data_type=="continuous")>1: raise ValueError("auto not thout for this. %s"%str(predictors_tuple))

                else:

                    for I, (feat, row) in enumerate(df_feats.iterrows()):

                        if I==0: xfield = row.feature
                        elif I==1: hue_field = row.feature
                        elif I==2: col_field = row.feature
                        elif I==3: marker_field = row.feature
                        elif I==4: row_field = row.feature
                        elif I>=5: skiped_feats.append(row.feature)

                palette = None  

            # redefine some
            if hue_field is None:
                if "strain" in xfield: hue_field = "strain"
                elif df_feats.loc[xfield, "type_feature"]=="WT_info": hue_field = "strain"
                elif df_feats.loc[xfield, "type_feature"] in {"pathway_info", "domain_info", "broad_resistance_genes_profile_hot_encoded"}: hue_field = "condition"
                else: raise ValueError("unconsidered situation. xfield=%s"%xfield)

            # define palette
            if hue_field=="condition": palette = condition_to_color
            elif hue_field=="strain": palette = "tab10"
            elif df_feats.loc[hue_field, "data_type"]=="continuous": palette = "rocket_r"

            # set some parameters, based on the xfield
            aspect = 1
            if row_field is None: height = 3.2
            else: height = 3.2

            # keep
            df_plot = df_data_all.copy()

            # change x if it is cathegoric
            all_xvals = sorted(set(df_plot[xfield]))
            xval_to_I = dict(zip(all_xvals, range(len(all_xvals))))
            df_plot[xfield] = df_plot[xfield].map(xval_to_I)

            # if len(skiped_feats)==0: continue

            # add jitter based on hue
            if not hue_field is None and len(df_plot[[hue_field, xfield]].drop_duplicates())!=len(set(df_plot[xfield])): 
                all_hue_vals = sorted(set(df_plot[hue_field]))
                array_offsets = np.linspace(-0.2, 0.2, len(all_hue_vals))
                huev_to_offset = dict(zip(all_hue_vals, array_offsets))
                df_plot[xfield] = df_plot.apply(lambda x: x[xfield] + huev_to_offset[x[hue_field]], axis=1)
                distance_hues = array_offsets[1] - array_offsets[0]

            else: distance_hues = 1

            # add jitter
            range_x = max(df_plot[xfield]) - min(df_plot[xfield])
            df_plot[xfield] = df_plot[xfield] + np.random.uniform(-distance_hues*0.2, distance_hues*0.2, len(df_plot))
            
            # make plot
            g = sns.relplot(kind="scatter", data=df_plot, hue=hue_field, x=xfield, y=fitness_y, row=row_field, col=col_field, aspect=aspect, height=height, palette=palette, edgecolor="black", s=25, style=marker_field)

            # define the pathway fields
            pathway_fields = list(df_feats[df_feats.type_feature=="pathway_info"].feature)

            # change axes
            nrows, ncols = g.axes.shape
            for row in range(g.axes.shape[0]):
                for col in range(g.axes.shape[1]):
                    ax = g.axes[row, col]

                    # change the text for the 1st row indicating the text
                    if row==0 and col==0:

                        if len(pathway_fields)>0: text_descriptions = "Pathway descriptions:\n" + "\n".join(["%s: %s"%(p, df_feats.loc[p, "description"]) for p in pathway_fields])
                        else: text_descriptions = ""

                        if len(skiped_feats)>0: text_skipped = "\nSkipped features in plot: %s"%(", ".join(skiped_feats))
                        else: text_skipped = ""

                        text_predictors = "%s ~ %s"%(fitness_y, " + ".join(["'%s'"%p for p in r.selected_predictors]))
                        text_model = ", ".join(["%s=%s"%(f, r[f]) for f in ["model_name", "only_predictors_univariate_corr", "type_feature_selection", 'tol_r2', 'consider_interactions', 'mean_mean_r2', 'ntries_predictors']])

                        ax.text(0, 1.4 + 0.3*len(pathway_fields), "%s\n%s%s\n\n%s"%(text_predictors, text_model, text_skipped, text_descriptions), transform=ax.transAxes, ha="left", va="top")

                    # change title in some cases
                    tit = ax.get_title().replace("resistance_genes_broad_", "rgenes_").replace("truncation", "trunc").replace("presence_variant-", "var-")
     
                    if " | " in tit: ax.set_title("\n".join(tit.split(" | ")), size=10)
                    elif "|||" in tit and tit.startswith("rgenes_"): ax.set_title("\n".join(tit.split("|||")), size=10)
                    else: ax.set_title(tit, size=10)


                    # change x vals in some cases
                    #if df_feature_info.loc[xfield, "type_feature"]=="broad_resistance_genes_profile":
                    ax.set_xticks([xval_to_I[x] for x in all_xvals])
                    ax.set_xticklabels(["%.2f"%x for x in all_xvals], rotation=90)



            # reposition legend
            if not hue_field is None: g.legend.set_bbox_to_anchor((1., 0.5))


            # show and save
            #plt.show()
            g.savefig("%s/%s_from_%s_r2=%.2f_ntries=%i.pdf"%(plots_dir, fitness_y, "|".join([f.replace("/", "-").replace("resistance_genes_broad_", "rgenes_").replace("truncation", "trunc").replace("presence_variant-", "var-") for f in r.    selected_predictors]), r.mean_mean_r2, r.ntries_predictors))
            #g.close()

############################






############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############

############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############

############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############

############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############

############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############

############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############

############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############
############### GRAVEYARD ############################## GRAVEYARD ###############



def get_df_used_predictors(df_data_all, df_feature_info_all, df_univariate_associations, only_predictors_univariate_corr=True):

    """Gets df_feature_info for the used predictors (one for each yfield)"""

    # keep
    df_data_all = cp.deepcopy(df_data_all)
    df_feature_info_all = cp.deepcopy(df_feature_info_all).sort_values(by="feature")
    df_univariate_associations = cp.deepcopy(df_univariate_associations)

    # init
    df_used_predictors_all = pd.DataFrame()

    for stress_condition in ['NaCl', 'CR', 'CFW', 'DTT', 'SDS', 'YPD', 'H2O2', 'CycA']:
        for fitness_estimate in ["M_rAUC_norm", "M_rAUC_diff", "M_AUC_norm_WT", "M_AUC_diff_WT"]:
            if stress_condition=="YPD" and not fitness_estimate in {"M_AUC_norm_WT", "M_AUC_diff_WT"}: continue

            # define yfield to predict
            yfield = "%s_%s"%(fitness_estimate, stress_condition)
            print("Predicting %s"%yfield)

            # check missing feats
            missing_feats = set(df_univariate_associations[df_univariate_associations.yfield==yfield].feature).difference(set(df_feature_info_all.feature))
            missing_feats_feats_info = set(df_feature_info_all.feature).difference(set(df_univariate_associations[df_univariate_associations.yfield==yfield].feature))
            if len(missing_feats)>0: raise ValueError("missing feats in df_feature_info_all")
            if len(missing_feats_feats_info)>0: raise ValueError("missing feats in df_univariate_associations. %s"%missing_feats_feats_info)

            # define WT-related info predictors that could be related to the resistance (fitness in the WT stress, afecting normalizations)
            potential_WT_predictors = {"M_rAUC_norm": {"WT_M_AUC", "WT_M_fAUC_norm"},
                                       "M_rAUC_diff": {"WT_M_AUC", "WT_M_fAUC_diff"},
                                       "M_AUC_norm_WT": {"WT_M_AUC"},
                                       "M_AUC_diff_WT": {"WT_M_AUC"}}[fitness_estimate]

            potential_WT_predictors = {"%s_%s"%(p, stress_condition) for p in potential_WT_predictors}
            potential_WT_predictors.add("WT_M_AUC_YPD") # add the WT fitness with no drug
            if len(potential_WT_predictors.difference(set(df_univariate_associations.feature))): raise ValueError("some feats not in df_univariate_associations")

            # define predictors that are correlated to resistance
            predictors_w_univariate_correlation = set(df_univariate_associations[(df_univariate_associations.yfield==yfield) & (df_univariate_associations.pval<0.05) & (df_univariate_associations.feature.isin(set(df_feature_info_all.feature)))].feature)
            if len(predictors_w_univariate_correlation)==0: raise ValueError("There can't be 0 univariate predictors")

            # define relevant predictor features (mutations, strain and WT fitness in YPD or the drug)
            df_feature_info = df_feature_info_all[(df_feature_info_all.type_feature.isin({"broad_resistance_genes_profile_hot_encoded", "WT_info_hot_encoded", "single_variants", "domain_info", "pathway_info"})) | (df_feature_info_all.feature.isin(potential_WT_predictors))]

            df_feature_info = df_feature_info[(df_feature_info.data_type=="continuous") | (df_feature_info.n_lineages_with_minor_cathegory_binary_features>=2)]
            df_feature_info = df_feature_info[~(df_feature_info.feature.isin({"resistance_genes_broad_ChrI_profile|||dup", "resistance_genes_broad_EPA13_profile|||truncation"}))] # remove predictors that are redundant
            if only_predictors_univariate_corr is True: df_feature_info = df_feature_info[df_feature_info.feature.isin(predictors_w_univariate_correlation)]
            if any(~df_feature_info.data_type.isin({"binary", "continuous"})): raise ValueError("data should be binary or continuous")

            # get predictors
            continuous_predictors = sorted(df_feature_info[df_feature_info.data_type=="continuous"].feature)
            binary_predictors = sorted(df_feature_info[df_feature_info.data_type=="binary"].feature)
            all_predictors = continuous_predictors + binary_predictors
            if len(all_predictors)==0: raise ValueError("predictors are 0")

            # Standardize the continuous predictors (optional).  this does not transform the linearity, but it allows to pick the most important predictors
            #if len(continuous_predictors)>0: df_data[continuous_predictors] = StandardScaler().fit_transform(df_data[continuous_predictors])

            # keep some fields
            df_data = df_data_all[[yfield, "exp_evol_lineage", "sampleID"] + all_predictors]

            # get all relevant interactions
            df_data, all_predictors, continuous_predictors, binary_predictors = get_df_data_and_predictors_with_interactions(df_data, all_predictors, continuous_predictors, binary_predictors, df_feature_info)

            # check that the predictors are not entirely correlated
            print("Checking %i predictors ..."%len(all_predictors))
            for p in all_predictors:
                if len(set(df_data[p]))==1: raise ValueError("all vals the same for %s"%p)

            for p1, p2 in itertools.combinations(continuous_predictors, 2):
                pearson_corr = np.corrcoef(df_data[p1], df_data[p2])[0, 1]
                #if abs(pearson_corr)>0.9: print("WARNING: r %s-%s = %.3f"%(p1, p2, pearson_corr))
                if abs(pearson_corr)==1: raise ValueError("cant be 1 corr\n%s"%(df_data[(df_data[p1]!=0) & (df_data[p2]!=0)][[p1, p2]]))

            for p1, p2 in itertools.combinations(binary_predictors, 2):
                jaccard_index = get_jaccard_index_samples(df_data, p1, p2)
                #if jaccard_index>0.9: print("WARNING: Jaccard Index %s   /   %s = %.3f"%(p1, p2, jaccard_index))
                if jaccard_index==1: raise ValueError("cant be 1 corr\n%s"%(df_data[[p1, p2]].drop_duplicates()))


            df_used_predictors = df_feature_info[df_feature_info.feature.isin(set(all_predictors))]

            df_used_predictors_all

            ddajkhgdajhghadgf

def old_predict_stress_from_features(df_data, df_feature_info, df_univariate_associations, fitness_estimate, stress_condition, PlotsDir, model_name="linear_regression"):

    """Does all the predictions for explaining the stress growth from the features"""

    # define yfield to predict
    yfield = "%s_%s"%(fitness_estimate, stress_condition)
    #print("Predicting %s"%yfield)

    # keep
    df_data = df_data.copy()
    df_feature_info = df_feature_info.copy()

    # for linear regression, do not consider the predictors


    # get predictors
    continuous_predictors = sorted(df_feature_info[df_feature_info.data_type=="continuous"].feature)
    binary_predictors = sorted(df_feature_info[df_feature_info.data_type=="binary"].feature)
    all_predictors = continuous_predictors + binary_predictors
    if len(all_predictors)==0: raise ValueError("predictors are 0")

    # keep some fields
    df_data = df_data[[yfield, "exp_evol_lineage", "sampleID"] + all_predictors]

    print(df_feature_info)
    mjadhmaghdagda


    # debug some strains (with only two points)
    # df_data = df_data[~df_data.strain.isin({"F15", "CST78"})]

    # check missing feats
    missing_feats = set(df_univariate_associations[df_univariate_associations.yfield==yfield].feature).difference(set(df_feature_info.feature))
    missing_feats_feats_info = set(df_feature_info.feature).difference(set(df_univariate_associations[df_univariate_associations.yfield==yfield].feature))
    if len(missing_feats)>0: raise ValueError("missing feats in df_feature_info")
    if len(missing_feats_feats_info)>0: raise ValueError("missing feats in df_univariate_associations. %s"%missing_feats_feats_info)

    # define heavily correlated matrices 
    # redundant_WT_predictors_max_corr_75 = {"WT_M_fAUC_NaCl", "WT_M_fAUC_H2O2", "WT_M_AUC_SDS", "WT_M_fAUC_CFW", "WT_M_AUC_DTT"}

    # define WT-related info predictors that could be related to the resistance (fitness in the WT stress, afecting normalizations)
    potential_WT_predictors = {"M_rAUC_norm": {"WT_M_AUC", "WT_M_fAUC_norm"},
                               "M_rAUC_diff": {"WT_M_AUC", "WT_M_fAUC_diff"},
                               "M_AUC_norm_WT": {"WT_M_AUC"},
                               "M_AUC_diff_WT": {"WT_M_AUC"}}[fitness_estimate]

    potential_WT_predictors = {"%s_%s"%(p, stress_condition) for p in potential_WT_predictors}
    potential_WT_predictors.add("WT_M_AUC_YPD") # add the WT fitness with no drug
    if len(potential_WT_predictors.difference(set(df_univariate_associations.feature))): raise ValueError("some feats not in df_univariate_associations")

    # define predictors that are correlated to resistance
    predictors_w_univariate_correlation = set(df_univariate_associations[(df_univariate_associations.yfield==yfield) & (df_univariate_associations.pval<0.05) & (df_univariate_associations.feature.isin(set(df_feature_info.feature)))].feature)
    if len(predictors_w_univariate_correlation)==0: raise ValueError("There can't be 0 univariate predictors")

    # define relevant predictor features (mutations, strain and WT fitness in YPD or the drug)
    df_feature_info = df_feature_info[(df_feature_info.type_feature.isin({"broad_resistance_genes_profile_hot_encoded", "WT_info_hot_encoded", "single_variants", "domain_info", "pathway_info"})) | (df_feature_info.feature.isin(potential_WT_predictors))]
    df_feature_info = df_feature_info[(df_feature_info.data_type=="continuous") | (df_feature_info.n_lineages_with_minor_cathegory_binary_features>=2)]
    df_feature_info = df_feature_info[~(df_feature_info.feature.isin({"resistance_genes_broad_ChrI_profile|||dup", "resistance_genes_broad_EPA13_profile|||truncation"}))] # remove predictors that are redundant
    if only_predictors_univariate_corr is True: df_feature_info = df_feature_info[df_feature_info.feature.isin(predictors_w_univariate_correlation)]
    if any(~df_feature_info.data_type.isin({"binary", "continuous"})): raise ValueError("data should be binary or continuous")

    # get predictors
    continuous_predictors = sorted(df_feature_info[df_feature_info.data_type=="continuous"].feature)
    binary_predictors = sorted(df_feature_info[df_feature_info.data_type=="binary"].feature)
    all_predictors = continuous_predictors + binary_predictors
    # for data_type in ["binary"]: print(yfield, data_type, "predictors", list(df_feature_info[df_feature_info.data_type==data_type].feature))
    if len(all_predictors)==0: raise ValueError("predictors are 0")

    # Standardize the continuous predictors (optional).  this does not transform the linearity, but it allows to pick the most important predictors
    #if len(continuous_predictors)>0: df_data[continuous_predictors] = StandardScaler().fit_transform(df_data[continuous_predictors])

    # keep some fields
    df_data = df_data[[yfield, "exp_evol_lineage", "sampleID"] + all_predictors]

    # get all relevant interactions
    print("adding interactiobns...")
    if model_name=="linear_regression": df_data, all_predictors, continuous_predictors, binary_predictors = get_df_data_and_predictors_with_interactions(df_data, all_predictors, continuous_predictors, binary_predictors, df_feature_info)

    # check that the predictors are not entirely correlated
    print("Checking predictor...")
    for p in all_predictors:
        if len(set(df_data[p]))==1: raise ValueError("all vals the same for %s"%p)

    for p1, p2 in itertools.combinations(continuous_predictors, 2):
        pearson_corr = np.corrcoef(df_data[p1], df_data[p2])[0, 1]
        #if abs(pearson_corr)>0.9: print("WARNING: r %s-%s = %.3f"%(p1, p2, pearson_corr))
        if abs(pearson_corr)==1: raise ValueError("cant be 1 corr\n%s"%(df_data[(df_data[p1]!=0) & (df_data[p2]!=0)][[p1, p2]]))

    for p1, p2 in itertools.combinations(binary_predictors, 2):
        jaccard_index = get_jaccard_index_samples(df_data, p1, p2)
        #if jaccard_index>0.9: print("WARNING: Jaccard Index %s   /   %s = %.3f"%(p1, p2, jaccard_index))
        if jaccard_index==1: raise ValueError("cant be 1 corr\n%s"%(df_data[[p1, p2]].drop_duplicates()))


    ###############################


    ##### MODELLING #######
    print("modelling with %i predictors..."%len(all_predictors))

    print(all_predictors)

    dmjbdamd
    adkgmhgdaad

    # keep data
    X = df_data[all_predictors].copy()
    y = df_data[yfield].copy()

    # run inhouse forward model selection
    tol = 0.05

    for Itry in range(5):
        important_feats = get_important_features_inHouse_forward_selection(model_name, X, y, tol)


        # get CV score    
        if len(important_features)>0:

            # get the performance of the model on independent datasets
            model_obj = get_predictor_model(model_name, X[important_features].copy(), y.copy())
            cv_scores = cross_val_score(model_obj, X[important_features].copy(), y.copy(), cv=get_CVindices_per_lineage(X[important_features].copy(), nsplits=10), scoring="r2")

        else: cv_scores = [0]


    #, np.mean(cv_scores), np.std(cv_scores)


    print(yfield, important_feats, "%.3f"%mean_r2)

    return 
    print()
    for Itry in range(20):

        sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select="auto", direction="forward", tol=tol, cv=get_CVindices_per_lineage(X, nsplits=10), scoring="r2") 
        sfs.fit(X, y)    
        important_feats_sklearn = list(X.columns[sfs.get_support()])

        print(important_feats_sklearn)

    admhjkjadh


    # define and parametrize the model
    if model_name=="regression_tree":

        for Itrt in range(4):

            model = get_DecisionTreeRegressor()
            ccp_alphas = model.cost_complexity_pruning_path(X, y).ccp_alphas


            cv_score_list = []
            for ccp_alpha in ccp_alphas:

                cv_scores = cross_val_score(get_DecisionTreeRegressor(ccp_alpha=ccp_alpha), X, y, cv=get_CVindices_per_lineage(X, test_size=0.25, nsplits=10), scoring="r2")
                cv_score_list.append({"ccp_alpha":ccp_alpha, "mean_r2":np.mean(cv_scores), "std_r2":np.std(cv_scores), "median_r2":np.median(cv_scores), "mad_r2":stats.median_abs_deviation(cv_scores)})

            df_scores = pd.DataFrame(cv_score_list)

            best_alpha = df_scores.sort_values(by=["median_r2", "mad_r2", "ccp_alpha"], ascending=[False, True, False]).iloc[0].ccp_alpha

            print(best_alpha)

            pseudocount = min(df_scores[df_scores.ccp_alpha!=0].ccp_alpha) / 2
            plt.errorbar(np.log10(df_scores.ccp_alpha+pseudocount), df_scores.median_r2, yerr=df_scores.mad_r2, fmt='o-', capsize=5, label='Mean R2 with Error Bars')
            plt.xlabel('CCP alpha only_predictors_univariate_corr=%s'%only_predictors_univariate_corr)
            plt.ylim([-1, 1])
            plt.ylabel('Median R2 %s'%yfield)
            plt.legend()
            plt.axvline(np.log10(best_alpha+pseudocount), color="gray")
            plt.grid(True)
            plt.show()

            final_model = get_DecisionTreeRegressor(ccp_alpha=best_alpha)
            res = final_model.fit(X, y)


            print(X.columns)
            df_feats = pd.DataFrame({"feature":res.feature_names_in_, "importance":res.feature_importances_}).sort_values(by="importance", ascending=False)
            df_feats = df_feats[df_feats.importance>0]


            print(df_feats.set_index("feature").importance)

            adahghjdag

            print

            print(res.feature_importances_, res.feature_names_in_)

    
            from sklearn import tree
            fig = plt.figure(figsize=(20, 10))  # You can adjust the size by changing the figsize values

            tree.plot_tree(res, feature_names=list(X.columns), filled=True)
            plt.show()
            fig.savefig("%s/regression_tree.pdf"%PlotsDir, bbox_inches="tight")

            dakadjhdkahjda

            ddakjhkjhad


        dakjadh


        ccp_alphas, impurities = path.ccp_alphas, path.impurities


        #min_weight_fraction_leaffloat, default=0.0
        # The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.

        #min_samples_split=2 is the minimum
        #>>> 



    if model_name=="linear_regression":

        for I in range(5):

            # run model selection
            print("getting important features not inhouse...")
            model = LinearRegression()
            sfs = SequentialFeatureSelector(model, n_features_to_select="auto", direction="forward", tol=tol, cv=get_CVindices_per_lineage(X, test_size=0.25, nsplits=20), scoring="r2") # n_features_to_select can be 3
            sfs.fit(X, y)
            
            selected_predictors = list(X.columns[sfs.get_support()])

            important_feats, median_r2, mad_r2 = get_important_features_inHouse_forward_selection(model_name, X, y, tol)


            print("sfs", selected_predictors)
            print("inhouse", important_feats, median_r2)

        ddakjhkjhad






        for direction, tol in [("forward", 0.1), ("backward", -0.05), ("backward", 0.05)]:
            if direction!="forward": continue

            for tryI in range(3):

                # reorder predictors (to check effect)
                shuffled_predictors = cp.deepcopy(all_predictors)
                random.shuffle(shuffled_predictors)
                X = X[shuffled_predictors].copy()
                #X = X.copy()
                y = y.copy()

                #random_indices = cp.deepcopy(sorted(X.index))
                #random.shuffle(random_indices)



                #random_indices = cp.deepcopy(sorted(X.index))
                #random.shuffle(random_indices)

                """
                cv_score = []
                for train_idx, test_idx in get_CVindices_per_lineage(X, test_size=0.25, nsplits=5):
                    # X_train, X_test, y_train, y_test = train_test_split(X.copy(), y.copy(), test_size=0.25) # random_state=42

                    X_train = X.loc[train_idx].copy()
                    y_train = y.loc[train_idx].copy()
                    X_test = X.loc[test_idx].copy()
                    y_test = y.loc[test_idx].copy()


                    model.fit(X_train[selected_predictors], y_train)
                    model_r2 = model.score(X_test[selected_predictors], y_test)
                    cv_score.append(model_r2)

                """


                cv_score = cross_val_score(model, X[selected_predictors].copy(), y.copy(), cv=get_CVindices_per_lineage(X, test_size=0.25, nsplits=10), scoring="r2")

                if np.mean(cv_score)>=0.3: text_r2 = "!!***!!"
                else: text_r2 = ""

                if text_r2!="": print("%s, %s tol=%s, r2=%.3f sd=%.3f"%(yfield, direction, tol, np.mean(cv_score), np.std(cv_score)), selected_predictors, text_r2)
                continue





                r2_dict = {}
                #for n_features_to_select in range(1, len(all_predictors)+1):
                previous_predictors = set()
                for n_features_to_select in range(1,11):


                    sfs = SequentialFeatureSelector(model, n_features_to_select=n_features_to_select, direction=direction, cv=4, scoring="r2") # n_features_to_select can be 3
                    sfs.fit(X, y)

                    selected_predictors = X.columns[sfs.get_support()]

        

                    cv_score = cross_val_score(model, X[selected_predictors].copy(), y.copy(), cv=4, scoring="r2")

                    new_predictor = [p for p in selected_predictors if p not in previous_predictors][0]
                    previous_predictors = set(selected_predictors)
                    print(n_features_to_select, np.mean(cv_score), np.std(cv_score), new_predictor)

                    r2_dict[n_features_to_select] = {"n_feats":n_features_to_select, "mean_r2":np.mean(cv_score), "std_r2":np.std(cv_score)}

                df_r2 = pd.DataFrame(r2_dict).transpose() 


                plt.errorbar(df_r2.n_feats, df_r2.mean_r2, yerr=df_r2.std_r2, fmt='o-', capsize=5, label='Mean R2 with Error Bars')
                plt.xlabel('Number of Features')
                plt.ylabel('Mean R2')
                plt.title('Number of Features vs Mean R2 with Error Bars')
                plt.legend()
                plt.grid(True)
                plt.show()

                print(df_r2)

                admhagdhamg

                return sfs
                # Get the number of features selected at each step
                num_features = np.arange(1, len(sfs.get_support()) + 1)



                # Get the cross-validated R^2 scores at each step
                cross_val_r2_scores = []
                for k_features in num_features:
                    selected_features = np.where(sfs.get_support()[:k_features])[0]
                    X_selected = X[:, selected_features]
                    
                    # Assuming 'model' is your linear regression model
                    cv_score = np.mean(cross_val_score(model, X_selected, y, cv=10, scoring="r2"))
                    cross_val_r2_scores.append(cv_score)

                # Plot the results
                plt.figure(figsize=(10, 6))
                plt.plot(num_features, cross_val_r2_scores, marker='o', linestyle='-')
                plt.title('Sequential Feature Selection - Number of Features vs Cross-validated R^2 Score')
                plt.xlabel('Number of Features Selected')
                plt.ylabel('Cross-validated R^2 Score')
                plt.grid(True)
                plt.show()



                # direction{‘forward’, ‘backward’}, adding feats, removing feats.
                # forward. tol is positive
                # backward. tol can be negative. can be negative when removing features using direction="backward". It can be useful to reduce the number of features at the cost of a small decrease in the score.


                selected_predictors = list(X.columns[sfs.get_support()])



                """
                
                rfecv = RFECV(estimator=model, step=1, cv=4, min_features_to_select=1, verbose=0)  # You can adjust the number of cross-validation folds. cv=5 means that the data is divided in 5 subsets
                rfecv.fit(X, y)

                # plot
                n_scores = len(rfecv.cv_results_["mean_test_score"])
                plt.figure(figsize=(3, 3))
                plt.xlabel("Number of features selected")
                plt.ylabel("Mean test accuracy")
                plt.errorbar(
                    range(1, n_scores + 1),
                    rfecv.cv_results_["mean_test_score"],
                    yerr=rfecv.cv_results_["std_test_score"],
                )
                plt.title("Recursive Feature Elimination \nwith correlated features")
                plt.show()

                # initeresting features
                selected_predictors = list(X.columns[rfecv.support_])

                """


                # test r2 of the model for CVs
                r2_list = []
                for Ic in range(10):
                    model = LinearRegression()
                    X_train, X_test, y_train, y_test = train_test_split(X.copy(), y.copy(), test_size=0.25) # random_state=42
                    model.fit(X_train[selected_predictors], y_train)
                    model_r2 = model.score(X_test[selected_predictors], y_test)
                    r2_list.append(model_r2)

                mean_r2 = np.mean(r2_list)
                if mean_r2>=0.25: text_r2 = "*"
                else: text_r2 = ""

                print(stress_condition, direction, tol, model_name, "predictors selected by direction / selection cv=5", selected_predictors, "r2_mean=%.2f, sd=%.2f"%(mean_r2, np.std(r2_list)), text_r2, text_r2, text_r2)


            # Split the data into training and testing sets

    ##########################

def predict_features_GLM_and_RFECV(df_data, df_feature_info, yfield="M_rAUC_H2O2"):

    """Uses GLM and RFECV to get predictive modules"""

    # keep
    df_data = cp.deepcopy(df_data)
    df_feature_info = cp.deepcopy(df_feature_info).sort_values(by="feature")

    # imports
    #import statsmodels.othermod.betareg.BetaModel as BetaModel

    # get some predictors, encoded properly if they are cathegoric
    df_feature_info = df_feature_info[(df_feature_info.type_feature.isin({"broad_resistance_genes_profile_hot_encoded", "WT_info_PC_continuous_data", "WT_info_hot_encoded"})) | (df_feature_info.feature_is_WT_continuous_info)]
    #df_feature_info = df_feature_info[(df_feature_info.type_feature.isin({"broad_resistance_genes_profile_hot_encoded", "WT_info_PC_continuous_data", "WT_info_hot_encoded"}))]
    
    #df_feature_info = df_feature_info[(df_feature_info.type_feature.isin({"WT_info_PC_continuous_data", "WT_info_hot_encoded"}))]
    #df_feature_info = df_feature_info[(df_feature_info.type_feature.isin({"WT_info_PC_continuous_data"}))]

    if any(~df_feature_info.data_type.isin({"binary", "continuous"})): raise ValueError("data should be binary or continuous")
    interesting_features = list(df_feature_info.feature)

    # prune
    #interesting_features = [f for f in interesting_features if f not in {"PC4_WT_continuous_variables", "PC5_WT_continuous_variables", "PC6_WT_continuous_variables"}]

    #interesting_features = ['PC1_WT_continuous_variables', 'PC2_WT_continuous_variables', 'PC3_WT_continuous_variables', "resistance_genes_broad_ERG11_chrE_profile|||ChrE dup"] #  , , , 'strain|||F15', 


    # define manually predictors
    #interesting_features = ["PC1_WT_continuous_variables", "PC2_WT_continuous_variables", "PC3_WT_continuous_variables", "resistance_genes_broad_ERG11_chrE_profile|||ChrE dup", "resistance_genes_broad_ERG11_chrE_profile|||ERG11 miss"]

    #interesting_features = ["resistance_genes_broad_ERG11_chrE_profile|||ChrE dup", "resistance_genes_broad_ERG11_chrE_profile|||ERG11 miss", "strain|||CST78"]

    # Select predictor and target varianbles
    X = df_data[interesting_features]
    y = df_data[yfield]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25) # random_state=42


    # Add a constant term to the features matrix for the intercept
    X_train = sm_api.add_constant(X_train)

    # Create and fit the model using statsmodels
    model = sm_api.OLS(y_train, X_train)
    results = model.fit()

    # Make predictions on the test set
    X_test = sm_api.add_constant(X_test)
    y_pred = results.predict(X_test)

    # make pred on the train
    y_pred = results.predict(X_train)


    # Evaluate the model
    plt.plot(y_test, y_pred, marker="o", linewidth=0)

    regression_line = np.polyfit(y_test, y_pred, 1)
    x_values = np.array([min(y_test), max(y_test)])
    y_values = np.polyval(regression_line, x_values)
    sns.lineplot(x=x_values, y=y_values, color='black')


    #print(results.summary())


    # Get the p-values
    p_values = results.pvalues

    # Print predictors with p-values less than 0.05
    print(p_values[p_values<0.05])


    sys.exit(0)


    # Create and train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)


    # Get the coefficients (weights) and intercept
    coefficients = model.coef_
    intercept = model.intercept_

    print(f"Coefficients: {coefficients}")
    print(f"Intercept: {intercept}")



    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    plt.plot(y_test, y_pred, marker="o", linewidth=0)

    adhdghjagd

    adkajhakdhkadjhad

    scaler = StandardScaler()

    reg.fit(X_scaled, y)

    coefficients = reg.coef_
    intercept = reg.intercept_

    return reg


    print(reg, coefficients, intercept)

    addkjdakahd


    akjdhkjhdaahjdah
    df_data[yfield] = df_data[yfield] + 0.01 # pseudocount
    max_val = max(df_data[yfield]) + 0.01

    y = (df_data[yfield] / max_val)   # scale by the df_data

    if list(X.index)!=list(y.index): raise ValueError("indeices should be the same")


    print(y, min(y), max(y))




    import statsmodels.othermod.betareg as betareg

    mod = betareg.BetaModel(y, X)
    results_mod = mod.fit()

    print(results_mod.summary())


    sys.exit(0)



    # Train-test split with 75% train and 25% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # GLM model
    glm_model = sm.GLM(y_train, sm.add_constant(X_train), family=sm.families.Gamma()).fit() # maybe I want to use another family (not Gamma)


    # predict on xtest
    predictions = glm_model.predict(sm.add_constant(X_train))
    pearson_corr = np.corrcoef(predictions.values, y_train.values)[0, 1]
    print("R2 of the model (pearson): %s"%(pearson_corr**2))

    # Print summary of the model
    print(glm_model.summary())



    dajagjagdhjagjhgdajg

    # Use RFECV for feature selection
    selector = RFECV(estimator=sm.GLM(y_train, sm.add_constant(X_train), family=sm.families.Gamma()),
                     step=1, cv=5)
    selector = selector.fit(X_train, y_train)

    # Selected features
    selected_features = X.columns[selector.support_]

    # Print selected features
    print("Selected Features:", selected_features)

    # Evaluate model on test set
    X_test_selected = X_test[selected_features]
    predictions = glm_model.predict(sm.add_constant(X_test_selected))

    # Assess model performance
    # You can use metrics like mean squared error, R-squared, etc. based on your problem




def get_CVindices_per_lineage_old(X, test_size=0.25, nsplits=10):

    """Takes an X df that has the samples as indices and returns an iterable of train / test indices for X """

    this_should_be_reworked

    # get a df that reflects the lineage and sample
    X = X.copy()
    df_samples = pd.DataFrame({"sampleID":list(X.index)}).set_index("sampleID", drop=False)
    condition_to_lineage_condition = {"FLZ":"FLZ", "ANI":"ANI", "AinF":"ANI", "FinA":"FLZ", "ANIFLZ":"ANIFLZ"}
    df_samples["exp_evol_lineage"] = df_samples.sampleID.apply(lambda s: "%s_%s_%s"%(s.split("_")[0], s.split("_")[1], condition_to_lineage_condition[s.split("_")[2]]))
    # df_samples["condition"] = df_samples.sampleID.apply(lambda s: s.split("_")[2])

    # define all the indices, splitting by exp_evol_lineage
    all_lineages = list(set(df_samples.exp_evol_lineage))
    CV_indices = []
    for I in range(nsplits):
        test_lineages = set(random.sample(all_lineages, int(round(len(all_lineages)*test_size, 0))))
        train_lineages = set(all_lineages).difference(test_lineages)

        test_samples = set(df_samples[df_samples.exp_evol_lineage.isin(test_lineages)].sampleID)
        train_samples = set(df_samples[df_samples.exp_evol_lineage.isin(train_lineages)].sampleID)

        test_idx = df_samples.index.isin(test_samples)
        train_idx = df_samples.index.isin(train_samples)
        if any(test_idx==train_idx): raise ValueError("All should be in different")
        if (sum(test_idx)+sum(train_idx)) != len(X): raise ValueError("all samples should be in some")

        CV_indices.append((train_idx, test_idx))

    return CV_indices




def old_predict_stress_from_features_old(df_data, df_feature_info, df_univariate_associations, yfield, model_name="linear_regression"):

    """Does all the predictions for explaining the stress growth from the features"""

    # keep
    df_data = cp.deepcopy(df_data)
    df_feature_info = cp.deepcopy(df_feature_info).sort_values(by="feature")

    # debug some strains (with only two points)
    # df_data = df_data[~df_data.strain.isin({"F15", "CST78"})]

    # check missing feats
    missing_feats = set(df_univariate_associations.feature).difference(set(df_feature_info.feature))
    if len(missing_feats): raise ValueError("missing feats")
    
    # define heavily correlated matrices 
    # redundant_WT_predictors_max_corr_75 = {"WT_M_fAUC_NaCl", "WT_M_fAUC_H2O2", "WT_M_AUC_SDS", "WT_M_fAUC_CFW", "WT_M_AUC_DTT"}

    # define predictors that are correlated to resistance
    predictors_w_univariate_correlation = set(df_univariate_associations[(df_univariate_associations.yfield==yfield) & (df_univariate_associations.pval<0.05) & (df_univariate_associations.feature.isin(set(df_feature_info.feature)))].feature)

    # define relevant predictor features
    df_feature_info = df_feature_info[(df_feature_info.type_feature.isin({"broad_resistance_genes_profile_hot_encoded", "WT_info_PC_continuous_data", "WT_info_hot_encoded", "single_variants"})) | (df_feature_info.feature_is_WT_continuous_info)]
    df_feature_info = df_feature_info[(df_feature_info.data_type=="continuous") | (df_feature_info.n_lineages_with_minor_cathegory_binary_features>=2)]
    df_feature_info = df_feature_info[~(df_feature_info.feature.isin({"PC4_WT_continuous_variables", "PC5_WT_continuous_variables", "PC6_WT_continuous_variables"}))]
    #df_feature_info = df_feature_info[df_feature_info.data_type=="continuous"]
    if any(~df_feature_info.data_type.isin({"binary", "continuous"})): raise ValueError("data should be binary or continuous")

    # add extra conditons
    #df_feature_info = df_feature_info[(df_feature_info.data_type=="binary") | (df_feature_info.feature.isin({"PC1_WT_continuous_variables", "PC2_WT_continuous_variables", "PC3_WT_continuous_variables"}))]
    #df_feature_info = df_feature_info[(df_feature_info.data_type=="binary") | (df_feature_info.feature.isin({"AUCs_log2_concentration_median_FLZ_WT_Ksiezopolska2021", "AUCs_log2_concentration_median_ANI_WT_Ksiezopolska2021", "fitness_conc0_median_WT_Ksiezopolska2021"}))]
    #df_feature_info = df_feature_info[(df_feature_info.data_type=="binary") | (df_feature_info.feature.isin({"AUCs_log2_concentration_median_FLZ_WT_Ksiezopolska2021", "AUCs_log2_concentration_median_ANI_WT_Ksiezopolska2021", "fitness_conc0_median_WT_Ksiezopolska2021", "WT_M_fAUC_H2O2", "WT_M_fAUC_NaCl", "WT_M_fAUC_SDS", 'WT_M_fAUC_CFW', 'WT_M_fAUC_CR', 'WT_M_fAUC_DTT'}))]
    #['', '', '', 'PC2_WT_continuous_variables', 'PC3_WT_continuous_variables', 'WT_M_AUC_CFW', 'WT_M_AUC_CR', 'WT_M_AUC_DTT', 'WT_M_AUC_H2O2', 'WT_M_AUC_NaCl', 'WT_M_AUC_SDS', 'WT_M_AUC_YPD', 'WT_M_fAUC_CFW', 'WT_M_fAUC_CR', 'WT_M_fAUC_DTT', 'WT_M_fAUC_H2O2', 'WT_M_fAUC_NaCl', 'WT_M_fAUC_SDS', 'fitness_conc0_median_WT_Ksiezopolska2021'] "AUCs_log2_concentration_median_ANI_WT_Ksiezopolska2021", "AUCs_log2_concentration_median_FLZ_WT_Ksiezopolska2021"
    #df_feature_info = df_feature_info[df_feature_info.data_type=="binary"]
    #df_feature_info = df_feature_info[~df_feature_info.feature.isin({"PC1_WT_continuous_variables", "PC2_WT_continuous_variables", "PC3_WT_continuous_variables"})]
    #df_feature_info = df_feature_info[~df_feature_info.feature.isin(redundant_WT_predictors_max_corr_75)]
    # df_feature_info = df_feature_info[df_feature_info.data_type=="continuous"]
    # df_feature_info = df_feature_info[df_feature_info.feature.apply(lambda x: x.endswith("*"))]
    df_feature_info = df_feature_info[df_feature_info.feature.isin(predictors_w_univariate_correlation)]
    df_feature_info = df_feature_info[(df_feature_info.data_type=="binary") & (df_feature_info.type_feature!="WT_info_hot_encoded")]

    # add valid predictor names
    df_feature_info["feature_formulas"] = df_feature_info.feature.apply(get_correct_feature_for_formulas)
    if len(df_feature_info)!=len(set(df_feature_info.feature_formulas)): raise ValueError("invalid formula vals")
    feature_to_feature_formula = dict(df_feature_info.set_index("feature").feature_formulas)
    feature_formula_to_feature = dict(df_feature_info.set_index("feature_formulas").feature)

    # get predictors
    continuous_predictors = sorted(df_feature_info[df_feature_info.data_type=="continuous"].feature)
    binary_predictors = sorted(df_feature_info[df_feature_info.data_type=="binary"].feature)
    all_predictors = continuous_predictors + binary_predictors
    # for data_type in ["binary", "continuous"]: print(data_type, list(df_feature_info[df_feature_info.data_type==data_type].feature))

    # add predictors
    #for feat_formula, feature in feature_formula_to_feature.items(): df_data[feat_formula] = df_data[feature].copy()
    if len(all_predictors)==0: raise ValueError("predictors are 0")

    ##### LMNM ######

    if model_name=="LMM":

        # keep data
        X = df_data[all_predictors].copy()
        y = df_data[yfield].copy()


        # imports
        import statsmodels.formula.api as smf
        import statsmodels.api as sm

        for tryI in range(1):


            shuffled_predictors = cp.deepcopy(all_predictors)
            random.shuffle(shuffled_predictors)
            predictors_text = " + ".join(["%s"%p for p in shuffled_predictors])

            """

            # test r2 of the model for CVs
            r2_list = []
            for Ic in range(10):

                # define the training_samples. 75% of each strain
                training_samples = set()
                testing_samples = set()

                for strain in sorted(set(df_data.strain)):
                    samples_strain = set(df_data[df_data.strain==strain].index)
                    training_samples_strain = set(random.sample(samples_strain, int(len(samples_strain) * 0.75)))
                    testing_samples_strain = samples_strain.difference(training_samples_strain)
                    if len(training_samples_strain)<3: raise ValueError("at least 3 samples")
                    if (training_samples_strain.union(testing_samples_strain))!=samples_strain: raise ValueError("invlaid samples")
                    training_samples.update(training_samples_strain)
                    testing_samples.update(testing_samples_strain)

                df_data_train = df_data.loc[list(training_samples)].copy()[shuffled_predictors + ["strain", yfield]]
                df_data_test = df_data.loc[list(testing_samples)].copy()[shuffled_predictors + ["strain", yfield]]

                # fit the model on train
                model = smf.mixedlm("%s~%s"%(yfield, predictors_text), df_data_train, groups=df_data_train.strain) # re_formula="~%s"%predictors_text
                res = model.fit() # method=["lbfgs"]

                model_r2 = get_r2_two_series(res.predict(df_data_test), df_data_test[yfield])

                print(model_r2)


                r2_list.append(model_r2)

            mean_r2 = np.mean(r2_list)


            print("r2=%s"%mean_r2)

            """

            model = smf.mixedlm("%s ~ %s"%(yfield, predictors_text), df_data, groups=df_data.strain) # re_formula="~%s"%predictors_text
            res = model.fit() # method=["lbfgs"]


            # Create a DataFrame
            coefficients = res.params
            p_values = res.pvalues

            result_df = pd.DataFrame({'Predictor': coefficients.index, 'Coefficient': coefficients.values, 'pval': p_values.values}).sort_values(by="pval", ascending=True)
            result_df  = result_df[result_df.pval<0.05]


            rsquare = get_r2_two_series(res.fittedvalues, df_data[yfield])

            # plt.scatter(df_data[yfield], res.fittedvalues)



            print(yfield, ["%s: %.2f"%(k, v) for k,v in dict(result_df.set_index("Predictor").Coefficient).items()], "r2=%s"%rsquare)







    #################
    ##### LINEAR MODEL #######

    if model_name=="linear_regression":

        # keep data
        X = df_data[all_predictors].copy()
        y = df_data[yfield].copy()

        # add interactions
        """
        interaction_terms_model = PolynomialFeatures(include_bias=False, interaction_only=True)
        X_interact = interaction_terms_model.fit_transform(X)
        interaction_predictors = interaction_terms_model.get_feature_names_out(all_predictors)

        X_interact = pd.DataFrame(X_interact, index=X.index, columns=continuous_interaction_predictors)


        print(all((X_interact.PC1_WT_continuous_variables* X_interact.PC2_WT_continuous_variables)==(X_interact["PC1_WT_continuous_variables PC2_WT_continuous_variables"]))
        jahgjhgad


        plt.scatter(X.PC1_WT_continuous_variables, X_interact.PC1_WT_continuous_variables)
        adjhgdajhgda

        # Get feature names for the interaction terms
        interaction_columns = interaction_terms.get_feature_names_out(continuous_predictors)



        print(X_interact, X_interact.shape, interaction_columns, len(interaction_columns), len(continuous_predictors))

        adadkhgjhgad

        adadkgaadgjhg

        daadkjahkhdakhadj

        """


        # Standardize the continuous predictors (optional).  this does not transform the linearity. but it allows to better know which predictors are more important
        if len(continuous_predictors)>0: X[continuous_predictors] = StandardScaler().fit_transform(X[continuous_predictors])
        
        print("\n")

        for direction, tol in [("forward", 0.1), ("backward", -0.1), ("backward", 0.1)]:
            print()

            for tryI in range(3):

                # reorder predictors (to check effect)
                shuffled_predictors = cp.deepcopy(all_predictors)
                random.shuffle(shuffled_predictors)
                X = X[shuffled_predictors].copy()
                y = y.copy()

                # run model selection
                model = LinearRegression()
                sfs = SequentialFeatureSelector(model, n_features_to_select="auto", direction=direction, tol=tol, cv=10, scoring="r2") # n_features_to_select can be 3

                sfs.fit(X.copy(), y.copy())


                # direction{‘forward’, ‘backward’}, adding feats, removing feats.
                # forward. tol is positive
                # backward. tol can be negative. can be negative when removing features using direction="backward". It can be useful to reduce the number of features at the cost of a small decrease in the score.


                selected_predictors = list(X.columns[sfs.get_support()])



                """
                
                rfecv = RFECV(estimator=model, step=1, cv=4, min_features_to_select=1, verbose=0)  # You can adjust the number of cross-validation folds. cv=5 means that the data is divided in 5 subsets
                rfecv.fit(X, y)

                # plot
                n_scores = len(rfecv.cv_results_["mean_test_score"])
                plt.figure(figsize=(3, 3))
                plt.xlabel("Number of features selected")
                plt.ylabel("Mean test accuracy")
                plt.errorbar(
                    range(1, n_scores + 1),
                    rfecv.cv_results_["mean_test_score"],
                    yerr=rfecv.cv_results_["std_test_score"],
                )
                plt.title("Recursive Feature Elimination \nwith correlated features")
                plt.show()

                # initeresting features
                selected_predictors = list(X.columns[rfecv.support_])

                """


                # test r2 of the model for CVs
                r2_list = []
                for Ic in range(10):
                    model = LinearRegression()
                    X_train, X_test, y_train, y_test = train_test_split(X.copy(), y.copy(), test_size=0.25) # random_state=42
                    model.fit(X_train[selected_predictors], y_train)
                    model_r2 = model.score(X_test[selected_predictors], y_test)
                    r2_list.append(model_r2)

                mean_r2 = np.mean(r2_list)
                if mean_r2>=0.25: text_r2 = "*"
                else: text_r2 = ""

                print(stress_condition, direction, tol, model_name, "predictors selected by direction / selection cv=5", selected_predictors, "r2_mean=%.2f, sd=%.2f"%(mean_r2, np.std(r2_list)), text_r2, text_r2, text_r2)


        # Split the data into training and testing sets


    ##########################


def get_df_data_and_predictors_with_interactions(df_data, all_predictors, continuous_predictors, binary_predictors, df_feature_info):

    """adds interactions"""

    print("adding interaction terms for %i preds..."%(len(all_predictors)))

    p_to_data_type = dict(df_feature_info.set_index("feature").data_type)
    p_to_nl = dict(df_feature_info[df_feature_info.data_type=="binary"].set_index("feature").n_lineages_with_minor_cathegory_binary_features)
    initial_binary_predictors = cp.deepcopy(binary_predictors)
    all_interaction_predictors = []


    for p1, p2 in itertools.combinations(all_predictors, 2):

        # discard some predictors that do not make sense in interaction. When there is a binary predictor that has only 2 lineages with the minor (least frequent cathegory)
        if any([p_to_data_type[p]=="binary" and p_to_nl[p]<3 for p in [p1, p2]]): continue

        # discard interaction between 'strain' and WT information (it is redudnat)
        if any([x.startswith("WT_") and y.startswith("strain|||") for x,y in [(p1, p2), (p2, p1)]]): continue

        # into df
        interaction_predictor = "%s * %s"%(p1, p2)
        # df_data[interaction_predictor] = df_data[p1] * df_data[p2] # not efficient
        df_data = df_data.assign(**{interaction_predictor: df_data[p1] * df_data[p2]})

        # check if the interaction term is correlated to other binary predictors
        if p_to_data_type[p1]=="binary" and p_to_data_type[p2]=="binary":
            if any(map(lambda p: get_jaccard_index_samples(df_data, p, interaction_predictor)==1, initial_binary_predictors)): continue

        # define the number of lineages that have each val            
        val_to_lineages = df_data[["exp_evol_lineage", interaction_predictor]].groupby(interaction_predictor).apply(lambda df_f: set(df_f.exp_evol_lineage))

        # discard specific interactions, because they are correlated to other interaction predictors
        #uninteresting_interactions = {"presence_variant-ChrE-DUP * presence_variant-FKS2-mis|p.1378|R/L", "presence_variant-ChrE-DUP * resistance_genes_broad_ERG3_profile|||truncation"}
        #if interaction_predictor in uninteresting_interactions: continue

        # keep or not, depending on the lineages
        n_lineages_with_minor_cathegory_binary_features = min(val_to_lineages.apply(len))
        if n_lineages_with_minor_cathegory_binary_features>=2 and len(val_to_lineages)>1: all_interaction_predictors.append(interaction_predictor)

    # keep the interaction predictors, if they are not correlated to other interaction predictors
    print("Prunning predictors...")
    interaction_predictors_already_kept = set()
    interact_p_to_samples = {}
    for interact_p in all_interaction_predictors:
        print(interact_p)

        # check
        p1, p2 = interact_p.split(" * ")
        if not p1 in all_predictors: raise ValueError("p1 not in all")
        if not p2 in all_predictors: raise ValueError("p2 not in all")

        # for continuous predictors, keep all predictors
        if p_to_data_type[p1]=="continuous" or p_to_data_type[p2]=="continuous": 
            continuous_predictors.append(interact_p)
            all_predictors.append(interact_p)

        # for binary predictors, only keep nonunique ones
        else:

            other_found = False
            for other_interact_p in interaction_predictors_already_kept:
                if get_jaccard_index_samples(df_data, other_interact_p, interact_p)==1: 
                    other_found = True
                    break
                    #print("WARNING: %s discarded because it is the same as %s"%(interact_p, other_interact_p)); break

            if other_found is False:
                binary_predictors.append(interact_p)
                all_predictors.append(interact_p)
                interaction_predictors_already_kept.add(interact_p)

    # checks
    if len(all_predictors)!=len(set(all_predictors)): raise ValueError("all should be unique")
    if len(continuous_predictors)!=len(set(continuous_predictors)): raise ValueError("continuous_predictors should be unique")
    if len(binary_predictors)!=len(set(binary_predictors)): raise ValueError("binary_predictors should be unique")
    if (set(continuous_predictors).union(set(binary_predictors)))!=set(all_predictors): raise ValueError("all should be cont + bin")    

    return df_data, all_predictors, continuous_predictors, binary_predictors

def get_p_to_renamed_p_plot(p, info_p):

    """Gets renamed predictor"""

    p_to_renamed_p = {"resistance_genes_broad_FKS_profile|||FKS1 truncation & FKS2 miss": "FKS1* / FKS2 miss",
                      "resistance_genes_broad_FKS_profile|||FKS2 miss" : "FKS2 miss",
                      "presence_variant-ChrE-DUP" : "ChrE dup",
                      "resistance_genes_broad_ERG11_chrE_profile|||ChrE dup" : "ChrE dup"
                      }

    if p in p_to_renamed_p.keys(): return p_to_renamed_p[p]
    elif any([p.startswith(x) for x in {"presence_variant-ERG3-mis|", "ERG3_miss_"}]): return "ERG3 miss"
    elif any([p.startswith(x) for x in {"presence_variant-FKS2-mis|", "FKS2_miss_"}]): return "FKS2 miss"
    elif any([p.startswith(x) for x in {"presence_variant-FKS1-mis|", "FKS1_miss_"}]): return "FKS1 miss"
    elif any([p.startswith(x) for x in {"presence_variant-PDR1-mis|", "PDR1_miss_"}]): return "PDR1 miss"

    elif p.startswith("WT_M_AUC_"): return "WT AUC"
    elif p.startswith("WT_M_fAUC_norm_"): return "WT fAUC"

    elif p.startswith("strain|||"): return " ".join(p.split("|||"))

    # elif info_p.type_feature=="pathway_info": return info_p.description.split(". Genes:")[0].replace("GO_CC, ","").replace("GO_BP, ","").replace("GO_MF, ","").replace(" in pathway ", " in ")
    elif info_p.type_feature=="pathway_info": 

        mut_type = {"truncation":"trunc", "miss":"miss"}[info_p.description.split()[0]]
        pathway_name =  info_p.description.split("'")[1].replace("GO_CC, ","").replace("GO_BP, ","").replace("GO_MF, ","").replace(" in pathway ", " in ").replace("endoplasmic reticulum", "ER").replace("(engineered)", "")
        pathway_ID = info_p.description.split("' (")[1].split()[1].split(")")[0]
        return "%s %s (%s)"%(mut_type, pathway_ID, pathway_name)

    else: raise ValueError("invalid %s %s"%(p, info_p))



def get_df_models_with_set_generic_features(df_models_real, df_feature_info):

    """adds the set of generic features"""

    # kepp
    df_feature_info = df_feature_info.copy().set_index("feature")
 
    # add the set of features, making some of them more generic
    df_models_real["set_features"] = df_models_real.selected_predictors.apply(lambda all_ps: set.union(*[set(p.split(" * ")) for p in all_ps]))
    all_predictors = sorted(set.union(*df_models_real.set_features))
    p_to_renamed_p = dict(zip(all_predictors, map(lambda p: get_p_to_renamed_p_plot(p, df_feature_info.loc[p]), all_predictors)))
    df_models_real["set_features"] = df_models_real["set_features"].apply(lambda all_ps: {p_to_renamed_p[p] for p in all_ps})

    return df_models_real

def adjust_cm_positions(cm, df_square, hm_height_multiplier=0.0002, hm_width_multiplier=0.01, cc_height_multiplier=0.017, rc_width_multiplier=0.017, idx_delimiter="-", distance_btw_boxes=0.0025, cd_height=0.07, rd_width=0.07, cbar_width=0.08, cbar_height=0.015, nrow_fields=None, ncol_fields=None):

    """ Adjusts cm positions """


    # define the hm width and height
    hm_height = len(df_square)*hm_height_multiplier
    hm_width = len(df_square.columns)*hm_width_multiplier

    # set heatmap position
    hm_pos = cm.ax_heatmap.get_position()
    hm_y0 = (hm_pos.y0+hm_pos.height)-hm_height
    cm.ax_heatmap.set_position([hm_pos.x0, hm_y0, hm_width, hm_height]); hm_pos = cm.ax_heatmap.get_position()

    # set the col colors position
    if cm.ax_col_colors is not None:
        cc_height = ncol_fields*cc_height_multiplier

        cm.ax_col_colors.set_position([hm_pos.x0, hm_pos.y0+hm_height+distance_btw_boxes, hm_width, cc_height]); cc_pos = cm.ax_col_colors.get_position()
        cd_y0 = cc_pos.y0 + cc_height + distance_btw_boxes

        top_object_pos = cm.ax_col_colors.get_position()

    else: 
        cd_y0 = hm_pos.y0 + hm_height + distance_btw_boxes
        top_object_pos = cm.ax_heatmap.get_position()


    # set the col dendrogram position
    if cm.ax_col_dendrogram is not None: cm.ax_col_dendrogram.set_position([hm_pos.x0, cd_y0, hm_width, cd_height])

    # set the row colors
    if cm.ax_row_colors is not None: 
        rc_width = nrow_fields*rc_width_multiplier

        cm.ax_row_colors.set_position([hm_pos.x0-rc_width-distance_btw_boxes, hm_pos.y0, rc_width, hm_height]); rc_pos = cm.ax_row_colors.get_position()
        rd_x0 = rc_pos.x0 - rd_width - distance_btw_boxes
        leftmost_object_pos = cm.ax_row_colors.get_position()


    else: 
        rd_x0 = hm_pos.x0 - rd_width - distance_btw_boxes
        leftmost_object_pos = cm.ax_heatmap.get_position()

    # set the row dendrogram
    if cm.ax_row_dendrogram is not None: 
        cm.ax_row_dendrogram.set_position([rd_x0, hm_pos.y0, rd_width, hm_pos.height]); rd_pos = cm.ax_row_dendrogram.get_position()


    # define the cbar position
    #cm.ax_cbar.set_position([hm_pos.x0, hm_pos.y0 - cbar_height - distance_btw_boxes, cbar_width, cbar_height]) # on the bottom
    #cm.ax_cbar.set_position([leftmost_object_pos.x0 - distance_btw_boxes*5 - cbar_width, leftmost_object_pos.y0, cbar_width, cbar_height]) # on the bottom left

    #cm.ax_cbar.set_position([top_object_pos.x0, top_object_pos.y0 +  top_object_pos.height + distance_btw_boxes*40, cbar_width, cbar_height])

    hm_pos = cm.ax_heatmap.get_position()
    cm.ax_cbar.set_position([hm_pos.x0 + hm_pos.width + hm_width_multiplier*3, hm_pos.y0, cbar_width, cbar_height])

def plot_heatmap_sig_features(df_models_real, PlotsDir_paper, df_feature_info):

    """Plot a heatmap with the most relevant features for each """

    # keep dfs
    df_sig_models = df_models_real[df_models_real.sig_model].copy()
    df_sig_models = df_sig_models[df_sig_models.apply(lambda r: r.fitness_estimate == condition_to_tradeoff_f[r.stress_condition], axis=1)]
    if len(df_sig_models)!=(len(df_sig_models[["modelID", "stress_condition"]].drop_duplicates())): raise ValueError("one mod for each s")

    # add set feats
    df_sig_models = get_df_models_with_set_generic_features(df_sig_models, df_feature_info)
    all_predictors = sorted(set.union(*df_sig_models.set_features))

    # create a df to plot the fraction of models with the feature
    df_plot = pd.DataFrame()

    for s in sorted(set(df_sig_models.stress_condition)):
        df_m = df_sig_models[df_sig_models.stress_condition==s]

        df_plot =  pd.concat([df_plot, pd.DataFrame({"feature" : all_predictors, "frac. models w/ feature" : [sum(df_m.set_features.apply(lambda all_fs: f in all_fs)) / len(df_m) for f in all_predictors], "# models w/feature" : [sum(df_m.set_features.apply(lambda all_fs: f in all_fs)) for f in all_predictors], "condition" : s})]).reset_index(drop=True)

    # get square df for the plot
    sorted_conditions = sorted(set(df_plot.condition))
    df_square = df_plot.pivot(index="condition", columns="feature", values="frac. models w/ feature").loc[sorted_conditions]

    df_row_colors = pd.DataFrame({"condition" : stress_condition_to_color}).loc[sorted_conditions]

    def get_number_as_str(x, max_nmodels):
        if x==0: return ""
        else: return "%i/%i"%(int(x), max_nmodels)

    df_anno = df_plot.pivot(index="feature", columns="condition", values="# models w/feature")

    for cond in df_anno.columns:

        max_nmodels = sum(df_sig_models.stress_condition==cond)
        df_anno[cond] = df_anno[cond].apply(get_number_as_str, max_nmodels=max_nmodels)    




    df_anno= df_anno.transpose().loc[df_square.index, df_square.keys()].copy()
    print(df_anno)
    # plotting
    cm = sns.clustermap(df_square, col_cluster=True, row_cluster=True, cmap="Greys", cbar_kws={"label":"fraction good models w/ feature", "orientation":'vertical'}, linewidth=.005, vmin=0, vmax=1.0, linecolor="gray", figsize=(8, 3), row_colors=df_row_colors, annot=df_anno, annot_kws={"size": 8}, fmt="", square=True)


    distance_btw_boxes = 0.005
    square_w = 0.03

    hm_height_multiplier = square_w*3
    #hm_height_multiplier  = square_w
    dendro_height_multiplier = square_w

    adjust_cm_positions(cm, df_square.copy(), hm_height_multiplier=hm_height_multiplier, hm_width_multiplier=square_w, cc_height_multiplier=dendro_height_multiplier, rc_width_multiplier=dendro_height_multiplier, cbar_width=hm_height_multiplier*0.4, cbar_height=hm_height_multiplier*6, distance_btw_boxes=distance_btw_boxes, nrow_fields=1, ncol_fields=0, rd_width=0.01, cd_height=0.03)



    cm.ax_heatmap.set_ylabel("")
    cm.ax_heatmap.set_xlabel("")
    #cm.ax_heatmap.set_yticklabels(sorted_conditions, rotation=0)
    plt.setp(cm.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

    #, row_colors=row_colors_df, col_colors=col_colors_df, cmap=cmap, vmin=vmin, vmax=vmax, xticklabels=False, cbar_kws={"label":zfield, "orientation":'horizontal'}, linewidth=0, yticklabels=1) #, linewidth=.01) # figsize=(30, 10),
    plt.show()


    cm.savefig("%s/sig_models_heatmaps.pdf"%PlotsDir_paper, bbox_inches="tight")


def get_name_for_plot_one_predictor(p, df_feats):

    """For one predictor, get the name for the scatterplot"""

    type_feature = df_feats.loc[p, "type_feature"]

    if type_feature in {"domain_info"}:
        return p.replace("_miss_", " ")

    elif type_feature=="pathway_info":

        pID = p.replace("GO_CC_","").replace("GO_BP_","").replace("GO_MF_","").replace("_truncation", " trunc").replace("MetaCyc_", "").replace("_miss", " miss")
        gene_names_list =  sorted(df_feats.loc[p, "description"].split("Genes: ")[1].replace("Scer_", "").split(", "))

        gene_str = ",\n".join([", ".join(c) for c in chunks(gene_names_list, 3)])

        return "%s\n(%s)"%(pID, gene_str)

    elif type_feature=="WT_info":

        dict_map_feats = {"WT_M_fAUC_norm_H2O2_above_0.5":"$fAUC_{WT}$ H2O2 $\geq 0.5$"}
        if p in dict_map_feats: return dict_map_feats[p]

        else:
            if p.startswith("WT_M_fAUC_norm"): fe_str = "$fAUC_{WT}$"
            elif p.startswith("WT_M_AUC"): fe_str = "$AUC_{WT}$"
            else: raise ValueError("%s"%p)
            return "%s %s"%(fe_str, [val for val in p.split("_") if val in stress_condition_to_color.keys()][0])

    elif type_feature=="single_variants": return p.split("presence_variant-")[1].replace("-DUP", "")

    else: raise ValueError(type_feature)



def find_nearest(a, a0):
    """Element in nd array `a` closest to the scalar value `a0`"""
    
    # Debug elements that are inf
    if a0 not in [np.inf, -np.inf]:
        a = np.array(a)
        idx = np.abs(a - a0).argmin()
        closest_in_a = a.flat[idx]
        
    elif a0==np.inf:
        closest_in_a = max(a)
        
    elif a0==-np.inf:
        closest_in_a = min(a)        

    return closest_in_a

def get_value_to_color(values, palette="mako", n=100, type_color="rgb", center=None):

    """TAkes an array and returns the color that each array has. Checj http://seaborn.pydata.org/tutorial/color_palettes.html"""

    # get the colors
    colors = sns.color_palette(palette, n)

    # change the colors
    if type_color=="rgb": colors = colors
    elif type_color=="hex": colors = [rgb_to_hex(c) for c in colors]
    else: raise ValueError("%s is not valid"%palette)

    # if they are strings
    if type(list(values)[0])==str:

        palette_dict = dict(zip(values, colors))
        value_to_color = palette_dict

    # if they are numbers
    else:

        # map eaqually distant numbers to colors
        if center==None:
            min_palette = min(values)
            max_palette = max(values)
        else: 
            max_deviation = max([abs(fn(values)-center) for fn in [min, max]])
            min_palette = center - max_deviation
            max_palette = center + max_deviation

        all_values_palette = list(np.linspace(min_palette, max_palette, n))
        palette_dict = dict(zip(all_values_palette, colors))

        # get value to color
        value_to_color = {v : palette_dict[find_nearest(all_values_palette, v)] for v in values}

    return value_to_color, palette_dict

def plot_best_multivariate_model(df_sig_models, stress_condition, PlotsDir_paper, df_feature_info, df_data_all):

    """Get the scatterplot of the best condition"""

    ########## PROCESSING DFS #############

    # keep
    df_feature_info = df_feature_info.copy().set_index("feature", drop=False)[["feature", "type_feature", "data_type", "description"]]
    df_data_all = df_data_all.copy()

    # filter
    df_sig_models = df_sig_models[(df_sig_models.apply(lambda r: r.fitness_estimate == condition_to_tradeoff_f[r.stress_condition], axis=1)) & (df_sig_models.stress_condition==stress_condition)].copy()

    # add missing predictors
    missing_feats = set.union(*df_sig_models.selected_predictors.apply(set)).difference(set(df_feature_info.feature))
    for f in missing_feats:
        if not " * " in f: raise ValueError("missing feats should only have *")
        f1, f2 = f.split(" * ")
        df_data_all[f] = df_data_all[f1] * df_data_all[f2]

    # define the fitness y
    fitness_y = "%s_%s"%(condition_to_tradeoff_f[stress_condition], stress_condition)

    # sort
    df_sig_models["npreds"] = df_sig_models.selected_predictors.apply(len)
    df_sig_models = df_sig_models.sort_values(by=["mean_mean_r2", "ntries_predictors", "npreds"], ascending=[False, False, True])

    for I, row in df_sig_models.iterrows():
        print(stress_condition, "mean_r2=%.2f"%row.mean_mean_r2, "min_r2=%.2f"%row.min_mean_r2, "sd_r2=%.2f"%row.std_mean_r2, row.ntries_predictors, row.selected_predictors, row.model_name, row.type_feature_selection, "tol_rw=%s"%row.tol_r2, "interactions=%s"%row.consider_interactions, "only univ pred=%s"%row.only_predictors_univariate_corr)

    # specific parse for H2O2
    if stress_condition=="H2O2": df_sig_models = df_sig_models[df_sig_models.ntries_predictors>8]

    # get the best model,
    best_model = df_sig_models.iloc[0]

    #######################################

    ##### PLOT PRED VS REAL #########

    # get the predicted y according to the model
    X = df_data_all[list(best_model.selected_predictors)].copy()
    y = df_data_all[fitness_y].copy()
    model_obj = get_predictor_model(best_model.model_name, X, y)
    model_obj.fit(X, y)
    y_pred = model_obj.predict(X)

    # plot
    fig = plt.figure(figsize=(2.5,2.5))
    ax = plt.scatter(y_pred, y.values, color="none", edgecolor=stress_condition_to_color[stress_condition])
    min_v = min([min(y.values), min(y_pred)]) - 0.1
    max_v = max([max(y.values), max(y_pred)]) + 0.1
    plt.plot([min_v, max_v], [min_v, max_v], color="gray", linewidth=.9, linestyle="--")
    ticks = [t for t in [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5] if t>=min_v and t<=max_v]

    ax = fig.axes[0]
    ax.set_xticks(ticks)
    ax.set_xticklabels(list(map(str, ticks)))
    ax.set_yticks(ticks)
    ax.set_yticklabels(list(map(str, ticks)))

    plt.xlim([min_v, max_v])
    plt.ylim([min_v, max_v])

    fitness_label = {"M_rAUC_norm":"$rAUC$", "M_AUC_norm_WT":"$AUC\ /\ AUC_{WT}$"}[condition_to_tradeoff_f[stress_condition]]
    plt.ylabel(fitness_label)
    plt.xlabel("predicted %s"%fitness_label)

    mod_dict = {"linear_regression":"lin. regression", "rf_regressor":"random forest"}
    plt.title("%s, mod=%s\n$r^2$=%.2f, consistency=%i"%(stress_condition, mod_dict[best_model.model_name], best_model.mean_mean_r2, best_model.ntries_predictors))

    plots_dir_pred = "%s/best_model_real_vs_predicted"%PlotsDir_paper
    make_folder(plots_dir_pred)
    fig.savefig("%s/best_model_%s.pdf"%(plots_dir_pred, fitness_y), bbox_inches="tight")
    plt.close(fig)

    #################################

    ####### PLOT THE MODEL RESULTS  #######


    # define all the predictors of this model (summarizing interactions)
    all_ps = []
    for p in best_model.selected_predictors:
        if " * " in p: all_ps += p.split(" * ")
        else: all_ps.append(p)

    # get initial
    initial_all_ps = cp.deepcopy(all_ps)

    # print the spearman corr
    df_feats = df_feature_info.loc[all_ps].copy()

    # get plot
    df_plot = df_data_all[[fitness_y] + all_ps].copy()


    # plot different cases
    for p in all_ps:

        fig = plt.figure(figsize=(.9,.9))
        ax = sns.scatterplot(data=df_plot, x=p, y=fitness_y, edgecolor="black", facecolor="none", linewidth=.5, s=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        pdir_single_ps = "%s/plots_single_predictors_bes_models"%PlotsDir_paper; make_folder(pdir_single_ps)
        ax.set_xlabel(get_name_for_plot_one_predictor(p, df_feats))
        ax.set_ylabel("%s %s"%(fitness_label, stress_condition))

        field_to_xvals_lines  = {"WT_M_fAUC_norm_H2O2":[0.5]}
        if p in field_to_xvals_lines:
            for x in field_to_xvals_lines[p]: ax.axvline(x, linewidth=.7, color="gray", linestyle="--")

        plt.show()
        fig.savefig("%s/%s_%s.pdf"%(pdir_single_ps, stress_condition, p),  bbox_inches='tight')

    # define fields depending on the condition
    if stress_condition=="CycA":

        xfield = "FKS2_miss_659-663"
        hue_field = "GO_CC_GO:0016020_truncation"
        marker_field = "GO_CC_GO:0031224_truncation"; markers_dict = {0:"o", 1:"X"}
        df_sort_fields = [marker_field, hue_field]
        row_field = None
        col_field = None
        legend_bbox_to_anchor = (0.9,1)
        plot_height = 3
        add_regression_lines_hue = False
        plot_width = 1.7


    elif stress_condition=="NaCl":

        xfield = "WT_M_fAUC_norm_NaCl"
        marker_field = None; markers_dict = None
        hue_field = "presence_variant-ChrE-DUP" 
        row_field = "GO_BP_GO:0046394_truncation"
        df_sort_fields = [hue_field]
        col_field = None
        legend_bbox_to_anchor = (1.3,1)
        plot_height = 1.5
        add_regression_lines_hue = False
        plot_width = 1.7

    elif stress_condition=="DTT":

        xfield = "WT_M_fAUC_norm_DTT"
        marker_field = None; markers_dict = None
        hue_field = "FKS2_miss_659-662" 
        df_sort_fields = [hue_field]
        row_field = None
        col_field = None
        legend_bbox_to_anchor = (0.9,1)
        plot_height = 3
        add_regression_lines_hue = True
        plot_width = 1.7

    elif stress_condition=="CR":

        xfield = "WT_M_fAUC_norm_CR"
        marker_field = None; markers_dict = None
        hue_field = "MetaCyc_PWY-6899_miss" 
        df_sort_fields = [hue_field]
        row_field = 'MetaCyc_Cellulose-Degradation_miss'
        col_field = 'MetaCyc_PWY-7385_truncation'

        plot_height = 1.5
        add_regression_lines_hue = False
        legend_bbox_to_anchor = (1.5,1)
        plot_width = 1.7

    elif stress_condition=="CFW":

        xfield = "GO_CC_GO:0031227_miss"
        marker_field = None; markers_dict = None
        hue_field = None        
        df_sort_fields = [xfield]
        row_field = None
        col_field = None

        plot_height = 1.5
        add_regression_lines_hue = False
        legend_bbox_to_anchor = (0.9, 1)
        plot_width = 0.9

    elif stress_condition=="YPD":

        xfield = "PDR1_miss_280-768"
        marker_field = None; markers_dict = None
        hue_field = "FKS2_miss_651-1378"        
        df_sort_fields = [hue_field]
        row_field = None
        col_field = None

        plot_height = 1.5
        add_regression_lines_hue = False
        legend_bbox_to_anchor = (0.9, 1)
        plot_width = 1

    elif stress_condition=="H2O2":

        # redefine ps to convert WT_M_fAUC_norm_H2O2 to binary
        """
        df_plot["WT_M_fAUC_norm_H2O2_above_0.5"] = (df_plot.WT_M_fAUC_norm_H2O2>=0.5).map({True:1, False:0})
        all_ps = [p for p in all_ps if p!="WT_M_fAUC_norm_H2O2"] + ["WT_M_fAUC_norm_H2O2_above_0.5"]
        dict_df = {"feature":"WT_M_fAUC_norm_H2O2_above_0.5", "type_feature":"WT_info", "data_type":"binary", "description":""}
        df_feats = pd.concat([df_feats, pd.DataFrame({"WT_M_fAUC_norm_H2O2_above_0.5" : dict_df}).transpose()])

        # define things
        xfield = "WT_M_AUC_YPD"
        marker_field = None; markers_dict = None
        hue_field = "WT_M_AUC_H2O2"        
        df_sort_fields = [hue_field]
        row_field = "WT_M_fAUC_norm_H2O2_above_0.5"
        col_field = "GO_CC_GO:0005576_truncation"

        plot_height = 1.5
        add_regression_lines_hue = False
        legend_bbox_to_anchor = (1.35, 1)
        plot_width = 1.7

        """

        xfield = "WT_M_AUC_H2O2"
        marker_field = None; markers_dict = None
        hue_field = None       
        df_sort_fields = [xfield]
        row_field = None
        col_field = None

        plot_height = 1.5
        add_regression_lines_hue = False
        legend_bbox_to_anchor = (0.9, 1)
        plot_width = 0.9
    
    # get as map
    data_type_map = dict(df_feats.data_type)

    # check that all fields are considered
    if set(all_ps)!={f for f in [xfield, hue_field, marker_field, col_field, row_field] if not f is None}:
        raise ValueError("unconsidered fields")

    # define fields depending on provided fields

    # palette
    if hue_field is None: 
        palette = None
        hue_order = None

    else:
   
        if data_type_map[hue_field]=="binary": 
            palette = {0:"gray", 1:stress_condition_to_color[stress_condition]}
            hue_order = [1,0]

        elif data_type_map[hue_field]=="continuous":
            palette, palette_dict = get_value_to_color(sorted(set(df_plot[hue_field])), palette="rocket", n=100, type_color="rgb", center=None)
            hue_order = None

    # add jitter based on hue and x
    if data_type_map[xfield]=="binary":
        jittering_val = 0.2

        # add for hue
        if not hue_field is None:
        
            all_hue_vals = sorted(set(df_plot[hue_field]))
            array_offsets = np.linspace(-jittering_val, jittering_val, len(all_hue_vals))
            huev_to_offset = dict(zip(all_hue_vals, array_offsets))
            df_plot[xfield] = df_plot.apply(lambda x: x[xfield] + huev_to_offset[x[hue_field]], axis=1)
            distance_hues = array_offsets[1] - array_offsets[0]

        else: distance_hues = 1

        # add jitter
        range_x = max(df_plot[xfield]) - min(df_plot[xfield])
        df_plot[xfield] = df_plot[xfield] + np.random.uniform(-distance_hues*jittering_val, distance_hues*jittering_val, len(df_plot))
    
    # lims
    range_x = (max(df_plot[xfield]) - min(df_plot[xfield]))
    x_suff = range_x*0.1
    xlim = [min(df_plot[xfield])-x_suff, max(df_plot[xfield])+x_suff]
    real_range_x = xlim[1]-xlim[0]

    range_y = (max(df_plot[fitness_y]) - min(df_plot[fitness_y]))
    y_suff = range_y*0.1
    ylim = [min(df_plot[fitness_y])-y_suff, max(df_plot[fitness_y])+y_suff]
    real_range_y = ylim[1]-ylim[0]

    # create fake col fields
    if row_field is None: 
        row_field = "dummy_row_field"
        df_plot[row_field] = "no_rows"

    if col_field is None: 
        col_field = "dummy_col_field"
        df_plot[col_field] = "no_cols"

    # define map dict to rename plot elements
    map_dict = {p : get_name_for_plot_one_predictor(p, df_feats) for p in (initial_all_ps + all_ps)}
    map_dict = {**map_dict, **{"0":"no", "1":"yes"}}

    # create the plot
    row_vals = sorted(set(df_plot[row_field]))
    col_vals = sorted(set(df_plot[col_field]))
    nrows = len(row_vals)
    ncols = len(col_vals)

    fig = plt.figure(figsize=(ncols*plot_width, nrows*plot_height)); Ip=1

    # add sublots
    for Ir, row_v in enumerate(row_vals):
        for Ic, col_v in enumerate(col_vals):

            # get sorted df
            df_p = df_plot[(df_plot[row_field]==row_v) & (df_plot[col_field]==col_v)].sort_values(by=df_sort_fields, ascending=True).copy()
            if len(df_p)==0: raise ValueError("no df_p")

            # get plot
            ax = plt.subplot(nrows, ncols, Ip); Ip+=1

            # add line
            ax.axhline(1.0, linestyle="--", color="black", linewidth=.7, zorder=0)

            # edit
            if hue_field is None: facecolor = stress_condition_to_color[stress_condition]
            else: facecolor = None


            # add scatter
            ax = sns.scatterplot(data=df_p, hue=hue_field, x=xfield, y=fitness_y, palette=palette, edgecolor="black", s=25, style=marker_field, alpha=1, hue_order=hue_order, markers=markers_dict, facecolor=facecolor)

            # Remove the top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # add reg line
            if add_regression_lines_hue is True:
                for hue_val in [0, 1]:
                    df_hue = df_p[df_p[hue_field] == hue_val]
                    sns.regplot(data=df_hue, x=xfield, y=fitness_y, ax=ax, scatter=False, line_kws={'linewidth': 1.5, 'color':palette[hue_val]})

            # set title with the model info
            if Ir==0 and Ic==0:

                if col_field=="dummy_col_field": suffix_tit = ""
                else: suffix_tit = "\n\n\n\n"

                mod_name_dict = {"linear_regression":"MLR", "rf_regressor":"RF"}
                ax.set_title("model (%s) features:\n%s\n%s"%(mod_name_dict[best_model.model_name],  "\n".join(["- "+" * ".join([map_dict[split_p].split("\n")[0] for split_p in p.split(" * ")],) for p in best_model.selected_predictors]), suffix_tit), fontsize=10, loc='left')

            # add xticks for binaries
            if data_type_map[xfield]=="binary":
                ax.set_xticks([0,1])
                ax.set_xticklabels(["no", "yes"])

            # define lims
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)


            # edit x and y ticks
            if Ir==(nrows-1):
                ax.set_xlabel(map_dict[xfield])
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])

            if Ic==0:
                ax.set_ylabel("%s %s"%(fitness_label, stress_condition))
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])

            # if there are rows, add them to the right of the plot
            if row_field!="dummy_row_field":

                # add general label
                if Ir==0 and Ic==(ncols-1):

                    if stress_condition=="NaCl": 
                        row_anno = map_dict[row_field].replace("\n", " ")
                        range_x_mult = 0.3

                    elif stress_condition=="CR": 
                        row_anno = map_dict[row_field].replace(",\n", ", ")
                        range_x_mult = 0.4

                    elif stress_condition=="H2O2":
                        row_anno = map_dict[row_field].replace(",\n", ", ")
                        range_x_mult = 0.3

                    else: raise ValueError("not considered %s. %s"%(stress_condition, map_dict[row_field]))

                    ax.text(xlim[1]+real_range_x*range_x_mult, ylim[0], row_anno, fontsize=10, rotation=270, ha="center", va="center")

                if Ic==(ncols-1):
                    ax.text(xlim[1]+real_range_x*0.1, ylim[0]+real_range_y*0.5, {0:"no", 1:"yes"}[row_v], fontsize=10, rotation=270, ha="center", va="center")

            # if there are cols, add them to the top
            if col_field!="dummy_col_field":

                # add general label
                if Ir==0 and Ic==0:

                    if stress_condition=="CR": 
                        col_anno = map_dict[col_field].replace(",\n", ", ")
                        range_y_mult = 0.4

                    elif stress_condition=="H2O2":
                        col_anno = map_dict[col_field].replace(",\n", ", ")
                        range_y_mult = 0.4

                    else: raise ValueError("not considered %s. %s"%(stress_condition, map_dict[col_field]))

                    ax.text(xlim[1], ylim[1]+real_range_y*range_y_mult, col_anno, fontsize=10, rotation=0, ha="center", va="center")

                if Ir==0:
                    ax.text(xlim[0]+real_range_x*0.5, ylim[1]+real_range_y*0.1, {0:"no", 1:"yes"}[col_v], fontsize=10, rotation=0, ha="center", va="center")



            # add legend
            if not hue_field is None:
                ax.get_legend().remove()

            if Ir==0 and Ic==(ncols-1) and not hue_field is None:

                # define functions
                def get_line(marker, facecolor, edgecolor, label): return Line2D([0], [0], marker=marker, color=facecolor, markerfacecolor=facecolor, markeredgewidth=.7, markeredgecolor=edgecolor, markersize=6, label=label, linewidth=0)

                def get_empty_line(lab): return get_line("s", "white", "white", lab)

                # get handle depending on the hue
                legend_handles = [get_empty_line(map_dict[hue_field])]

                if data_type_map[hue_field]=="binary":
                    legend_handles += [get_line("o", palette[hue_v], "gray", map_dict[str(hue_v)]) for hue_v in hue_order]

                elif data_type_map[hue_field]=="continuous":

                    # define vals for each
                    if hue_field=="WT_M_AUC_H2O2": vals_legend = sorted({float(round(x,1)) for x in df_plot[hue_field]})
                    else: raise ValueError("unconsidered: %s %s"%(hue_field, sorted(set(df_plot[hue_field]))))

                    # define the legend handles
                    legend_handles += [get_line("o", palette_dict[find_nearest(list(palette_dict.keys()), hue_v)], "gray", str(hue_v)) for hue_v in vals_legend]

                else: raise ValueError("invalid")

                # add marker
                if not marker_field is None:
                    legend_handles += [get_empty_line(""), get_empty_line(map_dict[marker_field])]
                    legend_handles += [get_line(markers_dict[mark_v], "black", "black", map_dict[str(mark_v)]) for mark_v in [1,0]]

                # add marker info
                ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=legend_bbox_to_anchor, frameon=False)#, title="model\nconsistency")

    # adjust
    plt.subplots_adjust(wspace=0.05, hspace=0.05)


    # save
    plt.show()
    pdir_results = "%s/best_models_feat_vals"%PlotsDir_paper
    make_folder(pdir_results)
    fig.savefig("%s/%s_best_model.pdf"%(pdir_results, stress_condition),  bbox_inches='tight')

    #######################################


def plot_selected_features_sig_models(df_sig_models, PlotsDir, df_feature_info, fitness_estimate):

    """Gets one plot with the fraction of significant models with each feature"""

    # keep df
    df_sig_models = df_sig_models.copy()
    df_feature_info = df_feature_info.copy().set_index("feature")
    df_sig_models = df_sig_models[df_sig_models.fitness_estimate==fitness_estimate]
    print(df_sig_models.groupby("stress_condition").apply(len))
    if len(df_sig_models)!=(len(df_sig_models[["modelID", "stress_condition"]].drop_duplicates())): raise ValueError("one mod for each s")


    # add the set of features, making some of them more generic
    df_sig_models["set_features"] = df_sig_models.selected_predictors.apply(lambda all_ps: set.union(*[set(p.split(" * ")) for p in all_ps]))
    all_predictors = sorted(set.union(*df_sig_models.set_features))
    p_to_renamed_p = dict(zip(all_predictors, map(lambda p: get_p_to_renamed_p_plot(p, df_feature_info.loc[p]), all_predictors)))
    df_sig_models["set_features"] = df_sig_models["set_features"].apply(lambda all_ps: {p_to_renamed_p[p] for p in all_ps})
    all_predictors = sorted(set.union(*df_sig_models.set_features))

    # create a df to plot the fraction of models with the feature
    df_plot = pd.DataFrame()

    for s in sorted(set(df_sig_models.stress_condition)):
        df_m = df_sig_models[df_sig_models.stress_condition==s]

        df_plot =  pd.concat([df_plot, pd.DataFrame({"feature" : all_predictors, "frac. models w/ feature" : [sum(df_m.set_features.apply(lambda all_fs: f in all_fs)) / len(df_m) for f in all_predictors], "condition" : s})]).reset_index(drop=True)

    print("plotting")

    fig = plt.figure(figsize=(5, 10))

    p_to_mean_frac = df_plot.groupby("feature").apply(lambda df_p: np.mean(df_p["frac. models w/ feature"]))
    sorted_ps = sorted(all_predictors, key=(lambda p: 1 / p_to_mean_frac[p]))

    cond_to_n_features = df_sig_models.groupby("stress_condition").apply(lambda df_c: len(set.union(*df_c.set_features)))

    sorted_conditions = sorted(set(df_plot.condition), key=(lambda c: 1 / cond_to_n_features[c]))
    ax = sns.barplot(data=df_plot, x="frac. models w/ feature", y="feature", hue="condition", order=sorted_ps, hue_order=sorted_conditions,  linewidth=.5, edgecolor="black")

    for I in range(len(sorted_ps)): plt.axhline(I + 0.5, color="gray", linestyle="--", linewidth=.7)

    plt.show()
    fig.savefig("%s/sig_model_results.pdf"%PlotsDir, bbox_inches="tight")


    """
    {'ERG3_miss_265-1094', 'FKS2_miss_201-662', 'resistance_genes_broad_ERG11_chrE_profile|||ChrE dup', 'WT_M_AUC_NaCl', 'MetaCyc_PWY-7921_miss', 'PDR1_miss_768-942', 'WT_M_fAUC_norm_DTT', 'PDR1_miss_280-376', 'resistance_genes_broad_FKS_profile|||FKS2 miss', 'ERG3_miss_267-300', 'GO_CC_GO:0031224_truncation', 'GO_CC_GO:0031227_miss', 'WT_M_fAUC_norm_CR', 'FKS1_miss_309-625', 'strain|||M12', 'PDR1_miss_280-768', 'GO_CC_GO:0005576_truncation', 'WT_M_AUC_H2O2', 'strain|||EF1620', 'strain|||CBS138', 'MetaCyc_ERGOSTEROL-SYN-PWY_miss', 'presence_variant-PDR1-mis|p.280|L/F', 'presence_variant-ChrE-DUP', 'FKS2_miss_666-667', 'WT_M_fAUC_norm_H2O2', 'presence_variant-FKS2-mis|p.662|L/W', 'MetaCyc_Cellulose-Degradation_miss', 'MetaCyc_PWY-6807_miss', 'PDR1_miss_339-376', 'WT_M_AUC_DTT', 'WT_M_AUC_YPD', 'MetaCyc_PWY-6899_miss', 'WT_M_fAUC_norm_NaCl', 'resistance_genes_broad_FKS_profile|||FKS1 truncation & FKS2 miss', 'FKS2_miss_659-662', 'GO_MF_GO:0004553_miss', 'GO_BP_GO:0044182_truncation', 'GO_BP_GO:0044182_miss', 'MetaCyc_PWY-7385_truncation', 'GO_BP_GO:0046394_truncation', 'ERG3_miss_207-243'}

    """




def plot_broad_results_multivariate_analysis(df_models_real, PlotsDir_paper):

    """Plots the distributions"""

    # keep and filter
    df_models_real = df_models_real.copy()
    df_models_real = df_models_real[df_models_real.apply(lambda r: r.fitness_estimate == condition_to_tradeoff_f[r.stress_condition], axis=1)]

    # add fields
    pval_pseudocount = min(df_models_real[df_models_real.pval_maxT!=0].pval_maxT) / 2
    df_models_real["minus_logp_maxT"] = -np.log10(df_models_real.pval_maxT + pval_pseudocount)
    df_models_real["sig. model"] = df_models_real.sig_model
    df_models_real["model consistency"] = df_models_real.ntries_predictors
        

    all_vals = list(range(min(df_models_real["ntries_predictors"]), max(df_models_real["ntries_predictors"])+1))
    val_to_color = {label : color for color, label in zip(sns.color_palette("rocket_r", n_colors=len(all_vals)), all_vals)}

    # make a plot for each case
    ncols = 2
    nrows = 4
    fig = plt.figure(figsize=(ncols*1.8, nrows*1.8)); Ip=0

    sorted_conditions = ["CFW", "CR", "CycA", "DTT", "H2O2", "NaCl", "SDS", "YPD"]
    for Ir in range(nrows):
        for Ic in range(ncols):

            cond = sorted_conditions[Ip]
            df_c = df_models_real[df_models_real.stress_condition==cond]
            ax = plt.subplot(nrows, ncols, Ip+1); Ip+=1

            if all(df_c.sig_model==False): mark_list = ["^"]
            else: mark_list = ["^", "o"]

            ax = sns.scatterplot(data=df_c.sort_values(by=["sig. model"], ascending=[True]), x="mean_mean_r2", y="minus_logp_maxT", hue="model consistency", palette=val_to_color, edgecolor="gray", style="sig. model", markers=mark_list, linewidth=.7, s=30)


            if not (Ic==1 and Ir==1): ax.get_legend().remove()
            else: 

                #legend_handles = [mpatches.Patch(color=color, edgecolor='gray', label=label) for color, label in zip(sns.color_palette("rocket_r"), [0, 2, 4, 6, 8, 10])]

                def get_line(marker, facecolor, edgecolor, label): return Line2D([0], [0], marker=marker, color=facecolor, markerfacecolor=facecolor, markeredgewidth=1, markeredgecolor=edgecolor, markersize=7, label=label, linewidth=0)

                def get_empty_line(lab): return get_line("s", "white", "white", lab)

                legend_handles = [get_empty_line("model\nconsistency")] + [get_line("s", val_to_color[label], "gray", "%s / 10"%label) for label in [0, 2, 4, 6, 8, 10]] + [get_empty_line(""), get_empty_line("good model"), get_line("o", "black", "black", "True"), get_line("^", "black", "black", "False")]


                ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.0, 1.0))#, title="model\nconsistency")
                #ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), edgecolor="gray", markerscale=1)

            if not (Ir==3): 
                ax.set_xlabel("")
                ax.set_xticklabels([])

            else: ax.set_xlabel("explained var. ($r^2$)")

            if not (Ic==0): 
                ax.set_ylabel("")
                ax.set_yticklabels([])

            else: ax.set_ylabel("$-log\ p\ _{max T}$")

            ax.set_xlim([-0.05, 0.7])
            ax.set_ylim([-0.15, 2.8])

            plt.axhline(-np.log10(0.05+pval_pseudocount), linewidth=.7, color="k", linestyle="--")

            for x in [0.1]:
                plt.axvline(x, linewidth=.7, color="k", linestyle="--")

            ax.text(0.67, 2.45, cond, horizontalalignment='right')

    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    fig.savefig("%s/results_multivar_models.pdf"%(PlotsDir_paper), bbox_inches='tight')



def plot_distribution_all_models(df_models_real, fitness_estimate, PlotsDir):

    """Plots the distributions"""

    # keep and filter
    df_models_real = df_models_real.copy()
    df_models_real = df_models_real[df_models_real.fitness_estimate==fitness_estimate]

    # make a plot for each case
    ncols = 3
    nrows = 2
    fig = plt.figure(figsize=(ncols*3, nrows*3)); Ip=0

    sorted_conditions = ["CFW", "H2O2", "DTT", "NaCl", "CR", "SDS", "CycA"]
    for Ir in range(nrows):
        for Ic in range(ncols):

            cond = sorted_conditions[Ip]
            df_c = df_models_real[df_models_real.stress_condition==cond]
            ax = plt.subplot(nrows, ncols, Ip+1); Ip+=1

            df_c["-log p(maxT)"] = -np.log10(df_c.pval_maxT + 0.005)
            distance_hues = 1
            df_c["ntries_predictors"] = df_c.ntries_predictors + + np.random.uniform(-distance_hues*0.2, distance_hues*0.2, len(df_c))

            ax = sns.scatterplot(data=df_c, x="ntries_predictors", y="mean_mean_r2", hue="pval_maxT", palette="rocket", edgecolor="gray", style="sig_model", markers=["^", "o"], linewidth=.7)


            if not (Ic==2 and Ir==0): ax.get_legend().remove()
            else: 

                #legend_handles = [mpatches.Patch(color=color, edgecolor='gray', label=label) for color, label in zip(sns.color_palette("rocket_r"), [0, 0.2, 0.4, 0.6, 0.8, 1.0])]
                #ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.0, 1.0), title="pval maxT")
                ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), title="pval maxT", edgecolor="gray", markerscale=1)

            if not (Ic==1 and Ir==1): ax.set_xlabel("")
            else: ax.set_xlabel("# tries with predictors")

            if not (Ic==0): ax.set_ylabel("")
            else: ax.set_ylabel("model r2")

            plt.axvline(min(df_models_real[df_models_real.sig_model].ntries_predictors), color="gray", linewidth=.7, linestyle="--")
            plt.axhline(min(df_models_real[df_models_real.sig_model].mean_mean_r2), color="gray", linewidth=.7, linestyle="--")

            ax.set_title(cond)
            ax.set_xlim([-0.5, max(df_models_real.ntries_predictors)+1])
            ax.set_ylim([-0.1, max(df_models_real.mean_mean_r2)+0.05])

    plt.subplots_adjust(wspace=0.3, hspace=0.3)


