import pandas as pd
import numpy as np

np.set_printoptions(suppress=True)

cl_names = ["A-673", "A375", "A549", "HCT116", "HS 578T", "HT29", "LNCAP", "LOVO", "MCF7", "PC-3", "RKO",
            "SK-MEL-28", "SW-620", "VCAP"]
cl_name_map = {"A-673": 'A673', "A375": 'A375', "A549": 'A549', "HCT116": 'HCT116', "HS 578T": 'HS578T',
               "HT29": 'HT29', "LNCAP": 'LNCAP', "LOVO": 'LOVO', "MCF7": 'MCF7', "PC-3": 'PC3', "RKO": 'RKO',
               "SK-MEL-28": 'SKMEL28', "SW-620": 'SW620', "VCAP": 'VCAP'}

def load_data(in_dir, feature_types, cell_line_name="all", score="S"):
    '''
    The feature_types parameter is used to select the feature types of samples

    The cell_line_name parameter is used to control the selected cell lines
    The types of parameters that can be entered are list and str
    If "all", select all cell lines (default)
    If cell line name, select a single cell line, such as "HT29"
    If list, select some cell lines, such as ["HT29","A375"]

    The score parameter is used to choose the synergistic scores of drug combinations
    '''

    # Load mf file
    mf_fea = pd.read_csv(f"{in_dir}/drugfeature1_finger_extract.csv")
    mf_fea.set_index(list(mf_fea.columns)[0], inplace=True)

    # Load pp file
    pp_fea = pd.read_csv(f"{in_dir}/drugfeature2_phychem_extract/drugfeature2_phychem_extract.csv")
    pp_fea.set_index(list(pp_fea.columns)[0], inplace=True)

    # Load cl file
    cl_exp_path = f"{in_dir}/drugfeature3_express_extract/"
    if cell_line_name == "all":
        sel_cl_names = cl_names
    elif type(cell_line_name) is list:
        sel_cl_names = cell_line_name
    elif type(cell_line_name) is str:
        sel_cl_names = [cell_line_name]
    else:
        raise ValueError(f"Invalid cell_line_name: {cell_line_name}")
    gp_fea_s = []
    for cl_name in sel_cl_names:
        gp_fea = pd.read_csv("{}{}.csv".format(cl_exp_path, cl_name))
        gp_fea.set_index(list(gp_fea.columns)[0], inplace=True)
        cl_col_name = [f"{cl_name}_{col_name}" for col_name in list(gp_fea.columns)]
        gp_fea.columns = cl_col_name
        gp_fea_s.append(gp_fea)
    gp_fea = pd.concat(gp_fea_s, axis=1).T

    # Load cl file
    cl_fea = pd.read_csv(f"{in_dir}/cell-line-feature_express_extract.csv")
    cl_fea = cl_fea.set_index(list(cl_fea.columns)[0]).T

    # Load drug file
    extract = pd.read_csv(f"{in_dir}/drugdrug_extract.csv")

    # Select drug combinations in the selected cell line
    drug_comb = extract.loc[extract["cell_line_name"].isin(sel_cl_names)]

    drug_a_id = drug_comb["drug_row_cid"]
    drug_b_id = drug_comb["drug_col_cid"]
    comb_cl_name = drug_comb["cell_line_name"].apply(lambda row: cl_name_map[row])
    drug_a_cl_id = drug_comb.apply(lambda row: f"{row['cell_line_name']}_{row['drug_row_cid']}", axis=1)
    drug_b_cl_id = drug_comb.apply(lambda row: f"{row['cell_line_name']}_{row['drug_col_cid']}", axis=1)
    comb_label = drug_comb[score]

    drug_a_mf = mf_fea.loc[drug_a_id] if 'MF' in feature_types else None
    drug_a_pp = pp_fea.loc[drug_a_id] if 'PP' in feature_types else None
    drug_a_gp = gp_fea.loc[drug_a_cl_id] if 'GEP' in feature_types else None
    drug_b_mf = mf_fea.loc[drug_b_id] if 'MF' in feature_types else None
    drug_b_pp = pp_fea.loc[drug_b_id] if 'PP' in feature_types else None
    drug_b_gp = gp_fea.loc[drug_b_cl_id] if 'GEP' in feature_types else None
    comb_cl = cl_fea.loc[comb_cl_name] if 'CL' in feature_types else None

    names = ["MF_A", "PP_A", "GEP_A", "MF_B", "PP_B", "GEP_B", "CL", "Label"]
    items = [drug_a_mf, drug_a_pp, drug_a_gp, drug_b_mf, drug_b_pp, drug_b_gp, comb_cl, comb_label]
    view_data = {}
    for i in range(len(items)):
        item = items[i]
        if item is not None:
            view_data[names[i]] = item.values
    return view_data
