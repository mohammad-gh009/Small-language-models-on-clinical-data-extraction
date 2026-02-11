import numpy as np
import pandas as pd
import os

directory_path = 'models_outputs'
all_files = os.listdir(directory_path)
parquet_files = [f for f in all_files if f.endswith('.parquet')]


grouped_files = {}

for file_name in parquet_files:
    full_path = os.path.join(directory_path, file_name)
    base_name = file_name.split('.parquet')[0]
    grouped_files[base_name] = grouped_files.get(base_name, []) + [full_path]

dataframes_dict = {}

for base_name, file_paths in grouped_files.items():
    list_of_dfs = [pd.read_parquet(path) for path in file_paths]
    concatenated_df = pd.concat(list_of_dfs, ignore_index=True)
    dataframes_dict[base_name] = concatenated_df



key_mapping = {
  "Patient requested a doctor's visit": "com_request_for_visit",
  "Complaints related to surgical wound site": "com_complaints_related_to_surgical_wound_site",
  "Drug reactions and side effects": "com_drug_reactions_and_side_effects",
  "Psychiatric and psychological complaints": "com_psychiatric_psychological_complaints",
  "Lymphedema": "com_lymphedema",
  "Sleep disorders": "com_sleep_disorders",
  "Loss of appetite": "com_loss_of_appetite",
  "Seizures": "com_seizures",
  "Weakness and fatigue": "com_weakness_and_fatigue",
  "Decreased level of consciousness": "com_decreased_level_of_consciousness",
  "Fever": "com_fever",
  "Respiratory complaints": "com_shortness_of_breath_oxygen_saturation_drop_respiratory_complaints",
  "Insurance/treatment cost issues": "com_issues_related_to_insurance_and_treatment_costs",
  "Urinary tract issues": "com_urinary_issues",
  "Pain": "com_pain",
  "Gastrointestinal issues": "com_gastrointestinal_issues"
}

def parse_complaints_to_json(text):

    text = text + ","
    out_json = {}

    for key in key_mapping.keys():
        start = text.find(key) + len(key)
        end = text.find(",", start)
        value = text[start:end].strip()

        if "True" in value:
            out_json[key_mapping[key]] = True
        elif "False" in value:
            out_json[key_mapping[key]] = False
        else:
            out_json[key_mapping[key]] = None


    return out_json

def expand_by_features(df):

    json_data = []
    for output in df["outputs"]:
        json_data.append(parse_complaints_to_json(output))

    json_df = pd.DataFrame(json_data)
    df_expanded = pd.concat([df.reset_index(drop=True), json_df], axis=1)
    df_expanded = df_expanded.drop('outputs', axis=1)
    return df_expanded


final_dfs = {}
for key , value in dataframes_dict.items():
    df_in = expand_by_features(value.sort_values(by="id").reset_index(drop=True))[["id"] + list(key_mapping.values())]
    final_dfs[key] = df_in


import pickle
with open("all_models_features.pkl", "wb") as f:
    pickle.dump(final_dfs, f, protocol=pickle.HIGHEST_PROTOCOL)