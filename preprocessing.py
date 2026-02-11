import torch
import numpy as np
import pandas as pd
from datasets import Dataset


split = "english"  # Or "persian"

df = pd.read_csv(
    "dataset/translated.csv"
)  ### change this
sample_df = pd.read_csv(r"sample_fewshot_df.csv")
sample_df_fa = pd.read_csv(
    r"sample_fewshot_persian_df.csv"
)


df["medical_history_en"] = df["medical_history_en"].apply(
    lambda x: x.split("<|CHATBOT_TOKEN|>")[1] if "<|CHATBOT_TOKEN|>" in x else x
)

complications_empty_prompt_en = f"""{{Patient requested for doctor's visit: ,
Psychiatric and psychological complaints: ,
Sleep disorders: ,
Loss of appetite: ,
Seizures: ,
Weakness and fatigue: ,
Decreased level of consciousness: ,
Fever: ,
Respiratory complaints: ,
Insurance/treatment cost issues: ,
Urinary tract issues: ,
Pain: ,
Gastrointestinal issues: ,}}"""

complications_empty_prompt_fa = f"""{{درخواست بیمار برای ویزیت پزشک: ,
شکایات روانی و روان‌شناختی: ,
اختلالات خواب: ,
از دست دادن اشتها: ,
تشنج: ,
ضعف و خستگی: ,
کاهش سطح هوشیاری: ,
تب: ,
شکایات تنفسی: ,
مشکلات بیمه/هزینه درمان: ,
مشکلات دستگاه ادراری: ,
درد: ,
مشکلات گوارشی: ,}}"""


def create_complications_en(row):
    return f"""Patient requested a doctor's visit: {row["com_request_for_visit"]},
Psychiatric and psychological complaints: {row["com_psychiatric_psychological_complaints"]},
Sleep disorders: {row["com_sleep_disorders"]},
Loss of appetite: {row["com_loss_of_appetite"]},
Seizures: {row["com_seizures"]},
Weakness and fatigue: {row["com_weakness_and_fatigue"]},
Decreased level of consciousness: {row["com_decreased_level_of_consciousness"]},
Fever: {row["com_fever"]},
Respiratory complaints: {row["com_shortness_of_breath_oxygen_saturation_drop_respiratory_complaints"]},
Insurance/treatment cost issues: {row["com_issues_related_to_insurance_and_treatment_costs"]},
Urinary tract issues: {row["com_urinary_issues"]},
Pain: {row["com_pain"]},
Gastrointestinal issues: {row["com_gastrointestinal_issues"]}, """


######################################## PERSIAN
def create_complications_fa(row):
    return f"""درخواست بیمار برای ویزیت پزشک: {row["com_request_for_visit"]},
شکایات روانی و روان‌شناختی: {row["com_psychiatric_psychological_complaints"]},
اختلالات خواب: {row["com_sleep_disorders"]},
از دست دادن اشتها: {row["com_loss_of_appetite"]},
تشنج: {row["com_seizures"]},
ضعف و خستگی: {row["com_weakness_and_fatigue"]},
کاهش سطح هوشیاری: {row["com_decreased_level_of_consciousness"]},
تب: {row["com_fever"]},
شکایات تنفسی: {row["com_shortness_of_breath_oxygen_saturation_drop_respiratory_complaints"]},
مشکلات بیمه/هزینه درمان: {row["com_issues_related_to_insurance_and_treatment_costs"]},
مشکلات دستگاه ادراری: {row["com_urinary_issues"]},
درد: {row["com_pain"]},
مشکلات گوارشی: {row["com_gastrointestinal_issues"]}, """


def convert_to_dataset(df):
    df["complications"] = df.apply(create_complications_fa, axis=1)
    df = df[["id", "medical_history", "complications"]]
    return df


def make_prompt(sample_df, df, index, complications_empty_prompt_en):
    prompt_list = [
        {
            "role": "system",
            "content": (
                "You are an expert in data extraction specializing in medical information. "
                "You are provided with clinical data about patients with cancer who require palliative care. "
                "Your task is to read the patient's condition and extract ONLY complications strictly into the predefined format below. "
                "Rules to follow:\n"
                "1. Output EXACTLY the same structure and order of fields.\n"
                "2. Fill each field with True or False only.\n"
                "3. Do NOT infer or assume any information that is not explicitly stated.\n"
                "4. Do NOT output anything outside the provided template.\n\n"
                "Required output format:\n"
                f"{complications_empty_prompt_en}"
            ),
        }
    ]

    for i in range(len(sample_df)):
        prompt_list.append(
            {
                "role": "user",
                "content": f"""The patient's condition is as follows: {sample_df["medical_history_en"][i]}""",
            }
        )
        prompt_list.append(
            {"role": "assistant", "content": f"""{sample_df["complications"][i]}"""}
        )

    prompt_list.append(
        {
            "role": "user",
            "content": f"""The patient's condition is as follows: {df["medical_history_en"].iloc[index]}""",
        }
    )
    return prompt_list


def make_prompt(sample_df, df, index, complications_empty_prompt_en):
    prompt_list = [
        {
            "role": "system",
            "content": (
                "شما یک متخصص در استخراج داده با تخصص در اطلاعات پزشکی هستید. "
                "داده‌های بالینی بیماران مبتلا به سرطان که نیاز به مراقبت‌های تسکینی دارند در اختیار شما قرار گرفته است. "
                "وظیفه شما خواندن وضعیت بیمار و استخراج عوارض به صورت دقیق و صرفاً در قالب از پیش تعیین‌شده زیر است. "
                "قوانینی که باید رعایت کنید:\n"
                "1. ساختار و ترتیب فیلدها را دقیقاً همان‌طور که هست خروجی دهید.\n"
                "2. هر فیلد را فقط با True یا False پر کنید.\n"
                "3. هیچ اطلاعاتی را که به صراحت بیان نشده است استنتاج یا فرض نکنید.\n"
                "4. هیچ چیزی خارج از الگوی ارائه‌شده خروجی ندهید.\n\n"
                "قالب خروجی مورد نیاز:\n"
                f"{complications_empty_prompt_en}"
            ),
        }
    ]

    for i in range(len(sample_df)):
        prompt_list.append(
            {
                "role": "user",
                "content": f"""وضعیت بیمار به شرح زیر است: {sample_df["medical_history"][i]}""",
            }
        )
        prompt_list.append(
            {"role": "assistant", "content": f"""{sample_df["complications"][i]}"""}
        )

    prompt_list.append(
        {
            "role": "user",
            "content": f"""وضعیت بیمار به شرح زیر است: {df["medical_history"].iloc[index]}""",
        }
    )
    return prompt_list


if __name__ == "__main__" :
    if split == "persian":
        all_prompts = []
        for i in range(len(df)):
            prompt = make_prompt(sample_df, df, i, complications_empty_prompt_en)
            all_prompts.append(prompt)
        df["prompt"] = all_prompts

        ds_prompt = Dataset.from_pandas(
            df[["id", "medical_history", "complications", "prompt"]]
        )
        ds_prompt.save_to_disk("fa_fewshot_prompt_ds")
    elif split == "english":
        all_prompts = []
        for i in range(len(df)):
            prompt = make_prompt(sample_df_fa, df, i, complications_empty_prompt_fa)
            all_prompts.append(prompt)
        df["prompt"] = all_prompts

        ds_prompt = Dataset.from_pandas(
            df[["id", "medical_history", "complications", "prompt"]]
        )
        ds_prompt.save_to_disk("en_fewshot_prompt_ds")