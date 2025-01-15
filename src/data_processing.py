import os
from typing import Dict

import pandas as pd
from dotenv import load_dotenv



def create_customer_info(row: pd.DataFrame, info_type: str, questions: Dict) -> pd.Series:

    if info_type == "scores":
        relevant_columns = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
        content = "\n".join([f"{questions[col]}: {row[col]}" for col in relevant_columns])
    elif info_type == 'nps_score':
        content = f"NPS Score: {row['nps_value']}"
    elif info_type == 'nps_type':
        content = f"NPS Type: {row['nps_type']}"
    elif info_type == 'open_responses':
        relevant_columns = ['Q6', 'Q7']
        content = "Open Responses: " + "\n".join([f"{questions[col]}: {row[col]}" for col in relevant_columns])
    else:
        raise ValueError(f"Unknown Info Type: {info_type}")

    return pd.Series(content)


def load_and_clean_data():
    # import environment variables
    load_dotenv()

    # read answers
    df_answers = pd.read_csv(os.getenv('ANSWERS_PATH'), delimiter=',')
    # Read questions as a dictionary
    questions_df = pd.read_csv(os.getenv('QUESTIONS_PATH'), delimiter=';')

    # manu
    translation_nps_map = {
        'neutro': 'neutral',
        'promotor': 'promoter',
        'detrator': 'detractor'
    }

    # manual translation to save API costs
    df_answers['nps_type'] = df_answers['nps_type'].map(translation_nps_map)

    # convert questions_df into questions dict
    questions_dict = dict(zip(questions_df.columns, questions_df.iloc[0]))

    df_answers['scores'] = df_answers.apply(lambda row: create_customer_info(row, 'scores', questions=questions_dict), axis=1)
    df_answers['nps_score'] = df_answers.apply(lambda row: create_customer_info(row, 'nps_score', questions=questions_dict), axis=1)
    df_answers['nps_type'] = df_answers.apply(lambda row: create_customer_info(row, 'nps_type', questions=questions_dict), axis=1)
    df_answers['open_responses'] = df_answers.apply(lambda row: create_customer_info(row, 'open_responses', questions=questions_dict), axis=1)

    return df_answers
