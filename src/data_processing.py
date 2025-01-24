import os
from typing import Dict

import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader


def create_customer_info(
    row: pd.DataFrame, info_type: str, questions: Dict
) -> pd.Series:

    if info_type == "scores":
        relevant_columns = ["Q1", "Q2", "Q3", "Q4", "Q5"]
        content = "\n".join(
            [f"{questions[col]}: {row[col]}" for col in relevant_columns]
        )
    elif info_type == "nps_score":
        content = f"NPS Score: {row['nps_value']}"
    elif info_type == "nps_type":
        content = f"NPS Type: {row['nps_type']}"
    elif info_type == "open_responses":
        relevant_columns = ["Q6", "Q7"]
        content = "Open Responses: " + "\n".join(
            [f"{questions[col]}: {row[col]}" for col in relevant_columns]
        )
    else:
        raise ValueError(f"Unknown Info Type: {info_type}")

    return pd.Series(content)


def load_and_clean_data(file_path):

    # read answers
    df_answers = pd.read_csv(file_path, delimiter=",")

    # manual transaltion nap just to save API costs
    translation_nps_map = {
        "neutro": "neutral",
        "promotor": "promoter",
        "detrator": "detractor",
    }

    # manual translation to save API costs. Temporary
    df_answers["nps_type"] = df_answers["nps_type"].map(translation_nps_map)

    # convert questions_df into questions dict
    #questions_dict = dict(zip(questions_df.columns, questions_df.iloc[0]))

    # For now hardcode questions_dict
    questions_dict = {
        "Q1": "How likely are you to recommend MDS to other organisations, family or friends?",
        "Q2": "How do you rate the speed and responsiveness of the MDS manager?",
        "Q3": "How do you rate the ease with which MDS resolves the situations you present?",
        "Q4": "Does the MDS Management Team follow up with the expected frequency and quality?",
        "Q5": "How do you rate the quality of the solutions presented by MDS?",
        "Q6": "Please let us know what we could improve by leaving your comment or suggestion.",
        "Q7": "If you feel it is important to pass on this satisfaction questionnaire to another member(s) of staff who can help you better understand the level of service provided by MDS, please fill in the details below."
    }


    df_answers["scores"] = df_answers.apply(
        lambda row: create_customer_info(row, "scores", questions=questions_dict),
        axis=1,
    )
    df_answers["nps_score"] = df_answers.apply(
        lambda row: create_customer_info(row, "nps_score", questions=questions_dict),
        axis=1,
    )
    df_answers["nps_type"] = df_answers.apply(
        lambda row: create_customer_info(row, "nps_type", questions=questions_dict),
        axis=1,
    )
    df_answers["open_responses"] = df_answers.apply(
        lambda row: create_customer_info(
            row, "open_responses", questions=questions_dict
        ),
        axis=1,
    )

    # Combine the customer information into a single string
    df_answers['customer_info'] = df_answers.apply(
        lambda row: f"Customer: {row['customer']}\n"
                    f"Scores: {row['scores']}\n"
                    f"NPS Score: {row['nps_score']}\n"
                    f"NPS Type: {row['nps_type']}\n"
                    f"Open Responses: {row['open_responses']}\n",
        axis = 1
    )

    return df_answers


if __name__ == '__main__':
    load_dotenv()
    file_path = os.getenv('ANSWERS_PATH')

    loader = CSVLoader(file_path= file_path, encoding='utf-8')
    print("Loading document")
    document = loader.load()

