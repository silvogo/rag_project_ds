import os
import pandas as pd

# read answers
df_answers = pd.read_csv(os.getenv('ANSWERS_PATH'), delimiter=';')
# Read questions
df_questions = pd.read_csv(os.getenv('QUESTIONS_PATH'), delimiter= ';')