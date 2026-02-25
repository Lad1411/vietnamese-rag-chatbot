import os
import pandas as pd
from google import genai
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric
from ragas.metrics.collections import ContextPrecision, ContextRecall
from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')


# Evaluation_dataset
gen_eval = pd.read_csv("Generation_eval_dataset.csv")
retrieval_eval = pd.read_csv("Evaluate_dataset.csv")


# Define the scorer
client = genai.Client(api_key=GOOGLE_API_KEY)
llm = llm_factory(
    "gemini-2.5-flash",
    provider="google",
    client=client
)

# Define correctness metric
correctness_metric = DiscreteMetric(
    name="correctness",
    prompt="""Compare the model response to the expected answer and determine if it's correct.
Consider the response correct if it:
1. Contains the key information from the expected answer
2. Is factually accurate based on the provided context
3. Adequately addresses the question asked

Return 'pass' if the response is correct, 'fail' if it's incorrect.

Question: {question}
Expected Answer: {expected_answer}
Model Response: {response}

Evaluation:""",
    allowed_values=["pass", "fail"],
)

recall_scorer = ContextRecall(llm=llm)
precision_scorer = ContextPrecision(llm=llm)


def evaluate_generation(row):
    question = row.question
    model_response = row.answer
    expected_ans = row.ground_truth

    score = correctness_metric.score(
        question=question,
        expected_answer=expected_ans,
        response=model_response,
        llm=llm
    )
    return 1 if score == "pass" else 0

def evaluate_retrieve(row):
    question = row.question
    expected_ans = row.ground_truth
    retrieved_contexts = row.contexts

    recall_res = recall_scorer.score(
        user_input=question,
        retrieved_contexts=retrieved_contexts,
        reference=expected_ans
    )

    precision_res = precision_scorer.score(
        user_input=question,
        reference=expected_ans,
        retrieved_contexts = retrieved_contexts
    )

    return precision_res.value, recall_res.value

def run_evaluation():
    print("Generation evaluation")
    generation_scores = [evaluate_generation(row) for row in gen_eval.itertuples()]
    # print(generation_scores)
    print("Retrieval evaluation")
    retrieval_results = [evaluate_retrieve(row) for row in retrieval_eval.itertuples()]

    context_pre_scores = [res[0] for res in retrieval_results]
    context_rec_scores = [res[1] for res in retrieval_results]

    print("RESULT")
    print("Generation_score = {}/100".format(sum(generation_scores)*100/ len(generation_scores)))
    print("Context precision = {}/100".format(sum(context_pre_scores)*100 / len(context_pre_scores)))
    print("Context recall = {}/100".format(sum(context_rec_scores)*100 / len(context_rec_scores)))

if __name__ == '__main__':
    run_evaluation()




