import pandas as pd
from ragas import experiment
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric
from openai import OpenAI, AsyncOpenAI
import asyncio
import numpy as np
import time

# Evaluation_dataset
gen_eval = pd.read_csv("Generation_eval_dataset.csv")
retrieval_eval = pd.read_csv("Evaluate_dataset.csv")


# Define the scorer
client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

llm = llm_factory("llama3:8b", client=client)

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
    allowed_values=["pass", "fail"]
)

# recall_scorer = ContextRecall(llm=llm)
# precision_scorer = ContextPrecision(llm=llm)

context_metric = DiscreteMetric(
    name="context",
    prompt="""Compare the model retrieved context to the expected answer and determine if it is adequate.
Consider the context adequate if it:
1. Contains the key facts necessary to answer the question correctly.
2. The information in the expected answer can be fully supported using only the retrieved context.
3. Minor wording differences or paraphrasing are acceptable.

Return 'pass' if the response is adequate, 'fail' if important information is missing

Question: {question}
Expected Answer: {expected_answer}
Retrieved context = {contexts}
Evaluation:""",
    allowed_values=["pass", "fail"]
)
@experiment()
async def evaluate_generation(row):
    question = row.question
    model_response = row.answer
    expected_ans = row.ground_truth

    score = await correctness_metric.ascore(
        question=question,
        expected_answer=expected_ans,
        response=model_response,
        llm=llm
    )
    return 1 if score == "pass" else 0

@experiment()
async def evaluate_retrieve_e2e(row):
    question = row.question
    expected_ans = row.ground_truth
    retrieved_contexts = row.contexts
    ans = row.answer

    # recall_res = await recall_scorer.ascore(
    #     user_input=question,
    #     retrieved_contexts=retrieved_contexts,
    #     reference=expected_ans
    # )

    # precision_res = await precision_scorer.ascore(
    #     user_input=question,
    #     reference=expected_ans,
    #     retrieved_contexts = retrieved_contexts
    # )

    context_res = await context_metric.ascore(
        question = question,
        expected_answer = expected_ans,
        contexts = retrieved_contexts,
        llm=llm
    )
    context_score = 1 if context_res == "pass" else 0

    e2e_res = await correctness_metric.ascore(
        question=question,
        expected_answer=expected_ans,
        response=ans,
        llm=llm
    )
    e2e_score = 1 if e2e_res == "pass" else 0

    # return [precision_res.value, recall_res.value, e2e_score]
    return [context_score, e2e_score]

async def run_evaluation():
    print("Generation evaluation")
    gen_task = [evaluate_generation(row) for row in gen_eval.itertuples()]
    gen_scores = await asyncio.gather(*gen_task)
    scores = np.array(gen_scores)
    generation_scores = scores.mean()

    print("Retrieval evaluation and End-to-end pipeline")
    task = [evaluate_retrieve_e2e(row) for row in retrieval_eval.itertuples()]
    retrieval_results = await asyncio.gather(*task)
    scores = np.array(retrieval_results)
    # context_pre_scores, context_rec_scores, e2e_scores = scores.mean(axis=0)
    context_scores, e2e_scores = scores.mean(axis=0)

    print("RESULT")
    print("Generation_score = {}/100".format(generation_scores*100))
    print("Retrieval score = {}/100".format(context_scores*100))
    # print("Context recall = {}/100".format(context_rec_scores))
    print("End-to-end pipeline = {}/100".format(e2e_scores*100))

if __name__ == '__main__':
    asyncio.run(run_evaluation())




