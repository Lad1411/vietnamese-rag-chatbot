from Retrieval_pipeline import Retriever
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import Qwen2TokenizerFast
import torch


class RAG_pipeline:
    def __init__(self, keyword_retriever, vector_db, chunking_func, llm_model=r'/home/lad/AI/vietnamese-rag-chatbot/Vi-Qwen2-7B-RAG'):
        # Define LLM model
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # Normalized Float 4
            bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in bfloat16 for speed
            bnb_4bit_use_double_quant=True,  # Saves a bit more memory
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model,
            dtype=torch.bfloat16,
            quantization_config=bnb_config,
            device_map="auto",
            use_cache=True,
            local_files_only = True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model, local_files_only=True)

        # Define retriever
        self.retriever = Retriever.RAGRetriever(keyword_retriever=keyword_retriever, chunking_func=chunking_func, vectordb=vector_db)


    def generate(self, query, context_text=None, evaluation=False):
        # Retrieve
        if context_text is None:
            relevant_docs = self.retriever.retrieve(query=query)
            docs = [doc.page_content for doc in relevant_docs]
            context_text = "\n\n".join(docs)

        # Prompt
        system_prompt = "Bạn là một trợ lý AI hữu ích. Hãy ưu tiên sử dụng [Ngữ cảnh] dưới đây để trả lời câu hỏi của người dùng."
        template = '''Chú ý các yêu cầu sau:
        - Nếu [Ngữ cảnh] có chứa thông tin, HÃY trích dẫn dựa trên đó.
        - Nếu [Ngữ cảnh] không có thông tin, nhưng câu hỏi thuộc về kiến thức chung phổ thông (địa lý, lịch sử cơ bản), bạn được phép dùng kiến thức của mình để trả lời.
        - Nếu câu hỏi đòi hỏi tính cập nhật, số liệu chuyên sâu mà [Ngữ cảnh] không có, hãy lịch sự từ chối và nói rằng bạn chưa có đủ thông tin.
        Hãy trả lời câu hỏi dựa trên ngữ cảnh:
        ### Ngữ cảnh :
        {context}

        ### Câu hỏi :
        {question}

        ### Trả lời :'''
        conversation = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": template.format(context=context_text, question=query)}]
        text = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True)
        # print(text)
        model_inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=2048,
            temperature=0.1,
            pad_token_id=self.tokenizer.eos_token_id
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        if evaluation:
            return response, docs

        return response



