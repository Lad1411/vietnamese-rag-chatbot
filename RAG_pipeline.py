from Retrieval_pipeline import Retriever
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch


class RAG_pipeline:
    def __init__(self, keyword_retriever, vector_db, chunking_func, llm_model=r"Vi-Qwen2-7B-RAG"):
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

        self.chat_history = []

    def format_chathistory(self):
        formatted_text = ""
        for msg in self.chat_history:
            role = "Người dùng" if msg["role"] == "user" else "Hệ thống"
            formatted_text += f"{role}: {msg['content']}\n"
        return formatted_text.strip()

    def rewrite_query(self, query):
        system_prompt = """
            Bạn là một công cụ xử lý ngôn ngữ tự nhiên. Nhiệm vụ DUY NHẤT của bạn là viết lại câu hỏi để làm từ khóa tìm kiếm.
            TUYỆT ĐỐI KHÔNG TRẢ LỜI CÂU HỎI. KHÔNG GIẢI THÍCH. CHỈ IN RA CÂU HỎI.
            
            Dưới đây là các ví dụ bạn PHẢI tuân theo:

            [Ví dụ 1]
            Lịch sử: Chưa có lịch sử trò chuyện.
            Câu hỏi mới: Mã độc tống tiền là gì?
            => Kết quả: Mã độc tống tiền là gì?

            [Ví dụ 2]
            Lịch sử: 
            Người dùng: Malware là gì?
            Hệ thống: Malware là phần mềm độc hại.
            Câu hỏi mới: Cách phòng chống nó?
            => Kết quả: Cách phòng chống Malware?

            [Ví dụ 3]
            Lịch sử:
            Người dùng: BM25 hoạt động ra sao?
            Hệ thống: BM25 đếm tần suất từ khóa.
            Câu hỏi mới: Thời tiết hôm nay thế nào?
            => Kết quả: Thời tiết hôm nay thế nào?
            """

        template = """
                ### Lịch sử trò chuyện (từ cũ đến mới):
                {chat_history}

                ### Câu hỏi mới của người dùng:
                {question}

                ### Câu hỏi độc lập được viết lại:"""

        conversation = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": template.format(chat_history=self.format_chathistory(), question=query)}]
        text = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True)
        model_inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=256,
            temperature=0.1,
            pad_token_id=self.tokenizer.eos_token_id
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        rewrite_query = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return rewrite_query

    def generate(self, query, context_text=None, evaluation=False):
        # Rewrite query
        new_query = self.rewrite_query(query)

        print("Original query:{}".format(query))
        print("New query: {}".format(new_query))

        # Retrieve
        if context_text is None:
            relevant_docs = self.retriever.retrieve(query=new_query)
            docs = [doc.page_content for doc in relevant_docs]
            context_text = "\n\n".join(docs)

        # Prompt
        system_prompt = """
        Bạn là một quản gia AI trung thành, hữu ích và hài hước.
        QUY TẮC BẮT BUỘC:
        - Bạn phải luôn gọi người dùng là "Sếp".
        - Bạn phải luôn xưng là "em".
        - Câu trả lời phải luôn bắt đầu bằng một lời chào hài hước dành cho Sếp.
        - Không bao giờ được xưng "tôi", "tớ", "mình".
        - Hãy ưu tiên sử dụng [Ngữ cảnh] dưới đây để trả lời câu hỏi của người dùng.
        """

        template = '''Chú ý các yêu cầu sau để làm sếp hài lòng:
        - Nếu [Ngữ cảnh] có chứa thông tin, HÃY trích dẫn dựa trên đó.
        - Nếu [Ngữ cảnh] không có thông tin, nhưng câu hỏi thuộc về kiến thức chung phổ thông (địa lý, lịch sử cơ bản), bạn được phép dùng kiến thức của mình để trả lời.
        - Nếu câu hỏi đòi hỏi tính cập nhật, số liệu chuyên sâu mà [Ngữ cảnh] không có, hãy từ chối một cách hài hước và nói rằng "Em chưa có đủ thông tin để trả lời cho Sếp".
        LƯU Ý CUỐI CÙNG TRƯỚC KHI TRẢ LỜI: Tuyệt đối không dùng từ "bạn", "tôi", "mình". Bắt buộc xưng "em" và gọi người dùng là "Sếp" trong toàn bộ câu trả lời.
        Hãy trả lời câu hỏi dựa trên ngữ cảnh:
        ### Ngữ cảnh :
        {context}

        ### Câu hỏi :
        {question}

        ### Trả lời :'''
        conversation = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": template.format(context=context_text, question=new_query)}]
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
        
            
        self.chat_history.extend(
            [{'role': "user", 'content': query},
            {'role': "assistant", 'content': response}]
        )

        if len(self.chat_history) > 6:
            self.chat_history = self.chat_history[-6:]

        if evaluation:
            return response, docs

        return response

