import os
import nest_asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from RAG_pipeline import RAG_pipeline
from utils import vietnamese_tokenizer
from Ingestion import ingestion_pipeline, chunking
from dotenv import load_dotenv
load_dotenv()

nest_asyncio.apply()
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 1. Run Ingestion (Load DB and Retriever)
print("=========================== Ingestion ===========================")
ingest_pip = ingestion_pipeline.IngestionPipeline(tokenizer=vietnamese_tokenizer)
vectordb, keyword_retriever = ingest_pip.ingest()
print('Initialize ingestion successfully')

# 2. Initialize RAG Pipeline
print("=========================== Retriever ===========================")
rag_pip = RAG_pipeline(
    keyword_retriever=keyword_retriever,
    vector_db=vectordb,
    chunking_func= chunking.chunking_func
)
print('Initialize retriever successfully')

# Hàm xử lý khi người dùng gõ /start
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    loi_chao = "Dạ bẩm Sếp, em là quản gia AI đã được lên sóng Telegram! Sếp cần tra cứu thông tin gì cứ ném vào đây cho em nhé!"
    await update.message.reply_text(loi_chao)


# Hàm xử lý mọi tin nhắn văn bản Sếp gửi tới
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query_cua_sep = update.message.text

    # Hiển thị trạng thái "đang gõ phím..." trên Telegram cho chuyên nghiệp
    await update.message.chat.send_action(action="typing")

    try:
        response = rag_pip.generate(query=query_cua_sep)

        # Trả kết quả về cho Sếp trên Telegram
        await update.message.reply_text(response)

    except Exception as e:
        # Bắt lỗi lỡ hệ thống RAG bị sập để bot không bị crash
        await update.message.reply_text(f"Ối, hệ thống tra cứu của em bị hắt hơi rồi Sếp ơi. Lỗi đây ạ: {e}")


def main():
    TELEGRAM_TOKEN = os.getenv('TELEGRAM')

    # Khởi tạo con Bot
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # Đăng ký các hàm xử lý
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("🚀 Quản gia Telegram đã lên đồ, đang vểnh tai nghe lệnh Sếp...")

    # Vòng lặp liên tục để nghe tin nhắn
    app.run_polling(poll_interval=1.0)


if __name__ == '__main__':
    main()