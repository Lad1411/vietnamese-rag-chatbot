from langchain_community.document_loaders import TextLoader, JSONLoader, CSVLoader, PDFMinerLoader
import os


def document_loader(docs_path):
    """
        Load documents (text, PDF, JSON, csv)

        Args:
            docs_path: directory path
    """

    if not os.path.exists(docs_path):
        raise FileNotFoundError('The directory "{}" is not exist'.format(docs_path))

    documents = []

    for root, _, files in os.walk(docs_path, topdown=True):
        for file in files:
            filepath = os.path.join(root, file)

            _, file_extension = os.path.splitext(filepath)
            loader = None
            try:
                if file_extension == '.txt':
                    loader = TextLoader(file_path=filepath, autodetect_encoding=True)

                elif file_extension == '.pdf':
                    loader = PDFMinerLoader(file_path=filepath)

                elif file_extension == '.json':
                    loader = JSONLoader(file_path=filepath, jq_schema='.', text_content=False)

                elif file_extension == '.csv':
                    loader = CSVLoader(file_path=filepath)

            except:
                print('Error loading {}'.format(filepath))

            if loader:
                documents.extend(loader.load())
    print('Loaded {} documents'.format(len(documents)))
    return documents

