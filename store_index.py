from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
import yaml
import warnings
from src import logger

# Filter out the specific warning
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

# Your PyTorch code goes here
# warnings.resetwarnings()

# Load Pinecone configuration from YAML file
with open('api-key.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Access Pinecone API key and environment
pinecone_api_key = config['pinecone']['api_key']
pinecone_api_env = config['pinecone']['api_env']

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

extracted_data = load_pdf("data/")
logger.info("Data is extracted")

text_chunks = text_split(extracted_data)
logger.info("Text chunks are extracted")

embeddings = download_hugging_face_embeddings()
logger.info("Embedding model downloaded from huggingface hub")

#Initializing the Pinecone
pinecone.init(api_key=pinecone_api_key,
              environment=pinecone_api_env)


index_name="vectordb"
logger.info("----------------------Ingestion of vector to pinecone  Started ----------------")
#Creating Embeddings for Each of The Text Chunks & storing
docsearch=Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)
logger.info("----------------------Ingestion of vector to pinecone ended---------")