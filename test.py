from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

texts = [
    'OpenAIのEmbeddings APIを使ってテキストをEmbedding化するサンプルコードです。',
    'このサンプルコードはLangChainのFAISSを使ってEmbeddingをベクトルDBに保存します。',
    'FaissはMeta社 (facebook) 謹製のベクトルDBで、検索パフォーマンスは良好です。'
]

# テキストをEmbeddingに変換してベクトルDBに保存する

# 'from_text'はベクトルDBを初期化して作成する。追加する際は 'add_texts' を使う
# Document Loaderを入力する際は 'from_texts' ではなく 'from_documents' を使う
vectorstore = FAISS.from_texts(
    texts,
    OpenAIEmbeddings(model="text-embedding-3-small")
)

# ベクトルDBに保存されたEmbeddingを類似度(L2距離)付きで検索する
# 類似度不要なら: vectorstore.similarity_search(query)
query = "日本の首相は?"
doc_and_scores = vectorstore.similarity_search_with_score(query)
print(doc_and_scores)
