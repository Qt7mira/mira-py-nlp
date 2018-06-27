from summary.textrank import TextRankSentences
from summary.textrank import TextParser
import pandas as pd
import numpy as np

tp = TextParser()

data = pd.read_excel("../data/news_data.xlsx", sheet_name="Sheet3")
doc_id = np.array(data['文章编号']).tolist()
doc_content = np.array(data['文章']).tolist()


for i in doc_content[7:8]:
    print(i)
    trh = TextRankSentences(tp.generate_docs(i))
    print(trh.get_top_n())
