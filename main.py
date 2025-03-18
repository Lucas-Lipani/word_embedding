import time
import spacy
import pandas as pd
from tqdm import tqdm

def main():
    start_time = time.time()
    
    # Carregar spaCy
    nlp = spacy.load("en_core_web_sm")
    
    # Carregar o DataFrame
    df = pd.read_parquet("wos_sts_journals.parquet")
    
    # Selecionar uma amostra do DataFrame    
    df = df.sample(n=300, random_state=42)
    
    # Dicionário para armazenar os termos e suas quantidades por título
    map_term = {}
    
    for index, row in tqdm(df.iterrows(), desc="DF Iteration", total=len(df)):
        title = row["title"]
        doc = nlp(row["abstract"])
        
        if title not in map_term:
            map_term[title] = {}
        
        for token in doc:
            if not token.is_stop and not token.is_punct:
                term = token.text.lower()
                
                if term in map_term[title]:
                    map_term[title][term] += 1
                else:
                    map_term[title][term] = 1
    
    print(dict(list(map_term.items())[:1]))
    print(f"\nO tempo total de execução desse código foi de :{time.time() - start_time:.2f} segundos")

# if __name__ == "__main__":
#     main()