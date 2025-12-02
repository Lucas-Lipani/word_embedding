"""
Script simples para testar a visualização dos grafos com um pequeno corpus.
"""
import sys
from pathlib import Path
import pandas as pd
import spacy

sys.path.insert(0, "src")
from word_embedding import graph_build

def create_mini_corpus():
    """Cria um DataFrame pequeno para testes."""
    return pd.DataFrame({
        "abstract": [
            "Machine learning is a subset of artificial intelligence that focuses on data.",
            "Deep learning uses neural networks to process information.",
            "Natural language processing helps computers understand human language.",
        ]
    })

def main():
    print("=== Teste de Visualização de Grafos ===\n")
    
    # Carrega spaCy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("[ERROR] spaCy model não instalado. Execute:")
        print("  python -m spacy download en_core_web_sm")
        sys.exit(1)
    
    # Cria corpus mini
    df = create_mini_corpus()
    
    # Tokeniza
    print("Tokenizando abstracts...")
    tokens_all = []
    for abstract in df["abstract"]:
        doc = nlp(abstract)
        tokens = [
            t.text.lower().strip()
            for t in doc
            if not t.is_stop and not t.is_punct
        ]
        tokens_all.append(tokens)
    df["tokens"] = tokens_all
    
    print(f"  {len(df)} documentos processados")
    print(f"  Tokens por doc: {[len(t) for t in tokens_all]}\n")
    
    # Testa diferentes tamanhos de janela
    for window in [3, 5, "full"]:
        print(f"\n--- Construindo grafos para janela={window} ---")
        
        try:
            g_full, g_slide = graph_build.build_window_graph_and_sliding(
                df, nlp, window, 
                save_visualizations=True  # ← SALVA OS PDFs
            )
            
            print(f"  ✓ g_full: {g_full.num_vertices()} vértices, {g_full.num_edges()} arestas")
            print(f"  ✓ g_slide: {g_slide.num_vertices()} vértices, {g_slide.num_edges()} arestas")
        except Exception as e:
            print(f"  ✗ ERRO ao construir grafos: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✅ Testes completos!")
    print("📁 PDFs salvos em: ../outputs/graphs/")
    print("   - 01_Document-Window-Term_windowX.pdf")
    print("   - 02_Document-SlideWindow-Term_windowX.pdf")

if __name__ == "__main__":
    main()
