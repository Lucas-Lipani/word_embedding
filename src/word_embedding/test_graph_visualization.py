"""
Script para testar a visualização de TODOS os tipos de grafos com um pequeno corpus.
"""

import sys
from pathlib import Path
import pandas as pd
import spacy

sys.path.insert(0, "src")
from word_embedding import graph_build


def create_mini_corpus():
    """Cria um DataFrame pequeno para testes."""
    return pd.DataFrame(
        {
            "abstract": [
                "Machine learning enables systems to make autonomous decisions.",
                # "Deep learning uses neural networks to process information.",
                # "Natural language processing helps computers understand text.",
            ]
        }
    )


def tokenize_corpus(df, nlp):
    """Tokeniza os abstracts e adiciona coluna 'tokens'."""
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
    df = df.copy()
    df["tokens"] = tokens_all
    return df


def test_document_slidewindow_term(df, nlp, windows):
    """Testa o grafo Document-SlideWindow-Term (janelas deslizantes)."""
    print(f"\n{'='*70}")
    print(f"Testando: Document-SlideWindow-Term")
    print(f"{'='*70}")

    results = []
    for window in windows:
        try:
            g_full, g_slide = graph_build.build_window_graph_and_sliding(
                df, nlp, window, save_visualizations=True
            )

            print(f"✓ window={window}")
            print(
                f"  g_full: {g_full.num_vertices()} vértices, {g_full.num_edges()} arestas"
            )
            print(
                f"  g_slide: {g_slide.num_vertices()} vértices, {g_slide.num_edges()} arestas"
            )

            results.append(True)
        except Exception as e:
            print(f"✗ window={window} ERRO: {e}")
            import traceback

            traceback.print_exc()
            results.append(False)

    return all(results)


def test_document_window_term(df, nlp, windows):
    """Testa o grafo Document-Window-Term (janelas centradas, não deslizantes)."""
    print(f"\n{'='*70}")
    print(f"Testando: Document-Window-Term")
    print(f"{'='*70}")

    results = []
    for window in windows:
        try:
            g = graph_build.initialize_graph()
            g = graph_build.build_window_graph(g, df, nlp, window)
            g_win_term = graph_build.extract_window_term_graph(g)

            print(f"✓ window={window}")
            print(
                f"  g_full: {g.num_vertices()} vértices, {g.num_edges()} arestas"
            )
            print(
                f"  g_win_term: {g_win_term.num_vertices()} vértices, {g_win_term.num_edges()} arestas"
            )

            # Salvar visualizações
            graph_build.save_graph_visualization(
                g, f"03_Document-Window-Term_window{window}"
            )
            graph_build.save_graph_visualization(
                g_win_term, f"04_Window-Term_window{window}"
            )

            results.append(True)
        except Exception as e:
            print(f"✗ window={window} ERRO: {e}")
            import traceback

            traceback.print_exc()
            results.append(False)

    return all(results)


def test_document_context_window_term(df, nlp, windows):
    """Testa o grafo Document-Context-Window-Term (tripartido com contexto explícito)."""
    print(f"\n{'='*70}")
    print(f"Testando: Document-Context-Window-Term")
    print(f"{'='*70}")

    results = []
    for window in windows:
        try:
            # Primeiro construir o grafo Document-SlideWindow-Term
            g_full, g_slide = graph_build.build_window_graph_and_sliding(
                df, nlp, window, save_visualizations=False
            )

            # Extrair Window-Term
            g_win_term = graph_build.extract_window_term_graph(g_full)

            # Transformar em Context-Window-Term
            g_context = graph_build.extract_context_window_term_graph(
                g_win_term
            )

            print(f"✓ window={window}")
            print(
                f"  g_context: {g_context.num_vertices()} vértices, {g_context.num_edges()} arestas"
            )

            # Salvar visualização
            graph_build.save_graph_visualization(
                g_context, f"05_Document-Context-Window-Term_window{window}"
            )

            results.append(True)
        except Exception as e:
            print(f"✗ window={window} ERRO: {e}")
            import traceback

            traceback.print_exc()
            results.append(False)

    return all(results)


def test_document_term(df, nlp):
    """Testa o grafo Document-Term (bipartido simples, agregado)."""
    print(f"\n{'='*70}")
    print(f"Testando: Document-Term (bipartido agregado)")
    print(f"{'='*70}")

    try:
        # Criar grafo Document-SlideWindow-Term primeiro
        g_full, _ = graph_build.build_window_graph_and_sliding(
            df, nlp, 5, save_visualizations=False
        )

        # Extrair Doc-Term agregado
        g_dt = graph_build.extract_doc_term_graph(g_full)

        print(
            f"✓ g_dt: {g_dt.num_vertices()} vértices, {g_dt.num_edges()} arestas"
        )

        # Salvar visualização
        graph_build.save_graph_visualization(
            g_dt, "06_Document-Term_aggregated"
        )

        return True
    except Exception as e:
        print(f"✗ ERRO: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    print("=" * 70)
    print("TESTE DE VISUALIZAÇÃO DE TODOS OS TIPOS DE GRAFOS")
    print("=" * 70)

    # Carrega spaCy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("[ERROR] spaCy model não instalado. Execute:")
        print("  python -m spacy download en_core_web_sm")
        sys.exit(1)

    # Cria corpus mini
    df = create_mini_corpus()
    df = tokenize_corpus(df, nlp)

    print(f"\nCorpus criado:")
    print(f"  {len(df)} documentos")
    print(f"  Tokens por doc: {[len(t) for t in df['tokens']]}")

    # Testar todos os tipos com múltiplas janelas
    windows = [3]
    results = {}

    results["Document-SlideWindow-Term"] = test_document_slidewindow_term(
        df, nlp, windows
    )
    results["Document-Window-Term"] = test_document_window_term(
        df, nlp, windows
    )
    results["Document-Context-Window-Term"] = (
        test_document_context_window_term(df, nlp, windows)
    )
    results["Document-Term"] = test_document_term(df, nlp)

    # Resumo
    print(f"\n{'='*70}")
    print("RESUMO DOS TESTES")
    print(f"{'='*70}")

    for graph_type, success in results.items():
        status = "✓ SUCESSO" if success else "✗ FALHA"
        print(f"{graph_type:<40} {status}")

    print(f"\n📁 PDFs salvos em: ../outputs/graphs/")
    print(f"   - 01_Document-Window-Term_windowX.pdf (g_full)")
    print(f"   - 02_Document-SlideWindow-Term_windowX.pdf (g_slide)")
    print(f"   - 03_Document-Window-Term_windowX.pdf (grafo centrado)")
    print(f"   - 04_Window-Term_windowX.pdf (subgrafo)")
    print(f"   - 05_Document-Context-Window-Term_windowX.pdf (tripartido)")
    print(f"   - 06_Document-Term_aggregated.pdf (bipartido)")

    all_passed = all(results.values())
    print(
        f"\n{'✓ TODOS OS TESTES PASSARAM!' if all_passed else '✗ ALGUNS TESTES FALHARAM'}"
    )

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
