import streamlit as st
import pickle
import numpy as np
from scipy.spatial.distance import cosine

# Cache the model loading
@st.cache_resource
def load_model(model_name='skipgram_neg_w2'):
    with open('models/vocabulary.pkl', 'rb') as f:
        vocab_data = pickle.load(f)
    
    with open(f'models/{model_name}.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    return (model_data['embeddings'], 
            vocab_data['word2index'], 
            vocab_data['index2word'],
            model_data)

def find_similar_words(word, embeddings, word2index, index2word, top_k=10):
    if word.lower() not in word2index:
        return None
    
    word_idx = word2index[word.lower()]
    word_vec = embeddings[word_idx]
    
    # Calculate cosine similarities
    similarities = []
    for idx in range(len(embeddings)):
        if idx == word_idx:
            continue
        sim = 1 - cosine(word_vec, embeddings[idx])
        similarities.append((index2word[idx], sim))
    
    # Sort and return top k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

def solve_analogy(a, b, c, embeddings, word2index, index2word, top_k=5):
    words = [w.lower() for w in [a, b, c]]
    if any(w not in word2index for w in words):
        return None
    
    # Vector arithmetic: b - a + c
    vec_a = embeddings[word2index[words[0]]]
    vec_b = embeddings[word2index[words[1]]]
    vec_c = embeddings[word2index[words[2]]]
    
    target = vec_b - vec_a + vec_c
    
    # Find closest words (excluding a, b, c)
    similarities = []
    for idx in range(len(embeddings)):
        word = index2word[idx]
        if word in words:
            continue
        sim = 1 - cosine(target, embeddings[idx])
        similarities.append((word, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

def compute_word_similarity(word1, word2, embeddings, word2index):
    words = [word1.lower(), word2.lower()]
    if any(w not in word2index for w in words):
        return None
    
    vec1 = embeddings[word2index[words[0]]]
    vec2 = embeddings[word2index[words[1]]]
    
    return float(1 - cosine(vec1, vec2))  # Convert to native Python float

# Streamlit App
def main():
    st.set_page_config(
        page_title="Word Embeddings Explorer",
        page_icon="",
        layout="wide"
    )
    
    st.title("Word Embeddings Explorer")
    st.markdown("---")
    
    # Sidebar - Model Selection
    with st.sidebar:
        st.header("Settings")
        model_choice = st.selectbox(
            "Select Model",
            options=['skipgram_neg_w2', 'glove', 'skipgram_w2'],
            format_func=lambda x: {
                'skipgram_neg_w2': 'Skipgram (Negative Sampling)',
                'glove': 'GloVe',
                'skipgram_w2': 'Skipgram (Full Softmax)'
            }[x]
        )
        
        st.markdown("---")
        st.markdown("### Model Info")
        
    # Load model
    try:
        embeddings, word2index, index2word, model_info = load_model(model_choice)
        
        with st.sidebar:
            st.info(f"""
            **Vocabulary Size:** {model_info['vocab_size']}  
            **Embedding Dim:** {model_info['embedding_dim']}  
            **Training Time:** {model_info['train_time']:.2f}s  
            **Final Loss:** {model_info['final_loss']:.4f}
            """)
        
    except FileNotFoundError:
        st.error("Model files not found! Please run the notebook to train and save models first.")
        st.stop()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Similar Words", 
        "Word Analogies", 
        "Word Similarity",
        "Vocabulary Browser"
    ])
    
    # Tab 1: Similar Words
    with tab1:
        st.header("Find Similar Words")
        st.markdown("Enter a word to find the most similar words in the vocabulary.")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            query_word = st.text_input("Enter a word:", value="oil", key="similar")
        with col2:
            top_k = st.slider("Number of results:", 1, 20, 10, key="topk_similar")
        
        if st.button("Find Similar Words", type="primary"):
            result = find_similar_words(query_word, embeddings, word2index, index2word, top_k)
            
            if result is None:
                st.warning(f"Word '{query_word}' not found in vocabulary.")
            else:
                st.success(f"Top {top_k} words similar to **'{query_word}'**:")
                
                # Display results in a nice format
                for i, (word, similarity) in enumerate(result, 1):
                    progress = similarity if similarity > 0 else 0
                    st.metric(
                        label=f"{i}. {word}",
                        value=f"{similarity:.4f}",
                        delta=None
                    )
    
    # Tab 2: Word Analogies
    with tab2:
        st.header("Solve Word Analogies")
        st.markdown("**Format:** A is to B as C is to **?**")
        
        st.info("**Tip:** This model is trained on Reuters financial news corpus. Try words like: oil, crude, company, market, bank, trade, price, year, dollar, etc.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            word_a = st.text_input("Word A:", value="oil", key="a")
        with col2:
            word_b = st.text_input("Word B:", value="crude", key="b")
        with col3:
            word_c = st.text_input("Word C:", value="bank", key="c")
        
        top_k_analogy = st.slider("Number of results:", 1, 10, 5, key="topk_analogy")
        
        if st.button("Solve Analogy", type="primary"):
            result = solve_analogy(word_a, word_b, word_c, embeddings, word2index, index2word, top_k_analogy)
            
            if result is None:
                st.error("One or more words not found in vocabulary.")
                st.markdown("**Words entered:**")
                for word in [word_a, word_b, word_c]:
                    if word.lower() in word2index:
                        st.markdown(f"'{word}' - Found")
                    else:
                        st.markdown(f"'{word}' - Not found")
                
                # Show some example words from vocabulary
                st.markdown("**Sample words from vocabulary:**")
                sample_words = list(word2index.keys())[:30]
                st.write(", ".join(sample_words))
            else:
                st.success(f"**{word_a}** : **{word_b}** :: **{word_c}** : **?**")
                
                st.markdown("### Top Predictions:")
                for i, (word, similarity) in enumerate(result, 1):
                    st.metric(
                        label=f"{i}. {word}",
                        value=f"{similarity:.4f}",
                        delta=None
                    )
    
    # Tab 3: Word Similarity
    with tab3:
        st.header("Compute Word Similarity")
        st.markdown("Calculate cosine similarity between two words.")
        
        col1, col2 = st.columns(2)
        with col1:
            word1 = st.text_input("First word:", value="bank", key="w1")
        with col2:
            word2 = st.text_input("Second word:", value="money", key="w2")
        
        if st.button("Compute Similarity", type="primary"):
            similarity = compute_word_similarity(word1, word2, embeddings, word2index)
            
            if similarity is None:
                st.warning("One or both words not found in vocabulary.")
            else:
                st.success(f"Similarity between **'{word1}'** and **'{word2}'**:")
                
                # Display with a progress bar
                st.metric(
                    label="Cosine Similarity",
                    value=f"{similarity:.4f}"
                )
                
                # Visual representation - convert to float
                similarity_percent = float(max(0, min(100, (similarity + 1) * 50)))
                st.progress(similarity_percent / 100)
                
                # Interpretation
                if similarity > 0.7:
                    st.info("Very similar words")
                elif similarity > 0.4:
                    st.info("Moderately similar words")
                elif similarity > 0.1:
                    st.info("Somewhat similar words")
                else:
                    st.info("Not very similar words")
    
    # Tab 4: Vocabulary Browser
    with tab4:
        st.header("Vocabulary Browser")
        st.markdown(f"Browse the vocabulary of {len(word2index)} words from the Reuters corpus.")
        
        # Search vocabulary
        search_term = st.text_input("Search for words containing:", "")
        
        if search_term:
            matching_words = [w for w in word2index.keys() if search_term.lower() in w]
            st.success(f"Found {len(matching_words)} words containing '{search_term}':")
            
            # Display in columns
            cols = st.columns(4)
            for idx, word in enumerate(sorted(matching_words)[:100]):  # Limit to 100 results
                cols[idx % 4].write(f"• {word}")
            
            if len(matching_words) > 100:
                st.info(f"Showing first 100 of {len(matching_words)} matches")
        else:
            st.markdown("**Sample words from vocabulary:**")
            sample_words = sorted(list(word2index.keys()))[:200]
            
            cols = st.columns(4)
            for idx, word in enumerate(sample_words):
                cols[idx % 4].write(f"• {word}")
            
            st.info(f"Showing 200 of {len(word2index)} words. Use the search box to find specific words.")

if __name__ == "__main__":
    main()
