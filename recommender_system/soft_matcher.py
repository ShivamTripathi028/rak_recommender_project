# recommender_system/soft_matcher.py
from sentence_transformers import SentenceTransformer, util
import numpy as np
import logging
from . import config # Use relative import
import pandas as pd # <--- ADD THIS LINE
import re 

logger = logging.getLogger(__name__)

class SoftMatcher:
    def __init__(self):
        self.model_name = config.SENTENCE_BERT_MODEL
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"SentenceTransformer model '{self.model_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model '{self.model_name}': {e}")
            self.model = None # Fallback or raise error

    def build_product_corpus(self, product_row, features_df, product_feature_map_df):
        if product_row is None: return ""
        texts = []
        
        # Ensure data types are string to avoid issues with .get() on non-dict/Series or float NaNs
        desc_app = product_row.get('Description_And_Application', '')
        notes = product_row.get('Notes', '')
        connectivity = product_row.get('Connectivity', '') # Raw connectivity string

        texts.append(str(desc_app) if pd.notna(desc_app) else '')
        texts.append(str(notes) if pd.notna(notes) else '')
        texts.append(str(connectivity) if pd.notna(connectivity) else '')

        # Get associated features
        if features_df is not None and not features_df.empty and \
           product_feature_map_df is not None and not product_feature_map_df.empty:
            product_id = str(product_row.get('Product_ID', ''))
            if product_id:
                associated_feature_ids = product_feature_map_df[
                    product_feature_map_df['Product_ID'] == product_id
                ]['Feature_ID']
                
                if not associated_feature_ids.empty:
                    # Ensure Feature_ID in features_df is also string for consistent merging/filtering
                    features_df['Feature_ID'] = features_df['Feature_ID'].astype(str)
                    feature_descriptions = features_df[
                        features_df['Feature_ID'].isin(associated_feature_ids)
                    ]['Feature_Description']
                    texts.extend(feature_descriptions.dropna().astype(str).tolist())
        
        # Clean up, join, and lowercase
        full_text = " ".join(filter(None, texts)) # Joins non-empty strings
        full_text = re.sub(r'\s+', ' ', full_text).strip() # Normalize whitespace
        return full_text.lower()


    def build_requirement_query(self, requirements_json):
        query_parts = []
        
        app_info = requirements_json.get('application', {})
        query_parts.append(str(app_info.get('type', '')))
        
        subtypes = app_info.get('subtypes', [])
        if subtypes:
            query_parts.append(" ".join(subtypes))
        
        other_subtype = app_info.get('otherSubtype', '')
        if other_subtype:
            query_parts.append(other_subtype)
            
        query_parts.append(str(requirements_json.get('additionalDetails', '')))
        
        full_query = " ".join(filter(None, query_parts))
        full_query = re.sub(r'\s+', ' ', full_query).strip()
        return full_query.lower()

    def get_embeddings(self, texts: list):
        if not self.model:
            logger.error("SentenceTransformer model not available for generating embeddings.")
            return None
        if not texts or all(not text for text in texts) : # if list is empty or all texts are empty
            logger.warning("No valid text provided for embedding.")
            return [] # Return empty list or handle as appropriate
        
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False, show_progress_bar=False)
            return embeddings
        except Exception as e:
            logger.error(f"Error during embedding generation: {e}")
            return None

    def calculate_similarity(self, query_embedding, product_embeddings_list):
        if query_embedding is None or not product_embeddings_list:
            return np.array([]) # Return empty array if no query or product embeddings
        
        # Ensure product_embeddings_list is a 2D numpy array for util.cos_sim
        # It might be a list of 1D arrays from precomputation
        product_embeddings_matrix = np.array([emb for emb in product_embeddings_list if emb is not None and emb.ndim > 0])
        
        if product_embeddings_matrix.ndim == 1 : # if only one product embedding was valid
             product_embeddings_matrix = product_embeddings_matrix.reshape(1, -1)

        if product_embeddings_matrix.shape[0] == 0: # No valid product embeddings
            return np.array([])

        try:
            # util.cos_sim expects tensors or numpy arrays.
            # query_embedding is 1D, product_embeddings_matrix is 2D
            # The result will be a 2D tensor/array of shape (1, num_product_embeddings)
            similarities = util.pytorch_cos_sim(query_embedding, product_embeddings_matrix)
            return similarities.cpu().numpy().flatten() # Get as 1D numpy array
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return np.array([])