# recommender_system/recommender.py
import pandas as pd
import numpy as np
import logging
from . import config # Use relative import
from .hard_matcher import HardConstraintMatcher
from .soft_matcher import SoftMatcher

logger = logging.getLogger(__name__)

class ProductRecommender:
    def __init__(self, products_df, features_df, product_feature_map_df):
        self.products_df = products_df
        self.features_df = features_df
        self.product_feature_map_df = product_feature_map_df
        
        self.hard_matcher = HardConstraintMatcher()
        self.soft_matcher = SoftMatcher()
        self.weights = config.WEIGHTS

        self._precompute_product_data()

    def _precompute_product_data(self):
        """Precomputes combined text and embeddings for all products."""
        logger.info("Starting precomputation of product corpuses and embeddings...")
        if self.products_df is None or self.products_df.empty:
            logger.error("Products DataFrame is empty. Cannot precompute.")
            self.products_df['combined_text'] = pd.Series(dtype='str')
            self.products_df['embedding'] = pd.Series(dtype='object')
            return

        self.products_df['combined_text'] = self.products_df.apply(
            lambda row: self.soft_matcher.build_product_corpus(row, self.features_df, self.product_feature_map_df),
            axis=1
        )
        
        product_corpuses = self.products_df['combined_text'].tolist()
        if not product_corpuses or all(not text for text in product_corpuses):
            logger.warning("All product corpuses are empty. Embeddings will not be generated.")
            self.products_df['embedding'] = [np.array([]) for _ in range(len(self.products_df))]
        else:
            embeddings = self.soft_matcher.get_embeddings(product_corpuses)
            if embeddings is not None and len(embeddings) == len(self.products_df):
                 # Store embeddings as list of numpy arrays
                self.products_df['embedding'] = [emb for emb in embeddings]
            else:
                logger.error("Failed to generate or align embeddings for products. Embedding column will be empty.")
                # Fill with empty arrays to avoid errors later, or handle more gracefully
                self.products_df['embedding'] = [np.array([]) for _ in range(len(self.products_df))]
        
        logger.info("Precomputation of product data (text & embeddings) complete.")


    def recommend(self, requirements_json, top_n=5):
        if self.products_df is None or self.products_df.empty:
            logger.warning("No product data available for recommendations.")
            return []

        # 1. Hard Constraint Filtering
        candidate_products_indices = []
        hard_constraint_results = []

        for index, product_row in self.products_df.iterrows():
            passed_all, hc_score, hc_details, hc_explanation = self.hard_matcher.check_constraints(product_row, requirements_json)
            if passed_all:
                candidate_products_indices.append(index)
                hard_constraint_results.append({
                    'index': index,
                    'hc_score': hc_score,
                    'hc_details': hc_details,
                    'hc_explanation': hc_explanation
                })
        
        if not candidate_products_indices:
            logger.info("No products passed the hard constraints.")
            # Optionally provide feedback on why no products matched
            return []
        
        candidate_products_df = self.products_df.loc[candidate_products_indices].copy()
        # Merge hc_results back to candidate_products_df for easy access
        hc_results_df = pd.DataFrame(hard_constraint_results).set_index('index')
        candidate_products_df = candidate_products_df.join(hc_results_df)

        logger.info(f"{len(candidate_products_df)} products passed hard constraints.")

        # 2. Soft Matching for candidate products
        requirement_query_str = self.soft_matcher.build_requirement_query(requirements_json)
        logger.info(f"Requirement query for soft match: '{requirement_query_str}'")

        recommendation_data = []

        if not requirement_query_str.strip() or self.soft_matcher.model is None:
            logger.warning("Requirement query is empty or soft matcher model not available. Skipping similarity calculation.")
            for idx, row in candidate_products_df.iterrows():
                recommendation_data.append({
                    'product_id': row['Product_ID'],
                    'product_name': row['Product_Name'],
                    'hard_constraint_score': row['hc_score'],
                    'hard_constraint_details': row['hc_details'],
                    'hard_constraint_explanation': row['hc_explanation'],
                    'similarity_score': 0.0,
                    'final_score': row['hc_score'],
                    'explanation': row['hc_explanation'] + ["Text Similarity: 0.00 (Query empty or model issue)"]
                })
        else:
            query_embedding = self.soft_matcher.get_embeddings([requirement_query_str])
            if query_embedding is None or query_embedding.shape[0] == 0:
                 logger.error("Failed to generate embedding for the requirement query.")
                 # Handle as if no similarity (similar to empty query string)
                 for idx, row in candidate_products_df.iterrows():
                    recommendation_data.append({
                        'product_id': row['Product_ID'],
                        'product_name': row['Product_Name'],
                        'hard_constraint_score': row['hc_score'],
                        'hard_constraint_details': row['hc_details'],
                        'hard_constraint_explanation': row['hc_explanation'],
                        'similarity_score': 0.0,
                        'final_score': row['hc_score'],
                        'explanation': row['hc_explanation'] + ["Text Similarity: 0.00 (Query embedding failed)"]
                    })
            else:
                query_embedding = query_embedding[0] # We only have one query
                
                product_embeddings_for_similarity = [
                    emb for emb in candidate_products_df['embedding'].tolist() if emb is not None and emb.size > 0
                ]
                
                # Keep track of original indices for products with valid embeddings
                valid_embedding_indices = [
                    idx for idx, emb in enumerate(candidate_products_df['embedding'].tolist()) if emb is not None and emb.size > 0
                ]

                if not product_embeddings_for_similarity:
                    logger.warning("No valid product embeddings found among candidates for similarity calculation.")
                    cosine_similarities = []
                else:
                    cosine_similarities = self.soft_matcher.calculate_similarity(query_embedding, product_embeddings_for_similarity)

                # Map similarities back to the candidate_products_df
                # Initialize similarity scores to 0 for all candidates
                candidate_products_df['similarity_score_temp'] = 0.0 
                for i, sim_score in enumerate(cosine_similarities):
                    original_candidate_idx = valid_embedding_indices[i]
                    # Get the actual DataFrame index from the candidate_products_df using its iloc
                    df_index = candidate_products_df.index[original_candidate_idx]
                    candidate_products_df.loc[df_index, 'similarity_score_temp'] = sim_score


                for idx, row in candidate_products_df.iterrows():
                    similarity = row.get('similarity_score_temp', 0.0) # Default to 0 if not set
                    scaled_similarity_score = similarity * self.weights["text_similarity_scale"]
                    final_score = row['hc_score'] + scaled_similarity_score
                    
                    full_explanation = list(row['hc_explanation']) # Make a copy
                    full_explanation.append(f"Text Similarity Score: {similarity:.2f} (scaled: {scaled_similarity_score:.2f})")

                    recommendation_data.append({
                        'product_id': row['Product_ID'],
                        'product_name': row['Product_Name'],
                        'hard_constraint_score': row['hc_score'],
                        'hard_constraint_details': row['hc_details'],
                        'similarity_score': round(similarity, 4),
                        'final_score': round(final_score, 2),
                        'explanation': full_explanation
                    })

        # Sort products by final score
        ranked_products = sorted(recommendation_data, key=lambda x: x['final_score'], reverse=True)

        # Format output
        output_recommendations = []
        for p_data in ranked_products[:top_n]:
            output_recommendations.append({
                "Product_ID": p_data['product_id'],
                "Product_Name": p_data['product_name'],
                "Hard_Constraints_Passed_Details": p_data['hard_constraint_details'],
                "Text_Similarity": p_data['similarity_score'],
                "Final_Score": p_data['final_score'],
                "Explanation_Details": p_data['explanation']
            })
            
        return output_recommendations