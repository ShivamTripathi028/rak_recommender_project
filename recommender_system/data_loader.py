# recommender_system/data_loader.py
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, product_file, feature_file, mapping_file):
        self.product_file = product_file
        self.feature_file = feature_file
        self.mapping_file = mapping_file
        self.products_df = None
        self.features_df = None
        self.product_feature_map_df = None

    def load_data(self):
        """Loads data from CSV files into pandas DataFrames."""
        try:
            self.products_df = pd.read_csv(self.product_file)
            self.features_df = pd.read_csv(self.feature_file)
            self.product_feature_map_df = pd.read_csv(self.mapping_file)
            logger.info("All data files loaded successfully.")
            return True
        except FileNotFoundError as e:
            logger.error(f"Error loading data: {e}. Please ensure file paths are correct.")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred during data loading: {e}")
            return False

    def preprocess_data(self):
        """Preprocesses the loaded data."""
        if self.products_df is None or self.features_df is None or self.product_feature_map_df is None:
            logger.error("Data not loaded. Cannot preprocess.")
            return False

        # Fill NaNs in text columns
        text_cols_products = ['Product_Name', 'Description_And_Application', 'Notes', 'Connectivity', 'Deployment_Environment', 'Region Support']
        for col in text_cols_products:
            if col in self.products_df.columns:
                self.products_df[col] = self.products_df[col].fillna('')
            else:
                logger.warning(f"Expected column '{col}' not found in products table. Skipping NaN fill for it.")
                self.products_df[col] = '' # Add empty column to prevent downstream errors


        text_cols_features = ['Feature_Name', 'Feature_Description']
        for col in text_cols_features:
            if col in self.features_df.columns:
                self.features_df[col] = self.features_df[col].fillna('')
            else:
                logger.warning(f"Expected column '{col}' not found in features table. Skipping NaN fill for it.")
                self.features_df[col] = ''


        # Normalize and split 'Region Support'
        if 'Region Support' in self.products_df.columns:
            self.products_df['Region_Support_List'] = self.products_df['Region Support'].apply(
                lambda x: [region.strip().lower() for region in str(x).split(',') if region.strip()]
            )
        else:
            self.products_df['Region_Support_List'] = pd.Series([[] for _ in range(len(self.products_df))])


        # Normalize 'Connectivity' for keyword search AND list creation
        if 'Connectivity' in self.products_df.columns:
            self.products_df['Connectivity_Lower_Text'] = self.products_df['Connectivity'].str.lower()
            self.products_df['Connectivity_List'] = self.products_df['Connectivity'].apply(
                lambda x: [conn.strip().lower() for conn in str(x).split(',') if conn.strip()]
            )
        else:
            self.products_df['Connectivity_Lower_Text'] = ''
            self.products_df['Connectivity_List'] = pd.Series([[] for _ in range(len(self.products_df))])


        # Lowercase 'Deployment_Environment'
        if 'Deployment_Environment' in self.products_df.columns:
            self.products_df['Deployment_Environment_Lower'] = self.products_df['Deployment_Environment'].str.lower()
        else:
             self.products_df['Deployment_Environment_Lower'] = ''


        # Ensure Product_ID is consistent for merging
        self.products_df['Product_ID'] = self.products_df['Product_ID'].astype(str)
        self.product_feature_map_df['Product_ID'] = self.product_feature_map_df['Product_ID'].astype(str)
        self.features_df['Feature_ID'] = self.features_df['Feature_ID'].astype(str)
        self.product_feature_map_df['Feature_ID'] = self.product_feature_map_df['Feature_ID'].astype(str)

        logger.info("Data preprocessing complete.")
        return True

    def get_data(self):
        return self.products_df, self.features_df, self.product_feature_map_df