# main_production.py
import json
import logging
import sys
# Add the recommender_system directory to Python's path
import os
# Ensure the recommender_system package can be found
# This assumes main_production.py is in the parent directory of recommender_system
# If your structure is different, you might need to adjust this
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))


from recommender_system.data_loader import DataLoader
from recommender_system.recommender import ProductRecommender
from recommender_system import config as sys_config # Import your system's config

# --- Setup Logging ---
# Basic configuration for logging
logging.basicConfig(
    level=logging.INFO, # Change to DEBUG for more verbosity
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout) # Log to console
        # logging.FileHandler("recommender.log") # Optionally log to a file
    ]
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logger.info("Starting RAKWireless Product Recommendation System...")

    # --- Load Data ---
    data_loader = DataLoader(
        product_file=sys_config.PRODUCT_FILE,
        feature_file=sys_config.FEATURE_FILE,
        mapping_file=sys_config.MAPPING_FILE
    )
    if not data_loader.load_data() or not data_loader.preprocess_data():
        logger.error("Failed to initialize data. Exiting.")
        sys.exit(1)
    
    products_df, features_df, product_feature_map_df = data_loader.get_data()

    if products_df is None or products_df.empty:
         logger.error("Product data is empty after loading/preprocessing. Exiting.")
         sys.exit(1)

    # --- Initialize Recommender ---
    # This will also precompute embeddings, which might take a moment the first time
    # or if the model needs to be downloaded.
    try:
        recommender = ProductRecommender(products_df, features_df, product_feature_map_df)
        logger.info("ProductRecommender initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize ProductRecommender: {e}", exc_info=True)
        sys.exit(1)


    # --- PASTE YOUR JSON INPUT HERE ---
    # Replace the content of this multi-line string with your JSON requirements.
    # Make sure it's valid JSON.
    # Example:
    # json_input_string = """
    # {
    #   "clientInfo": { "name": "Test Customer"},
    #   "region": {"frequencyBand": "US915"},
    #   "deployment": {"environment": "Outdoor"},
    #   "application": {
    #     "type": "Monitoring",
    #     "subtypes": ["Water Quality"],
    #     "otherSubtype": ""
    #   },
    #   "connectivity": {
    #     "elaborate": {
    #       "wirelessCommunication": ["lorawan"],
    #       "protocolsDataBuses": ["sdi12"]
    #     }
    #   },
    #   "power": ["Battery Powered"],
    #   "additionalDetails": "Need a robust outdoor sensor for water quality monitoring using LoRaWAN and SDI-12 interface. Must be battery efficient."
    # }
    # """

    # Default JSON string if you don't paste anything over it (uses the first example from previous code)
    json_input_string = """
    {
  "clientInfo": {
    "name": "Enterprise Solutions",
    "email": "enterprise@example.com",
    "company": "Industrial IoT Corp.",
    "contactNumber": "+1-555-001-0001"
  },
  "region": {
    "selected": "USA / Canada / South America (915 MHz)",
    "frequencyBand": "US915"
  },
  "deployment": {
    "environment": "Outdoor"
  },
  "application": {
    "type": "Communication",
    "subtypes": ["IoT Gateway", "Connectivity Solutions"],
    "otherSubtype": ""
  },
  "scale": "Large Deployment (50+ devices)",
  "connectivity": {
    "elaborate": {
      "wirelessCommunication": ["lorawan", "lte"],
      "gnssGps": ["gps"],
      "wiredInterfaces": ["ethernet", "poe"],
      "protocolsDataBuses": ["mqtt"]
    }
  },
  "power": ["PoE (Power over Ethernet)", "Solar Power"],
  "additionalDetails": "We require a very reliable and high-performance outdoor LoRaWAN gateway with LTE backhaul for a large-scale industrial deployment in the US. The gateway must handle high device density and harsh environmental conditions. PoE power is preferred. Built-in network server capabilities would be a plus."
}
    """

    # --- Process the JSON Input ---
    try:
        client_requirements_json = json.loads(json_input_string)
        logger.info("\n--- Parsed Client Requirements ---")
        logger.info(json.dumps(client_requirements_json, indent=2))
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON provided: {e}")
        logger.error("Please check the format of the JSON string you pasted.")
        sys.exit(1)


    logger.info("\n--- Generating Recommendations ---")
    recommendations = recommender.recommend(client_requirements_json, top_n=3)

    logger.info("\n--- Top Recommendations ---")
    if recommendations:
        for rec in recommendations:
            print(f"\nProduct ID: {rec['Product_ID']}") # Using print for final user-facing output
            print(f"  Name: {rec['Product_Name']}")
            print(f"  Final Score: {rec['Final_Score']:.2f}")
            print(f"  Text Similarity: {rec['Text_Similarity']:.4f}")
            print(f"  Explanation:")
            for item in rec['Explanation_Details']:
                print(f"    - {item}")
    else:
        print("No suitable products found based on the criteria.")

    # You can comment out or remove the second example if you only want to test the pasted JSON
    run_second_example = False # Set to True if you want to run the hardcoded second example as well
    if run_second_example:
        client_requirements_json_2 = {
            "region": {"frequencyBand": "EU868"},
            "deployment": {"environment": "Indoor"},
            "application": {"type": "Asset Tracking"},
            "connectivity": {"elaborate": {"wirelessCommunication": ["ble", "wifi"]}},
            "power": ["USB Powered"],
            "additionalDetails": "Small indoor tracker for assets using Bluetooth and WiFi."
        }
        logger.info("\n\n--- Generating Recommendations for Second Example (Indoor Tracker) ---")
        logger.info(json.dumps(client_requirements_json_2, indent=2))
        recommendations_2 = recommender.recommend(client_requirements_json_2, top_n=3)

        logger.info("\n--- Top Recommendations (Indoor Tracker) ---")
        if recommendations_2:
            for rec in recommendations_2:
                print(f"\nProduct ID: {rec['Product_ID']}")
                print(f"  Name: {rec['Product_Name']}")
                print(f"  Final Score: {rec['Final_Score']:.2f}")
                print(f"  Text Similarity: {rec['Text_Similarity']:.4f}")
                print(f"  Explanation:")
                for item in rec['Explanation_Details']:
                    print(f"    - {item}")
        else:
            print("No suitable products found for indoor tracker criteria.")