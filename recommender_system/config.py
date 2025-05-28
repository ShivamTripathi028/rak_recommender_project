# recommender_system/config.py

# --- Model Configuration ---
# Recommended: 'all-MiniLM-L6-v2' (good balance of speed and quality)
# or 'paraphrase-multilingual-MiniLM-L12-v2' if multilingual product data
SENTENCE_BERT_MODEL = 'all-MiniLM-L6-v2'

# --- Scoring Weights ---
WEIGHTS = {
    "frequency_band": 5,
    "environment": 3,
    "connectivity_option": 2, # Per matched option
    "power_keyword": 1,
    "text_similarity_scale": 10 # Similarity (0-1) will be multiplied by this
}

# --- Mappings ---
# Mapping from user-friendly power labels to keywords for searching
POWER_KEYWORD_MAPPING = {
    "DC Power": ["dc power", "dc supply"],
    "AC Power": ["ac power", "ac supply"],
    "Battery Powered": ["battery", "batteries", "battery-powered"],
    "USB Powered": ["usb power", "usb powered"],
    "Solar Power": ["solar", "solar power", "solar-powered"],
    "PoE (Power over Ethernet)": ["poe", "power over ethernet"],
    "Other / Not Specified": []
}

# Mapping from JSON connectivity IDs to potential keywords in product data
CONNECTIVITY_JSON_TO_PRODUCT_KEYWORDS = {
    "lorawan": ["lorawan", "lora wan"],
    "lora_p2p": ["lora p2p", "lora point-to-point"],
    "lora": ["lora"],
    "meshtastic": ["meshtastic"],
    "wifi": ["wi-fi", "wifi", "wlan", "wireless lan"],
    "ble": ["ble", "bluetooth low energy", "bluetooth le"],
    "nfc": ["nfc", "near field communication"],
    "uwb": ["uwb", "ultra-wideband"],
    "lte": ["lte", "4g"],
    "5g": ["5g", "5g nr"], # Added 5G based on product P3
    "lte_m_cat_m1": ["lte-m", "cat-m1", "lte cat m1"],
    "nb_iot": ["nb-iot", "narrowband iot"],
    "gsm": ["gsm"],
    "agw": ["agw"],
    "lpwan_other": [],
    "gps": ["gps", "global positioning system"],
    "gnss": ["gnss"],
    "ethernet": ["ethernet", "rj45", "lan port"],
    "poe": ["poe", "power over ethernet"],
    "usb": ["usb"],
    "pcie": ["pcie", "pci express"],
    "twisted_pair": ["twisted pair"],
    "coaxial_cable": ["coaxial"],
    "i2c": ["i2c", "i²c"],
    "spi": ["spi"],
    "uart_serial": ["uart", "serial port", "rs232", "rs-232"],
    "rs485": ["rs485", "rs-485"],
    "sdi12": ["sdi12", "sdi-12"],
    "can_bus": ["can bus", "canbus"],
    "lin_bus": ["lin bus"],
    "mqtt": ["mqtt"],
    "adc": ["adc", "analog-to-digital", "analog input"],
    "digital_io": ["digital i/o", "digital io", "gpio", "digital input", "digital output"]
}

# --- Data File Paths ---
# Assuming data files are in a 'data' subdirectory relative to where main_production.py is run
PRODUCT_FILE = 'data/product_table.csv'
FEATURE_FILE = 'data/feature_table.csv'
MAPPING_FILE = 'data/mapping_table.csv'