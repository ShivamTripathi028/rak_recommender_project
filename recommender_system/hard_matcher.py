# recommender_system/hard_matcher.py
import re
import logging
from . import config # Use relative import

logger = logging.getLogger(__name__)

class HardConstraintMatcher:
    def __init__(self):
        self.weights = config.WEIGHTS
        self.power_keyword_mapping = config.POWER_KEYWORD_MAPPING
        self.connectivity_json_to_product_keywords = config.CONNECTIVITY_JSON_TO_PRODUCT_KEYWORDS

    def check_constraints(self, product_series, requirements_json):
        constraints_passed_details = {}
        overall_pass = True
        score = 0
        explanation = []

        # 1. Frequency Band
        req_freq_band = requirements_json.get('region', {}).get('frequencyBand', '').lower()
        product_regions = product_series.get('Region_Support_List', [])
        if req_freq_band:
            if req_freq_band in product_regions:
                constraints_passed_details['frequency_band'] = True
                score += self.weights["frequency_band"]
                explanation.append(f"Matched frequency band: {req_freq_band.upper()}")
            else:
                constraints_passed_details['frequency_band'] = False
                overall_pass = False
                explanation.append(f"FAILED frequency band: Product supports {product_regions}, required {req_freq_band.upper()}")
        else:
            constraints_passed_details['frequency_band'] = None
            explanation.append("Frequency band not specified by user.")


        # 2. Deployment Environment
        req_env = requirements_json.get('deployment', {}).get('environment', '').lower()
        product_env = product_series.get('Deployment_Environment_Lower', '')
        if req_env:
            matched_env = False
            if product_env == req_env:
                matched_env = True
            elif req_env == "both" and product_env in ["indoor", "outdoor", "both", ""]: # if user wants both, product can be indoor, outdoor, both, or even unspecified
                matched_env = True
            elif product_env == "both" and req_env in ["indoor", "outdoor"]: # if product is 'both', it matches specific indoor/outdoor requests
                 matched_env = True

            if matched_env:
                constraints_passed_details['environment'] = True
                score += self.weights["environment"]
                explanation.append(f"Matched environment: User '{req_env.capitalize()}', Product '{product_env.capitalize()}'")
            else:
                constraints_passed_details['environment'] = False
                overall_pass = False
                explanation.append(f"FAILED environment: Product is '{product_env.capitalize()}', required '{req_env.capitalize()}'")
        else:
            constraints_passed_details['environment'] = None
            explanation.append("Deployment environment not specified by user.")


        # 3. Connectivity
        user_connectivity_reqs_ids = []
        elaborate_conn = requirements_json.get('connectivity', {}).get('elaborate', {})
        for conn_category_key, conn_ids in elaborate_conn.items():
            user_connectivity_reqs_ids.extend([item.lower() for item in conn_ids])
        user_connectivity_reqs_ids = list(set(user_connectivity_reqs_ids))

        matched_conn_options_found = []
        if user_connectivity_reqs_ids:
            product_conn_text_lower = product_series.get('Connectivity_Lower_Text', '')
            product_conn_list_lower = product_series.get('Connectivity_List', [])
            
            any_user_req_matched = False
            for req_conn_id in user_connectivity_reqs_ids:
                keywords_to_check = self.connectivity_json_to_product_keywords.get(req_conn_id, [req_conn_id])
                req_conn_id_matched_this_product = False
                for kw in keywords_to_check:
                    # Check in the pre-split list or the raw text
                    if kw in product_conn_list_lower or (product_conn_text_lower and kw in product_conn_text_lower):
                        matched_conn_options_found.append(req_conn_id.upper())
                        score += self.weights["connectivity_option"]
                        any_user_req_matched = True
                        req_conn_id_matched_this_product = True
                        break # Matched this req_conn_id, move to the next one
                # logger.debug(f"Product {product_series['Product_ID']} for conn_id {req_conn_id}: checked {keywords_to_check}, matched: {req_conn_id_matched_this_product}")


            if any_user_req_matched: # At least one of the user's requested connectivities matched
                constraints_passed_details['connectivity'] = True
                explanation.append(f"Matched connectivity options: {', '.join(sorted(list(set(matched_conn_options_found))))}")
            else: # User specified connectivities, but NONE of them matched this product
                constraints_passed_details['connectivity'] = False
                overall_pass = False
                explanation.append(f"FAILED connectivity: No product match for user required options: {', '.join(user_connectivity_reqs_ids)}")
        else:
            constraints_passed_details['connectivity'] = None
            explanation.append("Connectivity not specified by user.")


        # 4. Power
        req_power_options_labels = requirements_json.get('power', [])
        product_text_for_power = (str(product_series.get('Description_And_Application', '')) + " " + \
                                  str(product_series.get('Notes', ''))).lower()
        
        power_keywords_found_in_product = []
        if req_power_options_labels:
            any_power_match = False
            for user_power_label in req_power_options_labels:
                power_keywords_for_label = self.power_keyword_mapping.get(user_power_label, [])
                for p_keyword in power_keywords_for_label:
                    if re.search(r'\b' + re.escape(p_keyword) + r'\b', product_text_for_power):
                        power_keywords_found_in_product.append(f"'{p_keyword}' (for {user_power_label})")
                        any_power_match = True
                        # Score for power is +1 if *any* keyword matches, not per keyword.
                        # So we break after the first keyword for this label, and then the outer loop handles the score once.
                        break 
                if any_power_match: # If one keyword for this label matched, we might not need to check other keywords for the same label
                    pass # continue to next user_power_label, or if we want one score for ANY power match, we can break here too

            if power_keywords_found_in_product: # If list is not empty, at least one keyword matched
                constraints_passed_details['power'] = True
                score += self.weights["power_keyword"] # Add score once if any power option matched
                explanation.append(f"Power requirement(s) met: Found {', '.join(list(set(power_keywords_found_in_product)))}")
            else: # User specified power, but no keywords matched
                constraints_passed_details['power'] = False
                overall_pass = False
                explanation.append(f"FAILED power: No product match for user required power options ({', '.join(req_power_options_labels)}) in product description/notes.")
        else:
            constraints_passed_details['power'] = None
            explanation.append("Power requirements not specified by user.")

        return overall_pass, score, constraints_passed_details, explanation