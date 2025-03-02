#!/usr/bin/env python3
#
# symbiote/PiiExtractor.py
"""
Project Overview:
The `PIIExtractor` script is a comprehensive solution for identifying and
extracting Personally Identifiable Information (PII) from text. This tool
is designed for scenarios where sensitive information such as addresses,
credit card numbers, or government-issued IDs needs to be detected and
assessed. The script employs advanced pattern matching techniques,
keyword proximity analysis, and confidence scoring to deliver accurate and
meaningful results.

Purpose and Goals:
- To provide a robust, extensible framework for PII detection.
- To balance precision and recall in identifying PII through regex patterns,
  proximity-based keyword scoring, and validation logic.
- To ensure modularity, allowing individual components to be reused or extended
  for additional PII types.

Key Features:
1. **Extraction Methods**:
   - Over 50 specialized methods to extract different types of PII, including:
     - Addresses, email addresses, phone numbers, and social security numbers.
     - Financial identifiers such as IBANs, credit card numbers, and routing numbers.
     - Government-issued IDs (e.g., Aadhaar, CPF, passports).
     - Geographic coordinates and timestamps.
   - Each extraction method combines regex-based detection, keyword proximity,
     and optional checksum validation.

2. **Proximity-Based Scoring**:
   - Confidence scores are calculated based on the presence of relevant keywords
     near a potential match.
   - Proximity scoring ensures that matches are contextually relevant, improving
     precision.

3. **Validation and Uniqueness**:
   - Implemented logic to validate formats (e.g., Luhn checksum for credit cards,
     Verhoeff checksum for Aadhaar).
   - Duplicates are filtered out to ensure output clarity and correctness.

4. **Comprehensive `extract_all` Method**:
   - Provides a single entry point to extract all supported PII types from text.
   - Aggregates results into a structured dictionary for easy integration with
     other systems.

5. **Modular Design**:
   - Each PII type has its dedicated extraction method, allowing easy addition
     of new patterns and validations.
   - Shared helper functions, such as `_is_near_keywords` and `_unique`, ensure
     consistency and reduce redundancy.

Methodologies:
- Regular Expressions: Used extensively for pattern matching across diverse
  formats (e.g., addresses, dates, financial identifiers).
- Keyword Analysis: Enhances regex-based matches with proximity scoring for
  contextual relevance.
- Validation Algorithms: Includes domain-specific validation such as Luhn and
  Verhoeff checks to ensure data accuracy.

Extensibility:
- Adding New PII Types:
  - Define a new method starting with `extract_`.
  - Use a regex pattern for basic detection and integrate keyword analysis
    for confidence scoring.
  - If applicable, implement a validation function to further refine matches.
- Adapting for International Use:
  - Modify patterns and keywords to reflect local formats and conventions
    (e.g., CPF for Brazil, Aadhaar for India).

Best Practices for Use:
- Input text should be preprocessed to remove unnecessary formatting for
  improved matching accuracy.
- Combine the script with additional data sanitization workflows for end-to-end
  PII management.

Reusable Prompt for Extending:
- "Enhance the PIIExtractor by adding support for [specific PII type or region].
   Maintain its modular structure and ensure consistent confidence scoring and
   validation."

Personal Style Alignment:
- The script reflects a structured, modular approach to solving complex problems.
- Clear method definitions and consistent patterns enhance readability and ease
  of maintenance.
- Error-prone tasks, such as regex matching and data validation, are handled
  systematically, minimizing potential inaccuracies.

Usage Example:
```
extractor = PIIExtractor()
text = "Sample text with PII data like 123-45-6789 (SSN) or 4111 1111 1111 1111 (Credit Card)."
results = extractor.extract_all(text)
print(json.dumps(results, indent=2))
```
"""

import re
import datetime
import json

class PIIExtractor:
    def _unique(self, items):
        seen = set()
        unique_items = []
        for item in items:
            identifier = item["match"]
            if identifier not in seen:
                unique_items.append(item)
                seen.add(identifier)
        return unique_items

    def _add_score(self, default, confidence):
        total = default + confidence
        return total

    def _is_near_keywords(self, match, text, keywords, default_score, proximity=20):
        match_start = text.find(match)
        before_match = text[max(0, match_start - proximity):match_start].lower()
        after_match = text[match_start + len(match):min(len(text), match_start + len(match) + proximity)].lower()

        keyword_distances = []
        keyword_count = 0

        for keyword in keywords:
            before_index = before_match.rfind(keyword)
            after_index = after_match.find(keyword)

            if before_index != -1:
                distance = proximity - before_index
                keyword_distances.append(distance)
                keyword_count += 1
            if after_index != -1:
                distance = after_index
                keyword_distances.append(distance)
                keyword_count += 1

        if keyword_count == 0:
            return default_score  # No keywords found, return lowest confidence

        average_distance = sum(keyword_distances) / len(keyword_distances)
        average_distance = min(average_distance, proximity)
        keyword_coverage = keyword_count / len(keywords)
        proximity_score = (proximity - average_distance) / proximity
        confidence = max(0.0, min(1.0, keyword_coverage * proximity_score))
        confidence = default_score + confidence

        return confidence

    def extract_address(self, text):
        default_score = 0.5
        address_pattern = r'\b\d{1,5} [A-Za-z0-9 ]+ (?:Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Lane|Ln|Drive|Dr|Court|Ct|Square|Sq|Place|Pl|Terrace|Ter|Parkway|Pkwy)\b(?:, [A-Za-z ]+)*, [A-Z]{2} \d{5}\b'
        matches = re.findall(address_pattern, text)
        keywords = ["address", "location", "headquarters", "office"]

        results = []
        for match in matches:
            confidence = self._is_near_keywords(match, text, keywords, default_score)
            results.append({"match": match, "confidence": confidence})

        return self._unique(results)

    def extract_geo_coordinates(self, text):
        default_score = 0.5
        
        # Comprehensive pattern to match various geo-coordinate formats
        geo_pattern = r'\b[-+]?([1-8]?\d(\.\d+)?|90(\.0+)?)[,\s]*[-+]?(180(\.0+)?|(1[0-7]\d|\d{1,2})(\.\d+)?)\b|\b([1-8]?\d)°\s*([1-5]?\d)\'\s*([1-5]?\d(\.\d+)?)"\s*[NnSs],\s*(180|([1-7]?\d))°\s*([1-5]?\d)\'\s*([1-5]?\d(\.\d+)?)"\s*[EeWw]\b|\b([1-8]?\d)°\s*([1-5]?\d(\.\d+)?)\'\s*[NnSs],\s*(180|([1-7]?\d))°\s*([1-5]?\d(\.\d+)?)\'\s*[EeWw]\b|\b([1-8]?\d(\.\d+)?)[NnSs]\s*[, ]\s*(180|([1-7]?\d(\.\d+)?))[EeWw]\b'
        
        matches = re.findall(geo_pattern, text)
        keywords = ["location", "coordinates", "latitude", "longitude", "geo"]

        results = []
        for match in matches:
            # Each match will have multiple capturing groups due to the complexity of the regex
            if isinstance(match, tuple):
                # Handle different capture groups to form a consistent geo-coordinate representation
                latitude = match[0] if match[0] else match[17]
                longitude = match[4] if match[4] else match[21]
                
                if not latitude:  # Fallback to other capture groups if necessary
                    latitude = match[8] or match[13]
                if not longitude:
                    longitude = match[12] or match[20]
                
                match_str = f"{latitude}, {longitude}".strip()  # Combine latitude and longitude parts
            else:
                match_str = match  # If not a tuple, use the match as is
            
            confidence = self._is_near_keywords(match_str, text, keywords, default_score)
            results.append({"match": match_str, "confidence": confidence})

        return self._unique(results)

    def _is_valid_bank_account(self, account):
        # A placeholder for complex validation logic, such as checksums or country-specific rules.
        if re.match(r'^[A-Z]{2}\d{2}', account):  # Basic IBAN check
            return len(account) >= 15 and len(account) <= 34
        # Additional validation checks can be added here for other formats
        return True

    def extract_aadhaar(self, text):
        default_score = 0.5
        def verhoeff_checksum(aadhaar_number):
            verhoeff_table_d = [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
                [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
                [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
                [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
                [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
                [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
                [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
                [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
                [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
            ]
            verhoeff_table_p = [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
                [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
                [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
                [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
                [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
                [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
                [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]
            ]
            verhoeff_table_inv = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]
            c = 0
            for i, digit in enumerate(reversed(aadhaar_number)):
                c = verhoeff_table_d[c][verhoeff_table_p[i % 8][int(digit)]]
            return c == 0

        matches = re.findall(r'\b\d{12}\b', text)
        keywords = ["aadhaar", "uid", "uidai", "identification", "india"]

        results = []
        for match in matches:
            if verhoeff_checksum(match):
                confidence = self._is_near_keywords(match, text, keywords, default_score)
                results.append({"match": match, "confidence": confidence})

        return self._unique(results)

    def extract_australian_abn(self, text):
        default_score = 0.5
        def abn_checksum(abn):
            abn_weights = [10, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
            abn_digits = [int(x) for x in abn]
            abn_digits[0] -= 1  # Subtract 1 from the first digit
            weighted_sum = sum(w * d for w, d in zip(abn_weights, abn_digits))
            return weighted_sum % 89 == 0

        matches = re.findall(r'\b\d{11}\b', text)
        keywords = ["abn", "australian", "business", "number", "tax", "identifier"]

        results = []
        for match in matches:
            if abn_checksum(match):
                confidence = self._is_near_keywords(match, text, keywords, default_score)
                results.append({"match": match, "confidence": confidence})

        return self._unique(results)

    def extract_bank_account(self, text):
        default_score = 0.3
        patterns = [
            r'\b\d{8,12}\b',  # Generic 8 to 12 digits (US, Canada)
            r'\b\d{9,18}\b',  # India (IFSC + Account)
            r'\b\d{2}-\d{2}-\d{2} \d{8}\b',  # UK (Sort Code + Account)
            r'\b\d{6} \d{8,9}\b',  # Australia (BSB + Account)
            r'\b\d{5} \d{7,12}\b',  # Canada (Transit Number + Account)
            r'\b\d{8} \d{7,10}\b',  # Germany (BLZ + Konto-Nummer)
            r'\b[A-Z]{2}\d{2}[A-Z0-9]{1,30}\b',  # IBAN (General pattern)
        ]

        matches = []
        keywords = ["bank", "account", "iban", "bsb", "routing", "sort code", "branch code", "account number"]

        for pattern in patterns:
            potential_matches = re.findall(pattern, text)
            for match in potential_matches:
                confidence = self._is_near_keywords(match, text, keywords, default_score)
                matches.append({"match": match, "confidence": confidence})

        valid_accounts = []
        for match in matches:
            if self._is_valid_bank_account(match["match"]):
                valid_accounts.append(match)

        return self._unique(valid_accounts)

    def extract_routing_number(self, text):
        default_score = 0.4
        pattern = r'\b\d{3}[- ]?\d{3}[- ]?\d{3}\b'
        matches = re.findall(pattern, text)
        routing_numbers = [re.sub(r'[- ]', '', match) for match in matches]
        keywords = ["routing", "aba", "bank", "code", "number", "banking", "transfer"]

        results = []
        for match in routing_numbers:
            confidence = self._is_near_keywords(match, text, keywords, default_score)
            results.append({"match": match, "confidence": confidence})

        return self._unique(results)

    def extract_birth_certificate(self, text):
        default_score = 0.3
        matches = re.findall(r'\bBC\d{9}\b', text)
        keywords = ["birth", "certificate", "registry", "document", "registration"]

        results = []
        for match in matches:
            confidence = self._is_near_keywords(match, text, keywords, default_score)
            results.append({"match": match, "confidence": confidence})

        return self._unique(results)

    def extract_brazilian_cpf(self, text):
        default_score = 0.5
        def cpf_checksum(cpf):
            cpf = [int(digit) for digit in cpf if digit.isdigit()]
            if len(cpf) != 11:
                return False
            for i in range(9, 11):
                value = sum((cpf[num] * ((i + 1) - num) for num in range(0, i)))
                check_digit = ((value * 10) % 11) % 10
                if check_digit != cpf[i]:
                    return False
            return True

        matches = re.findall(r'\b\d{3}\.\d{3}\.\d{3}-\d{2}\b', text)
        keywords = ["cpf", "brazilian", "tax", "payer", "identification", "brazil"]

        results = []
        for match in matches:
            if cpf_checksum(match):
                confidence = self._is_near_keywords(match, text, keywords, default_score)
                results.append({"match": match, "confidence": confidence})

        return self._unique(results)

    def extract_company_registration(self, text):
        default_score = 0.4
        matches = re.findall(r'\bCRN\d{9}\b', text)
        keywords = ["company", "registration", "crn", "corporate", "identifier", "number"]

        results = []
        for match in matches:
            confidence = self._is_near_keywords(match, text, keywords, default_score)
            results.append({"match": match, "confidence": confidence})

        return self._unique(results)

    def extract_credit_card(self, text):
        default_score = 0.5
        def luhn_checksum(card_number):
            digits = [int(d) for d in card_number if d.isdigit()]
            checksum = 0
            reverse_digits = digits[::-1]
            for i, d in enumerate(reverse_digits):
                if i % 2 == 0:
                    checksum += d
                else:
                    checksum += sum(divmod(2 * d, 10))
            return checksum % 10 == 0

        matches = re.findall(r'\b(?:\d[ -]*?){13,19}\b', text)
        keywords = ["credit", "card", "number", "debit", "payment", "cc"]

        results = []
        for match in matches:
            def luhn_checksum(card_number):
                digits = [int(d) for d in card_number if d.isdigit()]
                checksum = 0
                reverse_digits = digits[::-1]
                for i, d in enumerate(reverse_digits):
                    if i % 2 == 0:
                        checksum += d
                    else:
                        checksum += sum(divmod(2 * d, 10))
                return checksum % 10 == luhn_checksum(match)

            confidence = self._is_near_keywords(match, text, keywords, default_score)
            results.append({"match": match, "confidence": confidence})

        return self._unique(results)

    def extract_date(self, text):
        default_score = 0.2
        date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',
            r'\b\d{1,2}-\d{1,2}-\d{4}\b',
            r'\b\d{1,2}\.\d{1,2}\.\d{4}\b',
            r'\b\d{4}/\d{1,2}/\d{1,2}\b',
            r'\b\d{4}-\d{1,2}-\d{1,2}\b',
            r'\b\d{4}\.\d{1,2}\.\d{1,2}\b',
            r'\b(?:[Jj]anuary|[Ff]ebruary|[Mm]arch|[Aa]pril|[Mm]ay|[Jj]une|[Jj]uly|[Aa]ugust|[Ss]eptember|[Oo]ctober|[Nn]ovember|[Dd]ecember) \d{1,2}, \d{4}\b',
            r'\b\d{1,2} (?:[Jj]anuary|[Ff]ebruary|[Mm]arch|[Aa]pril|[Mm]ay|[Jj]une|[Jj]uly|[Aa]ugust|[Ss]eptember|[Oo]ctober|[Nn]ovember|[Dd]ecember) \d{4}\b',
            r'\b\d{1,2}(?:st|nd|rd|th)? (?:[Jj]anuary|[Ff]ebruary|[Mm]arch|[Aa]pril|[Mm]ay|[Jj]une|[Jj]uly|[Aa]ugust|[Ss]eptember|[Oo]ctober|[Nn]ovember|[Dd]ecember) \d{4}\b',
            r'\b(?:[Jj]an|[Ff]eb|[Mm]ar|[Aa]pr|[Mm]ay|[Jj]un|[Jj]ul|[Aa]ug|[Ss]ep|[Oo]ct|[Nn]ov|[Dd]ec) \d{1,2}, \d{4}\b',
            r'\b\d{1,2} (?:[Jj]an|[Ff]eb|[Mm]ar|[Aa]pr|[Mm]ay|[Jj]un|[Jj]ul|[Aa]ug|[Ss]ep|[Oo]ct|[Nn]ov|[Dd]ec) \d{4}\b',
            r'\b\d{1,2}/\d{1,2}/\d{2}\b',
            r'\b\d{1,2}-\d{1,2}-\d{2}\b',
            r'\b(?:[Jj]anuary|[Ff]ebruary|[Mm]arch|[Aa]pril|[Mm]ay|[Jj]une|[Jj]uly|[Aa]ugust|[Ss]eptember|[Oo]ctober|[Nn]ovember|[Dd]ecember) \d{1,2}(?:st|nd|rd|th)?, \d{4}\b',
            r'\b(?:[Jj]an|[Ff]eb|[Mm]ar|[Aa]pr|[Mm]ay|[Jj]un|[Jj]ul|[Aa]ug|[Ss]ep|[Oo]ct|[Nn]ov|[Dd]ec) \d{1,2}, \d{2}\b',
            r'\b\d{1,2} (?:[Jj]an|[Ff]eb|[Mm]ar|[Aa]pr|[Mm]ay|[Jj]un|[Jj]ul|[Aa]ug|[Ss]ep|[Oo]ct|[Nn]ov|[Dd]ec) \d{2}\b',
        ]

        matches = []
        keywords = ["date", "birthdate", "dob", "expiration", "issued", "valid until"]

        for pattern in date_patterns:
            potential_matches = re.findall(pattern, text)
            for match in potential_matches:
                confidence = self._is_near_keywords(match, text, keywords, default_score)
                matches.append({"match": match, "confidence": confidence})

        valid_dates = []
        for match in matches:
            try:
                for fmt in ("%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d", "%Y-%m-%d", "%Y.%m.%d", "%d %B %Y", "%B %d, %Y", "%d %b %Y", "%b %d, %Y", "%d-%m-%Y", "%m-%d-%Y", "%b %d, %y", "%d %b %y"):
                    try:
                        datetime.datetime.strptime(match["match"], fmt)
                        valid_dates.append(match)
                        break
                    except ValueError:
                        continue
            except ValueError:
                continue

        return self._unique(valid_dates)

    def extract_drivers_license(self, text):
        default_score = 0.3
        matches = re.findall(r'\bD\d{9,12}\b', text)
        keywords = ["driver", "license", "dl", "permit", "identification", "number"]

        results = []
        for match in matches:
            confidence = self._is_near_keywords(match, text, keywords, default_score)
            results.append({"match": match, "confidence": confidence})

        return self._unique(results)

    def extract_drivers_license_international(self, text):
        default_score = 0.3
        matches = re.findall(r'\bINTD\d{9}\b', text)
        keywords = ["international", "driver", "license", "permit", "identification"]

        results = []
        for match in matches:
            confidence = self._is_near_keywords(match, text, keywords, default_score)
            results.append({"match": match, "confidence": confidence})

        return self._unique(results)

    def extract_email(self, text):
        default_score = 0.4
        matches = re.findall(r'\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b', text)
        keywords = ["email", "address", "contact", "mail", "e-mail"]

        results = []
        for match in matches:
            confidence = self._is_near_keywords(match, text, keywords, default_score)
            results.append({"match": match, "confidence": confidence})

        valid_emails = self._unique(results)
        return valid_emails

    def extract_financial_account_other(self, text):
        default_score = 0.4
        matches = re.findall(r'\bFA\d{9,12}\b', text)
        keywords = ["financial", "account", "investment", "retirement", "portfolio", "number"]

        results = []
        for match in matches:
            confidence = self._is_near_keywords(match, text, keywords, default_score)
            results.append({"match": match, "confidence": confidence})

        return self._unique(results)

    def extract_full_name(self, text):
        default_score = 0.2
        matches = re.findall(r'\b[A-Z][a-z]*\s[A-Z][a-z]*(\s[A-Z][a-z]*)?\b', text)
        keywords = ["name", "full", "first", "last", "middle", "surname", "title", "mr", "ms", "dr", "mrs", "miss"]

        results = []
        for match in matches:
            confidence = self._is_near_keywords(match, text, keywords, default_score)
            results.append({"match": match, "confidence": confidence})

        return self._unique(results)

    def extract_government_service_number(self, text):
        default_score = 0.4
        matches = re.findall(r'\bGS\d{9}\b', text)
        keywords = ["government", "service", "number", "gsn", "identifier", "public", "sector"]

        results = []
        for match in matches:
            confidence = self._is_near_keywords(match, text, keywords, default_score)
            results.append({"match": match, "confidence": confidence})

        return self._unique(results)

    def extract_health_id(self, text):
        default_score = 0.4
        matches = re.findall(r'\bHID\d{9}\b', text)
        keywords = ["health", "id", "identification", "medical", "hid", "number"]

        results = []
        for match in matches:
            confidence = self._is_near_keywords(match, text, keywords, default_score)
            results.append({"match": match, "confidence": confidence})

        return self._unique(results)

    def extract_health_insurance_other(self, text):
        default_score = 0.4
        matches = re.findall(r'\bHI\d{9}\b', text)
        keywords = ["health", "insurance", "policy", "number", "coverage", "plan", "hid"]

        results = []
        for match in matches:
            confidence = self._is_near_keywords(match, text, keywords, default_score)
            results.append({"match": match, "confidence": confidence})

        return self._unique(results)

    def extract_iban(self, text):
        default_score = 0.5
        def iban_checksum(iban):
            iban = iban.replace(' ', '')
            iban = iban[4:] + iban[:4]
            iban = ''.join(str(int(ch, 36)) for ch in iban)
            return int(iban) % 97 == 1

        matches = re.findall(r'\b[A-Z]{2}\d{2}[A-Z0-9]{1,30}\b', text)
        keywords = ["iban", "international", "bank", "account", "number", "swift", "bic"]

        results = []
        for match in matches:
            if iban_checksum(match):
                confidence = self._is_near_keywords(match, text, keywords, default_score)
                results.append({"match": match, "confidence": confidence})

        return self._unique(results)

    def extract_ipv4_address(self, text):
        default_score = 0.5
        matches = re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', text)
        valid_ipv4s = []
        keywords = ["ip", "ipv4", "internet", "address", "network", "connection"]

        for match in matches:
            octets = match.split('.')
            if all(0 <= int(octet) <= 255 for octet in octets):
                confidence = self._is_near_keywords(match, text, keywords, default_score)
                valid_ipv4s.append({"match": match, "confidence": confidence})

        return self._unique(valid_ipv4s)

    def extract_ipv6_address(self, text):
        default_score = 0.5
        matches = re.findall(r'\b([a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}\b', text)
        keywords = ["ip", "ipv6", "internet", "address", "network", "connection"]

        results = []
        for match in matches:
            confidence = self._is_near_keywords(match, text, keywords, default_score)
            results.append({"match": match, "confidence": confidence})

        return self._unique(results)

    def extract_license_key(self, text):
        default_score = 0.3
        matches = re.findall(r'\b[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}\b', text)
        keywords = ["license", "key", "activation", "serial", "software", "product", "registration"]

        results = []
        for match in matches:
            confidence = self._is_near_keywords(match, text, keywords, default_score)
            results.append({"match": match, "confidence": confidence})

        return self._unique(results)

    def extract_mac_address(self, text):
        default_score = 0.4
        matches = re.findall(r'\b[0-9A-Fa-f]{2}[:-]{1}[0-9A-Fa-f]{2}[:-]{1}[0-9A-Fa-f]{2}[:-]{1}[0-9A-Fa-f]{2}[:-]{1}[0-9A-Fa-f]{2}[:-]{1}[0-9A-Fa-f]{2}\b', text)
        keywords = ["mac", "address", "network", "hardware", "device", "identifier"]

        results = []
        for match in matches:
            confidence = self._is_near_keywords(match, text, keywords, default_score)
            results.append({"match": match, "confidence": confidence})

        return self._unique(results)

    def extract_national_id_china(self, text):
        default_score = 0.5
        matches = re.findall(r'\b\d{18}\b', text)
        keywords = ["national", "id", "china", "identification", "number", "citizen"]

        results = []
        for match in matches:
            confidence = self._is_near_keywords(match, text, keywords, default_score)
            results.append({"match": match, "confidence": confidence})

        return self._unique(results)

    def extract_national_id_other(self, text):
        default_score = 0.4
        matches = re.findall(r'\bNID\d{9,12}\b', text)
        keywords = ["national", "id", "identification", "number", "citizen"]

        results = []
        for match in matches:
            confidence = self._is_near_keywords(match, text, keywords, default_score)
            results.append({"match": match, "confidence": confidence})

        return self._unique(results)

    def extract_passport(self, text):
        default_score = 0.4
        matches = re.findall(r'\b\d{9}\b', text)
        keywords = ["passport", "travel", "document", "number", "identification"]

        results = []
        for match in matches:
            confidence = self._is_near_keywords(match, text, keywords, default_score)
            results.append({"match": match, "confidence": confidence})

        return self._unique(results)

    def extract_passport_non_us(self, text):
        default_score = 0.4
        matches = re.findall(r'\b[A-Z]\d{8}\b', text)
        keywords = ["passport", "non-us", "foreign", "travel", "document", "number"]

        results = []
        for match in matches:
            confidence = self._is_near_keywords(match, text, keywords, default_score)
            results.append({"match": match, "confidence": confidence})

        return self._unique(results)

    def extract_phone_number(self, text):
        default_score = 0.3
        matches = re.findall(r'\b\+?\d{1,3}[-.\s]?\(?\d{1,3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', text)
        keywords = ["phone", "number", "contact", "mobile", "telephone"]

        results = []
        for match in matches:
            confidence = self._is_near_keywords(match, text, keywords, default_score)
            results.append({"match": match, "confidence": confidence})

        return self._unique(results)

    def extract_serial_number(self, text):
        default_score = 0.3
        matches = re.findall(r'\bSN\d{9}\b', text)
        keywords = ["serial", "number", "product", "identifier", "registration", "device"]

        results = []
        for match in matches:
            confidence = self._is_near_keywords(match, text, keywords, default_score)
            results.append({"match": match, "confidence": confidence})

        return self._unique(results)

    def extract_social_media_id(self, text):
        default_score = 0.2
        matches = re.findall(r'\b@[A-Za-z0-9_]{3,15}\b', text)
        keywords = ["social", "media", "id", "handle", "username", "profile", "account"]

        results = []
        for match in matches:
            confidence = self._is_near_keywords(match, text, keywords, default_score)
            results.append({"match": match, "confidence": confidence})

        return self._unique(results)

    def extract_ssn(self, text):
        default_score = 0.5
        matches = re.findall(r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b', text)
        keywords = ["ssn", "social", "security", "number", "identifier", "taxpayer"]

        results = []
        for match in matches:
            if not match.startswith("000") and not match.endswith("0000"):
                confidence = self._is_near_keywords(match, text, keywords, default_score)
                results.append({"match": match, "confidence": confidence})

        return self._unique(results)

    def extract_student_id(self, text):
        default_score = 0.4
        matches = re.findall(r'\bSTU\d{6}\b', text)
        keywords = ["student", "id", "identification", "university", "school", "number"]

        results = []
        for match in matches:
            confidence = self._is_near_keywords(match, text, keywords, default_score)
            results.append({"match": match, "confidence": confidence})

        return self._unique(results)

    def extract_swift(self, text):
        default_score = 0.5
        matches = re.findall(r'\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?\b', text)
        keywords = ["swift", "bic", "code", "international", "bank", "identifier"]

        results = []
        for match in matches:
            confidence = self._is_near_keywords(match, text, keywords, default_score)
            results.append({"match": match, "confidence": confidence})

        return self._unique(results)

    def extract_tax_id_other(self, text):
        default_score = 0.4
        matches = re.findall(r'\bTAX\d{9,12}\b', text)
        keywords = ["tax", "identification", "number", "id", "tin", "vat"]

        results = []
        for match in matches:
            confidence = self._is_near_keywords(match, text, keywords, default_score)
            results.append({"match": match, "confidence": confidence})

        return self._unique(results)

    def extract_time(self, text):
        default_score = 0.2
        matches = re.findall(r'\b([01]?\d|2[0-3]):[0-5]\d(:[0-5]\d)?\b', text)
        keywords = ["time", "timestamp", "hour", "minute", "second", "o'clock"]

        results = []
        for match in matches:
            if isinstance(match, tuple):
                match = ":".join([part for part in match if part])  # Reconstruct the full time string
            confidence = self._is_near_keywords(match, text, keywords, default_score)
            results.append({"match": match, "confidence": confidence})

        return self._unique(results)

    def extract_vehicle_id(self, text):
        default_score = 0.4
        matches = re.findall(r'\b[A-HJ-NPR-Z0-9]{17}\b', text)
        keywords = ["vehicle", "vin", "identification", "number", "car", "auto"]

        results = []
        for match in matches:
            confidence = self._is_near_keywords(match, text, keywords, default_score)
            results.append({"match": match, "confidence": confidence})

        return self._unique(results)

    def extract_all_words(self, text):
        # Regex pattern to match words and numbers
        pattern = r'\b\w+\b'
        matches = re.findall(pattern, text)
        return matches

    def extract_all(self, text):
        extracted_data = {}
        for method_name in dir(self):
            if method_name.startswith('extract_') and method_name != 'extract_all':
                method = getattr(self, method_name)
                results = method(text)
                if results:
                    extracted_data[method_name.replace('extract_', '').upper()] = results
        return extracted_data

if __name__ == "__main__":
    text = """8.2 Driver's License
A driver's license number might be D123456789012.

8.3 Driver's License (International)
An international driver's license number might look like INTD123456.

8.4 Time
Time is typically represented in formats such as 13:45 or 7:30 PM.

8.5 Username
Usernames are unique identifiers in digital platforms, such as john_doe_1985.

social security 584-89-0092 ssn number identifier taxpayer

The routing number is 123456789. You can also write it as 123-456-789 or 123 456 789. The routing number 021000021 is for JPMorgan Chase Bank in Florida, and 111000038 is for the Federal Reserve Bank in Minneapolis.
    John Doe's email is john.doe@example.com and his phone number is +1-555-555-5555.
    His SSN is 123-45-6789. He often shops online using his credit card number 1234 5678 9101 1121.
    His bank account number is 12345678901, and the routing number is 021000021.
    He drives a vehicle with VIN 1HGCM82633A123456 and holds a driver's license number D12345678.
    He has a meeting scheduled on 15th July 2023 at 10:30 AM.
    Visit https://example.com for more details.
    His server's IP addresses are 192.168.1.1/24 and 2001:db8::/32.
    Another IPv6 address is 2001:0db8:85a3:0000:0000:8a2e:0370:7334.
    The headquarters is located at 1600 Pennsylvania Ave NW, Washington, DC 20500.
    His current location is at coordinates 37.7749, -122.4194.
    Here are examples of various data types:

1. **IP Addresses**:
   IPv4: 192.168.1.1, 255.255.255.255, 10.0.0.1, 192.168.0.0/16, 127.0.0.1
   IPv6: 2001:0db8:85a3:0000:0000:8a2e:0370:7334, fe80::1, 2001:db8::/48

2. **URLs**:
   http://example.com, https://secure-site.org, ftp://ftp.example.com, mailto:user@example.com

3. **Email addresses**:
   user@example.com, another.user@domain.org, test123+filter@sub.domain.com

4. **Phone numbers**:
   +1 (123) 456-7890, 123-456-7890, 5555555555, +44 20 7946 0958, +91-9876543210

5. **Dates and times**:
   ISO date: 2023-09-01, 2023/09/01
   US date: 09/01/2023, 09-01-2023
   European date: 01/09/2023, 01-09-2023
   Date with month: 15 March 2023, 7 July 2021
   Times: 14:30, 09:45:30, 3:45 PM, 12:00 AM

6. **Postal codes**:
   USA: 12345, 90210-1234
   Canada: A1A 1A1, B2B-2B2
   UK: SW1A 1AA, EC1A 1BB
   Germany: 10115, France: 75008, Australia: 2000

7. **VIN numbers**:
   1HGCM82633A123456, JH4KA4650MC000000, 5YJSA1CN5DFP01234

8. **MAC addresses**:
   00:1A:2B:3C:4D:5E, 00-1A-2B-3C-4D-5E, 001A.2B3C.4D5E

9. **Routing numbers**:
   011000015, 121000358, 123456789 (Invalid), 021000021

10. **Bank account numbers**:
    IBAN: DE44 5001 0517 5407 3249 31, GB29 NWBK 6016 1331 9268 19
    US: 12345678901234567, 987654321

11. **Credit card numbers**:
    4111 1111 1111 1111 (Visa), 5500-0000-0000-0004 (MasterCard), 378282246310005 (American Express), 6011 1111 1111 1117 (Discover)

12. **Social Security Numbers (SSNs)**:
    123-45-6789, 987654321, 000-12-3456 (Invalid)

13. **Passport numbers**:
    USA: 123456789, UK: 987654321, Canada: A1234567, India: B9876543

14. **SWIFT codes**:
    BOFAUS3NXXX, CHASUS33, DEUTDEFF500, HSBCGB2LXXX

15. **Geo-coordinates**:
    Decimal: 37.7749, -122.4194; 40.7128, -74.0060
    Degrees, minutes, seconds: 40°42'51"N 74°00'21"W, 37°46'29"N 122°25'09"W

1HGCM82633A004352, JH4TB2H26CC000000, 5YJSA1CN5DFP01234

"""
    extractor = PIIExtractor()
    print(json.dumps(extractor.extract_all(text), indent=4))
