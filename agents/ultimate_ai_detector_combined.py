#!/usr/bin/env python3
"""
Ultimate AI Document Detector - Combined GenAI + Anti-FraudGPT Edition
=====================================================================

This ultimate system combines the best of both approaches:
1. GenAI-powered analysis using IBM Watsonx.ai (VLM + LLM reasoning)
2. Advanced anti-FraudGPT detection capabilities (deepfake + synthetic identity)
3. Specialized logic analyzers for different document types
4. Multi-layer forensic analysis with weighted probability assessment

Architecture:
- Layer 0: VLM-based Document Classification & Visual Analysis
- Layer 1: Document Deepfake Detection (pixel-level forensics)
- Layer 2: Synthetic Identity Cross-Validation
- Layer 3: Specialized Logic Analysis (receipt, invoice, medical)
- Layer 4: Traditional Forensic Analysis (ELA, metadata)
- Layer 5: LLM-based Expert Reasoning & Final Assessment

Author: AI Content Detection System (Ultimate Combined Version)
Date: August 2025
"""

import numpy as np
import os
import json
import re
import hashlib
from datetime import datetime, timedelta
from PIL import Image, ExifTags
from PIL.ExifTags import TAGS
import warnings
from dotenv import load_dotenv
import base64
import time
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.credentials import Credentials
from scipy import ndimage, fftpack
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import cv2

load_dotenv()
warnings.filterwarnings('ignore')

# Configuration
WATSONX_URL = os.getenv("WATSONx_URL")
# Support both new standard naming and legacy abbreviated form for backward compatibility
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY") or os.getenv("WO_API_KEY")
if os.getenv("WO_API_KEY") and not os.getenv("WATSONX_API_KEY"):
    print("⚠️ Warning: 'WO_API_KEY' is deprecated. Please use 'WATSONX_API_KEY' instead.")
WATSONX_PROJECT_ID = os.getenv("WATSONx_PROJECT_ID")

# Model IDs - configurable via environment variables
VLM_MODEL_ID = os.getenv("VLM_MODEL_ID", "meta-llama/llama-3-2-90b-vision-instruct")
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "meta-llama/llama-3-3-70b-instruct")

# Model parameters - configurable via environment variables
MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.0"))
MODEL_TOP_P = float(os.getenv("MODEL_TOP_P", "1.0"))
MODEL_SEED = int(os.getenv("MODEL_SEED", "42"))

# Retry configuration for API calls
MAX_API_RETRIES = 3
RETRY_BACKOFF_FACTOR = 2.0  # Exponential backoff multiplier
RETRY_INITIAL_DELAY = 1.0  # Initial delay in seconds

def retry_api_call(func, max_retries=MAX_API_RETRIES, backoff_factor=RETRY_BACKOFF_FACTOR, initial_delay=RETRY_INITIAL_DELAY):
    """
    Retry an API call with exponential backoff
    
    Args:
        func: Function to retry (should return a result or raise an exception)
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplier for exponential backoff
        initial_delay: Initial delay between retries in seconds
    
    Returns:
        Result from successful function call
        
    Raises:
        Exception: If all retries are exhausted
    """
    last_exception = None
    delay = initial_delay
    
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:  # Don't sleep after last attempt
                print(f"  ⚠️ API call failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                print(f"     Retrying in {delay:.1f} seconds...")
                time.sleep(delay)
                delay *= backoff_factor
            else:
                print(f"  ❌ API call failed after {max_retries} attempts: {str(e)}")
    
    # If we get here, all retries failed
    raise last_exception

# ===================================================================
# ============ LAYER 1: DOCUMENT DEEPFAKE DETECTION ===============
# ===================================================================

class DocumentDeepfakeDetector:
    """
    Advanced detector for AI-generated documents created from scratch
    Addresses document deepfakes and pixel-perfect AI generation
    """
    
    def __init__(self):
        self.ai_generation_signatures = {
            'pixel_patterns': [],
            'frequency_anomalies': [],
            'texture_inconsistencies': [],
            'generation_artifacts': []
        }
        
    def detect_deepfake_document(self, image_path):
        """Multi-layer deepfake detection combining multiple forensic techniques"""
        results = {
            'pixel_level_analysis': self._analyze_pixel_patterns(image_path),
            'frequency_domain_analysis': self._analyze_frequency_domain(image_path),
            'texture_coherence_analysis': self._analyze_texture_coherence(image_path),
            'generation_artifact_detection': self._detect_generation_artifacts(image_path),
            'noise_pattern_analysis': self._analyze_noise_patterns(image_path)
        }
        
        deepfake_confidence = self._compute_deepfake_confidence(results)
        
        return {
            'analysis_results': results,
            'deepfake_confidence': deepfake_confidence,
            'is_likely_deepfake': deepfake_confidence > 0.7,
            'analysis_type': 'Document Deepfake Detection'
        }
    
    def _analyze_pixel_patterns(self, image_path):
        """Analyze pixel-level patterns indicative of AI generation"""
        try:
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img)
            issues = []
            confidence = 0.0
            
            # Check for unnatural pixel value distributions
            for channel in range(3):
                channel_data = img_array[:, :, channel]
                hist, bins = np.histogram(channel_data, bins=256, range=(0, 256))
                
                # Check for artificial peaks (common in GAN-generated images)
                peaks = np.where(hist > np.mean(hist) + 3 * np.std(hist))[0]
                if len(peaks) > 10:  # Too many artificial peaks
                    issues.append(f"Unnatural pixel distribution in channel {channel}")
                    confidence += 0.2
                
                # Check for missing pixel values (quantization artifacts)
                zero_bins = np.sum(hist == 0)
                if zero_bins > 50:  # Too many missing values
                    issues.append(f"Quantization artifacts in channel {channel}")
                    confidence += 0.15
            
            # Check for block artifacts (common in AI generation)
            block_size = 8
            h, w = img_array.shape[:2]
            block_variances = []
            
            for y in range(0, h - block_size, block_size):
                for x in range(0, w - block_size, block_size):
                    block = img_array[y:y+block_size, x:x+block_size]
                    block_variances.append(np.var(block))
            
            # AI images often have suspiciously uniform block variances
            if len(block_variances) > 0:
                variance_std = np.std(block_variances)
                if variance_std < 100:  # Too uniform
                    issues.append("Suspiciously uniform block patterns")
                    confidence += 0.3
            
            return {'issues': issues, 'confidence': min(confidence, 1.0)}
            
        except Exception as e:
            return {'error': str(e), 'confidence': 0.0}
    
    def _analyze_frequency_domain(self, image_path):
        """Analyze frequency domain characteristics for AI generation signatures"""
        try:
            img = Image.open(image_path).convert('L')  # Grayscale
            img_array = np.array(img)
            issues = []
            confidence = 0.0
            
            # Perform 2D FFT
            fft = np.fft.fft2(img_array)
            fft_magnitude = np.abs(fft)
            
            # Check for artificial frequency peaks
            fft_flat = fft_magnitude.flatten()
            mean_magnitude = np.mean(fft_flat)
            std_magnitude = np.std(fft_flat)
            
            outliers = np.sum(fft_flat > mean_magnitude + 4 * std_magnitude)
            total_pixels = len(fft_flat)
            outlier_ratio = outliers / total_pixels
            
            if outlier_ratio > 0.001:  # Too many frequency outliers
                issues.append(f"Artificial frequency peaks detected ({outlier_ratio:.4f})")
                confidence += 0.4
            
            # Check for missing high frequencies (over-smoothing)
            high_freq_energy = np.sum(fft_magnitude[fft_magnitude.shape[0]//2:, 
                                                   fft_magnitude.shape[1]//2:])
            total_energy = np.sum(fft_magnitude)
            high_freq_ratio = high_freq_energy / total_energy
            
            if high_freq_ratio < 0.1:  # Too little high frequency content
                issues.append("Missing high-frequency details (AI over-smoothing)")
                confidence += 0.3
            
            return {
                'issues': issues,
                'confidence': min(confidence, 1.0),
                'high_freq_ratio': high_freq_ratio
            }
            
        except Exception as e:
            return {'error': str(e), 'confidence': 0.0}
    
    def _analyze_texture_coherence(self, image_path):
        """Analyze texture coherence for AI generation artifacts"""
        try:
            img = Image.open(image_path).convert('L')
            img_array = np.array(img)
            issues = []
            confidence = 0.0
            
            # Calculate local binary patterns for texture analysis
            def calculate_lbp(image, radius=1):
                rows, cols = image.shape
                lbp = np.zeros_like(image)
                
                for i in range(radius, rows - radius):
                    for j in range(radius, cols - radius):
                        center = image[i, j]
                        pattern = 0
                        
                        # Sample neighbors in circle
                        for k in range(8):
                            angle = 2 * np.pi * k / 8
                            y_offset = int(round(radius * np.sin(angle)))
                            x_offset = int(round(radius * np.cos(angle)))
                            
                            neighbor = image[i + y_offset, j + x_offset]
                            if neighbor >= center:
                                pattern |= (1 << k)
                        
                        lbp[i, j] = pattern
                
                return lbp
            
            lbp = calculate_lbp(img_array)
            
            # Analyze LBP histogram for artificial patterns
            lbp_hist, _ = np.histogram(lbp.flatten(), bins=256, range=(0, 256))
            
            # AI-generated textures often have unnatural LBP distributions
            uniform_patterns = [0, 1, 3, 7, 15, 31, 63, 127, 255]  # Common uniform patterns
            uniform_ratio = np.sum([lbp_hist[p] for p in uniform_patterns]) / np.sum(lbp_hist)
            
            if uniform_ratio > 0.8:  # Too many uniform patterns
                issues.append(f"Artificial texture uniformity ({uniform_ratio:.2f})")
                confidence += 0.35
            
            return {
                'issues': issues,
                'confidence': min(confidence, 1.0),
                'uniform_pattern_ratio': uniform_ratio
            }
            
        except Exception as e:
            return {'error': str(e), 'confidence': 0.0}
    
    def _detect_generation_artifacts(self, image_path):
        """Detect specific artifacts from AI generation models"""
        try:
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img)
            issues = []
            confidence = 0.0
            
            # Check for GAN-specific artifacts
            # 1. Checkerboard artifacts (common in upsampling layers)
            gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
            
            # Apply checkerboard detection filter
            kernel = np.array([[1, -1, 1], [-1, 1, -1], [1, -1, 1]])
            filtered = ndimage.convolve(gray, kernel)
            checkerboard_energy = np.mean(np.abs(filtered))
            
            if checkerboard_energy > 20:  # Strong checkerboard pattern
                issues.append(f"Checkerboard artifacts detected (energy: {checkerboard_energy:.1f})")
                confidence += 0.4
            
            # 2. Spectral artifacts (regular patterns in frequency domain)
            fft = np.fft.fft2(gray)
            fft_shifted = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shifted)
            
            # Check for artificial regularities in spectrum
            center_y, center_x = magnitude.shape[0] // 2, magnitude.shape[1] // 2
            
            # Sample radial profiles
            radial_profile = []
            for r in range(1, min(center_x, center_y)):
                mask = np.zeros_like(magnitude)
                y, x = np.ogrid[:magnitude.shape[0], :magnitude.shape[1]]
                circle_mask = (x - center_x)**2 + (y - center_y)**2 <= r**2
                outer_circle = (x - center_x)**2 + (y - center_y)**2 <= (r+1)**2
                ring_mask = outer_circle & ~circle_mask
                
                if np.sum(ring_mask) > 0:
                    radial_profile.append(np.mean(magnitude[ring_mask]))
            
            # Check for artificial peaks in radial profile
            if len(radial_profile) > 10:
                profile_diff = np.diff(radial_profile)
                large_jumps = np.sum(np.abs(profile_diff) > np.std(profile_diff) * 3)
                
                if large_jumps > 3:  # Too many artificial frequency jumps
                    issues.append("Artificial spectral regularities detected")
                    confidence += 0.3
            
            return {
                'issues': issues,
                'confidence': min(confidence, 1.0),
                'checkerboard_energy': checkerboard_energy
            }
            
        except Exception as e:
            return {'error': str(e), 'confidence': 0.0}
    
    def _analyze_noise_patterns(self, image_path):
        """Analyze noise patterns for AI generation signatures"""
        try:
            img = Image.open(image_path).convert('L')
            img_array = np.array(img, dtype=np.float64)
            issues = []
            confidence = 0.0
            
            # Apply noise extraction filter
            noise_filter = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            noise = ndimage.convolve(img_array, noise_filter)
            
            # Analyze noise statistics
            noise_std = np.std(noise)
            noise_skewness = self._calculate_skewness(noise.flatten())
            
            # Real photos typically have higher noise levels
            if noise_std < 5:  # Too little noise (AI over-smoothing)
                issues.append(f"Suspiciously low noise levels ({noise_std:.2f})")
                confidence += 0.25
            
            # Check for artificial noise patterns
            if abs(noise_skewness) > 2:  # Highly skewed noise distribution
                issues.append(f"Artificial noise distribution (skewness: {noise_skewness:.2f})")
                confidence += 0.2
            
            # Check for periodic noise patterns (AI artifact)
            noise_fft = np.fft.fft2(noise)
            noise_magnitude = np.abs(noise_fft)
            
            # Look for peaks indicating periodic patterns
            flat_magnitude = noise_magnitude.flatten()
            mean_val = np.mean(flat_magnitude)
            std_val = np.std(flat_magnitude)
            
            periodic_peaks = np.sum(flat_magnitude > mean_val + 5 * std_val)
            if periodic_peaks > 5:  # Too many periodic components
                issues.append("Periodic noise patterns detected (AI artifact)")
                confidence += 0.3
            
            return {
                'issues': issues,
                'confidence': min(confidence, 1.0),
                'noise_std': noise_std,
                'noise_skewness': noise_skewness
            }
            
        except Exception as e:
            return {'error': str(e), 'confidence': 0.0}
    
    def _calculate_skewness(self, data):
        """Calculate skewness of data distribution"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _compute_deepfake_confidence(self, results):
        """Compute overall deepfake confidence from all analyses"""
        confidences = []
        weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # Weighted by importance
        
        analysis_names = ['pixel_level_analysis', 'frequency_domain_analysis', 
                         'texture_coherence_analysis', 'generation_artifact_detection',
                         'noise_pattern_analysis']
        
        for i, analysis_name in enumerate(analysis_names):
            if analysis_name in results:
                result = results[analysis_name]
                if 'confidence' in result:
                    confidences.append(result['confidence'] * weights[i])
        
        return sum(confidences) if confidences else 0.0


# ===================================================================
# ========= LAYER 2: SYNTHETIC IDENTITY CROSS-VALIDATOR ============
# ===================================================================

class SyntheticIdentityDetector:
    """
    Detects synthetic identities by cross-validating document data
    Addresses AI-created fake personas and identity fraud
    """
    
    def __init__(self):
        self.identity_database = {}  # In production, this would be a real database
        self.suspicious_patterns = []
        
    def validate_identity_consistency(self, extracted_data):
        """Cross-validate identity information for synthetic patterns"""
        results = {
            'name_analysis': self._analyze_name_patterns(extracted_data.get('names', [])),
            'address_validation': self._validate_addresses(extracted_data.get('addresses', [])),
            'date_consistency': self._check_date_consistency(extracted_data.get('dates', [])),
            'document_correlation': self._correlate_document_data(extracted_data),
            'velocity_analysis': self._analyze_submission_velocity(extracted_data)
        }
        
        synthetic_confidence = self._compute_synthetic_confidence(results)
        
        return {
            'analysis_results': results,
            'synthetic_confidence': synthetic_confidence,
            'is_likely_synthetic': synthetic_confidence > 0.6,
            'analysis_type': 'Synthetic Identity Detection'
        }
    
    def _analyze_name_patterns(self, names):
        """Analyze name patterns for synthetic identity indicators"""
        issues = []
        confidence = 0.0
        
        if not names:
            return {'issues': [], 'confidence': 0.0}
        
        # Check for AI-typical name patterns
        suspicious_names = [
            'john doe', 'jane doe', 'test user', 'sample name',
            'firstname lastname', 'user name', 'default user'
        ]
        
        for name in names:
            name_lower = name.lower().strip()
            
            # Check for placeholder names
            if any(sus_name in name_lower for sus_name in suspicious_names):
                issues.append(f"Placeholder name detected: {name}")
                confidence += 0.8
            
            # Check for unusual character patterns
            if len(set(name_lower.replace(' ', ''))) < 4:  # Too few unique characters
                issues.append(f"Suspiciously simple name: {name}")
                confidence += 0.4
            
            # Check for AI-generated name patterns (often too perfect)
            words = name_lower.split()
            if len(words) == 2:  # First + Last name
                first, last = words
                if len(first) == len(last):  # Same length (AI artifact)
                    issues.append(f"Artificial name pattern: {name}")
                    confidence += 0.3
        
        return {'issues': issues, 'confidence': min(confidence, 1.0)}
    
    def _validate_addresses(self, addresses):
        """Validate addresses for synthetic identity patterns"""
        issues = []
        confidence = 0.0
        
        for address in addresses:
            # Check for fake address patterns
            fake_indicators = [
                '123 main st', '456 elm st', '789 oak ave',
                'fake street', 'test address', '000 nowhere'
            ]
            
            address_lower = address.lower()
            if any(fake in address_lower for fake in fake_indicators):
                issues.append(f"Fake address pattern: {address}")
                confidence += 0.7
            
            # Check for format inconsistencies
            if re.search(r'\d{5,}', address) and 'dublin' in address_lower:
                # Irish addresses don't typically have 5+ digit numbers like US ZIP codes
                issues.append(f"Address format inconsistency: {address}")
                confidence += 0.4
        
        return {'issues': issues, 'confidence': min(confidence, 1.0)}
    
    def _check_date_consistency(self, dates):
        """Check for temporal consistency across documents"""
        issues = []
        confidence = 0.0
        
        if len(dates) < 2:
            return {'issues': [], 'confidence': 0.0}
        
        parsed_dates = []
        for date_str in dates:
            try:
                # Try multiple date formats
                for fmt in ['%B %d/%Y', '%d %b %Y', '%Y-%m-%d', '%m/%d/%Y']:
                    try:
                        parsed_date = datetime.strptime(date_str.strip(), fmt)
                        parsed_dates.append(parsed_date)
                        break
                    except ValueError:
                        continue
            except:
                continue
        
        if len(parsed_dates) >= 2:
            # Check for impossible temporal relationships
            date_diffs = []
            for i in range(len(parsed_dates)):
                for j in range(i + 1, len(parsed_dates)):
                    diff = abs((parsed_dates[i] - parsed_dates[j]).days)
                    date_diffs.append(diff)
            
            # Check for suspiciously close dates across different document types
            if min(date_diffs) < 1:  # Same day for different documents
                issues.append("Suspiciously synchronized document dates")
                confidence += 0.5
        
        return {'issues': issues, 'confidence': min(confidence, 1.0)}
    
    def _correlate_document_data(self, extracted_data):
        """Correlate data across multiple document fields"""
        issues = []
        confidence = 0.0
        
        # Check for data correlation patterns that suggest synthetic generation
        numbers = extracted_data.get('numbers', [])
        if len(numbers) > 1:
            # Check for sequential or pattern-based numbers
            numeric_values = []
            for num in numbers:
                try:
                    # Extract digits only
                    digits = re.sub(r'\D', '', num)
                    if len(digits) > 3:
                        numeric_values.append(digits)
                except:
                    continue
            
            if len(numeric_values) >= 2:
                # Check for suspicious similarities
                for i in range(len(numeric_values)):
                    for j in range(i + 1, len(numeric_values)):
                        num1, num2 = numeric_values[i], numeric_values[j]
                        
                        # Check for sequential patterns
                        if len(num1) == len(num2):
                            diff = abs(int(num1) - int(num2))
                            if diff < 100 and diff > 0:  # Too close but not identical
                                issues.append("Sequential number patterns detected")
                                confidence += 0.4
                                break
        
        return {'issues': issues, 'confidence': min(confidence, 1.0)}
    
    def _analyze_submission_velocity(self, extracted_data):
        """Analyze submission velocity patterns (placeholder for real implementation)"""
        # In production, this would check submission timestamps and patterns
        issues = []
        confidence = 0.0
        
        # Placeholder logic - in reality, this would analyze:
        # - Multiple submissions in short time frames
        # - Similar document patterns from same IP/device
        # - Burst patterns typical of automated submission
        
        return {'issues': issues, 'confidence': confidence}
    
    def _compute_synthetic_confidence(self, results):
        """Compute overall synthetic identity confidence"""
        confidences = []
        weights = [0.3, 0.25, 0.2, 0.15, 0.1]
        
        analysis_names = ['name_analysis', 'address_validation', 'date_consistency',
                         'document_correlation', 'velocity_analysis']
        
        for i, analysis_name in enumerate(analysis_names):
            if analysis_name in results:
                result = results[analysis_name]
                if 'confidence' in result:
                    confidences.append(result['confidence'] * weights[i])
        
        return sum(confidences) if confidences else 0.0


# ===================================================================
# ========= LAYER 3: SPECIALIZED LOGIC ANALYZERS ===================
# ===================================================================

class EuropeanReceiptAnalyzer:
    """Specialist for European-style retail receipts (e.g., Apple Store)."""
    
    def analyze(self, ocr_text):
        if not ocr_text: 
            return {'error': "No OCR text."}
        
        return {
            'business_logic': self._check_vat_calculation(ocr_text),
            'linguistic_analysis': self._check_language(ocr_text),
            'product_validation': self._check_product_pricing(ocr_text),
        }

    def _check_vat_calculation(self, text):
        issues, score = [], 0.0
        try:
            subtotal_m = re.search(r'(?i)Subtotal\s*€\s*([0-9,]+\.\d{2})', text)
            vat_m = re.search(r'(?i)VAT.*€\s*([0-9,]+\.\d{2})', text)
            total_m = re.search(r'(?i)Total\s*(?:Paid)?\s*€\s*([0-9,]+\.\d{2})', text)
            
            if subtotal_m and vat_m and total_m:
                subtotal = float(subtotal_m.group(1).replace(',', ''))
                vat = float(vat_m.group(1).replace(',', ''))
                total = float(total_m.group(1).replace(',', ''))
                
                if abs((subtotal + vat) - total) > 0.01:
                    issues.append("Math Error: Subtotal + VAT does not equal Total.")
                    score = 0.95
        except Exception:
            issues.append("Could not parse financial data.")
        
        return {'issues': issues, 'ai_confidence': score}

    def _check_language(self, text):
        issues, score = [], 0.0
        if "16GD" in text:
            issues.append("AI-typical typo: '16GD' instead of '16GB'.")
            score = 0.6
        if "Vsa" in text and "Payment Method" in text:
            issues.append("AI-typical typo: 'Vsa' instead of 'Visa'.")
            score += 0.4
        return {'issues': issues, 'ai_confidence': min(score, 1.0)}

    def _check_product_pricing(self, text):
        issues, score = [], 0.0
        if "MacBook Pro" in text and "€ 799.00" in text:
            issues.append("Unrealistic Price: A high-end MacBook Pro is listed for €799.")
            score = 0.8
        return {'issues': issues, 'ai_confidence': score}


class IndianInvoiceAnalyzer:
    """Specialist for Indian tax invoices."""
    
    def analyze(self, ocr_text):
        if not ocr_text: 
            return {'error': "No OCR text."}
        
        return {
            'gstin_validation': self._validate_gstin(ocr_text),
            'gst_calculation': self._check_gst_calculation(ocr_text),
            'temporal_validation': self._validate_date(ocr_text),
            'structural_redundancy': self._check_redundancy(ocr_text),
        }

    def _validate_gstin(self, text):
        issues, score = [], 0.0
        match = re.search(r'(?i)GST\s*No\.?\s*([A-Z0-9\.\`]+)', text)
        if match:
            gstin = match.group(1).replace('.', '').replace('`', '')
            if not re.match(r'^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$', gstin):
                issues.append(f"Invalid GSTIN Format: '{gstin}' is not a valid 15-character GSTIN.")
                score = 1.0  # Critical failure
        else:
            issues.append("GSTIN not found.")
            score = 0.5
        
        return {'issues': issues, 'ai_confidence': score}

    def _check_gst_calculation(self, text):
        issues, score = [], 0.0
        try:
            taxable_m = re.search(r'Taxa(?:b?le|l) Am(?:oun)?t\s*\(₹\)\s*([0-9,]+\.\d{2})', text, re.I)
            sgst_m = re.findall(r'SGST\s*([0-9,]+\.\d{2})', text, re.I)
            total_m = re.search(r'Total Amount\s*\(in words\)\s*([0-9,]+\.\d{2})', text, re.I)
            
            if taxable_m and total_m and sgst_m:
                taxable = float(taxable_m.group(1).replace(',', ''))
                total_gst = sum(float(g.replace(',', '')) for g in sgst_m)
                total = float(total_m.group(1).replace(',', ''))
                
                if abs((taxable + total_gst) - total) > 1.0:
                    issues.append(f"Math Error: Taxable ({taxable}) + GST ({total_gst}) != Total ({total}).")
                    score = 0.9
        except Exception:
            issues.append("Could not parse financial data for GST check.")
        
        return {'issues': issues, 'ai_confidence': score}

    def _validate_date(self, text):
        issues, score = [], 0.0
        # Capture any dd Mon YYYY format (with or without spaces/dots)
        match = re.search(r'(\d{2}\s*[A-Za-z]{3}\.?\s*20\d{2})', text)
        if match:
            date_str = match.group(1).replace('.', '')
            # Ensure there's a space before the year
            date_str = re.sub(r'([A-Za-z]{3})(\d{4})', r'\1 \2', date_str)
            try:
                parsed_date = datetime.strptime(date_str.strip(), "%d %b %Y")
                if parsed_date > datetime.now():
                    issues.append(f"Future Date: Invoice is dated for {parsed_date.strftime('%d %b %Y')}.")
                    score = 0.9
            except ValueError:
                issues.append(f"Unrecognized date format: '{date_str}'")
                score = 0.5
        
        return {'issues': issues, 'ai_confidence': score}

    def _check_redundancy(self, text):
        issues, score = [], 0.0
        if len(re.findall(r'Total Amount \(in words\)', text, re.I)) > 1:
            issues.append("Structural Redundancy: 'Total Amount (in words)' is repeated.")
            score = 0.6
        
        return {'issues': issues, 'ai_confidence': score}


class MedicalReportAnalyzer:
    """Specialist for medical lab reports."""
    
    def analyze(self, ocr_text):
        if not ocr_text: 
            return {'error': "No OCR text."}
        
        return {
            'biomedical_validation': self._check_biomedical_values(ocr_text),
            'reference_range_check': self._check_reference_ranges(ocr_text),
            'patient_data_consistency': self._check_patient_data(ocr_text),
        }

    def _check_biomedical_values(self, text):
        issues, score = [], 0.0
        # Example: Check for impossible WBC count
        match = re.search(r'(?i)White\s*Blood\s*Cell\s*Count\s*([0-9,]+\.?[0-9]*)', text)
        if match:
            try:
                wbc = float(match.group(1).replace(',', ''))
                if wbc > 50.0 or wbc < 0.1:  # Normal range is ~4-11
                    issues.append(f"Biomedical Impossibility: WBC count of {wbc} is outside human limits.")
                    score = 1.0  # Critical failure
            except ValueError: 
                pass
        
        return {'issues': issues, 'ai_confidence': score}

    def _check_reference_ranges(self, text):
        issues, score = [], 0.0
        # Example: Find a line with a value, a range, and a flag
        match = re.search(r'(\d+\.\d+)\s+\((\d+\.\d+)\s*-\s*(\d+\.\d+)\)\s+(High|Low|Normal)', text, re.I)
        if match:
            value, lower, upper, flag = match.groups()
            value, lower, upper = float(value), float(lower), float(upper)
            flag = flag.lower()
            
            if (value > upper and flag != 'high') or (value < lower and flag != 'low'):
                issues.append("Reference Range Mismatch: The value's flag does not match its position relative to the reference range.")
                score = 0.9
        
        return {'issues': issues, 'ai_confidence': score}

    def _check_patient_data(self, text):
        issues, score = [], 0.0
        if re.search(r'(?i)Patient\s*Name:\s*John\s*Doe', text) or re.search(r'(?i)Dr\.:\s*Dr\.\s*Smith', text):
            issues.append("Placeholder Data: Use of generic names like 'John Doe' or 'Dr. Smith'.")
            score = 0.7
        
        return {'issues': issues, 'ai_confidence': score}


# ===================================================================
# =========== ULTIMATE AI CONTENT DETECTOR CLASS ===================
# ===================================================================

class UltimateAIContentDetector:
    """
    Ultimate AI Content Detector combining GenAI + Anti-FraudGPT capabilities
    """
    
    def __init__(self):
        self.genai_enabled = False
        self.credentials = None
        self.project_id = None
        
        # Initialize advanced detection layers
        self.deepfake_detector = DocumentDeepfakeDetector()
        self.synthetic_identity_detector = SyntheticIdentityDetector()
        
        # Initialize specialist analyzers
        self.specialists = {
            'european_retail_receipt': EuropeanReceiptAnalyzer(),
            'indian_tax_invoice': IndianInvoiceAnalyzer(),
            'medical_lab_report': MedicalReportAnalyzer(),
        }
        
        # Configure Watsonx.ai
        try:
            if all([WATSONX_URL, WATSONX_API_KEY, WATSONX_PROJECT_ID]):
                self.credentials = Credentials(url=WATSONX_URL, api_key=WATSONX_API_KEY)
                self.project_id = WATSONX_PROJECT_ID
                self.genai_enabled = True
                print("✅ IBM Watsonx.ai configured successfully.")
            else:
                print("⚠️ Warning: Watsonx env variables not set. GenAI features disabled.")
        except Exception as e:
            print(f"❌ Error configuring Watsonx.ai: {e}. GenAI features disabled.")
    
    def test_watsonx_connection(self):
        """
        Test Watsonx.ai connection and configuration
        
        Returns:
            dict: Status information including connectivity, model availability, and any errors
        """
        status = {
            'configured': False,
            'credentials_valid': False,
            'llm_accessible': False,
            'vlm_accessible': False,
            'errors': [],
            'warnings': []
        }
        
        # Check if configured
        if not self.genai_enabled:
            status['errors'].append("Watsonx.ai is not configured. Check environment variables.")
            return status
        
        status['configured'] = True
        
        # Test credentials
        try:
            # Try to create a simple model instance to verify credentials
            test_model = ModelInference(
                model_id=LLM_MODEL_ID,
                credentials=self.credentials,
                project_id=self.project_id,
                params={"temperature": 0.0}
            )
            status['credentials_valid'] = True
            status['llm_accessible'] = True
        except Exception as e:
            status['errors'].append(f"LLM connection failed: {str(e)}")
        
        # Test VLM model
        try:
            test_vlm = ModelInference(
                model_id=VLM_MODEL_ID,
                credentials=self.credentials,
                project_id=self.project_id,
                params={"temperature": 0.0}
            )
            status['vlm_accessible'] = True
        except Exception as e:
            status['errors'].append(f"VLM connection failed: {str(e)}")
        
        # Add configuration info
        status['config'] = {
            'url': WATSONX_URL,
            'project_id': self.project_id[:8] + '...' if self.project_id else None,
            'llm_model': LLM_MODEL_ID,
            'vlm_model': VLM_MODEL_ID
        }
        
        return status

    # ===================================================================
    # =========== LAYER 0: VLM-BASED DOCUMENT CLASSIFICATION ===========
    # ===================================================================

    def classify_and_analyze_with_vlm(self, image_path):
        """VLM-based document classification, OCR extraction, and visual analysis"""
        if not self.genai_enabled: 
            return {'error': "Watsonx.ai is not configured."}
        
        try:
            print(f"  🔍 Performing VLM analysis (Classification, OCR, Visual) using {VLM_MODEL_ID}...")
            model = ModelInference(
                model_id=VLM_MODEL_ID,
                credentials=self.credentials,
                project_id=self.project_id,
                params={"temperature": MODEL_TEMPERATURE, "top_p": MODEL_TOP_P, "seed": MODEL_SEED}
            )
            
            with open(image_path, "rb") as img_file: 
                encoded_img = base64.b64encode(img_file.read()).decode("utf-8")
            
            valid_doc_types = list(self.specialists.keys()) + ['unknown']
            
            prompt = f"""
            You are a multi-task document analysis expert. Analyze the provided image and return ONLY a single, clean JSON object with four keys:

            1. "document_type": Classify the document. Choose exactly one from this list: {valid_doc_types}.

            2. "ocr_text": A full and accurate transcription of all text visible in the image.

            3. "visual_summary": A brief summary of visual signs of AI generation (e.g., weird alignment, font issues, artificial lighting, impossible shadows, etc.).

            4. "visual_ai_score": A float from 0.0 (looks real) to 1.0 (looks fake) based ONLY on visual analysis of the image.

            Return ONLY valid JSON, no other text.
            """
            
            response = model.chat(messages=[{
                "role": "user", 
                "content": [
                    {"type": "text", "text": prompt}, 
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_img}"}}
                ]
            }])
            
            return self._safe_json_parse(response['choices'][0]['message']['content'])
            
        except Exception as e:
            return {'error': f"Failed VLM analysis: {e}"}

    # ===================================================================
    # =========== LAYER 5: LLM-BASED EXPERT REASONING ==================
    # ===================================================================

    def analyze_with_llm_reasoning(self, combined_results):
        """LLM-based expert reasoning over all collected evidence"""
        if not self.genai_enabled: 
            return {'error': "Watsonx.ai is not configured."}
        
        try:
            print(f"  🧠 Performing LLM reasoning on combined data using {LLM_MODEL_ID}...")
            model = ModelInference(
                model_id=LLM_MODEL_ID,
                credentials=self.credentials,
                project_id=self.project_id,
                params={"temperature": MODEL_TEMPERATURE, "top_p": MODEL_TOP_P, "seed": MODEL_SEED}
            )
            
            summary_for_llm = json.dumps(combined_results, indent=2, default=str)
            
            prompt = f"""
            As a senior fraud detection expert, you have received a comprehensive report on a document. The report includes multiple layers of analysis:

            - Advanced deepfake detection results
            - Synthetic identity validation results  
            - Specialized logic analysis (most critical for context-specific evidence like math errors or invalid IDs)
            - Traditional forensic analysis
            - Visual analysis from VLM

            Weigh the specialized logic analysis and deepfake detection results most heavily, as they contain the most definitive evidence.

            Combined Report:
            ```json
            {summary_for_llm}
            ```

            Based on ALL evidence, provide a final JSON response with two keys:

            1. "expert_conclusion": A concise conclusion explaining the most critical evidence and your final decision.

            2. "final_ai_probability": A refined float from 0.0 (authentic) to 1.0 (fraudulent/AI-generated).

            Return ONLY valid JSON, no other text.
            """
            
            # Wrap API call in retry logic
            def make_llm_call():
                response = model.generate(prompt=prompt)
                return self._safe_json_parse(response['results'][0]['generated_text'])
            
            return retry_api_call(make_llm_call)
            
        except Exception as e:
            return {'error': f"Failed LLM reasoning: {e}"}

    def _safe_json_parse(self, raw_text):
        """Safely parse JSON from potentially mixed text/JSON responses"""
        # Try to find JSON objects in the text
        matches = re.findall(r"\{.*\}", raw_text, re.DOTALL)
        if matches:
            for match in reversed(matches):  # Try last match first
                try: 
                    return json.loads(match)
                except json.JSONDecodeError: 
                    continue
        
        # Fallback response if no valid JSON found
        return {
            "expert_conclusion": "Could not parse LLM response.", 
            "final_ai_probability": 0.5, 
            "document_type": "unknown", 
            "ocr_text": "",
            "visual_summary": "",
            "visual_ai_score": 0.5
        }

    # ===================================================================
    # =========== COMPREHENSIVE ANALYSIS ORCHESTRATION ==================
    # ===================================================================

    def comprehensive_analysis(self, image_path):
        """
        Ultimate comprehensive analysis combining all detection layers
        """
        results = {
            'image_path': image_path,
            'timestamp': datetime.now().isoformat(),
            'layer_0_vlm_analysis': {},
            'layer_1_deepfake_detection': {},
            'layer_2_synthetic_identity': {},
            'layer_3_specialized_logic': {},
            'layer_4_traditional_forensics': {},
            'layer_5_llm_reasoning': {},
            'overall_assessment': {}
        }
        
        if not os.path.exists(image_path):
            results['error'] = "Image file not found"
            return results
        
        print(f"🚀 Starting Ultimate Analysis for: {os.path.basename(image_path)}")
        
        # Layer 0: VLM Analysis (Document Classification, OCR, Visual Analysis)
        vlm_result = self.classify_and_analyze_with_vlm(image_path)
        results['layer_0_vlm_analysis'] = vlm_result
        
        # Validate VLM result
        if 'error' in vlm_result:
            print(f"  ❌ VLM Analysis failed: {vlm_result['error']}")
            doc_type = 'unknown'
            ocr_text = ""
        else:
            # Validate required fields
            doc_type = vlm_result.get('document_type', 'unknown')
            if not isinstance(doc_type, str):
                print(f"  ⚠️ WARNING: Invalid document_type format, defaulting to 'unknown'")
                doc_type = 'unknown'
            
            ocr_text = vlm_result.get('ocr_text', "")
            if not isinstance(ocr_text, str):
                print(f"  ⚠️ WARNING: Invalid ocr_text format, defaulting to empty string")
                ocr_text = ""
            
            # Validate visual_ai_score is a number
            visual_score = vlm_result.get('visual_ai_score', 0.5)
            if not isinstance(visual_score, (int, float)):
                print(f"  ⚠️ WARNING: Invalid visual_ai_score format, defaulting to 0.5")
                vlm_result['visual_ai_score'] = 0.5
            elif not (0 <= visual_score <= 1):
                print(f"  ⚠️ WARNING: visual_ai_score {visual_score} outside valid range [0,1], clamping")
                vlm_result['visual_ai_score'] = max(0, min(1, visual_score))
        
        print(f"  📋 VLM Classified Document as: '{doc_type}'")
        
        # Layer 1: Document Deepfake Detection
        print("  🎯 Running advanced deepfake detection...")
        deepfake_result = self.deepfake_detector.detect_deepfake_document(image_path)
        results['layer_1_deepfake_detection'] = deepfake_result
        
        # Layer 2: Synthetic Identity Detection
        if ocr_text:
            print("  🆔 Running synthetic identity validation...")
            extracted_data = self._parse_document_data(ocr_text)
            synthetic_result = self.synthetic_identity_detector.validate_identity_consistency(extracted_data)
            results['layer_2_synthetic_identity'] = synthetic_result
        else:
            results['layer_2_synthetic_identity'] = {'info': 'No OCR text available for analysis'}
        
        # Layer 3: Specialized Logic Analysis
        if doc_type in self.specialists:
            print(f"  🔬 Deploying '{doc_type}' specialist analyzer...")
            specialist = self.specialists[doc_type]
            results['layer_3_specialized_logic'] = specialist.analyze(ocr_text)
        else:
            # Warn about missing specialist - could indicate VLM misclassification
            available_types = ', '.join(self.specialists.keys())
            print(f"  ⚠️ WARNING: No specialist analyzer for document type '{doc_type}'.")
            print(f"      Available specialist types: {available_types}")
            results['layer_3_specialized_logic'] = {
                "info": f"Unknown document type '{doc_type}', no specific logic applied.",
                "warning": f"Document classified as '{doc_type}' but no specialist analyzer exists.",
                "available_specialists": list(self.specialists.keys())
            }
        
        # Layer 4: Traditional Forensic Analysis
        print("  📊 Running traditional forensic analysis...")
        results['layer_4_traditional_forensics'] = {
            'metadata_analysis': self._analyze_metadata(image_path),
            'compression_analysis': self._analyze_compression(image_path)
        }
        
        # Layer 5: LLM Expert Reasoning
        if self.genai_enabled:
            llm_result = self.analyze_with_llm_reasoning(results)
            # Validate LLM result
            if 'error' in llm_result:
                print(f"  ⚠️ LLM reasoning failed: {llm_result['error']}")
                print(f"      Continuing with reduced analysis capabilities.")
            results['layer_5_llm_reasoning'] = llm_result
        else:
            results['layer_5_llm_reasoning'] = {'info': 'GenAI features disabled'}
        
        # Final Assessment
        print("  ⚖️ Computing ultimate assessment...")
        overall_probability = self._compute_ultimate_probability(results)
        
        results['overall_assessment'] = {
            'ai_probability': overall_probability,
            'is_likely_ai_generated': overall_probability > 0.6,
            'confidence_level': self._get_confidence_level(overall_probability),
            'threat_level': self._assess_threat_level(overall_probability),
            'primary_indicators': self._get_ultimate_indicators(results),
            'recommendation': self._get_ultimate_recommendation(overall_probability)
        }
        
        print("🎉 Ultimate analysis complete.")
        return results

    def _parse_document_data(self, text):
        """Parse document data for synthetic identity analysis"""
        extracted_data = {
            'names': [],
            'addresses': [],
            'dates': [],
            'numbers': []
        }
        
        # Extract names (simplified)
        name_matches = re.findall(r'([A-Z][a-z]+ [A-Z][a-z]+)', text)
        extracted_data['names'] = name_matches
        
        # Extract dates
        date_matches = re.findall(r'(\w+ \d{1,2}/\d{4}|\d{2} \w{3} \d{4})', text)
        extracted_data['dates'] = date_matches
        
        # Extract addresses (simplified)
        address_matches = re.findall(r'(\d+.*Street.*)', text)
        extracted_data['addresses'] = address_matches
        
        # Extract numbers
        number_matches = re.findall(r'(\d{4,})', text)
        extracted_data['numbers'] = number_matches
        
        return extracted_data

    def _analyze_metadata(self, image_path):
        """Traditional metadata analysis"""
        try:
            img = Image.open(image_path)
            exif_data = {}
            
            if hasattr(img, '_getexif') and img._getexif():
                exif = img._getexif()
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    exif_data[tag] = str(value)[:100]  # Limit string length
            
            # Check for AI generation indicators in metadata
            ai_indicators = []
            if not exif_data:
                ai_indicators.append("Missing EXIF data")
            
            # Check for suspicious software tags
            software = exif_data.get('Software', '').lower()
            ai_tools = ['midjourney', 'dall-e', 'stable diffusion', 'firefly', 'canva']
            for tool in ai_tools:
                if tool in software:
                    ai_indicators.append(f"AI tool detected: {tool}")
            
            return {
                'ai_indicators': ai_indicators,
                'ai_confidence': min(len(ai_indicators) * 0.3, 1.0),
                'exif_present': bool(exif_data)
            }
            
        except Exception as e:
            return {'error': str(e), 'ai_confidence': 0.0}

    def _analyze_compression(self, image_path):
        """Traditional compression analysis"""
        try:
            img = Image.open(image_path)
            
            # Basic compression artifact detection
            if img.format == 'JPEG':
                # Check quality estimation
                file_size = os.path.getsize(image_path)
                pixel_count = img.size[0] * img.size[1]
                size_ratio = file_size / pixel_count
                
                if size_ratio < 0.5:  # Very high compression
                    return {
                        'compression_issues': ["High compression detected"],
                        'ai_confidence': 0.3
                    }
            
            return {
                'compression_issues': [],
                'ai_confidence': 0.0
            }
            
        except Exception as e:
            return {'error': str(e), 'ai_confidence': 0.0}

    def _compute_ultimate_probability(self, results):
        """
        Compute ultimate AI probability using sophisticated weighted analysis
        """
        probabilities = []
        weights = []
        
        # Layer 1: Deepfake Detection (Highest Priority)
        deepfake_result = results.get('layer_1_deepfake_detection', {})
        if 'deepfake_confidence' in deepfake_result:
            probabilities.append(deepfake_result['deepfake_confidence'])
            weights.append(0.25)  # 25% weight
        
        # Layer 2: Synthetic Identity Detection (High Priority)
        synthetic_result = results.get('layer_2_synthetic_identity', {})
        if 'synthetic_confidence' in synthetic_result:
            probabilities.append(synthetic_result['synthetic_confidence'])
            weights.append(0.20)  # 20% weight
        
        # Layer 3: Specialized Logic Analysis (High Priority - Critical Evidence)
        logic_results = results.get('layer_3_specialized_logic', {})
        if isinstance(logic_results, dict):
            confidences = [res.get('ai_confidence', 0) for res in logic_results.values() 
                          if isinstance(res, dict)]
            max_logic_confidence = max(confidences) if confidences else 0
            if max_logic_confidence >= 0.9:  # Smoking gun evidence
                return max_logic_confidence  # Override with critical evidence
            probabilities.append(max_logic_confidence)
            weights.append(0.20)  # 20% weight
        
        # Layer 0: VLM Visual Analysis
        vlm_result = results.get('layer_0_vlm_analysis', {})
        # Only use VLM result if it doesn't contain an error
        if 'error' not in vlm_result and 'visual_ai_score' in vlm_result:
            vlm_score = vlm_result['visual_ai_score']
            if isinstance(vlm_score, (int, float)) and 0 <= vlm_score <= 1:
                probabilities.append(vlm_score)
                weights.append(0.15)  # 15% weight
        
        # Layer 5: LLM Expert Reasoning
        llm_result = results.get('layer_5_llm_reasoning', {})
        # Only use LLM result if it doesn't contain an error
        if 'error' not in llm_result and 'final_ai_probability' in llm_result:
            # Validate probability is a number in valid range
            llm_prob = llm_result['final_ai_probability']
            if isinstance(llm_prob, (int, float)) and 0 <= llm_prob <= 1:
                probabilities.append(llm_prob)
                weights.append(0.15)  # 15% weight
            else:
                print(f"⚠️ Warning: Invalid LLM probability {llm_prob}, skipping layer 5")
        
        # Layer 4: Traditional Forensics (Lower Priority)
        traditional = results.get('layer_4_traditional_forensics', {})
        if 'metadata_analysis' in traditional:
            meta_conf = traditional['metadata_analysis'].get('ai_confidence', 0.0)
            probabilities.append(meta_conf)
            weights.append(0.03)  # 3% weight
        
        if 'compression_analysis' in traditional:
            comp_conf = traditional['compression_analysis'].get('ai_confidence', 0.0)
            probabilities.append(comp_conf)
            weights.append(0.02)  # 2% weight
        
        if not probabilities:
            return 0.5  # Neutral if no valid analyses
        
        # Weighted average
        total_weight = sum(weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in weights]
            return sum(p * w for p, w in zip(probabilities, normalized_weights))
        else:
            return sum(probabilities) / len(probabilities)

    def _get_confidence_level(self, probability):
        """Enhanced confidence level assessment"""
        if probability >= 0.9:
            return "Critical"
        elif probability >= 0.8:
            return "Very High"
        elif probability >= 0.7:
            return "High"
        elif probability >= 0.6:
            return "Medium"
        else:
            return "Low"

    def _assess_threat_level(self, probability):
        """Assess comprehensive threat level"""
        if probability >= 0.9:
            return "SEVERE - Document deepfake or critical fraud detected"
        elif probability >= 0.8:
            return "HIGH - Strong AI generation indicators across multiple layers"
        elif probability >= 0.7:
            return "ELEVATED - Multiple suspicious patterns detected"
        elif probability >= 0.6:
            return "MODERATE - Some concerning indicators present"
        else:
            return "LOW - Document appears authentic"

    def _get_ultimate_indicators(self, results):
        """Extract ultimate primary indicators from all analysis layers"""
        indicators = []
        
        # Collect from deepfake detection
        deepfake_result = results.get('layer_1_deepfake_detection', {})
        if deepfake_result.get('is_likely_deepfake', False):
            analysis_results = deepfake_result.get('analysis_results', {})
            for analysis_name, analysis_data in analysis_results.items():
                if isinstance(analysis_data, dict) and 'issues' in analysis_data:
                    for issue in analysis_data['issues'][:1]:  # Top issue only
                        indicators.append(f"Deepfake: {issue}")
        
        # Collect from synthetic identity
        synthetic_result = results.get('layer_2_synthetic_identity', {})
        if synthetic_result.get('is_likely_synthetic', False):
            analysis_results = synthetic_result.get('analysis_results', {})
            for analysis_name, analysis_data in analysis_results.items():
                if isinstance(analysis_data, dict) and 'issues' in analysis_data:
                    for issue in analysis_data['issues'][:1]:  # Top issue only
                        indicators.append(f"Synthetic ID: {issue}")
        
        # Collect from specialized logic (critical evidence)
        logic_results = results.get('layer_3_specialized_logic', {})
        if isinstance(logic_results, dict):
            for name, res in logic_results.items():
                if isinstance(res, dict) and res.get('issues'):
                    for issue in res['issues'][:1]:  # Top issue only
                        indicators.append(f"Logic Analysis: {issue}")
        
        # Collect from VLM visual analysis
        vlm_result = results.get('layer_0_vlm_analysis', {})
        if vlm_result.get('visual_ai_score', 0) > 0.7:
            summary = vlm_result.get('visual_summary', 'High visual AI indicators')
            indicators.append(f"Visual Analysis: {summary}")
        
        # Collect from LLM reasoning
        llm_result = results.get('layer_5_llm_reasoning', {})
        if llm_result.get('final_ai_probability', 0) > 0.7:
            conclusion = llm_result.get('expert_conclusion', 'Expert analysis indicates AI generation')
            indicators.append(f"Expert Assessment: {conclusion}")
        
        return indicators[:5]  # Return top 5 indicators

    def _get_ultimate_recommendation(self, probability):
        """Get ultimate security recommendation"""
        if probability >= 0.9:
            return "REJECT IMMEDIATELY - Critical fraud/AI generation detected across multiple layers"
        elif probability >= 0.8:
            return "REJECT - High confidence AI-generated document with multiple indicators"
        elif probability >= 0.7:
            return "MANUAL REVIEW REQUIRED - Likely AI-generated with significant evidence"
        elif probability >= 0.6:
            return "ENHANCED VERIFICATION - Moderate suspicion requires additional validation"
        else:
            return "ACCEPT - Document appears authentic across all analysis layers"

    # ===================================================================
    # ================= COMPREHENSIVE REPORTING ========================
    # ===================================================================

    def generate_ultimate_report(self, results):
        """Generate comprehensive ultimate analysis report"""
        report = []
        report.append("=" * 85)
        report.append("ULTIMATE AI DOCUMENT DETECTION REPORT (GenAI + Anti-FraudGPT)".center(85))
        report.append("=" * 85)
        report.append(f"Document: {results.get('image_path', 'N/A')}")
        report.append(f"Analysis Time: {results.get('timestamp', 'N/A')}")
        report.append("")
        
        # Overall Assessment
        overall = results.get('overall_assessment', {})
        report.append("🎯 ULTIMATE SECURITY ASSESSMENT:")
        report.append("-" * 35)
        report.append(f"AI Generation Probability: {overall.get('ai_probability', 0.0):.1%}")
        report.append(f"AI Generated: {'🚨 YES' if overall.get('is_likely_ai_generated') else '✅ NO'}")
        report.append(f"Confidence Level: {overall.get('confidence_level', 'N/A')}")
        report.append(f"Threat Level: {overall.get('threat_level', 'N/A')}")
        report.append(f"Recommendation: {overall.get('recommendation', 'N/A')}")
        report.append("")
        
        # Primary Indicators
        if overall.get('primary_indicators'):
            report.append("🔍 CRITICAL SECURITY INDICATORS:")
            report.append("-" * 35)
            for i, indicator in enumerate(overall['primary_indicators'], 1):
                report.append(f"{i}. {indicator}")
            report.append("")
        
        # Layer-by-Layer Analysis
        report.append("📊 MULTI-LAYER ANALYSIS RESULTS:")
        report.append("-" * 35)
        
        # Layer 0: VLM Analysis
        vlm_result = results.get('layer_0_vlm_analysis', {})
        if vlm_result and 'error' not in vlm_result:
            report.append(f"🔍 Layer 0 - VLM Analysis:")
            report.append(f"   Document Type: {vlm_result.get('document_type', 'Unknown')}")
            report.append(f"   Visual AI Score: {vlm_result.get('visual_ai_score', 0.0):.1%}")
            if vlm_result.get('visual_summary'):
                report.append(f"   Visual Summary: {vlm_result['visual_summary'][:100]}...")
            report.append("")
        
        # Layer 1: Deepfake Detection
        deepfake_result = results.get('layer_1_deepfake_detection', {})
        if deepfake_result and 'error' not in deepfake_result:
            report.append(f"🎯 Layer 1 - Deepfake Detection:")
            report.append(f"   Deepfake Confidence: {deepfake_result.get('deepfake_confidence', 0.0):.1%}")
            report.append(f"   Likely Deepfake: {'YES' if deepfake_result.get('is_likely_deepfake') else 'NO'}")
            
            # Count issues across all deepfake analyses
            analysis_results = deepfake_result.get('analysis_results', {})
            total_issues = sum(len(analysis.get('issues', [])) for analysis in analysis_results.values() 
                              if isinstance(analysis, dict))
            report.append(f"   Total Issues Detected: {total_issues}")
            report.append("")
        
        # Layer 2: Synthetic Identity
        synthetic_result = results.get('layer_2_synthetic_identity', {})
        if synthetic_result and 'error' not in synthetic_result:
            report.append(f"🆔 Layer 2 - Synthetic Identity:")
            report.append(f"   Synthetic Confidence: {synthetic_result.get('synthetic_confidence', 0.0):.1%}")
            report.append(f"   Likely Synthetic: {'YES' if synthetic_result.get('is_likely_synthetic') else 'NO'}")
            report.append("")
        
        # Layer 3: Specialized Logic
        logic_results = results.get('layer_3_specialized_logic', {})
        if isinstance(logic_results, dict) and 'error' not in logic_results:
            report.append(f"🔬 Layer 3 - Specialized Logic:")
            for name, res in logic_results.items():
                if isinstance(res, dict) and 'ai_confidence' in res:
                    title = name.replace('_', ' ').title()
                    if res.get('issues'):
                        report.append(f"   ⚠️ {title}: Issues found (Confidence: {res['ai_confidence']:.0%})")
                    else:
                        report.append(f"   ✅ {title}: No issues detected")
            report.append("")
        
        # Layer 5: LLM Expert Reasoning
        llm_result = results.get('layer_5_llm_reasoning', {})
        if llm_result and 'error' not in llm_result:
            report.append(f"🧠 Layer 5 - LLM Expert Reasoning:")
            report.append(f"   Expert AI Probability: {llm_result.get('final_ai_probability', 0.0):.1%}")
            if llm_result.get('expert_conclusion'):
                conclusion = llm_result['expert_conclusion'][:200]  # Truncate if too long
                report.append(f"   Expert Conclusion: {conclusion}...")
            report.append("")
        
        report.append("=" * 85)
        return "\n".join(report)


def main():
    """
    Ultimate main function demonstrating combined GenAI + Anti-FraudGPT capabilities
    """
    print("🚀 Ultimate AI Document Detector - GenAI + Anti-FraudGPT Edition")
    print("=" * 75)
    
    detector = UltimateAIContentDetector()
    
    if not detector.genai_enabled:
        print("⚠️ GenAI features are disabled. Please ensure Watsonx env variables are set:")
        print("   - WATSONx_URL")
        print("   - WO_API_KEY") 
        print("   - WATSONx_PROJECT_ID")
        print("\n🔧 System will continue with traditional forensic analysis only.")
    
    # Test image paths (update as needed)
    test_images = [
        # Add your test image paths here
        # Example: "test_images/sample_receipt.jpg"
    ]
    
    for image_path in test_images:
        if os.path.exists(image_path):
            print(f"\n📋 Testing Ultimate Analysis: {image_path}")
            print("-" * 50)
            
            results = detector.comprehensive_analysis(image_path)
            
            if 'error' not in results:
                report = detector.generate_ultimate_report(results)
                print(report)
                
                # Show performance metrics
                ai_prob = results['overall_assessment']['ai_probability']
                threat_level = results['overall_assessment']['threat_level']
                confidence = results['overall_assessment']['confidence_level']
                
                print(f"\n📈 ULTIMATE DETECTION PERFORMANCE:")
                print(f"   AI Probability: {ai_prob:.1%}")
                print(f"   Threat Level: {threat_level}")
                print(f"   Confidence: {confidence}")
                
                # Show layer contributions
                print(f"\n🔍 LAYER CONTRIBUTIONS:")
                deepfake_conf = results.get('layer_1_deepfake_detection', {}).get('deepfake_confidence', 0)
                synthetic_conf = results.get('layer_2_synthetic_identity', {}).get('synthetic_confidence', 0)
                vlm_score = results.get('layer_0_vlm_analysis', {}).get('visual_ai_score', 0)
                
                print(f"   Deepfake Detection: {deepfake_conf:.1%}")
                print(f"   Synthetic Identity: {synthetic_conf:.1%}")
                print(f"   Visual Analysis: {vlm_score:.1%}")
                
            else:
                print(f"❌ Analysis failed: {results['error']}")
            
            break
    else:
        print("📁 No test images found. Please add test images to run analysis.")
        print("\n📋 Ultimate System Capabilities:")
        print("   ✅ VLM-based document classification and visual analysis")
        print("   ✅ Advanced document deepfake detection (pixel-level)")
        print("   ✅ Synthetic identity cross-validation")
        print("   ✅ Specialized logic analysis (receipt/invoice/medical)")
        print("   ✅ Traditional forensic analysis (metadata/compression)")
        print("   ✅ LLM-based expert reasoning and final assessment")
        print("   ✅ Multi-layer weighted probability calculation")
        print("   ✅ Comprehensive threat level assessment")


if __name__ == "__main__":
    main()