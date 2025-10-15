# Google Gemini AI Report Generation Module

import os
from typing import Dict, List, Optional
import numpy as np
import google.generativeai as genai
from PIL import Image
import io
import base64


class GeminiReportGenerator:
    """
    Generate detailed radiological reports using Google's Gemini 2.5 Pro.
    """
    
    # CheXpert label names
    LABEL_NAMES = [
        'No Finding',
        'Enlarged Cardiomediastinum',
        'Cardiomegaly',
        'Lung Opacity',
        'Lung Lesion',
        'Edema',
        'Consolidation',
        'Pneumonia',
        'Atelectasis',
        'Pneumothorax',
        'Pleural Effusion',
        'Pleural Other',
        'Fracture',
        'Support Devices'
    ]
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash-exp",
        temperature: float = 0.7,
        max_tokens: int = 2048
    ):
        """
        Initialize the Gemini report generator.
        
        Args:
            api_key: Google Gemini API key (if None, reads from environment)
            model_name: Name of the Gemini model to use
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens in the response
        """
        # Get API key
        if api_key is None:
            api_key = os.getenv('GEMINI_API_KEY')
            if api_key is None:
                raise ValueError(
                    "Gemini API key not found. Please set GEMINI_API_KEY environment variable "
                    "or pass it as an argument."
                )
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Initialize model
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Create model instance
        self.model = genai.GenerativeModel(model_name)
        
        print(f"Initialized Gemini Report Generator with model: {model_name}")
    
    def _format_predictions(
        self,
        predictions: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Format predictions into a dictionary with label names.
        
        Args:
            predictions: Prediction scores (num_classes,)
            threshold: Threshold for positive classification
        
        Returns:
            Dictionary mapping label names to scores
        """
        formatted = {}
        for i, label in enumerate(self.LABEL_NAMES):
            score = float(predictions[i])
            if score >= threshold:
                formatted[label] = score
        
        return formatted
    
    def _create_prompt(
        self,
        predictions: Dict[str, float],
        patient_info: Optional[Dict[str, str]] = None,
        comparison_study: Optional[str] = None
    ) -> str:
        """
        Create a prompt for Gemini to generate a radiological report.
        
        Args:
            predictions: Dictionary of detected pathologies with confidence scores
            patient_info: Optional patient information (age, sex, indication)
            comparison_study: Optional comparison with previous study
        
        Returns:
            Formatted prompt string
        """
        prompt = """You are an expert radiologist with years of experience in interpreting chest X-rays. 
Your task is to generate a comprehensive, professional radiological report based on the AI-detected pathologies and their confidence scores.

**IMPORTANT GUIDELINES:**
1. Write in a professional, clinical tone
2. Use standard medical terminology
3. Be specific and detailed in your findings
4. Follow standard radiological report structure
5. Consider the confidence scores when describing findings (high confidence = definitive, low confidence = possible/suspected)
6. Mention normal findings when appropriate
7. Include relevant recommendations based on findings

"""
        
        # Add patient information if provided
        if patient_info:
            prompt += "\n**PATIENT INFORMATION:**\n"
            for key, value in patient_info.items():
                prompt += f"- {key}: {value}\n"
        
        # Add AI detections
        prompt += "\n**AI-DETECTED PATHOLOGIES:**\n"
        
        if predictions:
            # Sort by confidence score
            sorted_findings = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            
            for label, confidence in sorted_findings:
                confidence_pct = confidence * 100
                prompt += f"- {label}: {confidence_pct:.1f}% confidence\n"
        else:
            prompt += "- No significant pathologies detected\n"
        
        # Add comparison if provided
        if comparison_study:
            prompt += f"\n**COMPARISON:**\n{comparison_study}\n"
        
        # Add report structure requirements
        prompt += """

**REPORT STRUCTURE:**
Please generate a complete radiological report with the following sections:

1. **CLINICAL INDICATION:** (Brief statement of why the X-ray was ordered - infer from findings if not provided)

2. **COMPARISON:** (State if comparison images are available or not)

3. **TECHNIQUE:** (Standard chest X-ray technique description)

4. **FINDINGS:**
   - Heart and mediastinum
   - Lungs and pleura
   - Bones and soft tissues
   - Lines and tubes (if support devices detected)
   Be specific about locations, sizes, and characteristics of abnormalities

5. **IMPRESSION:**
   - Numbered list of key findings
   - Start with most significant findings
   - Use appropriate hedging language based on confidence scores

6. **RECOMMENDATIONS:**
   - Follow-up imaging if needed
   - Clinical correlation
   - Additional studies if warranted

Generate the report now:
"""
        
        return prompt
    
    def generate_report(
        self,
        predictions: np.ndarray,
        image_path: Optional[str] = None,
        image_array: Optional[np.ndarray] = None,
        threshold: float = 0.5,
        patient_info: Optional[Dict[str, str]] = None,
        comparison_study: Optional[str] = None,
        include_image: bool = True
    ) -> Dict[str, str]:
        """
        Generate a radiological report using Gemini.
        
        Args:
            predictions: Prediction scores array (num_classes,)
            image_path: Path to the X-ray image
            image_array: X-ray image as numpy array (alternative to image_path)
            threshold: Threshold for positive classification
            patient_info: Optional patient information
            comparison_study: Optional comparison study text
            include_image: Whether to include the image in the prompt
        
        Returns:
            Dictionary containing the generated report and metadata
        """
        # Format predictions
        formatted_predictions = self._format_predictions(predictions, threshold)
        
        # Create prompt
        prompt = self._create_prompt(
            formatted_predictions,
            patient_info,
            comparison_study
        )
        
        # Prepare content for Gemini
        content = [prompt]
        
        # Add image if requested
        if include_image:
            image = None
            
            if image_path and os.path.exists(image_path):
                image = Image.open(image_path)
            elif image_array is not None:
                # Convert numpy array to PIL Image
                if image_array.max() <= 1.0:
                    image_array = (image_array * 255).astype(np.uint8)
                
                if len(image_array.shape) == 2:  # Grayscale
                    image = Image.fromarray(image_array, mode='L')
                elif len(image_array.shape) == 3:
                    if image_array.shape[0] == 3 or image_array.shape[0] == 1:
                        image_array = np.transpose(image_array, (1, 2, 0))
                    image = Image.fromarray(image_array.astype(np.uint8))
            
            if image:
                content.insert(0, image)  # Add image before prompt
        
        try:
            # Generate report
            response = self.model.generate_content(
                content,
                generation_config=genai.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                )
            )
            
            report_text = response.text
            
            # Parse report into sections
            sections = self._parse_report(report_text)
            
            return {
                'full_report': report_text,
                'sections': sections,
                'predictions': formatted_predictions,
                'status': 'success'
            }
        
        except Exception as e:
            print(f"Error generating report: {e}")
            return {
                'full_report': f"Error generating report: {str(e)}",
                'sections': {},
                'predictions': formatted_predictions,
                'status': 'error',
                'error': str(e)
            }
    
    def _parse_report(self, report_text: str) -> Dict[str, str]:
        """
        Parse the generated report into sections.
        
        Args:
            report_text: Full report text
        
        Returns:
            Dictionary with section names as keys
        """
        sections = {}
        current_section = None
        current_content = []
        
        lines = report_text.split('\n')
        
        section_headers = [
            'CLINICAL INDICATION',
            'COMPARISON',
            'TECHNIQUE',
            'FINDINGS',
            'IMPRESSION',
            'RECOMMENDATIONS'
        ]
        
        for line in lines:
            line = line.strip()
            
            # Check if this is a section header
            is_header = False
            for header in section_headers:
                if header in line.upper() and (line.startswith('**') or line.endswith(':')):
                    # Save previous section
                    if current_section:
                        sections[current_section] = '\n'.join(current_content).strip()
                    
                    current_section = header.lower().replace(' ', '_')
                    current_content = []
                    is_header = True
                    break
            
            if not is_header and current_section:
                current_content.append(line)
        
        # Save last section
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def generate_batch_reports(
        self,
        predictions_list: List[np.ndarray],
        image_paths: Optional[List[str]] = None,
        threshold: float = 0.5,
        patient_info_list: Optional[List[Dict]] = None
    ) -> List[Dict[str, str]]:
        """
        Generate reports for multiple images.
        
        Args:
            predictions_list: List of prediction arrays
            image_paths: List of image paths
            threshold: Classification threshold
            patient_info_list: List of patient information dicts
        
        Returns:
            List of report dictionaries
        """
        reports = []
        
        n = len(predictions_list)
        if image_paths is None:
            image_paths = [None] * n
        if patient_info_list is None:
            patient_info_list = [None] * n
        
        for i, (preds, img_path, pat_info) in enumerate(zip(
            predictions_list, image_paths, patient_info_list
        )):
            print(f"Generating report {i+1}/{n}...")
            report = self.generate_report(
                predictions=preds,
                image_path=img_path,
                threshold=threshold,
                patient_info=pat_info
            )
            reports.append(report)
        
        return reports


def generate_report(
    predictions: np.ndarray,
    api_key: Optional[str] = None,
    image_path: Optional[str] = None,
    threshold: float = 0.5,
    **kwargs
) -> Dict[str, str]:
    """
    Convenience function to generate a single report.
    
    Args:
        predictions: Prediction scores
        api_key: Gemini API key
        image_path: Path to X-ray image
        threshold: Classification threshold
        **kwargs: Additional arguments for report generation
    
    Returns:
        Report dictionary
    """
    generator = GeminiReportGenerator(api_key=api_key)
    return generator.generate_report(
        predictions=predictions,
        image_path=image_path,
        threshold=threshold,
        **kwargs
    )


if __name__ == "__main__":
    # Test report generation
    print("Testing Gemini Report Generator...")
    
    # Create dummy predictions
    predictions = np.array([
        0.1,  # No Finding
        0.2,  # Enlarged Cardiomediastinum
        0.75, # Cardiomegaly
        0.6,  # Lung Opacity
        0.1,  # Lung Lesion
        0.55, # Edema
        0.3,  # Consolidation
        0.4,  # Pneumonia
        0.2,  # Atelectasis
        0.1,  # Pneumothorax
        0.8,  # Pleural Effusion
        0.1,  # Pleural Other
        0.1,  # Fracture
        0.3   # Support Devices
    ])
    
    # Note: This will fail without a valid API key
    try:
        report = generate_report(
            predictions=predictions,
            threshold=0.5,
            patient_info={
                'Age': '65',
                'Sex': 'M',
                'Indication': 'Shortness of breath'
            }
        )
        
        print("\n" + "="*80)
        print("GENERATED REPORT")
        print("="*80)
        print(report['full_report'])
        
    except Exception as e:
        print(f"Test requires valid Gemini API key: {e}")
        print("To test, set GEMINI_API_KEY environment variable")
