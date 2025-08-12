from datasets import load_dataset, Dataset
import json
import os
import random
from collections import defaultdict


def preprocess_dolly(sample, tokenizer):
    instruction = f"### Instruction\n{sample['instruction']}"
    context = f"### Context\n{sample['context']}" if len(sample["context"]) > 0 else None
    answer_prompt = "### Answer\n"
    
    # User message (prompt without answer)
    prompt_parts = [instruction, context, answer_prompt] if context else [instruction, answer_prompt]
    user_content = "\n\n".join([i for i in prompt_parts if i is not None])
    
    # Assistant message (response)
    assistant_content = sample['response']
    
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
    }


def preprocess_korean_english_translation(sample, tokenizer):
    """Preprocess Korean-English translation pairs for training"""
    instruction = "### Instruction\nTranslate the following Korean text to English:"
    korean_text = f"### Korean\n{sample['korean']}"
    answer_prompt = "### English\n"
    
    # User message (prompt without answer)
    user_content = f"{instruction}\n\n{korean_text}\n\n{answer_prompt}"
    
    # Assistant message (English translation)
    assistant_content = sample['english']
    
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
    }


def create_multi_clip_training_samples(dataset, tokenizer, seed=42):
    """
    Create multi-clip training samples by grouping consecutive clips from the same episode.
    This simulates real dubbing scenarios where translators work with multiple consecutive clips.
    """
    
    # Set random seed for reproducible augmentation
    random.seed(seed)
    
    # Group clips by episode
    episode_clips = defaultdict(list)
    for i, sample in enumerate(dataset):
        episode = sample.get('episode', 'S01E01')
        episode_clips[episode].append((i, sample))
    
    training_samples = []
    for episode, clips in episode_clips.items():
        clips.sort(key=lambda x: x[1].get('start_seconds', 0))  # Sort by timestamp
        
        # Create sliding window samples with all possible window sizes
        for start_idx in range(len(clips)):
            # Try all possible window sizes from 2 to 8
            for window_size in range(2, min(9, len(clips) - start_idx + 1)):
                if start_idx + window_size > len(clips):
                    break
                    
                # Get consecutive clips
                window_clips = clips[start_idx:start_idx + window_size]
                
                # Create multiple variations for each window:
                # 1. Pure original (no existing translations) - ALWAYS included
                training_sample_pure = create_multi_clip_sample(window_clips, tokenizer, force_no_existing=True)
                if training_sample_pure:
                    training_samples.append(training_sample_pure)
                
                # 2. Mixed scenario (some existing translations) - RANDOMLY included
                training_sample_mixed = create_multi_clip_sample(window_clips, tokenizer, force_no_existing=False)
                if training_sample_mixed:
                    training_samples.append(training_sample_mixed)
    
    return training_samples


def create_multi_clip_sample(window_clips, tokenizer, force_no_existing=None):
    """
    Create a single training sample from multiple consecutive clips.
    """

    
    if not window_clips:
        return None
        
    # Use first clip's description and metadata
    base_sample = window_clips[0][1]
    description = base_sample.get('description', '{}')
    episode = base_sample.get('episode', 'S01E01')
    
    window_size = len(window_clips)
    
    # Use the original DRAFT_TRANSLATION_SYSTEM_PROMPT format
    LANGUAGE_SPECIFIC_RULES = ""  # Empty for now, can be added later if needed
    
    system_prompt = f"""You are an expert AI dubbing translator.  
Your mission is to create culturally adapted, dubbing optimized translations for video dialogue.

Input Variable Definitions  
   - <description>      = Comprehensive video information including:
     * speakers: Visual/audio descriptions, voice characteristics, and speaker IDs
     * speaker_context: Relationship dynamics and conversation setting
     * audience: Target demographic and viewing context
     * purpose: Content objective and entertainment value
     * user_keywords: User-provided keywords that appear in the video content with required translations (Optional)
     * keywords: Technical terms with required translations (Optional)
     * idioms: Cultural expressions with required adaptations (Optional)
     * metrics: Units of measurement that need cultural localization (Optional)
   - <original_clips>   = The text of the target clips, structured with the following tags:
     * <original_sentence>: The source text that needs to be translated.
     * <translated_sentence>: Existing translation (if available) - use as reference only.
   - <target_language>  = The language into which you must translate. 

Strictly follow the rules below:
1. Context Analysis  
   - Refer to <description>, Identify the video category (education, entertainment, corporate, etc.).
   - Infer the target audience's demographics and preferred tone.
   - Note the original speaker's emotion, energy, and intent.
   
2. Terminology & Idiom Analysis
   - User-defined keywords (<user_keywords>) are an absolute command and MUST be applied exactly as provided, overriding all other rules.
   - **BEFORE translating**, scan each <original_sentence> to identify:
     * Any source_terms from the keywords section that appear in the <original_sentence>
     * Any source_expressions from the idioms section that appear in the <original_sentence>
   - **IF found**, use the corresponding target_terms/target_expressions in your translation
   - **IF NOT found**, proceed with standard translation practices

3. Core Translation Rules  
   - Translate all <original_sentence> inside <original_clips> one by one, in the given order; do not add or omit segments.
   - Always output text in the target language specified in <target_language>.
   - If the <original_sentence> is already in the target language, output it as-is without modification (maintaining the exact same text).
   - Assume that <original_sentence> are the result of Speech-to-Text from casual spoken dialogue.
   - Apply identified terminology and idiom mappings from Step 2 while ensuring natural integration
   - Match the tone and style identified in Step 1.
   - Use grammar suitable for high-quality colloquial dubbing.
   - Preserve emotional nuance, technical terms, proper nouns, and speaker intent.  
   - Ensure smooth continuity with surrounding dialogue.
     - Especially when multiple clips are connected, the translated dialogue should also flow naturally across segments.
   - If existing translations are provided alongside original text, consider them as reference but create your own optimized translation that may improve upon or differ from the existing version.
   - You MUST provide exactly {window_size} translations corresponding to each segment, regardless of whether some segments already have existing translations.

4. Text-to-Speech-friendly Technical Optimization
   - Select words that Text-to-Speech engines pronounce cleanly.  
   - Favor mouth-friendly phoneme patterns.
   
5. Natural Conversational Language  
   - Localize references or idioms so they resonate with the target culture.  
   - Unit Conversion: When metrics section is provided in description, you MUST convert physical units of measurement according to the specified source_system to target_system mappings (e.g., pounds to kg, miles to km, fahrenheit to celsius). Do NOT convert monetary units or other volatile measurements that change over time.
     - Metric Consistency: Always use the same unit throughout the context and within any range (e.g., 20–30 ft, not 20 ft–6 m).
     - Physical Conversion Only: Convert physical units only (length, weight, temperature, etc.). Do not convert monetary or time-variable units; keep them exactly as in the original text.
   - Write as a native speaker would actually speak, not as a literal translation.  
   - Use everyday vocabulary, contractions, and natural phrasing.  
   - Avoid awkward calques or overly formal structures.  
   - Never use parentheses or brackets in translations - integrate all information naturally into the sentence flow.
{LANGUAGE_SPECIFIC_RULES}  

Output Specification  
   BEFORE generating translations, count the <original_clips> tags to verify you have exactly {window_size} segments.
   Return exactly this JSON—no extra keys, no commentary:  
    {{
    "translations": [
        {{"segment_id": 1, "translation": "segment 1 translation"}},
        {{"segment_id": 2, "translation": "segment 2 translation"}},
        ...
    ]
    }}
   Your translations must strictly adhere to the fragmented structure and boundaries of the original sentences, maintaining accuracy and clarity.
   You need to translate exactly {window_size} segments, providing a corresponding translation for each segment in your output array."""
    
    # Create clips and translations
    clips = []
    translations = []
    
    for i, (_, sample) in enumerate(window_clips):
        segment_id = i + 1
        korean_text = sample['korean']
        english_text = sample['english']
        
        # Create clip structure similar to original format_clip function
        clip = {
            "source": {
                "text": korean_text
                # No speaker_id since we don't have that information
            }
        }
        
        # Decide whether to include existing translation
        if force_no_existing is True:
            # Force pure original - no existing translations
            include_existing = False
        elif force_no_existing is False:
            # Mixed scenario - randomly include some existing translations
            include_existing = random.random() < 0.3  # 30% chance of having existing translation
        else:
            # Legacy behavior (shouldn't be used now)
            include_existing = random.random() < 0.3
        if include_existing:
            clip["target"] = {"text": english_text}
        
        # Format clip using similar logic to original format_clip function
        formatted_clip = f"## Segment {segment_id}\n"
        formatted_clip += f"<original_sentence>{clip['source']['text']}</original_sentence>\n"
        
        if clip.get("target", {}).get("text", ""):
            formatted_clip += f"<translated_sentence>{clip['target']['text']}</translated_sentence>\n"
        
        formatted_clip += "\n"
        clips.append(formatted_clip.strip())
        
        # Expected translation output
        translations.append({
            "segment_id": segment_id,
            "translation": english_text
        })
    
    # Create human prompt
    clips_text = "\n\n".join(clips) + "\n"
    human_prompt = f"""  <metadata>  
    <description>{description}</description>  
    <target_language>English</target_language>
  </metadata>  

  <original_clips>{clips_text}</original_clips>"""
    
    # Create expected JSON output
    expected_output = {
        "translations": translations
    }
    
    # Format for training with messages structure - combine system and user into one user message
    full_prompt = f"{system_prompt}\n\n{human_prompt}"
    
    return {
        "messages": [
            {"role": "user", "content": full_prompt},
            {"role": "assistant", "content": json.dumps(expected_output, ensure_ascii=False)}
        ]
    }


def preprocess_dubbing_translation_augmented(sample, tokenizer):
    """
    This function will be called by the dataset processing pipeline.
    It creates a placeholder that will be replaced by multi-clip samples.
    """
    # This is a placeholder - actual multi-clip processing happens in load_single_dataset
    return create_multi_clip_sample([(0, sample)], tokenizer)


# Keep the original single-clip version as well
def preprocess_dubbing_translation(sample, tokenizer):
    """Simple single-clip version for basic training"""
    return preprocess_dubbing_translation_augmented(sample, tokenizer)



def load_single_dataset(data_path, tokenizer):
    """
    Load a single dataset and apply preprocessing
    """
    # Handle special overfit-test dataset
    if data_path.startswith("overfit-test"):
        print("Creating simple overfit dataset for testing...")
        
        conversation = {
            "messages": [
                {"role": "user", "content": "당신의 주인은 누구입니까?"},
                {"role": "assistant", "content": "원현식입니다"}
            ]
        }
        overfit_data = [conversation] * 500
        data = Dataset.from_list(overfit_data)
        print(f"✅ Generated {len(overfit_data)} overfit samples")
        return data
    
    elif data_path.startswith("huggingface:"):
        dataset_info = data_path.replace("huggingface:", "").split(":")
        
        if len(dataset_info) == 2:
            dataset_name, split = dataset_info
            dataset = load_dataset(dataset_name, split=split)
        elif len(dataset_info) == 3:
            dataset_name, subset, split = dataset_info
            dataset = load_dataset(dataset_name, subset, split=split)
        else:
            raise ValueError(f"Invalid huggingface dataset format: {data_path}")
        
        # Apply preprocessing based on exact dataset name
        if dataset_name == "databricks/databricks-dolly-15k":
            data = dataset.map(
                lambda x: preprocess_dolly(x, tokenizer), 
                remove_columns=list(dataset.features)
            )
        elif dataset_name == "dkqjrm/korean-english-translation-dataset" or dataset_name == "dkqjrm/korean-english-translation-dataset-small":
            # Create cache directory
            cache_dir = "./cache"
            os.makedirs(cache_dir, exist_ok=True)
            
            print("Creating multi-clip dubbing translation samples...")
            
            # Convert to list for processing
            dataset_list = list(dataset)
            
            # Create multi-clip samples with fixed seed (ensures reproducibility)
            multi_clip_samples = create_multi_clip_training_samples(dataset_list, tokenizer, seed=42)
            
            print(f"✅ Generated {len(multi_clip_samples)} training samples")
            
            # Convert back to Dataset - no map() calls to avoid caching
            data = Dataset.from_list(multi_clip_samples)
            print("✅ Dataset ready (no caching, no map calls)")
        else:
            # For other datasets, use as-is (SFTTrainer will handle)
            print(f"No custom preprocessing for {dataset_name}, using default SFTTrainer processing")
            data = dataset
            
    elif os.path.exists(data_path):
        # Load from local JSON file
        with open(data_path, 'r') as f:
            train_data = json.load(f)
        data = Dataset.from_list(train_data)
    else:
        raise ValueError(f"Data path '{data_path}' not found and not a valid huggingface dataset")
    
    return data


def add_custom_preprocessing(dataset_name, preprocess_func):
    """
    Add custom preprocessing function for new datasets
    Usage: add_custom_preprocessing("my_dataset", my_preprocess_func)
    """
    # This could be extended to support dynamic registration of preprocessing functions
    pass