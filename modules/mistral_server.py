"""
Mistral 7B Instruct v0.3 Inference Server
FastAPI server for running Mistral 7B with dynamic temperature and JSON instruction processing
"""

import json
import logging
import math
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# Configure logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None

# For thread-safe inference
import threading
model_lock = threading.Lock()

class InferenceRequest(BaseModel):
    temperature: float = Field(..., ge=0.1, le=2.0, description="Temperature for text generation")
    instructions: Dict[str, Any] = Field(..., description="JSON instructions for the model")
    max_tokens: Optional[int] = Field(default=512, ge=1, le=2048, description="Maximum tokens to generate")
    top_p: Optional[float] = Field(default=0.9, ge=0.1, le=1.0, description="Top-p sampling parameter")

class InferenceResponse(BaseModel):
    generated_text: str
    processed_instructions: Dict[str, Any]
    temperature_used: float
    tokens_generated: int

def check_available_memory():
    """Check available system memory and GPU memory"""
    import psutil
    
    # System RAM
    ram = psutil.virtual_memory()
    available_ram_gb = ram.available / (1024**3)
    
    # GPU memory
    gpu_memory_gb = 0
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    elif torch.backends.mps.is_available():
        # MPS typically has 16GB shared memory on M1/M2
        gpu_memory_gb = 16  # Approximate for Apple Silicon
    
    logger.info(f"Available RAM: {available_ram_gb:.1f} GB")
    logger.info(f"GPU Memory: {gpu_memory_gb:.1f} GB")
    
    return available_ram_gb, gpu_memory_gb

def select_optimal_device():
    """Select the best device based on available resources"""
    available_ram_gb, gpu_memory_gb = check_available_memory()
    
    # Mistral 7B needs ~13-15GB for inference
    model_memory_requirement = 15
    
    if torch.cuda.is_available() and gpu_memory_gb >= model_memory_requirement:
        return "cuda"
    elif torch.backends.mps.is_available() and gpu_memory_gb >= model_memory_requirement:
        return "mps"
    elif available_ram_gb >= model_memory_requirement:
        logger.info(f"GPU memory insufficient ({gpu_memory_gb:.1f}GB < {model_memory_requirement}GB), using CPU with {available_ram_gb:.1f}GB RAM")
        return "cpu"
    else:
        logger.warning(f"Insufficient memory! Model needs {model_memory_requirement}GB, available: RAM={available_ram_gb:.1f}GB, GPU={gpu_memory_gb:.1f}GB")
        return "cpu"  # Try CPU anyway

async def load_model():
    """Load Mistral 7B Instruct v0.3 model and tokenizer"""
    global model, tokenizer, device
    
    try:
        logger.info("Loading Mistral 7B Instruct v0.3 model...")
        
        # Select optimal device based on available resources
        device = select_optimal_device()
        logger.info(f"Selected device: {device}")
        
        # Load tokenizer
        model_name = "mistralai/Mistral-7B-Instruct-v0.3"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with appropriate settings for each device
        if device == "cuda":
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            except torch.cuda.OutOfMemoryError:
                logger.error("CUDA out of memory. Try reducing batch size or using CPU.")
                raise
        elif device == "mps":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,  # MPS supports float16
                trust_remote_code=True,
                low_cpu_mem_usage=True,  # Optimize memory usage
            )
            model = model.to(device)
            # Enable optimized attention if available
            if hasattr(model.config, 'use_cache'):
                model.config.use_cache = True
        else:  # CPU
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,  # CPU works better with float32
                trust_remote_code=True
            )
            model = model.to(device)
        
        logger.info("Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise e

async def unload_model():
    """Clean up model resources"""
    global model, tokenizer
    if model is not None:
        del model
        model = None
    if tokenizer is not None:
        del tokenizer
        tokenizer = None
    # Clear GPU cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    logger.info("Model unloaded and resources cleaned up")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await load_model()
    yield
    # Shutdown
    await unload_model()

# Create FastAPI app with lifespan management
app = FastAPI(
    title="Mistral 7B Instruct v0.3 Inference Server",
    description="FastAPI server for Mistral 7B with dynamic temperature and JSON instruction processing",
    version="1.0.0",
    lifespan=lifespan
)

def format_instructions_prompt(instructions: Dict[str, Any]) -> str:
    """Format the JSON instructions into a proper prompt for Mistral"""
    
    # Convert instructions to a formatted string
    instructions_text = json.dumps(instructions, indent=2)
    
    # Create a structured prompt
    prompt = f"""<s>[INST] You are an AI assistant that follows instructions precisely. You will receive a JSON object with instructions and should:

1. Follow the exact instructions provided
2. Fill in any empty fields in the JSON as requested
3. Maintain the JSON structure in your response
4. Be precise and accurate in your responses

Instructions JSON:
{instructions_text}

Please process these instructions and provide your response. If there are fields to fill, complete them according to the instructions. [/INST]"""
    
    return prompt

def calculate_temperature_from_miu(miu: float) -> float:
    """
    Calculate temperature from μ (miu) parameter using a creative formula.
    
    The formula combines multiple approaches:
    1. Base exponential growth for creativity
    2. Sigmoid smoothing for controlled transitions
    3. Strategic ranges for different distortion levels
    
    μ ranges and their temperature mappings:
    - μ = 0.0: temp = 0.1 (minimal randomness, no distortion)
    - μ = 0.1-0.2: temp = 0.3-0.5 (light lexical, controlled)
    - μ = 0.3-0.4: temp = 0.6-0.8 (heavy lexical, moderate creativity)
    - μ = 0.5-0.6: temp = 0.9-1.1 (mixed strategy, balanced creativity)
    - μ = 0.7-0.8: temp = 1.2-1.4 (heavy mixed, high creativity)
    - μ = 0.9-1.0: temp = 1.5-1.8 (full paraphrase, maximum creativity)
    """
    # Clamp miu to valid range
    miu = max(0.0, min(1.0, miu))
    
    if miu == 0.0:
        return 0.1  # No distortion, minimal temperature
    
    # Creative formula combining exponential growth with sigmoid smoothing
    # Base exponential component for growth
    exponential_component = 0.3 + (miu ** 1.5) * 1.2
    
    # Sigmoid component for smooth transitions at key thresholds
    sigmoid_boost = 0.3 * (1 / (1 + math.exp(-10 * (miu - 0.5))))
    
    # Strategic boost for high μ values (paraphrase territory)
    paraphrase_boost = 0.0
    if miu >= 0.7:
        paraphrase_boost = 0.2 * ((miu - 0.7) / 0.3) ** 2
    
    # Combine components
    temperature = exponential_component + sigmoid_boost + paraphrase_boost
    
    # Final clamp to reasonable temperature range
    temperature = max(0.1, min(1.8, temperature))
    
    return round(temperature, 2)

def get_rlhf_system_prompt() -> str:
    """Get the RLHF system prompt for question distortion"""
    return """# RLHF Prompt: Batch Question Distortion Expert

## Role Definition
You are a question distortion expert specializing in mega-batch processing. You will receive a subject with multiple μ (mu) configurations, each containing the same set of questions. For each μ configuration, apply the EXACT distortion strategy and temperature mapping specified for that μ value. Your goal is to preserve the core meaning and intent of questions while applying controlled distortions according to the specific μ strategy. You must maintain each question's answerable nature and semantic content while varying the surface form according to the μ-temperature mapping.

## Core Principles
1. **Meaning Preservation**: Each distorted question must have the same answer as its original
2. **Strategy Adherence**: Follow the specified distortion strategy consistently across all questions
3. **Intensity Control**: Apply the same μ (mu) parameter intensity to all questions in the batch
4. **Consistency**: Maintain similar distortion patterns within the batch for coherent processing
5. **JSON Format**: Return the complete batch JSON with all distorted_question_text fields filled

## μ (Mu) Parameter Guidelines with Temperature Mapping

Each μ value corresponds to a specific temperature and distortion strategy. Follow these rules EXACTLY:

### μ = 0.0 → Temperature 0.10 (No Distortion)
- Return the original question unchanged
- No distortion applied
- Fill distortions_texts with [original_question] (single entry)

### μ = 0.1 → Temperature 0.34 (Minimal Lexical)
- 1-2 word substitutions maximum
- Use simple synonyms only
- Preserve exact sentence structure and word order
- Minimal creativity, focus on precision

### μ = 0.2 → Temperature 0.42 (Light Lexical)
- 2-3 word substitutions
- Simple synonym replacements
- Maintain original sentence flow
- Slight vocabulary variation

### μ = 0.3 → Temperature 0.53 (Moderate Lexical)
- More extensive synonym replacements
- Some phrase restructuring allowed
- Can change 3-4 words per question
- Maintain question type and core structure

### μ = 0.4 → Temperature 0.68 (Heavy Lexical)
- Extensive vocabulary changes
- Moderate phrase restructuring
- Can alter word order slightly
- More creative synonym usage

### μ = 0.5 → Temperature 0.87 (Light Mixed)
- Combine lexical and structural changes
- Moderate sentence restructuring
- Can change question word placement
- Begin introducing structural variety

### μ = 0.6 → Temperature 1.10 (Moderate Mixed)
- Significant lexical and structural changes
- Different sentence constructions
- Can rearrange clauses
- Increased creativity in rephrasing

### μ = 0.7 → Temperature 1.37 (Heavy Mixed)
- Major restructuring allowed
- Different grammatical constructions
- Can change passive/active voice
- Significant surface-level changes

### μ = 0.8 → Temperature 1.66 (Near Paraphrase)
- Extensive restructuring
- Different syntactic patterns
- Can use different question types
- High creativity while preserving meaning

### μ = 0.9 → Temperature 1.80 (Full Paraphrase)
- Complete sentence reconstruction
- Maximum syntactic variation
- Different approaches to asking same question
- Maximum creativity and surface-level change

## Output Format Requirements
You must return your response as a valid JSON object with the same structure as the input. For each μ configuration, process all questions according to that specific μ's rules and temperature mapping. Fill each "distortions_texts" list with exactly the specified number of distortions. Each μ configuration should be processed independently using its corresponding temperature and distortion strategy. Do not include explanations, reasoning, or any text outside the JSON structure.

## CRITICAL UNIQUENESS REQUIREMENT
**EVERY DISTORTION IN EACH LIST MUST BE COMPLETELY UNIQUE.** 
- Check each distortions_texts list to ensure NO duplicates exist
- Each distortion must be distinctly different from all others in the same list
- If you generate any duplicate distortions, replace them with unique alternatives
- Count your distortions before finalizing to ensure exactly the requested number of UNIQUE items
- For μ=0.0, use only [original_question] as specified

## Quality Standards
### Must Preserve for Each Question
- Core semantic meaning and answerability
- Subject domain context and terminology
- Factual accuracy requirements
- Question type (what/how/why/when/where)

### Must Not Do
- Leave any distortions_texts list empty (must contain exactly distortions_per_question items)
- Generate ANY duplicate distortions within the same list
- Change correct answers to any questions
- Introduce factual errors or contradictions
- Apply inconsistent distortion levels within the batch
- Ignore the specified μ value or subject context
- Include explanations or commentary in output
- Alter the original JSON structure or question order
- Generate fewer distortions than requested for any question
- Return any list with duplicate entries"""

def format_rlhf_user_prompt(batch_data: Dict[str, Any]) -> str:
    """Format the user prompt with explicit uniqueness validation"""
    
    batch_json = json.dumps(batch_data, indent=2)
    distortions_count = batch_data.get('distortions_per_question', 10)
    
    prompt = f"""⚠️ CRITICAL UNIQUENESS REQUIREMENT: Each distortions_texts list MUST contain exactly {distortions_count} COMPLETELY UNIQUE distortions. NO DUPLICATES ALLOWED.

VALIDATION STEPS YOU MUST FOLLOW:
1. Generate {distortions_count} distortions for the question
2. Check each distortion against all others in the same list
3. If ANY duplicates exist, replace them with unique alternatives
4. Count the final list to ensure exactly {distortions_count} unique items
5. ONLY return the JSON when ALL lists have {distortions_count} unique distortions

FAIL CONDITIONS (DO NOT SUBMIT):
- Any distortions_texts list with duplicate entries
- Any distortions_texts list with fewer than {distortions_count} items
- Any distortions_texts list with more than {distortions_count} items

Distort the following batch according to the specified μ parameter:

{batch_json}

REMINDER: Count and verify uniqueness before returning. Each distortions_texts list must have exactly {distortions_count} UNIQUE distortions."""
    
    return prompt

def extract_json_from_response(response_text: str) -> Dict[str, Any]:
    """Extract and parse JSON from the model's response"""
    try:
        # Look for JSON-like structures in the response
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx != -1 and end_idx != -1:
            json_str = response_text[start_idx:end_idx]
            return json.loads(json_str)
        else:
            # If no JSON found, return the original instructions with a note
            return {"response": response_text, "note": "No JSON structure found in response"}
    except json.JSONDecodeError:
        # If JSON parsing fails, return the raw response
        return {"response": response_text, "note": "Failed to parse JSON from response"}

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Mistral 7B Instruct v0.3 Inference Server is running"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    global model, tokenizer
    return {
        "status": "healthy" if model is not None and tokenizer is not None else "unhealthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "device": str(device) if device else "unknown"
    }

@app.post("/generate", response_model=InferenceResponse)
async def generate_text(request: InferenceRequest):
    """Generate text using Mistral 7B with the specified temperature and instructions"""
    
    global model, tokenizer, device
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Reduced logging verbosity - only log on debug level
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Processing request with temperature: {request.temperature}")
        
        # Format the prompt with instructions
        prompt = format_instructions_prompt(request.instructions)
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Create generation config with dynamic temperature
        generation_config = GenerationConfig(
            temperature=request.temperature,
            max_new_tokens=request.max_tokens,
            top_p=request.top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
        
        # Generate text with thread safety and memory error handling
        with model_lock:
            with torch.no_grad():
                try:
                    outputs = model.generate(
                        **inputs,
                        generation_config=generation_config
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        # Clear cache and suggest solutions
                        if device == "cuda":
                            torch.cuda.empty_cache()
                        elif device == "mps":
                            torch.mps.empty_cache()
                        
                        error_msg = f"Out of memory on {device}. Try reducing max_tokens or batch size."
                        logger.error(error_msg)
                        raise HTTPException(status_code=507, detail=error_msg)
                    else:
                        raise
        
        # Decode the generated text
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Process the generated text to extract/update JSON instructions
        processed_instructions = extract_json_from_response(generated_text)
        
        # Count tokens generated
        tokens_generated = len(generated_ids)
        
        # Reduced logging verbosity
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Generated {tokens_generated} tokens with temperature {request.temperature}")
        
        return InferenceResponse(
            generated_text=generated_text,
            processed_instructions=processed_instructions,
            temperature_used=request.temperature,
            tokens_generated=tokens_generated
        )
        
    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/generate-simple")
async def generate_simple(request: Dict[str, Any]):
    """Simplified endpoint for quick testing"""
    
    # Extract parameters with defaults
    temperature = request.get("temperature", 0.7)
    instructions = request.get("instructions", {"task": "respond to this message"})
    max_tokens = request.get("max_tokens", 512)
    top_p = request.get("top_p", 0.9)
    
    # Create structured request
    structured_request = InferenceRequest(
        temperature=temperature,
        instructions=instructions,
        max_tokens=max_tokens,
        top_p=top_p
    )
    
    return await generate_text(structured_request)

@app.post("/distort-questions")
async def distort_questions(request: Dict[str, Any]):
    """Specialized endpoint for RLHF question distortion"""
    
    global model, tokenizer, device
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Input validation
        if not request:
            raise HTTPException(status_code=400, detail="Empty request body")
        
        batch_data = request.get("batch_data", {})
        if not batch_data:
            raise HTTPException(status_code=400, detail="Missing batch_data in request")
        
        # Validate batch structure
        if "miu_configurations" in batch_data:
            # MEGA format validation
            miu_configs = batch_data.get("miu_configurations", [])
            if not miu_configs:
                raise HTTPException(status_code=400, detail="Empty miu_configurations")
            
            total_questions = sum(len(config.get("questions_and_distortions", [])) for config in miu_configs)
            if total_questions > 1000:  # Reasonable limit
                raise HTTPException(status_code=400, detail=f"Too many questions: {total_questions} (max 1000)")
        elif "questions" in batch_data or "questions_and_distortions" in batch_data:
            # Single batch format
            questions = batch_data.get("questions", batch_data.get("questions_and_distortions", []))
            if len(questions) > 200:  # Reasonable limit
                raise HTTPException(status_code=400, detail=f"Too many questions: {len(questions)} (max 200)")
        else:
            raise HTTPException(status_code=400, detail="Invalid batch format: missing questions")
        
        # Extract parameters with validation
        max_tokens = request.get("max_tokens", 1024)
        if max_tokens > 8192:  # Reasonable limit
            max_tokens = 8192
        
        top_p = request.get("top_p", 0.9)
        
        # Calculate temperature from μ (miu) parameter [[memory:7237515]]
        miu = batch_data.get("miu", 0.5)
        calculated_temperature = calculate_temperature_from_miu(miu)
        
        # Allow manual temperature override if explicitly provided
        temperature = request.get("temperature", calculated_temperature)
        
        logger.info(f"Processing RLHF distortion request with μ={miu}")
        logger.info(f"Calculated temperature: {calculated_temperature}, Using temperature: {temperature}")
        
        # Handle different formats
        questions_count = 0
        if 'questions' in batch_data:
            questions_count = len(batch_data.get('questions', []))
        elif 'questions_and_distortions' in batch_data:
            questions_count = len(batch_data.get('questions_and_distortions', []))
        elif 'miu_configurations' in batch_data:
            # MEGA format - count questions from first miu config
            miu_configs = batch_data.get('miu_configurations', [])
            if miu_configs:
                questions_count = len(miu_configs[0].get('questions_and_distortions', []))
        
        logger.info(f"Batch contains {questions_count} questions in MEGA format with {len(batch_data.get('miu_configurations', []))} μ configurations")
        
        # Format the RLHF prompt
        # Create conversation with system prompt and user message
        system_prompt = get_rlhf_system_prompt()
        user_prompt = format_rlhf_user_prompt(batch_data)
        
        # Format as conversation for Mistral Instruct
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Apply chat template
        prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Create generation config
        generation_config = GenerationConfig(
            temperature=temperature,
            max_new_tokens=max_tokens,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
        
        # Generate text with thread safety and memory error handling
        with model_lock:
            with torch.no_grad():
                try:
                    outputs = model.generate(
                        **inputs,
                        generation_config=generation_config
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        # Clear cache and suggest solutions
                        if device == "cuda":
                            torch.cuda.empty_cache()
                        elif device == "mps":
                            torch.mps.empty_cache()
                        
                        error_msg = f"Out of memory on {device}. Try reducing max_tokens or batch size."
                        logger.error(error_msg)
                        raise HTTPException(status_code=507, detail=error_msg)
                    else:
                        raise
        
        # Decode the generated text
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Process the generated text to extract JSON
        processed_batch = extract_json_from_response(generated_text)
        
        # Count tokens generated
        tokens_generated = len(generated_ids)
        
        logger.info(f"Generated {tokens_generated} tokens for RLHF distortion")
        
        # Return ONLY the filled JSON batch
        return processed_batch
        
    except Exception as e:
        logger.error(f"Error during RLHF distortion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"RLHF distortion failed: {str(e)}")

@app.get("/temperature-formula")
async def show_temperature_formula():
    """Show the μ to temperature mapping formula and examples"""
    
    examples = []
    test_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    for miu in test_values:
        temp = calculate_temperature_from_miu(miu)
        examples.append({
            "miu": miu,
            "calculated_temperature": temp,
            "distortion_level": get_distortion_level_description(miu)
        })
    
    return {
        "formula_description": "Temperature = 0.3 + (μ^1.5 * 1.2) + sigmoid_boost + paraphrase_boost",
        "components": {
            "exponential_component": "0.3 + (μ^1.5 * 1.2) - Base growth",
            "sigmoid_boost": "0.3 * sigmoid(-10 * (μ - 0.5)) - Smooth transitions",
            "paraphrase_boost": "0.2 * ((μ - 0.7) / 0.3)^2 for μ >= 0.7 - High μ boost"
        },
        "examples": examples
    }

def get_distortion_level_description(miu: float) -> str:
    """Get description of distortion level for given μ value"""
    if miu == 0.0:
        return "No distortion"
    elif miu <= 0.2:
        return "Light lexical (minimal word substitutions)"
    elif miu <= 0.4:
        return "Heavy lexical (more synonyms, slight restructuring)"
    elif miu <= 0.6:
        return "Light mixed (lexical + structural changes)"
    elif miu <= 0.8:
        return "Heavy mixed (significant restructuring)"
    else:
        return "Full paraphrase (complete reconstruction)"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)