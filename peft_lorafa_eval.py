import torch
import json
import ollama
import numpy as np
import os
from tqdm import tqdm
from typing import Dict, Any, Tuple
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
