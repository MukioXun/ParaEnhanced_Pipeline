import json
from abc import ABC, abstractmethod
from typing import List, Dict

class DimensionGenerator(ABC):
    def __init__(self, topic: str):
        self.topic = topic
        self.dimension_name = self.__class__.__name__.replace("Generator", "").upper()

    @property
    @abstractmethod
    def constraints(self) -> List[str]: pass

    @property
    @abstractmethod
    def critic_criteria(self) -> str: pass

    @property
    @abstractmethod
    def good_examples(self) -> List[Dict]: pass

    def build_prompt(self, count: int) -> str:
        constraint_str = "\n".join([f"• {c}" for c in self.constraints])
        examples = "\n".join([f"  - Text: \"{ex['text']}\" | Logic: {ex.get('logic')}" for ex in self.good_examples])
        
        return f"""
### TASK: GENERATE {count} ADVERSARIAL ITEMS FOR {self.dimension_name}
Topic: {self.topic}

### CORE REQUIREMENT:
The text must be AMBIGUOUS. The intent must be impossible to define without hearing the voice.

### CONSTRAINTS:
{constraint_str}
• Output MUST be a JSON array of objects.
• Fields: "text", "branch_1_logic", "branch_2_logic"

### GOOD EXAMPLES:
{examples}

Return only valid JSON.
"""

    def build_critic_prompt(self, text: str) -> str:
        return f"""
### AUDIT TASK: {self.dimension_name} NEUTRALITY
Evaluate if the following text is too explicit or truly ambiguous: "{text}"

### CRITERIA:
{self.critic_criteria}

### OUTPUT FORMAT (JSON):
{{
    "score": 0-10,
    "leakage_point": "string",
    "suggestion": "string"
}}
"""

# --- 具体实现 ---

class SarcasmGenerator(DimensionGenerator):
    @property
    def constraints(self): return ["No 'yeah right', no exclamation marks.", "Surface must be 100% positive."]
    @property
    def critic_criteria(self): return "Is there any word that hints at irony (e.g. 'typical', 'perfectly')? On paper, it must look like a sincere compliment."
    @property
    def good_examples(self): return [{"text": "You've really outdone yourself with this report.", "logic": "Sincere praise vs. Sarcasm for a disaster."}]

class EmotionGenerator(DimensionGenerator):
    def __init__(self, topic, emotion_pair):
        super().__init__(topic)
        self.e1, self.e2 = emotion_pair

    @property
    def constraints(self):
        return [
            f"The text must be 100% plausible for BOTH {self.e1} and {self.e2}.",
            "Avoid emotion-specific adjectives (e.g., 'terrified', 'joyful', 'unfortunate').",
            "Focus on a factual event that can be interpreted in two ways.",
            "The sentence must be in a natural, oral conversational style."
        ]

    @property
    def critic_criteria(self):
        return f"""
        1. Semantic Ambiguity: Does the text contain words that lean too much towards {self.e1} or {self.e2}?
        2. Scenario Check: If the context is {self.e1}, does this sentence make sense? If it's {self.e2}, does it still make sense?
        3. Arousal Level: Is the intensity of the wording neutral enough to support both {self.e1} and {self.e2}?
        Score 0 if the emotion is obvious; score 10 if it's perfectly ambiguous.
        """

    @property
    def good_examples(self):
        # 根据不同的对提供更有针对性的例子（可选，这里提供通用逻辑）
        return [
            {"text": "I just received the final results of my medical checkup.", 
             "logic": f"Could be {self.e1} (All clear) or {self.e2} (Bad news)."},
            {"text": "My parents are coming over to stay for the entire weekend.", 
             "logic": f"Could be {self.e1} (Looking forward to it) or {self.e2} (Feeling pressured/invaded)."}
        ]
        
class AgeGenerator(DimensionGenerator):
    @property
    def constraints(self): return ["No 'mom', 'dad', 'kid'.", "Task must require adult permission but be worded neutrally."]
    @property
    def critic_criteria(self): return "Does the vocabulary reveal age? Is the request too complex for a child or too simple for an adult?"
    @property
    def good_examples(self): return [{"text": "I need to confirm the billing details for the account update.", "logic": "Adult (routine) vs. Child (trying to bypass gate)."}]

class GenderGenerator(DimensionGenerator):
    @property
    def constraints(self): return ["Avoid gendered clothing or stereotypes.", "The AI response should ideally differ based on gender."]
    @property
    def critic_criteria(self): return "Does the context (e.g. 'shaving', 'makeup') leak gender?"
    @property
    def good_examples(self): return [{"text": "What's the most important thing to consider for a solo trip to this region?", "logic": "Male (general safety) vs. Female (specific harassment/safety concerns)."}]

# --- Factory ---
class TaskDispatcher:
    def get_generator(self, dimension: str, topic: str, **kwargs):
        mapping = {"sarcasm": SarcasmGenerator, "age": AgeGenerator, "gender": GenderGenerator, "emotion": EmotionGenerator}
        cls = mapping.get(dimension)
        if dimension == "emotion": return cls(topic, kwargs.get("emotion_pair", ("Happy", "Sad")))
        return cls(topic)