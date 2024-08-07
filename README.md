# Overview

| Developed by | Guardrails AI |
| Date of development | Feb 15, 2024 |
| Validator type | Format |
| Blog |  |
| License | Apache 2 |
| Input/Output | Output |

## Description

### Intended Use
This validator checks if the input value contains sensitive topics. The default behavior first runs a Zero-Shot model, and then falls back to ask OpenAI's `gpt-3.5-turbo` if the Zero-Shot model is not confident in the topic classification (score < 0.5). In our experiments this LLM fallback increases accuracy by 15% but also increases latency (more than doubles the latency in the worst case). Both the Zero-Shot classification and the GPT classification may be toggled.

### Requirements

* Dependencies: 
    - guardrails-ai>=0.4.0

## Installation

```bash
$ guardrails hub install hub://guardrails/sensitive_topics
```

## Usage Examples

### Validating string output via Python

In this example, we apply the validator to a string output generated by an LLM.

```python
# Import Guard and Validator
from guardrails import Guard
from guardrails.hub import SensitiveTopic

# Setup Guard
guard = Guard().use(
    SensitiveTopic,
    sensitive_topics=["politics"],
    disable_classifier=False,
    disable_llm=False,
    on_fail="exception",
)

# Test passing response
guard.validate(
    "San Francisco is known for its cool summers, fog, steep rolling hills, eclectic mix of architecture, and landmarks, including the Golden Gate Bridge, cable cars, the former Alcatraz Federal Penitentiary, Fisherman's Wharf, and its Chinatown district.",
)

try:
    # Test failing response
    guard.validate(
        """
        Donald Trump is one of the most controversial presidents in the history of the United States.
        He has been impeached twice, and is running for re-election in 2024.
        """
    )
except Exception as e:
    print(e)
```
Output:
```console
Validation failed for field with errors: Sensitive topics detected: politics
```
# API Reference

**`__init__(self, sensitive_topics=["holiday or anniversary of the trauma or loss", "certain sounds, sights, smells, or tastes related to the trauma", "loud voices or yelling", "loud noises", "arguments", "being ridiculed or judged", "being alone", "getting rejected", "being ignored", "breakup of a relationship", "violence in the news", "sexual harassment or unwanted touching", "physical illness or injury",], device=-1, model="facebook/bart-large-mnli", llm_callable="gpt-3.5-turbo", disable_classifier=False, disable_llm=False, model_threshold=0.5, on_fail="noop")`**
<ul>
Initializes a new instance of the Validator class.

**Parameters**
- **`sensitive_topics`** *(List[str], Optional)*: topics that the text should not contain.
- **`device`** *(int, Optional)*: Device ordinal for CPU/GPU supports for Zero-Shot classifier. Setting this to -1 will leverage CPU, a positive will run the Zero-Shot model on the associated CUDA device id.
- **`model`** *(str, Optional)*: The Zero-Shot model that will be used to classify the topic. See a list of all models here: https://huggingface.co/models?pipeline_tag=zero-shot-classification
- **`llm_callable`** *(Optional[Union[str, Callable]])*: Either the name of the OpenAI model, or a callable that takes a prompt and returns a response.
- **`disable_classifier`** *(bool, Optional)*: controls whether to use the Zero-Shot model. At least one of disable_classifier and disable_llm must be False.
- **`classifier_api_endpoint`** *(str, Optional)*: An API endpoint to receive post requests that will be used when provided. If not provided, a local model will be initialized.
- **`disable_llm`** *(bool, Optional)*: controls whether to use the LLM fallback. At least one of disable_classifier and disable_llm must be False.
- **`zero_shot_threshold`** *(float, Optional)*: The threshold used to determine whether to accept a topic from the Zero-Shot model. Must be a number between 0 and 1.
- **`llm_threshold`** *(float, Optional)*: The threshold used to determine if a topic exists based on the provided llm api. Must be between 0 and 5.
- **`on_fail`** *(str, Callable)*: The policy to enact when a validator fails. If `str`, must be one of `reask`, `fix`, `filter`, `refrain`, `noop`, `exception` or `fix_reask`. Otherwise, must be a function that is called when the validator fails.
</ul>
<br/>

**`__call__(self, value, metadata={}) -> ValidationResult`**
<ul>
Validates the given `value` using the rules defined in this validator, relying on the `metadata` provided to customize the validation process. This method is automatically invoked by `guard.parse(...)`, ensuring the validation logic is applied to the input data.

Note:

1. This method should not be called directly by the user. Instead, invoke `guard.parse(...)` where this method will be called internally for each associated Validator.
2. When invoking `guard.parse(...)`, ensure to pass the appropriate `metadata` dictionary that includes keys and values required by this validator. If `guard` is associated with multiple validators, combine all necessary metadata into a single dictionary.

**Parameters**
- **`value`** *(Any)*: The input value to validate.
- **`metadata`** *(dict)*: A dictionary containing metadata required for validation. No additional metadata keys are needed for this validator.

</ul>
