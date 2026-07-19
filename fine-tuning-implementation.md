# Whisper tiny fine-tuning implementation

## Objective and success criteria

Fine-tune `openai/whisper-tiny` for English ASR on the `en-US` configuration of `PolyAI/minds14`. The assignment requires the first **450** examples for training, the remaining **113** examples for evaluation, and `num_proc=1` on every pre-processing `.map()` call. Report fractional WER values (for example `0.37`, not `37.0`); the required normalised `wer` is below `0.37`.

The notebook currently contains the assignment only, so add the following cells to `src/fine-tuning/fine-tuning.ipynb` in this order.

## 1. Install the missing evaluation dependency

`evaluate` is required for the Hub WER metric but is not currently in `pyproject.toml`.

```powershell
uv add evaluate
uv run hf auth login  # only if this machine is not already logged in
```

## 2. Imports and reproducibility

```python
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Union

import evaluate
import torch
from datasets import Audio, load_dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    set_seed,
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

set_seed(42)
MODEL_ID = "openai/whisper-tiny"
LANGUAGE = "english"
TASK = "transcribe"
```

## 3. Load the exact required split

The dataset configuration has one `train` split. Select it first, then slice by position; do not shuffle before this split. The data card currently lists 563 `en-US` rows, yielding 450 training and 113 evaluation rows.

```python
dataset = load_dataset("PolyAI/minds14", "en-US", split="train")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

train_dataset = dataset.select(range(450))
eval_dataset = dataset.select(range(450, len(dataset)))

assert len(train_dataset) == 450
assert len(eval_dataset) == 113
```

Use `transcription`, not `english_transcription`: this configuration contains English speech, but `transcription` is the ASR target.

## 4. Load the processor and model

`WhisperProcessor` pairs the 16 kHz log-Mel feature extractor and Whisper tokenizer. Setting `language` and `task` while tokenising writes the proper decoder-prefix tokens into labels.

```python
processor = WhisperProcessor.from_pretrained(
    MODEL_ID, language=LANGUAGE, task=TASK
)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
```

## 5. Preprocess audio and labels

This is intentionally one process, as mandated by the assignment. Removing the original columns keeps only model inputs and labels in memory.

```python
def prepare_example(batch):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features
    batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
    return batch

train_dataset = train_dataset.map(
    prepare_example,
    num_proc=1,
    remove_columns=train_dataset.column_names,
    desc="Extracting train features",
)
eval_dataset = eval_dataset.map(
    prepare_example,
    num_proc=1,
    remove_columns=eval_dataset.column_names,
    desc="Extracting evaluation features",
)
```

## 6. Pad batches correctly

Audio features and text labels have different padding rules. Label padding must become `-100`, so PyTorch cross-entropy ignores it. The initial decoder token is removed only when every label has it, because the model adds it during training.

```python
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_features": feature["input_features"][0]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
```

## 7. Compute the two required WER metrics

`wer_ortho` is calculated on decoded text unchanged. `wer` normalises casing, punctuation, and number formatting before scoring. Both are deliberately returned as fractions, with **no `* 100`**. Empty normalised references are filtered to avoid an undefined WER denominator.

```python
wer_metric = evaluate.load("wer")
normalizer = BasicTextNormalizer()

def compute_metrics(pred):
    pred_ids = pred.predictions
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]

    label_ids = pred.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    predictions = processor.batch_decode(pred_ids, skip_special_tokens=True)
    references = processor.batch_decode(label_ids, skip_special_tokens=True)

    wer_ortho = wer_metric.compute(predictions=predictions, references=references)

    normalized_pairs = [
        (normalizer(prediction), normalizer(reference))
        for prediction, reference in zip(predictions, references)
        if normalizer(reference).strip()
    ]
    normalized_predictions, normalized_references = map(list, zip(*normalized_pairs))
    wer = wer_metric.compute(
        predictions=normalized_predictions, references=normalized_references
    )
    return {"wer": wer, "wer_ortho": wer_ortho}
```

## 8. Configure training and generation

Gradient checkpointing reduces activation memory by recomputing them in the backward pass. It conflicts with the decoder KV cache, so cache is disabled in training and re-enabled only for generation. The language is **English** for this assignment (not the Sinhala value in the course's worked example).

The enough ram RTX Nvidia Gpu should fit this tiny checkpoint at batch size 16 in mixed precision. If a driver/library combination produces an out-of-memory error, use 8 and set `gradient_accumulation_steps=2` to preserve the effective batch size.

```python
model.config.use_cache = False
model.generate = partial(
    model.generate,
    language=LANGUAGE,
    task=TASK,
    use_cache=True,
)

training_args = Seq2SeqTrainingArguments(
    output_dir="outputs/whisper-tiny-minds14-en-us",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=50,
    max_steps=500,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    logging_steps=25,
    predict_with_generate=True,
    generation_max_length=225,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    report_to=["tensorboard"],
    push_to_hub=True,
    hub_model_id="whisper-tiny-minds14-en-us",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor,
)
```

`max_steps=500` is a sensible first run (about 18 passes over the 450 examples at batch 16). If it misses the target, continue from the saved best checkpoint with `max_steps=1_000`; do not change the mandated split.

## 9. Train, validate, and publish

```python
trainer.train()
metrics = trainer.evaluate()
print(metrics)  # inspect eval_wer and eval_wer_ortho as fractions

kwargs = {
    "dataset_tags": "PolyAI/minds14",
    "finetuned_from": "openai/whisper-tiny",
    "tasks": "automatic-speech-recognition",
}
trainer.push_to_hub(**kwargs)
```

Record `eval_wer`, `eval_wer_ortho`, the random seed, package versions, and the Hub revision with the submission.

## RTX Nvidia Gpu time estimate

The RTX Nvidia Gpu has enough ram GDDR7, enough for the specified mixed-precision tiny run. For the 500-step configuration above, expect **about 8-18 minutes** of GPU training, plus **1-3 minutes** of generated evaluation at each checkpoint. The first run can additionally spend **2-10 minutes** downloading, decoding, and preprocessing audio. A practical wall-clock estimate is **18-40 minutes** from a cold cache, or **13-30 minutes** when the model and dataset are cached.

This is an estimate, not a benchmark: audio lengths, CPU decoding throughput, CUDA/PyTorch support for Blackwell, storage speed, and evaluation generation dominate variance on such a small run. Measure a 100-step run, then multiply its training portion by five for the most reliable estimate on the actual machine.

## Exact code and documentation references

1. [Assignment requirements and WER target](https://huggingface.co/learn/audio-course/chapter5/hands_on)
2. [Hugging Face's Whisper fine-tuning walkthrough: processor, collator, metrics, cache, and trainer configuration](https://huggingface.co/learn/audio-course/chapter5/fine-tuning)
3. [WER definitions and `BasicTextNormalizer` usage](https://huggingface.co/learn/audio-course/chapter5/evaluation)
4. [Current Transformers Whisper API and generation behaviour](https://huggingface.co/docs/transformers/en/model_doc/whisper)
5. [Current Transformers sequence-to-sequence speech-recognition training example](https://github.com/huggingface/transformers/blob/main/examples/pytorch/speech-recognition/run_speech_recognition_seq2seq.py)
6. [PolyAI/minds14 dataset card and `en-US` subset size](https://huggingface.co/datasets/PolyAI/minds14)
7. [NVIDIA RTX Nvidia Gpu specifications (enough ram GDDR7)](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5070-family/)
