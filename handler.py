"""Example handler file."""

import runpod
import torch


# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.
language = 'de'
model_id = 'v3_de'
device = torch.device('cpu')

model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
  model='silero_tts',
  language=language,
  speaker=model_id)
model.to(device)  # gpu or cpu

def handler(job):
    """Handler function that will be used to process jobs."""
    example_text = job["input"]
    sample_rate = 48000
    speaker = 'karlsson'
    put_accent=True
    put_yo=True
    example_text = '''Du meine Seele, du mein Herz,
    Du meine Wonn’, o du mein Schmerz,
    Du meine Welt, in der ich lebe,
    Mein Himmel du, darein ich schwebe,
    O du mein Grab, in das hinab
    Ich ewig meinen Kummer gab!
    Du bist die Ruh, du bist der Frieden,
    Du bist vom Himmel mir beschieden.
    Dass du mich liebst, macht mich mir wert,
    Dein Blick hat mich vor mir verklärt,
    Du hebst mich liebend über mich,
    Mein guter Geist, mein bess’res Ich!'''

    audio = model.apply_tts(text=example_text,
                            speaker=speaker,
                            sample_rate=sample_rate,
                            put_accent=put_accent,
                            put_yo=put_yo)

    return audio


runpod.serverless.start({"handler": handler})
