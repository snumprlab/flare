# MMP
This README provides instructions for generating subgoals with GPT-4 as part of our implementation of Multi-Modal Planner in FLARE.


## Embed language instructions
~~Run `embed-instructions.py` to embed language instructions with BERT encoder.~~


~~This will create `few_examples_from_song/train_few_instrucitons_emb.p` which contains language instruction embeddings.~~

We provide pre-generated `few_examples_from_song/train_few_instrucitons_emb.p`

## Embed language instructions
~~Run `embed-state.py` to embed agnet's egocentric view with CLIP encoder.~~

~~This will create `few_examples_from_song/train_few_clip_image_panoramic_emb.p` which contains agent's egocentrice view (panoramic) embeddings.~~

We provide pre-generated `few_examples_from_song/train_few_clip_image_panoramic_emb.p`

## Retrieve top-k in-context examples
Run `retriever.py` to retreive top-k (here. k=9) in-context examples for each tasks in valid and tests splits.

```
python retriever.py
```
This will create `few_examples_from_song/few-song-{sp}_retrieved_keys_clip_Img1_Txt1_panoramic.json` which contains retrieved in-context examples for each tasks.

## Generate plan with GPT-4
**Modify your openai API key** in `generate_plans.py`.
Then run `generate_plans.py` to generate plans with GPT-4
```
python generate_plans.py --dn dn
```

Finally, run `postprocess.py` to postprocess llm generated plans to ALFRED executable action sequences.
```
python postprocess.py --dn dn
```

This will create `.json` files in `planner_results/dn`.
You can use them by editting `read_test_dict` in `models/instructions_processed_LP/ALFRED_task_helper.py` to use your files.
## Hardware 
Tested on:
- **GPU** - RTX A6000
- **CPU** - Intel(R) Core(TM) i7-12700K CPU @ 3.60GHz
- **RAM** - 64GB
- **OS** - Ubuntu 20.04