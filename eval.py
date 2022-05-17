# %%
from torch.utils.data import DataLoader
import os
import torch
import string
import random
from models.data import parse_file, Collate_Fn_Manager
from models.util import set_seed
from transformers import AutoModel, AutoConfig, AutoTokenizer
from tqdm import tqdm
import argparse
import json

# %%
random_string = ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(10))
 
print(random_string)

parser = argparse.ArgumentParser()
parser.add_argument("--seed_model", type=int, default=123, help="random seed for model")

parser.add_argument("--batch_size_eval", type=int, default=4, help="eval batch size")

parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint path")
parser.add_argument("--model", type=str, default="base", help="model")
parser.add_argument("--candidate_downsampling", type=int, default=1000, help="number of candidate spans to use during training (-1 for no downsampling)")
parser.add_argument("--k_mentions", type=int, default=50, help="number of mention spans to perform relation extraction on")
parser.add_argument("--pooling", type=str, default="mean", help="mention pooling method (mean, max)")
parser.add_argument("--preprocessing", type=str, default=None, help="preprocessing (None, latex2text)")


args = parser.parse_args()
print(os.path.basename(args.checkpoint))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
set_seed(args.seed_model)

# -------------- data init --------------
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_cased")

files = os.listdir('data/raw_dev/')
parsed_files_dev = []
for file in files:
    if file.endswith(".json"):
        parsed_files_dev.extend(parse_file(f"data/raw_dev/{file}", tokenizer=tokenizer, preprocessing=args.preprocessing))


dev_loader = DataLoader(
    parsed_files_dev, 
    batch_size=args.batch_size_eval, 
    shuffle=True, 
    collate_fn=Collate_Fn_Manager(max_span_len=15).collate_fn)


files = os.listdir('data/raw_test/')
parsed_files_test = []
for file in files:
    if file.endswith(".json"):
        parsed_files_test.extend(parse_file(f"data/raw_test/{file}", tokenizer=tokenizer, preprocessing=args.preprocessing))


test_loader = DataLoader(
    parsed_files_test, 
    batch_size=args.batch_size_eval, 
    shuffle=True, 
    collate_fn=Collate_Fn_Manager(max_span_len=15).collate_fn)

# %%

# -------------- model loading --------------

# start by loading language model

lm_config = AutoConfig.from_pretrained(
    "allenai/scibert_scivocab_cased",
    num_labels=10,
)
lm_model = AutoModel.from_pretrained(
    "allenai/scibert_scivocab_cased",
    from_tf=False,
    config=lm_config,
)


if args.model == "base":
    from models.base_model import Encoder

model = Encoder(
    config=lm_config,
    model=lm_model, 
    cls_token_id=tokenizer.convert_tokens_to_ids(tokenizer.cls_token), 
    sep_token_id=tokenizer.convert_tokens_to_ids(tokenizer.sep_token),
    k_mentions=args.k_mentions,
    pooling=args.pooling,
)

model.to('cuda')
model.load_state_dict(torch.load(f"{args.checkpoint}"))

# %% --------------- TRAIN LOOP ------------------

avg_rloss = []
avg_mloss = []
avg_loss = []

# --------------- EVAL ON DEV ------------------
avg_loss = []
model.true_positives = 0
model.false_positives = 0
model.false_negatives = 0
model.true_negatives = 0
model.eval()
print("---- DEV EVAL -----")

with tqdm(dev_loader) as progress_bar:
    for b in progress_bar:
        input_ids, attention_masks, candidate_spans, relation_labels, offset_mapping, ids, sources = b

        if input_ids.size(1) > 4096:
            # skip an insanely long document...
            continue
        
        # pass data to model
        relation_loss, mention_loss, loss, output = model(input_ids.to('cuda'), attention_masks.to('cuda'), candidate_spans, relation_labels)

        # accumulate loss
        avg_loss.append(loss.item())
        del loss

        acc, p, r, f1 = 0, 0, 0, 0
        if model.true_positives != 0:
            p = model.true_positives/(model.true_positives+model.false_positives)
            r = model.true_positives/(model.true_positives+model.false_negatives)
            f1 = (2*p*r)/(p+r)
        acc = (model.true_positives + model.true_negatives)/(model.true_positives + model.true_negatives + model.false_negatives + model.false_positives)
        progress_bar.set_postfix({"L":f"{sum(avg_loss)/len(avg_loss):.2f}", "TP":f"{model.true_positives}", "A":f"{100*acc:.2f}", "P":f"{100*p:.2f}", "R":f"{100*r:.2f}", "F1":f"{100*f1:.2f}"})

# %%

print("---- TEST EVAL -----")
# --------------- EVAL ON TEST ------------------

model.eval()

outputs = {}


with tqdm(test_loader) as progress_bar:
    for b in progress_bar:

        input_ids, attention_masks, candidate_spans, relation_labels, offset_mapping, ids, sources = b

        # pass data to model
        _, _, _, output = model(input_ids.to('cuda'), attention_masks.to('cuda'), candidate_spans)

        # convert label indexing from tokens to chars and save in correct format
        for preds, offsets, doc_id, source in zip(output, offset_mapping, ids, sources):
            detected_entities = []
            detected_relations = {}
            for prediction in preds:
                (h_span, h_type), (t_span, t_type), r_type = prediction
                h_span = [offsets[tk] for tk in h_span]
                h_ent = {
                    "label": h_type,
                    "start": h_span[0][0],
                    "end": h_span[1][1],
                }
                t_span = [offsets[tk] for tk in t_span]
                t_ent = {
                    "label": t_type,
                    "start": t_span[0][0],
                    "end": t_span[1][1],
                }
                
                if h_ent in detected_entities:
                    h_id = detected_entities.index(h_ent)
                else:
                    h_id = len(detected_entities)
                    detected_entities.append(h_ent)    

                if t_ent in detected_entities:
                    t_id = detected_entities.index(t_ent)
                else:
                    t_id = len(detected_entities)
                    detected_entities.append(t_ent)

                detected_relations[f"R{len(detected_relations)+1}"] = {
                    "rid": f"R{len(detected_relations)+1}",
                    "label": r_type,
                    "arg0": f"T{h_id+1}",
                    "arg1": f"T{t_id+1}",
                }
            
            ent_dict = {}
            
            corefs = {f"T{x+1}": [] for x in range(len(detected_entities))}
            for key in detected_relations.keys():
                r = detected_relations[key]
                if r["label"] == "Corefer-Symbol":
                    corefs[r["arg0"]].append(r["arg1"])
                    corefs[r["arg1"]].append(r["arg0"])
            
            crs = corefs.copy()
            
            for cr in corefs.keys():
                other_mentions = set(crs[cr])
                for ment in other_mentions:
                    corefs[cr].extend(crs[ment])
                corefs[cr] = list(set(corefs[cr]))
            
                            
            for i, ent in enumerate(detected_entities):
                
                label = ent["label"]
                if label == "PRIMARY":
                    is_ordered = False
                    direct_connections = []
                    for key in detected_relations.keys():
                        r = detected_relations[key]
                        if r["arg1"] != f"T{i+1}" and r["arg0"] != f"T{i+1}":
                            continue
                        if r["label"] == "Direct" and r["arg1"] == f"T{i+1}":
                            direct_connections.append(r["arg0"])
                    
                    if len(direct_connections) > 1:
                        is_ordered = True
                        for cr in corefs.keys():
                            if all(x in corefs[cr] for x in direct_connections):
                                is_ordered = False
                                break

                        if is_ordered:
                            label = "ORDERED"
                    
                
                ent_dict[f"T{i+1}"] = {
                    "eid": f"T{i+1}",
                    "label": label,
                    "start": ent["start"],
                    "end": ent["end"],
                    "text": source["text"][ent["start"]:ent["end"]],
                }

            doc = source
            doc["entity"] = ent_dict
            doc["relation"] = detected_relations
            file_id = doc["file"]
            del doc["file"]
            if file_id not in outputs.keys():
                outputs[file_id] = {}
            outputs[file_id][doc_id] = doc

        del output

os.makedirs(f'outputs/{os.path.basename(args.checkpoint).split(".json")[0]}_typefix_{random_string}_test/')
tozip = []
for filename in outputs.keys():
    tozip.append(f'outputs/{os.path.basename(args.checkpoint).split(".json")[0]}_typefix_{random_string}_test/{filename}')
    with open(f'outputs/{os.path.basename(args.checkpoint).split(".json")[0]}_typefix_{random_string}_test/{filename}', 'w') as f:
        json.dump(outputs[filename], f, indent=4)

import zipfile
ZipFile = zipfile.ZipFile(f'outputs/{os.path.basename(args.checkpoint).split(".json")[0]}_typefix_{random_string}_test/{os.path.basename(args.checkpoint).split(".json")[0]}_typefix_{random_string}_test.zip', "w")
for a in tozip:
    ZipFile.write(a, os.path.basename(a), compress_type=zipfile.ZIP_DEFLATED)
ZipFile.close()