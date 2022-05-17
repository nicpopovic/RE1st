from models.util import process_long_input
import torch
import torch.nn as nn
from transformers import BertConfig, RobertaConfig, DistilBertConfig, XLMRobertaConfig
from models.losses import ATLoss

class Encoder(nn.Module):
    def __init__(self, config, model, cls_token_id=0, sep_token_id=0, negative_prob=1.0, k_mentions=50, pooling="mean"):
        super().__init__()
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size

        self.relation_embeddings = nn.Parameter(torch.zeros((4,1536)))
        torch.nn.init.uniform_(self.relation_embeddings, a=-1.0, b=1.0)            
        self.nota_embeddings = nn.Parameter(torch.zeros((20,1536)))
        torch.nn.init.uniform_(self.nota_embeddings, a=-1.0, b=1.0)

        # unused parameters (left in to prevent checkpoints from breaking)
        self.relation_classifier = None
        self.block_size = None

        self.emb_size = 768

        self.entity_anchor = nn.Parameter(torch.zeros((3, 768)))
        torch.nn.init.uniform_(self.entity_anchor, a=-1.0, b=1.0)

        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id

        self.false_positives = 0
        self.true_positives = 0
        self.false_negatives = 0
        self.true_negatives = 0

        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        self.negative_prob = negative_prob
        self.k_mentions = k_mentions
        self.pooling = pooling


    def encode(self, input_ids, attention_mask):
        config = self.config
        if type(config) == BertConfig or type(config) == DistilBertConfig:
            start_tokens = [self.cls_token_id]
            end_tokens = [self.sep_token_id]
        elif type(config) == RobertaConfig or type(config) == XLMRobertaConfig:
            start_tokens = [self.cls_token_id]
            end_tokens = [self.sep_token_id, self.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention

    def forward(self,
                input_ids, 
                attention_masks,
                candidate_spans,
                relation_labels=None):
        
        sequence_output, _ = self.encode(input_ids, attention_masks)

        relation_index = ['Direct', 'Count', 'Corefer-Symbol', 'Corefer-Description']
        entity_type_index = ['PRIMARY', 'SYMBOL', 'ORDERED']
        relation_loss = torch.zeros((1)).to(sequence_output)
        mention_loss = torch.zeros((1)).to(sequence_output)
        output = []

        for batch_i in range(sequence_output.size(0)):
            # ---------- Candidate span embeddings ------------
            mention_candidates = []
            for span in candidate_spans[batch_i]:
                if self.pooling == "mean":
                    mention_embedding = torch.mean(sequence_output[batch_i, span[0]:span[1]+1,:], 0)
                elif self.pooling == "max":
                    mention_embedding = torch.max(sequence_output[batch_i, span[0]:span[1]+1,:], 0)[0]                
                elif self.pooling == "logsumexp":
                    mention_embedding = torch.logsumexp(sequence_output[batch_i, span[0]:span[1]+1,:], 0)
                #print(sequence_output[batch_i, span[0]:span[1],:].size())
                mention_candidates.append(mention_embedding)
                
            embs = torch.stack(mention_candidates)

            # ---------- Soft mention detection (scores) ------------
            span_scores = embs.unsqueeze(1) * self.entity_anchor.unsqueeze(0)
            span_scores = torch.sum(span_scores, dim=-1)
            span_scores_max, class_for_span = torch.max(span_scores, dim=-1)
            scores_for_max, max_spans = torch.topk(span_scores_max.view(-1), min(self.k_mentions, embs.size(0)), dim=0)
            class_for_max_span = class_for_span[max_spans]

            # true entity spans
            relevant_spans = []
            if relation_labels != None and self.training:
                # ---------- Mention Loss and adding true spans during training ------------
                type_for_spans = []
                for label in relation_labels[batch_i]:
                    try:
                        h = candidate_spans[batch_i].index(label['h'])
                        type_h = entity_type_index.index(label['type_h'])
                        relevant_spans.append(h)
                        type_for_spans.append(type_h)
                    except:
                        pass
                    try:
                        t = candidate_spans[batch_i].index(label['t'])
                        type_t = entity_type_index.index(label['type_t'])
                        relevant_spans.append(t)
                        type_for_spans.append(type_t)
                    except:
                        pass

                relevant_spans = list(set(relevant_spans))
                negative_examples = [(x,i) for x, i in zip(max_spans.tolist(), class_for_max_span.tolist()) if x not in relevant_spans]
                if len(relevant_spans) != 0 and len(negative_examples) != 0:
                    anchors, positives, negatives = [], [], []
                    for span, span_type in zip(relevant_spans, type_for_spans):
                        for neg, i in negative_examples:
                            positives.append(embs[span,:])
                            negatives.append(embs[neg,:])
                            # change anchor choice to label!
                            anchors.append(self.entity_anchor[span_type,:])

                    mention_loss += self.triplet_loss(torch.stack(anchors), torch.stack(positives), torch.stack(negatives))

            # ---------- Soft mention detection (select spans) ------------
            selected_spans = list(set(max_spans.tolist() + relevant_spans))
            class_for_max_span = class_for_span[selected_spans]
            selected_candidates = [candidate_spans[batch_i][x] for x in selected_spans]
            embs = embs[selected_spans,:]

            # ---------- Relation Classification ------------
            # create matrix with all candidate pairs
            h_entities = torch.unsqueeze(embs, 1)
            h_entities = h_entities.repeat(1, h_entities.size()[0], 1)
            t_entities = h_entities.transpose(0,1)

            # save shape
            target_shape = h_entities.shape

            # flatten
            h_entities, t_entities = h_entities.flatten(0,1), t_entities.flatten(0,1)

            # relation classification
            candidates = torch.cat((h_entities, t_entities), dim=-1)
            scores = candidates.unsqueeze(1) * self.relation_embeddings.unsqueeze(0)
            scores = torch.sum(scores, dim=-1)

            nota_scores = candidates.unsqueeze(1) * self.nota_embeddings.unsqueeze(0)
            nota_scores = torch.sum(nota_scores, dim=-1)
            nota_scores = nota_scores.max(dim=-1,keepdim=True)[0]
            scores = torch.cat((nota_scores, scores), dim=-1)

            scores = scores.view(target_shape[0], target_shape[0], -1)

            # ---------- Output processing ------------
            # generate outputs
            predictions = torch.argmax(scores, dim=-1, keepdim=False)
            pred_rels=(predictions != 0).nonzero(as_tuple=True)
            predictions_triples = []
            predictions_triples = []
            types_for_triples = []
            scores_for_triples = []
            for x,y in zip(pred_rels[0].tolist(), pred_rels[1].tolist()):
                if x == y:
                    continue
                h = selected_candidates[x]
                t = selected_candidates[y]
                type_h = entity_type_index[class_for_max_span[x].item()]
                type_t = entity_type_index[class_for_max_span[y].item()]
                r = predictions[x,y].item()
                predictions_triples.append([h,t,r])
                types_for_triples.append([type_h, type_t])
                scores_for_triples.append(scores[x,y,r].item())

            if not self.training:
                # ---------- Test/Inference output ------------
                # for relations where both h ant t spans are overlapping, select the one with the highest scores for h and t
                selected_relations = []

                # https://stackoverflow.com/a/45441404
                overlapping = [ [x,y] for x in selected_candidates for y in selected_candidates if x is not y and x[1]>y[0] and x[0]<y[0] or x[0]==y[0] and x[1]==y[1] and x is not y]
                alternative_spans = {tuple(key): [key] for key in selected_candidates}
                for x in overlapping:
                    if x[1] not in alternative_spans[tuple(x[0])]:
                        alternative_spans[tuple(x[0])].append(x[1])                
                    if x[0] not in alternative_spans[tuple(x[1])]:
                        alternative_spans[tuple(x[1])].append(x[0])
                for cand1 in selected_candidates:
                    for cand2 in selected_candidates:
                        if cand1[0] == cand2[0] or cand1[1] == cand2[1]:
                            if cand2 not in alternative_spans[tuple(cand1)]:
                                alternative_spans[tuple(cand1)].append(cand2)                
                            if cand1 not in alternative_spans[tuple(cand2)]:
                                alternative_spans[tuple(cand2)].append(cand1)
                # print(overlapping)
                
                has_been_replaced = {}
                for i, pt1 in enumerate(predictions_triples):
                    h1, t1, r1 = pt1
                    #score1 = scores_for_max[selected_candidates.index(h1)] + scores_for_max[selected_candidates.index(t1)]
                    score1 = scores_for_triples[i]
                    highest = [score1, i]
                    for j, pt2 in enumerate(predictions_triples):
                        h2, t2, r2 = pt2
                        if r1 != r2 or h2 not in alternative_spans[tuple(h1)] or t2 not in alternative_spans[tuple(t1)] or pt1 == pt2:
                            #print(pt1, pt2, r1 != r2, h2 not in alternative_spans[tuple(h1)], t2 not in alternative_spans[tuple(t1)], pt1 == pt2)
                            continue
                        # score2 = scores_for_max[selected_candidates.index(h2)] + scores_for_max[selected_candidates.index(t2)]
                        score2 = scores_for_triples[j]
                        #print(pt1, score1.item(), pt2, score2.item())
                        if score2 > highest[0]:
                            highest = [score2, j]
                    
                    if highest[1] in has_been_replaced.keys():
                        # winner already has a replacement
                        replacement, rep_score = has_been_replaced[highest[1]]

                        if rep_score > highest[0]:
                            # score of replacement is higher
                            highest = [rep_score, replacement]
                        else:
                            # this score is higher, need to fix prior replacement
                            has_been_replaced[highest[1]] = [highest[1], highest[0]]
                            selected_relations.pop(selected_relations.index(replacement))

                    has_been_replaced[i] = [highest[1], highest[0]]
                    selected_relations.append(highest[1])
                    selected_relations = list(set(selected_relations))
                    #print(highest[0].item())

                predictions_triples = [x for i, x in enumerate(predictions_triples) if i in selected_relations]
                types_for_triples = []

                for h,t,r in predictions_triples:
                    if r == 1:
                        type_h = "PRIMARY"
                        type_t = "SYMBOL"
                    elif r == 2:
                        type_h = "PRIMARY"
                        type_t = "SYMBOL"
                    elif r == 3:
                        type_h = "SYMBOL"
                        type_t = "SYMBOL"
                    elif r == 4:
                        type_h = "PRIMARY"
                        type_t = "PRIMARY"

                    types_for_triples.append([type_h, type_t])
                

            output.append([[(a, ta), (b, tb), relation_index[c-1]] for (a,b,c),(ta, tb) in zip(predictions_triples, types_for_triples)])

            if relation_labels != None:
                # ---------- Calculating classification stats + loss ------------
                # labels
                labels = torch.zeros((target_shape[0], target_shape[0], 5)).to(sequence_output)
                labels[:,:,0] = 1.0
                mask = torch.ones((scores.size(0), scores.size(1)), dtype=torch.float, device="cuda")
                mask[:,:] = self.negative_prob

                tp, fn = 0, 0
                for label in relation_labels[batch_i]:
                    try:
                        h = selected_candidates.index(label['h'])
                        t = selected_candidates.index(label['t'])
                        labels[h,t,0] = 0.0
                        labels[h,t,relation_index.index(label['r'])+1] = 1.0
                        mask[h,t] = 1.0

                        #print([label['h'],label['t'],relation_index.index(label['r'])+1], predictions_triples)
                        if [label['h'],label['t'],relation_index.index(label['r'])+1] in predictions_triples:
                            tp += 1
                        else:
                            fn += 1

                        #print(labels[h,t], scores[h,t])
                    except:
                        #print("ignored:", label)
                        fn += 1
                        pass
                
                self.true_positives += tp
                self.false_negatives += fn
                self.false_positives += len(predictions_triples) - tp
                self.true_negatives += target_shape[0] * (target_shape[0]-1) - len(predictions_triples) - fn
                    
                for k in range(scores.size(0)):
                    mask[k,k] = 0
                
                mask = torch.bernoulli(mask).bool().unsqueeze(-1)

                relation_loss += ATLoss()(torch.masked_select(scores, mask).view(-1, scores.size(-1)), torch.masked_select(labels.float(), mask).view(-1, scores.size(-1)))



        return relation_loss, mention_loss, relation_loss+mention_loss, output
