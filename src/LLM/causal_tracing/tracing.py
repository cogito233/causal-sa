import sys
import pandas as pd
import os, re, json
import torch, numpy
from collections import defaultdict
from util import nethook
from util.globals import DATA_DIR
from experiments.causal_trace import (
    ModelAndTokenizer,
    guess_subject,
    plot_trace_heatmap,
)
from transformers import  GPT2Tokenizer, GPT2LMHeadModel
from transformers import LlamaForCausalLM, LlamaTokenizer
from experiments.causal_trace import (
    make_inputs,
    decode_tokens,
    predict_token,
    predict_from_input,
    collect_embedding_std,
)
from dsets import KnownsDataset
import pickle
torch.set_grad_enabled(False)

c1_0="""As a customer writing a review, I initially composed the following feedback: "{}"
After carefully considering the facts, I selected a star rating from the options of "1", "2", "3", "4", or "5". My final rating was:"""
c1_1="""As a customer sharing my experience, I crafted the following review: "{}"
Taking into account the details of my experience, I chose a star rating from the available options of "1", "2", "3", "4", or "5". My ultimate rating is:"""
c1_2="""As a client providing my opinion, I penned down the subsequent evaluation: "{}"
Upon thorough reflection of my encounter, I picked a star rating among the choices of "1", "2", "3", "4", or "5". My conclusive rating stands at:"""
c1_3="""As a patron expressing my thoughts, I drafted the ensuing commentary: "{}"
After meticulously assessing my experience, I opted for a star rating from the range of "1", "2", "3", "4", or "5". My definitive rating turned out to be:"""
c1_4="""As a consumer conveying my perspective, I authored the following assessment: "{}"
By carefully weighing the aspects of my interaction, I determined a star rating from the possibilities of "1", "2", "3", "4", or "5". My final verdict on the rating is:"""

c2_0="""As a customer writing a review, I initially selected a star rating from the options "1", "2", "3", "4", and "5", and then provided the following explanations in my review: "{}"
The review clarifies why I gave a rating of"""
c2_1="""As a customer sharing my experience, I first chose a star rating from the available choices of "1", "2", "3", "4", and "5", and subsequently elaborated on my decision with the following statement: "{}"
The review elucidates the reasoning behind my assigned rating of"""
c2_2="""As a client providing my opinion, I initially picked a star rating from the range of "1" to "5", and then proceeded to justify my selection with the following commentary: "{}"
The review sheds light on the rationale for my given rating of"""
c2_3="""As a patron expressing my thoughts, I started by selecting a star rating from the scale of "1" to "5", and then offered an explanation for my choice in the following review text: "{}"
The review expounds on the basis for my designated rating of"""
c2_4="""As a consumer conveying my perspective, I began by opting for a star rating within the "1" to "5" spectrum, and then detailed my reasoning in the subsequent review passage: "{}"
The review delineates the grounds for my conferred rating of"""

c0_0="""You are an experienced and responsible data annotator for natural language processing (NLP) tasks. In the following, you will annotate some data for sentiment classification. Specifically, given the task description and the review text, you need to annotate the sentiment in terms of "1" (most negative), "2", "3", "4", and "5" (most positive).
Review Text: "{}"
Sentiment:"""
c0_1="""As a proficient data annotator in natural language processing (NLP), your responsibility is to determine the sentiment of the given review text. Please assign a sentiment value from "1" (very negative) to "5" (very positive).
Review Text: "{}"
Sentiment Score:"""
c0_2="""As a skilled data annotator in the field of natural language processing (NLP), your task is to evaluate the sentiment of the given review text. Please classify the sentiment using a scale from "1" (highly negative) to "5" (highly positive).
Review Text: "{}"
Sentiment Rating:"""
c0_3="""As an expert data annotator for NLP tasks, you are required to assess the sentiment of the provided review text. Kindly rate the sentiment on a scale of "1" (extremely negative) to "5" (extremely positive).
Review Text: "{}"
Sentiment Evaluation:"""
c0_4="""As a proficient data annotator in natural language processing (NLP), your responsibility is to determine the sentiment of the given review text. Please assign a sentiment value from "1" (very negative) to "5" (very positive).
Review Text: "{}"
Sentiment Assessment:"""

def layername(model, num, kind=None):
    if hasattr(model, "transformer"):
        if kind == "embed":
            return "transformer.wte"
        return f'transformer.h.{num}{"" if kind is None else "." + kind}'
    if hasattr(model, "gpt_neox"):
        if kind == "embed":
            return "gpt_neox.embed_in"
        if kind == "attn":
            kind = "attention"
        return f'gpt_neox.layers.{num}{"" if kind is None else "." + kind}'
    if hasattr(model, "model"):
        if kind == "embed":
            return "model.embed_tokens"
        return f'model.layers.{num}{"" if kind is None else "." + kind}'
    assert False, "unknown transformer structure"

def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))
    return results

def find_token_range(tokenizer, token_array, substring):
    tr=list(token_array.cpu().numpy())
    subtokens=tokenizer.encode(substring)[1:]
    ind=find_sub_list(subtokens,tr)[0]
    return ind


def trace_with_patch(
    model,  # The model
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    answers_t,  # Answer probabilities to collect
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    noise=0.1,  # Level of noise to add
    trace_layers=None,  # List of traced outputs to return
):
    prng = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    embed_layername = layername(model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    def patch_rep(x, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                x[1:, b:e] += noise * torch.from_numpy(
                    prng.randn(x.shape[0] - 1, e - b, x.shape[2])
                ).to(x.device)
            return x
        if layer not in patch_spec:
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec[layer]:
            h[1:, t] = h[0, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    additional_layers = [] if trace_layers is None else trace_layers
    with torch.no_grad(), nethook.TraceDict(
        model,
        [embed_layername] + list(patch_spec.keys()) + additional_layers,
        edit_output=patch_rep,
    ) as td:
        outputs_exp = model(**inp)

    # We report softmax probabilities for the answers_t token predictions of interest.
    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]

    # If tracing all layers, collect all activations together to return.
    if trace_layers is not None:
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
        )
        return probs, all_traced

    return probs

def reorder_outputs(t):
    t=torch.permute(t,(2,0,1)).detach().cpu().numpy()
    out_dict={' 1':t[0],' 2':t[1],' 3':t[2],' 4':t[3],' 5':t[4],' one':t[5],' two':t[6],' three':t[7],' four':t[8],' five':t[9]}
    return out_dict


def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))
    return results

def find_token_range(tokenizer, token_array, substring):
    tr=list(token_array.cpu().numpy())
    subtokens=tokenizer.encode(substring)[1:]
    ind=find_sub_list(subtokens,tr)[0]
    return ind



def trace_with_patch(
    model,  # The model
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    answers_t,  # Answer probabilities to collect
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    noise=0.1,  # Level of noise to add
    trace_layers=None,  # List of traced outputs to return
):
    prng = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    embed_layername = layername(model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    def patch_rep(x, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                x[1:, b:e] += noise * torch.from_numpy(
                    prng.randn(x.shape[0] - 1, e - b, x.shape[2])
                ).to(x.device)
            return x
        if layer not in patch_spec:
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec[layer]:
            h[1:, t] = h[0, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    additional_layers = [] if trace_layers is None else trace_layers
    with torch.no_grad(), nethook.TraceDict(
        model,
        [embed_layername] + list(patch_spec.keys()) + additional_layers,
        edit_output=patch_rep,
    ) as td:
        outputs_exp = model(**inp)

    # We report softmax probabilities for the answers_t token predictions of interest.
    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]

    # If tracing all layers, collect all activations together to return.
    if trace_layers is not None:
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
        )
        return probs, all_traced

    return probs


def calculate_hidden_flow(
    mt, prompt, subject, samples=6, noise=0.1, window=6, kind=None
):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    """
    inp = make_inputs(mt.tokenizer, [prompt] * (samples + 1))
    with torch.no_grad():
        answer_o, base_score = [d[0] for d in predict_from_input(mt.model, inp)]
    answer_t=make_inputs(mt.tokenizer, ['1','2','3','4','5','one','two','three','four','five'])['input_ids']
    #answer_t=make_inputs(mt.tokenizer, [' 1',' 2',' 3',' 4',' 5',' one',' two',' three',' four',' five'])['input_ids']
    answer_t=answer_t[:,-1]
    answer = decode_tokens(mt.tokenizer, answer_t)
    answer_o_str = decode_tokens(mt.tokenizer, [answer_o])
    e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], subject)
    low_score = trace_with_patch(
        mt.model, inp, [], answer_o, e_range, noise=noise
    ).item()
    if not kind:
        differences = trace_important_states(
            mt.model, mt.num_layers, inp, e_range, answer_t, noise=noise
        )
    else:
        differences = trace_important_window(
            mt.model,
            mt.num_layers,
            inp,
            e_range,
            answer_t,
            noise=noise,
            window=window,
            kind=kind,
        )
    differences = differences.detach().cpu()
    structured_diffs=reorder_outputs(differences)
    return dict(
        scores=structured_diffs,
        low_score=low_score,
        high_score=base_score.detach().cpu(),
        input_ids=inp["input_ids"][0].detach().cpu(),
        input_tokens=decode_tokens(mt.tokenizer, inp["input_ids"][0]),
        subject_range=e_range,
        answer=answer_o_str,
        window=window,
        kind=kind or "",
    )


def trace_important_states(model, num_layers, inp, e_range, answer_t, noise=0.1):
    ntoks = inp["input_ids"].shape[1]
    table = []
    for tnum in range(ntoks):
        row = []
        for layer in range(0, 2):
            r = trace_with_patch(
                model,
                inp,
                [(tnum, layername(model, layer))],
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


def trace_important_window(
    model, num_layers, inp, e_range, answer_t, kind, window=6, noise=0.1
):
    ntoks = inp["input_ids"].shape[1]
    table = []
    for tnum in range(ntoks):
        row = []
        for layer in range(0, 2):
            layerlist = [
                (tnum, layername(model, L, kind))
                for L in range(
                    max(0, layer - window // 2), min(num_layers, layer - (-window // 2))
                )
            ]
            r = trace_with_patch(
                model, inp, layerlist, answer_t, tokens_to_mix=e_range, noise=noise
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)

def plot_hidden_flow(
    mt,
    prompt,
    subject=None,
    samples=6,
    noise=0.1,
    window=6,
    kind=None,
    modelname=None,
    savepdf=None,
):
    if subject is None:
        subject = guess_subject(prompt)
    result = calculate_hidden_flow(
        mt, prompt, subject, samples=samples, noise=noise, window=window, kind=kind
    )
    #plot_trace_heatmap(result, savepdf, modelname=modelname)
    return result


def plot_all_flow(mt, prompt, subject=None, noise=0.1, modelname=None):
    for kind in [None, "mlp", "attn"]:
        plot_hidden_flow(
            mt, prompt, subject, modelname=modelname, noise=noise, kind=kind
        )

tokenizer = LlamaTokenizer.from_pretrained("/path/to/llama")
model_name = "/path/to/llama" # or "EleutherAI/gpt-j-6B" or "EleutherAI/gpt-neox-20b"
mt = ModelAndTokenizer(
    model_name,
    low_cpu_mem_usage=False,
    torch_dtype=(torch.float16 if (("20b" in model_name) or ("llama" in model_name)) else None),
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
prompt_groups=[[c0_0,c0_1,c0_2,c0_3,c0_4],[c1_0,c1_1,c1_2,c1_3,c1_4],[c2_0,c2_1,c2_2,c2_3,c2_4]]

yelp=pd.read_csv("./test_yelp_1k.csv").head(100)
noise_level=0.1

noise=noise_level
#llama=[[0,0],[1,1],[2,1]]
llama=[[1,1],[2,1]]
gpt2xl=[[1,1],[2,4]]
outputs_final=[]
for i in range(0,yelp.shape[0],1):
    with torch.no_grad():
        for elem in llama:
            review=yelp['text'].values[i]
            prompt=prompt_groups[elem[0]][elem[1]]
            #prompts=prompt.format(review)
            prompts=prompt.format(review).strip()+" "
            str_rep='"{}"'.format(review)
            #str_rep=prompts
            inputs = tokenizer(prompts, return_tensors='pt').to(device)
            if inputs['input_ids'].shape[1]>2048:
                print(inputs['input_ids'].shape[1])
                continue
#            try:
            dict_res={}
            dict_res['review_id']=yelp['review_id'].values[i]
            dict_res['prompt_type']=elem[0]
            dict_res['prompt_id']=elem[1]
            for kind in [None]:
                result=plot_hidden_flow(
                    mt, prompts, str_rep, modelname=None, noise=noise, kind=kind
                )
                if kind:
                    l_name="res_"+kind
                else:
                    l_name="res"
                dict_res[l_name] =result
            outputs_final.append(dict_res)
#            except Exception as e:
#                print(e)
#                continue
            with open('./psy_res/2lay_trace_llama7b.pkl', 'wb') as file:
                pickle.dump(outputs_final, file)
        print("DONE",i)
        with open('./psy_res/2lay_trace_llama7b.pkl', 'wb') as file:
            pickle.dump(outputs_final, file)


