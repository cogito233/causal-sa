import torch
from transformers import  GPT2Tokenizer, GPT2LMHeadModel
from transformers import  LlamaForCausalLM, LlamaTokenizer
import csv
import pandas as pd
import pickle

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

prompt_c1="""I just finished eating at a restaurant. Then I opened my Yelp app. I first wrote the following review:
{}
Then I read my review and finally gave a rating of"""
prompt_c2="""I just finished eating at a restaurant. Then I opened my Yelp app. I first gave a rating, and then justified it with the following review:
{}
The review explains why I gave it a rating of"""

is_llama=True
token_limit=1024
def main():
    tokenizer = LlamaTokenizer.from_pretrained("/path/to/llama")
    model = LlamaForCausalLM.from_pretrained("/path/to/llama",device_map="auto")
 
    #tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
    #model = GPT2LMHeadModel.from_pretrained("gpt2-xl",device_map="auto")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ## read prompts from csv and generate predictions
    df=pd.read_csv("./test_yelp_1k.csv")

    prompt_groups=[[c0_0,c0_1,c0_2,c0_3,c0_4],[c1_0,c1_1,c1_2,c1_3,c1_4],[c2_0,c2_1,c2_2,c2_3,c2_4]]

    if is_llama:
        tokens_stars=['1','2','3','4','5','one','two','three','four','five']
        tokens_res=tokenizer(tokens_stars)
        last_elements=[]
        for i in range(10):
            last_elements.append(tokens_res['input_ids'][i][-1])
        ids=torch.tensor(last_elements)
        token_limit=2048
    else:
        tokens_stars=[' 1',' 2',' 3',' 4',' 5','1','2','3','4','5','one','two','three','four','five',' one',' two',' three',' four',' five']
        tokens_res=tokenizer(tokens_stars)
        ids=torch.squeeze(torch.tensor(tokens_res['input_ids']))
        token_limit=1024

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    with open('./psy_res/llama7b_test_1k_2.csv', 'w') as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        row=['pred','review_id','prompt_type','prompt_id']
        writer.writerow(row)
    outputs_final=[]
    for j,group in enumerate(prompt_groups):
        if j==0:
            limit=df.shape[0]
        else:
            limit=500
        for k,prompt in enumerate(group):
            with torch.no_grad():
                for i in range(0,limit,1):
                    review=df['text'].values[i]
                    prompts=prompt.format(review)
                    inputs = tokenizer([prompts], return_tensors='pt').to(device)
                    if inputs['input_ids'].shape[1]>token_limit:
                        continue
                    output_sequences = model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        do_sample=False, # disable sampling to test if batching affects output
                        max_new_tokens=15,temperature=0,output_scores=True,return_dict_in_generate=True
                    )
                    outputs=tokenizer.batch_decode(output_sequences['sequences'], skip_special_tokens=True)
                    probas=torch.softmax(output_sequences['scores'][0][0],0)[ids].cpu().numpy()
                    probas1=torch.softmax(output_sequences['scores'][1][0],0)[ids].cpu().numpy()
                    dict_res={}
                    dict_res0=dict(zip(tokens_stars,probas))
                    dict_res1=dict(zip(tokens_stars,probas1))
                    dict_res['probas_0']=dict_res0
                    dict_res['probas_1']=dict_res1
                    dict_res['review_id']=df['review_id'].values[i]
                    dict_res['prompt_type']=j
                    dict_res['prompt_id']=k
                    outputs_final.append(dict_res)
                    with open('./psy_res/llama7b_test_1k_2.csv', 'a') as csvoutput:
                        writer = csv.writer(csvoutput, lineterminator='\n')
                        writer.writerow([outputs[0],str(dict_res['review_id']),str(j),str(k)])
                with open('./psy_res/res_llama7b_test_1k_2.pkl', 'wb') as file:
                    pickle.dump(outputs_final, file)

## call main function
if __name__ == '__main__':
    main()
