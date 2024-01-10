import discord
import torch
import re
import os

from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv


# Setup the LLM for text generation
def prepare_llm(base_prompt):
    # Model changeable, though different Class may be needed than AutoModelForSeq2SeqLM (either from huggingface or loading a local model)
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Maximum token size of the chat history memory
    max_memory_size = 1850
    max_token_generated = 1024
    # fetch model_name from huggingface
    # fetch a tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./downloaded_models")
    # Fetch Model
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 trust_remote_code=True, cache_dir="./downloaded_models",
                                                 device_map="auto", quantization_config=bnb_config)
    # Tweak these params to change performance
    pipe = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=0.7,
        repetition_penalty=1.4,
        return_full_text=True,
        max_new_tokens=max_token_generated,
        do_sample=True,
    )

    # store history
    huggingface_pipe = HuggingFacePipeline(pipeline=pipe)
    prompt_template = (
            "Current Conversation:{history} \n\n" +
            "Instruction: " + base_prompt + "\n\n" +
            "QUESTION:\n" +
            "{input}" + "\n\n" +
            "Answer:\n"
    )

    memory = ConversationSummaryBufferMemory(llm=huggingface_pipe, max_token_limit=max_memory_size)
    # base conversational prompt for the chatbot, change to change the "personality" of the bot
    chain = LLMChain(verbose=True, llm=huggingface_pipe, memory=memory,
                     prompt=PromptTemplate(input_variables=['history', 'input'],
                                           template=prompt_template))
    return chain, huggingface_pipe


# Prepare the Neural network
# Change this base prompt to change teh "personality" of the chatbot
base_prompt = "You are a friendly AI that speaks like a pirate."
conversation_buf, llm = prepare_llm(base_prompt)
load_dotenv()
# Load token from .env file
TOKEN = os.getenv("DISCORD_TOKEN")
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)


@client.event
async def on_message(message):
    # Needs refactoring, bad code
    global conversation_buf
    if re.match('-chat', message.content):
        input_prompt = message.content[6:]
        res = conversation_buf.predict(input=input_prompt)
        ### Not needed for mistral
        # Remove padding tokens from output
        #res = re.sub(r"<+.*(pad)+.*>+|(pad)>+|<+(pad)", "", out)
        # For some reason, output contains double space
        #res = re.sub(r'\s+', ' ', res)
        ###
        await message.reply(res)
    elif re.match('-base_prompt', message.content):
        content = re.sub('-base_prompt ', "", message.content)
        conversation_buf.prompt = PromptTemplate(input_variables=['history', 'input'],
                                                 template=content + '\n\nCurrent conversation:\n{history}\nHuman: {input}\nAI:')


@client.event
async def on_ready():
    print('Ready')


# run the client
client.run(TOKEN)
