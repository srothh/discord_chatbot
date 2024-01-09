import discord
import torch
import re
import os

from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationChain
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
from dotenv import load_dotenv

# Setup the LLM for text generation
def prepare_llm(base_prompt):
    # Model changeable, though different Class may be needed than AutoModelForSeq2SeqLM (either from huggingface or loading a local model)
    model_name = "lmsys/fastchat-t5-3b-v1.0"
    # Maximum token size of the chat history memory
    max_memory_size = 1850
    max_token_generated = 1024
    # fetch model_name from huggingface
    # fetch a tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./downloaded_models")
    # Fetch Model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                                  trust_remote_code=True, cache_dir="./downloaded_models",
                                                  device_map="auto")
    # Create a pipeline with some params
    pipe = pipeline("text2text-generation", model=model, torch_dtype=torch.bfloat16,
                    device_map="auto", tokenizer=tokenizer, max_new_tokens=max_token_generated, do_sample=True, temperature=0.7,
                    top_k=50, top_p=0.95)

    # store history
    huggingface_pipe = HuggingFacePipeline(pipeline=pipe)

    memory = ConversationSummaryBufferMemory(llm=huggingface_pipe, max_token_limit=max_memory_size)
    # base conversational prompt for the chatbot, change to change the "personality" of the bot
    chain = ConversationChain(llm=huggingface_pipe, memory=memory,
                                         prompt=PromptTemplate(input_variables=['history', 'input'],
                                                               template=base_prompt + '\n\nCurrent conversation:\n{history}\nHuman: {input}\nAI:'))
    return chain


# Prepare the Neural network
base_prompt = "You are a friendly chatbot who always responds like a pirate."
conversation_buf = prepare_llm(base_prompt)
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
        out = conversation_buf.predict(input=input_prompt)
        # Remove padding tokens from output
        res = re.sub(r"<+.*(pad)+.*>+|(pad)>+|<+(pad)", "", out)
        # For some reason, output contains double space
        res = re.sub(r'\s+', ' ', res)
        await message.reply(res)
    elif re.match('-base_prompt', message.content):
        content = re.sub('-base_prompt ', "",message.content)
        conversation_buf.prompt = PromptTemplate(input_variables=['history', 'input'],
                                                               template=content + '\n\nCurrent conversation:\n{history}\nHuman: {input}\nAI:')

@client.event
async def on_ready():
    print('Ready')


# run the client
client.run(TOKEN)
