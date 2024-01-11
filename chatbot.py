import asyncio
from asyncio import sleep

import discord
import torch
import os

from discord.ext import commands
from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv

# params
max_memory_size = 1700
max_token_generated = 1024
base_prompt = "You are a friendly AI that speaks like a pirate."
temperature = 0.7
repetition_penalty = 1.4
discord_command_prefix = "-"


# Setup the LLM for text generation

def prepare_llm(base_prompt):
    # Model changeable, though different Class may be needed than AutoModelForSeq2SeqLM (either from huggingface or loading a local model)
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Maximum token size of the chat history memory
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
        temperature=temperature,
        repetition_penalty=repetition_penalty,
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
conversation_buf, llm = prepare_llm(base_prompt)
load_dotenv()
# Load token from .env file
TOKEN = os.getenv("DISCORD_TOKEN")
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(description="Discord Chatbot", command_prefix=discord_command_prefix, intents=intents)


@bot.command()
async def chat(ctx, *, arg):
    input_prompt = arg
    async with ctx.typing():
        res = conversation_buf.predict(input=input_prompt)

        # Split the result into chunks of 2000 characters (discord character limit)
        chunks = [res[i:i+2000] for i in range(0, len(res), 2000)]
        # Send each chunk individually
        for chunk in chunks:
            print(chunk)
            await ctx.send(chunk)


@bot.command()
async def base_prompt(ctx, *, arg):
    prompt_template = (
            "Current Conversation:{history} \n\n" +
            "Instruction: " + arg + "\n\n" +
            "QUESTION:\n" +
            "{input}" + "\n\n" +
            "Answer:\n"
    )
    conversation_buf.prompt = PromptTemplate(input_variables=['history', 'input'],
                                             template=prompt_template)
    conversation_buf.memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=max_memory_size)


@bot.command()
async def flush_memory(ctx):
    conversation_buf.memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=max_memory_size)


@bot.event
async def on_ready():
    print('Ready')


# run the client
bot.run(TOKEN)
