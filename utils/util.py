import os
import re
import glob
from abc import ABC, abstractmethod
import pyarrow.parquet as pq
import json
import numpy as np
import time
import pandas as pd
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.distributed as distributed
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Dict, Union, Optional, List, Protocol, Any, TypeVar, Generic
from openai import OpenAI
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from itertools import islice
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
import asyncio
import aiohttp
from functools import wraps
import logging
from datetime import datetime
import datasets

T = TypeVar('T', bound=Any)
ModelOutput = TypeVar('ModelOutput', bound=Dict[str, Any])
Prompt = str
Reference = str
Score = float

@dataclass(frozen=True)
class EvaluationResult:
    """
    Represents the result of an evaluation.

    Attributes:
        method_name (str): The name of the evaluation method used.
        model_name (str): The name of the model being evaluated.
        scores (List[float]): A list of scores from the evaluation.
        metadata (Dict[str, Any]): Additional metadata related to the evaluation. Defaults to an empty dictionary.
    """

    method_name: str
    model_name: str
    scores: List[float]
    metadata: Dict[str, Any] = field(default_factory=dcit)

    @property
    def mean_score(self) -> float:
        """
        Calculate the mean of the evaluation scores.

        Returns:
            float: The mean of the scores.
        """
        return float(np.mean(self.scores))

    @property
    def std_score(self) -> float:
        """
        Calculate the standard deviation of the evaluation scores.

        Returns:
            float: The standard deviation of the scores.
        """
        return float(np.std(self.scores))

def extract_tokens(data):
    """
    Recursively extracts tokens from nested data structures.

    This function traverses dictionaries and lists to extract tokens
    (keys from dictionaries nested within single-item lists) in a
    hierarchical structure.

    Parameters
    ----------
    data : dict or list
        The input data structure to traverse. It can be a nested combination 
        of dictionaries and lists.

    Yields
    ------
    token : str
        Tokens (dictionary keys) extracted from the input data structure.
    """
    if isinstance(data, dict):
        for value in data.values():
            yield from extract_tokens(value)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and len(item) == 1:
                inner_dict = next(iter(item.values()))
                if isinstance(inner_dict, dict):
                    for token in inner_dict.keys():
                        yield token


def extract_token_logprob_pairs(data):
    """
    Recursively extracts token-logprob pairs from nested data structures.

    This function traverses dictionaries and lists to extract key-value pairs
    (tokens and their associated log probabilities) from dictionaries nested
    within single-item lists.

    Parameters
    ----------
    data : dict or list
        The input data structure to traverse. It can be a nested combination 
        of dictionaries and lists.

    Yields
    ------
    tuple
        A tuple of (token, logprob), where `token` is a string representing 
        the key and `logprob` is the associated value.
    """
    if isinstance(data, dict):
        for value in data.values():
            yield from extract_token_logprob_pairs(value)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and len(item) == 1:
                inner_dict = next(iter(item.values()))
                if isinstance(inner_dict, dict):
                    for token, logprob in inner_dict.items():
                        yield (token, logprob)

class ModelPlatform(Enum):
    """
    Enumeration of model types supported in the evaluation.

    Attributes:
        HUGGINGFACE: Represents models from the Hugging Face library.
        OPENAI: Represents models from OpenAI.
        ANTHROPIC: Represents models from Anthropic.
    """
    HUGGINGFACE = auto()
    OPENAI = auto()
    ANTHROPIC = auto()

    @property
    def default_endpoint(self) -> Optional[str]:
        endpoints = {
            ModelPlatform.OPENAI: "https://api.openai.com/v1",
            ModelPlatform.ANTHROPIC: "https://api.anthropic.com/v1",
            ModelPlatform.HUGGINGFACE: None
        }
        return endpoints[self]

class BaseAgent(ABC):
    """Base class for creating different LLM-based agents"""
    API_RETRY_SLEEP = 10
    API_RESPONSE_ERROR = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 20.0

    def __init__(self, platform: ModelPlatform, model_name: str) -> None:
        self.platform = platform
        self.model_name = model_name

    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass

class OpenAIAgent(BaseAgent):
    def __init__(self,
                model_name,
                api_key: Optional[str]=None,
                temperature: float=0.5,
                max_tokens: int=256) -> None:
        super().__init__(ModelPlatform.OPENAI, model_name)
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=self.platform.default_endpoint
        )
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        """Generate a response using any OpenAI model"""
        response = self.API_RESPONSE_ERROR
        for _ in range(self.API_MAX_RETRY):
            try:
                chat = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    model=self.model_name,
                    logprobs=True, 
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    timeout=self.API_TIMEOUT
                )
                response = chat.choices[0].message.content
                break
            except Exception as e:
                print(f"Failed to generate response: {e}")
                time.sleep(self.API_RETRY_SLEEP)
        return response

    def retrieve_tokens_logprobs(self, prompt: str) -> List[Dict[int, Dict[str, float]]]:
        """Retrieve token logprobs for a given prompt"""
        chat = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            logprobs=True,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=self.API_TIMEOUT
        )
        return [{count: {value.token: value.logprob}} 
                for count, value in enumerate(chat.choices[0].logprobs.content)]

class GPT35Agent(BaseAgent):
    """GPT 3.5 Turbo Agent Class
    Some methods borrowed from work in 
    https://github.com/patrickrchao/JailbreakingLLMs/blob/main/language_models.py

    Parameters
    ----------
    BaseAgent : class
        Super class for all agents
    """
    API_RETRY_SLEEP = 10
    API_RESPONSE_ERROR = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 20.0
    API_KEY = os.getenv("OPENAI_API_KEY")
    def __init__(self, 
                 model_name: str, 
                 neo4j_uri: str, 
                 neo4j_username: str, 
                 neo4j_password: str,
                 api_key=API_KEY) -> None:
        """Constructor function for gpt-3.5-turbo agent.

        Parameters
        ----------
        model_name : str
            Name of the model agent based on the respective LLM agent.
        neo4j_uri : str
            URI for Neo4j database
        neo4j_username : str
            Username for Neo4j database
        neo4j_password : str
            Password for Neo4j database
        """
        super().__init__(model_name, neo4j_uri, neo4j_username, neo4j_password)
        self.client = OpenAI(api_key=api_key)

    def start_chat(self, prompt: str):
        """_summary_

        Parameters
        ----------
        message : str
            _description_

        Returns
        -------
        _type_
            _description_
        """
        response = self.API_RESPONSE_ERROR
        for _ in range(self.API_MAX_RETRY):
            try:
                chat = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    model=self.model_name,
                    logprobs=True, 
                    max_tokens=200,
                    temperature=.5, 
                    timeout=self.API_TIMEOUT
                )
                response = chat
                break
            except Exception as e:
                print(f"Failed to initialize OpenAI client: {e}")
                time.sleep(self.API_RETRY_SLEEP)
            time.sleep(self.API_RETRY_SLEEP)
        return response

    def give_response(self, chat) -> str:
        """_summary_

        Parameters
        ----------
        chat : _type_
            _description_

        Returns
        -------
        str
            _description_
        """
        content = chat.choices[0].message.content
        return content
    
    def retrieve_tokens_logprobs(self, chat) -> List[Dict[int, Dict[str, float]]]:
        """_summary_

        Parameters
        ----------
        chat : _type_
            _description_

        Returns
        -------
        dict[int, str]
            _description_
        """
        token_logprob = [{count: {value.token: value.logprob}} 
                         for count, value in enumerate(chat.choices[0].logprobs.content)]
        return token_logprob
    
    def get_nlp_properties(self, content):
        """_summary_

        Parameters
        ----------
        content : _type_
            _description_

        Returns
        -------
        Dict[int, str, Union[str, str]]
            _description_
        """
        nlp = spacy.load("en_core_web_sm")
        document = nlp(content)
        property_dict = {}
        for index, token in enumerate(document):
            property_dict[index] = {'text': token.text, 
                                    'lemma': token.lemma_, 
                                    'pos': token.pos_, 
                                    'tag': token.tag_, 
                                    'dep': token.dep_, 
                                    'shape': token.shape_, 
                                    'is_alpha': token.is_alpha, 
                                    'is_stop': token.is_stop}
        return property_dict
    
    def create_token_node(self, tx, token_id, token):
        """Creates or updates a node that from a token produced by an LLM

        Parameters
        ----------
        tx : transactional object
            The transactional context to execute the Cypher query.
        token : str
            The token from the LLM.
        """
        tx.run("MERGE (t:Token {id: $token_id, name: $token})", 
               token_id=token_id, 
               token=token) 

    def create_weighted_relationship(self, tx, token1_id, token1, token2_id, token2, weight):
        """Creates or updates a weighted relationship between two tokens, 
        and sets an additional property on the relationship indicating the model 
        from which the tokens originated.

        Parameters
        ----------
        tx : transactional object
            The transactional context to execute the Cypher query.
        token1 : str
            The name of the first token node.
        token2 : str
            The name of the second token node.
        weight : float
            The logprob of the relationship
        """
        tx.run("""
            MATCH (t1:Token {id: $token1_id, name: $token1}), (t2:Token {id: $token2_id, name: $token2})
            MERGE (t1)-[r:NEXT]->(t2)
            ON CREATE SET r.weight = $weight, r.model_name = $model_name
            ON MATCH SET r.weight = $weight, r.model_name = $model_name
        """, token1_id=token1_id, 
        token1=token1, 
        token2_id=token2_id, 
        token2=token2, 
        weight=weight, 
        model_name=self.model_name)

    def create_graph(self, generative_responses_dict):
        with self.driver.session() as session:
            
            tokens_logprobs_list = list(extract_token_logprob_pairs(generative_responses_dict))

            # Create nodes for each token
            for token, logprob in tokens_logprobs_list:
                session.write_transaction(lambda tx: self.create_token_node(tx, token))

            # Create weighted relationships between consecutive tokens
            for i in range(len(tokens_logprobs_list) - 1):
                token1, _ = tokens_logprobs_list[i]
                token2, weight = tokens_logprobs_list[i + 1]
                session.write_transaction(lambda tx: self.create_weighted_relationship(tx, token1, token2, weight))
        print("Graph created.")
         

class GPT4Agent(BaseAgent):
    """GPT 4 Agent Class
    Some methods borrowed from work in 
    https://github.com/patrickrchao/JailbreakingLLMs/blob/main/language_models.py

    Parameters
    ----------
    BaseAgent : class
        Super class for all agents
    """
    API_RETRY_SLEEP = 10
    API_RESPONSE_ERROR = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 20.0
    API_KEY = os.getenv("OPENAI_API_KEY")
    def __init__(self, 
                 model_name: str, 
                 neo4j_uri: str, 
                 neo4j_username: str, 
                 neo4j_password: str,
                 api_key=API_KEY) -> None:
        """_summary_

        Parameters
        ----------
        model_name : str
            _description_
        neo4j_uri : str
            _description_
        neo4j_username : str
            _description_
        neo4j_password : str
            _description_
        """
        super().__init__(model_name, neo4j_uri, neo4j_username, neo4j_password)
        self.client = OpenAI(api_key=api_key)

    def start_chat(self, prompt: str):
        """_summary_

        Parameters
        ----------
        message : str
            _description_

        Returns
        -------
        _type_
            _description_
        """
        response = self.API_RESPONSE_ERROR
        for _ in range(self.API_MAX_RETRY):
            try:
                chat = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    model=self.model_name,
                    logprobs=True, 
                    max_tokens=200,
                    temperature=.5, 
                    timeout=self.API_TIMEOUT
                )
                response = chat
                break
            except Exception as e:
                print(f"Failed to initialize OpenAI client: {e}")
                time.sleep(self.API_RETRY_SLEEP)
            time.sleep(self.API_RETRY_SLEEP)
        return response

    def give_response(self, chat) -> str:
        """_summary_

        Parameters
        ----------
        chat : _type_
            _description_

        Returns
        -------
        str
            _description_
        """
        content = chat.choices[0].message.content
        return content
    
    def retrieve_tokens_logprobs(self, chat) -> List[Dict[int, Dict[str, float]]]:
        """_summary_

        Parameters
        ----------
        chat : _type_
            _description_

        Returns
        -------
        dict[int, str]
            _description_
        """
        token_logprob = [{count: {value.token: value.logprob}} 
                         for count, value in enumerate(chat.choices[0].logprobs.content)]
        return token_logprob
    
    def get_nlp_properties(self, content):
        """_summary_

        Parameters
        ----------
        content : _type_
            _description_

        Returns
        -------
        Dict[int, str, Union[str, str]]
            _description_
        """
        nlp = spacy.load("en_core_web_sm")
        document = nlp(content)
        property_dict = {}
        for index, token in enumerate(document):
            property_dict[index] = {'text': token.text, 
                                    'lemma': token.lemma_, 
                                    'pos': token.pos_, 
                                    'tag': token.tag_, 
                                    'dep': token.dep_, 
                                    'shape': token.shape_, 
                                    'is_alpha': token.is_alpha, 
                                    'is_stop': token.is_stop}
        return property_dict
    
    def create_token_node(self, tx, token_id, token):
        """Creates or updates a node that from a token produced by an LLM

        Parameters
        ----------
        tx : transactional object
            The transactional context to execute the Cypher query.
        token : str
            The token from the LLM.
        """
        tx.run("MERGE (t:Token {id: $token_id, name: $token})", 
               token_id=token_id, 
               token=token) 

    def create_weighted_relationship(self, tx, token1_id, token1, token2_id, token2, weight):
        """Creates or updates a weighted relationship between two tokens, 
        and sets an additional property on the relationship indicating the model 
        from which the tokens originated.

        Parameters
        ----------
        tx : transactional object
            The transactional context to execute the Cypher query.
        token1 : str
            The name of the first token node.
        token2 : str
            The name of the second token node.
        weight : float
            The logprob of the relationship
        """
        tx.run("""
            MATCH (t1:Token {id: $token1_id, name: $token1}), (t2:Token {id: $token2_id, name: $token2})
            MERGE (t1)-[r:NEXT]->(t2)
            ON CREATE SET r.weight = $weight, r.model_name = $model_name
            ON MATCH SET r.weight = $weight, r.model_name = $model_name
        """, token1=token1, token2=token2, weight=weight, model_name = self.model_name)


    def create_graph(self, generative_responses_dict):
        """_summary_

        Parameters
        ----------
        truthfulqa_tokens_dict : _type_
            _description_
        """
        with self.driver.session() as session:
            tokens_logprobs_list = list(extract_token_logprob_pairs(generative_responses_dict))

            # Create nodes for each token
            for token, logprob in tokens_logprobs_list:
                session.write_transaction(lambda tx: self.create_token_node(tx, token))

            # Create weighted relationships between consecutive tokens
            for i in range(len(tokens_logprobs_list) - 1):
                token1, _ = tokens_logprobs_list[i]
                token2, weight = tokens_logprobs_list[i + 1]
                session.write_transaction(lambda tx: self.create_weighted_relationship(tx, token1, token2, weight))
        print("Graph created.")


class LlamaBaseAgent(BaseAgent):
    def __init__(self, 
                 model_name: str, 
                 neo4j_uri: str, 
                 neo4j_username: str, 
                 neo4j_password: str,
                 cuda_device: str="0",
                 cache_path: Optional[str]=None,
                 use_DDP=False) -> None:
        super().__init__(model_name, neo4j_uri, neo4j_username, neo4j_password)
        self.cuda_device = cuda_device
        self.cache_path = cache_path

    def get_devices(self):
        cuda = torch.cuda.is_available()
        device = (lambda: torch.device(f"cuda:{self.cuda_device}" if cuda else "cpu"))()
        num_gpus = (lambda: torch.cuda.device_count() if cuda else 0)()
        return device, num_gpus

    def generate_text(self, prompt: str, model_path: str, return_raw_outputs=False):
        torch.set_default_device("cuda")
        model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                     torch_dtype="auto", 
                                                     trust_remote_code=True,
                                                     cache_dir=self.cache_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
        if return_raw_outputs:
            outputs = model.generate(**inputs, 
                                     max_new_tokens=200, 
                                     return_dict_in_generate=True, 
                                     output_scores=True)        
            transition_scores = model.compute_transition_scores(
                outputs.sequences,
                outputs.scores,
                normalize_logits=True
            )
            input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
            generated_tokens = outputs.sequences[:, input_length:]
            output_structure = []
            for index, (token, logprob) in enumerate(zip(generated_tokens[0], transition_scores[0])):
                logprob = logprob.cpu().numpy()
                token_str = tokenizer.decode([token])
                output_structure.append({index: {token_str: float(logprob)}})
            return output_structure
        else:
            outputs = model.generate(**inputs, max_new_tokens=200)
            text = tokenizer.batch_decode(outputs)[0]
            return text
    
    def generate_logprobs(self, logits, labels):
        log_probs = F.log_softmax(logits, dim=-1)
        logprobs_label = torch.gather(log_probs, 2, labels.unsqueeze(2)).squeeze(-1)
        return logprobs_label
    
    def sequence_logprob(self, model, labels, input_len=0) -> npt.NDArray[np.float64]:
        with torch.no_grad():
            output = model(labels)
            log_probs = self.generate_logprobs(
                output.logits[:, :-1, :], labels[:, 1:]
            )
            seq_log_probs = torch.sum(log_probs[:, input_len:])
        return seq_log_probs.cpu.numpy()
    
    def create_token_node(self, tx, token_id, token):
        """Creates or updates a node that from a token produced by an LLM

        Parameters
        ----------
        tx : transactional object
            The transactional context to execute the Cypher query.
        token : str
            The token from the LLM.
        """
        tx.run("MERGE (t:Token {id: $token_id, name: $token})", 
               token_id=token_id, 
               token=token) 

    def create_weighted_relationship(self, tx, token1_id, token1, token2_id, token2, weight):
        """Creates or updates a weighted relationship between two tokens, 
        and sets an additional property on the relationship indicating the model 
        from which the tokens originated.

        Parameters
        ----------
        tx : transactional object
            The transactional context to execute the Cypher query.
        token1 : str
            The name of the first token node.
        token2 : str
            The name of the second token node.
        weight : float
            The logprob of the relationship
        """
        tx.run("""
            MATCH (t1:Token {id: $token1_id, name: $token1}), (t2:Token {id: $token2_id, name: $token2})
            MERGE (t1)-[r:NEXT]->(t2)
            ON CREATE SET r.weight = $weight, r.model_name = $model_name
            ON MATCH SET r.weight = $weight, r.model_name = $model_name
        """, token1=token1, token2=token2, weight=weight, model_name = self.model_name)

    def create_graph(self, generative_responses_dict):
        with self.driver.session() as session:
            tokens_logprobs_list = list(extract_token_logprob_pairs(generative_responses_dict))

            # Create nodes for each token
            for token, logprob in tokens_logprobs_list:
                session.write_transaction(lambda tx: self.create_token_node(tx, token))

            # Create weighted relationships between consecutive tokens
            for i in range(len(tokens_logprobs_list) - 1):
                token1, _ = tokens_logprobs_list[i]
                token2, weight = tokens_logprobs_list[i + 1]
                session.write_transaction(lambda tx: self.create_weighted_relationship(tx, token1, token2, weight))
        print("Graph created.")


class LlamaLargeAgent(BaseAgent):
    def __init__(self, 
                 model_name: str, 
                 neo4j_uri: str, 
                 neo4j_username: str, 
                 neo4j_password: str,
                 cuda_device: str="0",
                 cache_path: Optional[str]=None,
                 use_DDP=False) -> None:
        super().__init__(model_name, neo4j_uri, neo4j_username, neo4j_password)
        self.cuda_device = cuda_device
        self.cache_path = cache_path

    def get_devices(self):
        cuda = torch.cuda.is_available()
        device = (lambda: torch.device(f"cuda:{self.cuda_device}" if cuda else "cpu"))()
        num_gpus = (lambda: torch.cuda.device_count() if cuda else 0)()
        return device, num_gpus

    def generate_text(self, prompt: str, model_path: str, return_raw_outputs=False):
        torch.set_default_device("cuda")
        model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                     torch_dtype="auto", 
                                                     trust_remote_code=True,
                                                     cache_dir=self.cache_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
        outputs = model.generate(**inputs, max_new_tokens=5, return_dict_in_generate=True, output_scores=True)
        # If marked true, returns raw outputs, logits, and probabilities
        if return_raw_outputs:
            transition_scores = model.compute_transition_scores(
                outputs.sequences,
                outputs.scores,
                normalize_logits=True
            )
            input_length = inputs.input_ids.shape[1]
            generated_tokens = outputs.sequences[:, input_length:]
            for token, score in zip(generated_tokens[0], transition_scores[0]):
                return token, tokenizer.decode(token), score.numpy, np.exp(score.numpy())
        else:
            text = tokenizer.batch_decode(outputs["sequences"])[0]
            return text
    
    def generate_logprobs(self, logits, labels):
        log_probs = F.log_softmax(logits, dim=-1)
        logprobs_label = torch.gather(log_probs, 2, labels.unsqueeze(2)).squeeze(-1)
        return logprobs_label
    
    def sequence_logprob(self, model, labels, input_len=0) -> npt.NDArray[np.float64]:
        with torch.no_grad():
            output = model(labels)
            log_probs = self.generate_logprobs(
                output.logits[:, :-1, :], labels[:, 1:]
            )
            seq_log_probs = torch.sum(log_probs[:, input_len:])
        return seq_log_probs.cpu.numpy()
    
    def create_token_node(self, tx, token_id, token):
        """Creates or updates a node that from a token produced by an LLM

        Parameters
        ----------
        tx : transactional object
            The transactional context to execute the Cypher query.
        token : str
            The token from the LLM.
        """
        tx.run("MERGE (t:Token {id: $token_id, name: $token})", 
               token_id=token_id, 
               token=token) 

    def create_weighted_relationship(self, tx, token1_id, token1, token2_id, token2, weight):
        """Creates or updates a weighted relationship between two tokens, 
        and sets an additional property on the relationship indicating the model 
        from which the tokens originated.

        Parameters
        ----------
        tx : transactional object
            The transactional context to execute the Cypher query.
        token1 : str
            The name of the first token node.
        token2 : str
            The name of the second token node.
        weight : float
            The logprob of the relationship
        """
        tx.run("""
            MATCH (t1:Token {id: $token1_id, name: $token1}), (t2:Token {id: $token2_id, name: $token2})
            MERGE (t1)-[r:NEXT]->(t2)
            ON CREATE SET r.weight = $weight, r.model_name = $model_name
            ON MATCH SET r.weight = $weight, r.model_name = $model_name
        """, token1=token1, token2=token2, weight=weight, model_name = self.model_name)
    
    def create_graph(self, generative_responses_dict):
        with self.driver.session() as session:
            tokens_logprobs_list = list(extract_token_logprob_pairs(generative_responses_dict))

            # Create nodes for each token
            for token, logprob in tokens_logprobs_list:
                session.write_transaction(lambda tx: self.create_token_node(tx, token))

            # Create weighted relationships between consecutive tokens
            for i in range(len(tokens_logprobs_list) - 1):
                token1, _ = tokens_logprobs_list[i]
                token2, weight = tokens_logprobs_list[i + 1]
                session.write_transaction(lambda tx: self.create_weighted_relationship(tx, token1, token2, weight))
        print("Graph created.")

class MicrosoftAgent(BaseAgent):
    def __init__(self, 
                 model_name: str, 
                 neo4j_uri: str, 
                 neo4j_username: str, 
                 neo4j_password: str,
                 cuda_device: str="0",
                 cache_path: Optional[str]=None,
                 use_DDP=False) -> None:
        super().__init__(model_name, neo4j_uri, neo4j_username, neo4j_password)
        self.cuda_device = cuda_device
        self.cache_path = cache_path

    def get_devices(self):
        cuda = torch.cuda.is_available()
        device = (lambda: torch.device(f"cuda:{self.cuda_device}" if cuda else "cpu"))()
        num_gpus = (lambda: torch.cuda.device_count() if cuda else 0)()
        return device, num_gpus
    
    def generate_logprobs(self, logits, labels):
        log_probs = F.log_softmax(logits, dim=-1)
        logprobs_label = torch.gather(log_probs, 2, labels.unsqueeze(2)).squeeze(-1)
        return logprobs_label
    
    def sequence_logprob(self, model, labels, input_len=0) -> npt.NDArray[np.float64]:
        with torch.no_grad():
            output = model(labels)
            log_probs = self.generate_logprobs(
                output.logits[:, :-1, :], labels[:, 1:]
            )
            seq_log_probs = torch.sum(log_probs[:, input_len:])
        return seq_log_probs.cpu.numpy()

    def generate_text(self, prompt: str, model_path: str, return_raw_outputs=False):
        torch.set_default_device("cuda")
        model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                     torch_dtype="auto", 
                                                     trust_remote_code=True,
                                                     cache_dir=self.cache_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        inputs = tokenizer(prompt, 
                           add_special_tokens=False, 
                           truncation=True,
                           return_tensors="pt", 
                           return_attention_mask=False)
        if return_raw_outputs:
            outputs = model.generate(**inputs, 
                                    max_new_tokens=200, 
                                    return_dict_in_generate=True, 
                                    output_scores=True)        
            transition_scores = model.compute_transition_scores(
                outputs.sequences,
                outputs.scores,
                normalize_logits=True
            )
            input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
            generated_tokens = outputs.sequences[:, input_length:]
            output_structure = []
            for index, (token, logprob) in enumerate(zip(generated_tokens[0], transition_scores[0])):
                logprob = logprob.cpu().numpy()
                token_str = tokenizer.decode([token])
                output_structure.append({index: {token_str: float(logprob)}}) #List[Dict[int, Dict[str, float]]]
            return output_structure
        else:
            outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.5)
            input_length = inputs.input_ids.shape[1]
            text = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)[input_length:]
            return " ".join(text)
        
    def create_token_node(self, tx, token_id, token):
        """Creates or updates a node that from a token produced by an LLM

        Parameters
        ----------
        tx : transactional object
            The transactional context to execute the Cypher query.
        token : str
            The token from the LLM.
        """
        tx.run("MERGE (t:Token {id: $token_id, name: $token})", 
               token_id=token_id, 
               token=token) 

    def create_weighted_relationship(self, tx, token1_id, token1, token2_id, token2, weight):
        """Creates or updates a weighted relationship between two tokens, 
        and sets an additional property on the relationship indicating the model 
        from which the tokens originated.

        Parameters
        ----------
        tx : transactional object
            The transactional context to execute the Cypher query.
        token1 : str
            The name of the first token node.
        token2 : str
            The name of the second token node.
        weight : float
            The logprob of the relationship
        """
        tx.run("""
            MATCH (t1:Token {id: $token1_id, name: $token1}), (t2:Token {id: $token2_id, name: $token2})
            MERGE (t1)-[r:NEXT]->(t2)
            ON CREATE SET r.weight = $weight, r.model_name = $model_name
            ON MATCH SET r.weight = $weight, r.model_name = $model_name
        """, token1=token1, token2=token2, weight=weight, model_name = self.model_name)
    
    def create_graph(self, generative_responses_dict):
        with self.driver.session() as session:
            tokens_logprobs_list = list(extract_token_logprob_pairs(generative_responses_dict))

            # Create nodes for each token
            for token, logprob in tokens_logprobs_list:
                session.write_transaction(lambda tx: self.create_token_node(tx, token))

            # Create weighted relationships between consecutive tokens
            for i in range(len(tokens_logprobs_list) - 1):
                token1, _ = tokens_logprobs_list[i]
                token2, weight = tokens_logprobs_list[i + 1]
                session.write_transaction(lambda tx: self.create_weighted_relationship(tx, token1, token2, weight))
        print("Graph created.")
    
        # torch.set_default_device("cuda")
        # if self.get_devices()[1] >= 1:
        #     # torch.distributed.init_process_group(backend='nccl')
        #     # rank = distributed.get_rank()
        #     # world_size = distributed.get_world_size()
        #     # print(f"World size: {world_size}")
        #     device, _ = self.get_devices()
        #     torch.cuda.set_device(device)
        #     model.to(device)
        #     # model = DDP(model, device_ids=[rank])
        #     inputs = tokenizer(prompt, return_tensors="pt")
        #     input_ids = inputs['input_ids']
        #     with torch.no_grad():
        #         generated_outputs = model.generate(input_ids, max_length=max_length)
        #         generated_text = tokenizer.decode(generated_outputs[0], skip_special_tokens=True)
        #     combined_text = prompt + generated_text
        #     combined_inputs = tokenizer.encode(combined_text, return_tensors="pt")
        #     with torch.no_grad():
        #         outputs = model(combined_inputs, labels=combined_inputs)
        #         logits = outputs.logits
        #     probabilities = torch.nn.functional.softmax(logits, dim=-1)
        #     log_probabilities = torch.log(probabilities)
        #     tokens = tokenizer.convert_ids_to_tokens(combined_inputs[0])
        #     log_probs_dict = {token: log_prob.item() for token, log_prob in zip(tokens, log_probabilities[0])}
        #     torch.distributed.destroy_process_group()
        #     return generated_text, log_probs_dict

class MistralAgent(BaseAgent):
       
    def __init__(self, 
                 model_name: str, 
                 neo4j_uri: str, 
                 neo4j_username: str, 
                 neo4j_password: str,
                 cuda_device: str="0",
                 cache_path: Optional[str]=None,
                 use_DDP=False) -> None:
        super().__init__(model_name, neo4j_uri, neo4j_username, neo4j_password)
        self.cuda_device = cuda_device
        self.cache_path = cache_path

    def get_devices(self):
        cuda = torch.cuda.is_available()
        device = (lambda: torch.device(f"cuda:{self.cuda_device}" if cuda else "cpu"))()
        num_gpus = (lambda: torch.cuda.device_count() if cuda else 0)()
        return device, num_gpus

    def generate_text(self, prompt: str, model_path: str, return_raw_outputs=False):
        torch.set_default_device("cuda")
        model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                     torch_dtype="auto", 
                                                     trust_remote_code=True,
                                                     cache_dir=self.cache_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt", return_attention_mask=False)
        if return_raw_outputs:
            outputs = model.generate(**inputs, 
                                    max_new_tokens=200, 
                                    return_dict_in_generate=True, 
                                    output_scores=True)        
            transition_scores = model.compute_transition_scores(
                outputs.sequences,
                outputs.scores,
                normalize_logits=True
            )
            input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
            generated_tokens = outputs.sequences[:, input_length:]
            output_structure = []
            for index, (token, logprob) in enumerate(zip(generated_tokens[0], transition_scores[0])):
                logprob = logprob.cpu().numpy()
                token_str = tokenizer.decode([token])
                output_structure.append({index: {token_str: float(logprob)}}) #List[Dict[int, Dict[str, float]]]
            return output_structure
        else:
            outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.5)
            input_length = inputs.input_ids.shape[1]
            text = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)[input_length:]
            return " ".join(text)
    
    def generate_logprobs(self, logits, labels):
        log_probs = F.log_softmax(logits, dim=-1)
        logprobs_label = torch.gather(log_probs, 2, labels.unsqueeze(2)).squeeze(-1)
        return logprobs_label
    
    def sequence_logprob(self, model, labels, input_len=0) -> npt.NDArray[np.float64]:
        with torch.no_grad():
            output = model(labels)
            log_probs = self.generate_logprobs(
                output.logits[:, :-1, :], labels[:, 1:]
            )
            seq_log_probs = torch.sum(log_probs[:, input_len:])
        return seq_log_probs.cpu.numpy()
    
    def create_token_node(self, tx, token_id, token):
        """Creates or updates a node that from a token produced by an LLM

        Parameters
        ----------
        tx : transactional object
            The transactional context to execute the Cypher query.
        token : str
            The token from the LLM.
        """
        tx.run("MERGE (t:Token {id: $token_id, name: $token})", 
               token_id=token_id, 
               token=token) 

    def create_weighted_relationship(self, tx, token1_id, token1, token2_id, token2, weight):
        """Creates or updates a weighted relationship between two tokens, 
        and sets an additional property on the relationship indicating the model 
        from which the tokens originated.

        Parameters
        ----------
        tx : transactional object
            The transactional context to execute the Cypher query.
        token1 : str
            The name of the first token node.
        token2 : str
            The name of the second token node.
        weight : float
            The logprob of the relationship
        """
        tx.run("""
            MATCH (t1:Token {id: $token1_id, name: $token1}), (t2:Token {id: $token2_id, name: $token2})
            MERGE (t1)-[r:NEXT]->(t2)
            ON CREATE SET r.weight = $weight, r.model_name = $model_name
            ON MATCH SET r.weight = $weight, r.model_name = $model_name
        """, token1=token1, token2=token2, weight=weight, model_name = self.model_name)
    
    def create_graph(self, generative_responses_dict):
        with self.driver.session() as session:
            tokens_logprobs_list = list(extract_token_logprob_pairs(generative_responses_dict))

            # Create nodes for each token
            for token, logprob in tokens_logprobs_list:
                session.write_transaction(lambda tx: self.create_token_node(tx, token))

            # Create weighted relationships between consecutive tokens
            for i in range(len(tokens_logprobs_list) - 1):
                token1, _ = tokens_logprobs_list[i]
                token2, weight = tokens_logprobs_list[i + 1]
                session.write_transaction(lambda tx: self.create_weighted_relationship(tx, token1, token2, weight))
        print("Graph created.")

def read_questions(file, batch_size=1000):
    """Function that extracts questions from the TruthfulQA dataset as input to LLM agent

    Parameters
    ----------
    file : str
        Parquet file 
    batch_size : int, optional
        Number of lines to read, by default 1000

    Yields
    ------
    Generator 
        Question extracted from dataset
    """
    parquet_file = pq.ParquetFile(file)
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=['question']):
        table = batch.to_pandas()
        for question in table['question']:
            yield question

def read_bill(file, batch_size=1000):
    """Function that extracts bill from the billsum dataset as input to LLM agent

    Parameters
    ----------
    file : str
        Billsum Parquet file
    batch_size : int, optional
        Number of lines to read, by default 1000
    """
    parquet_file = pq.ParquetFile(file)
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=['text']):
        table = batch.to_pandas()
        if 'text' in table.columns:
            for bill in table['text']:
                yield bill
        else:
            print("Batch does not contain 'text' column.")

def read_article(file, batch_size=1000):
    """Function that extracts article from the cnn_dailymail dataset as input to LLM agent

    Parameters
    ----------
    file : str
        CNN_Dailymail Parquet file
    batch_size : int, optional
        Number of lines to read, by default 1000
    """
    parquet_file = pq.ParquetFile(file)
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=['article']):
        table = batch.to_pandas()
        if 'article' in table.columns:
            for article in table['article']:
                yield article
        else:
            print("Batch does not contain 'article' column.")

def clean_response(text: str) -> str:
    """_summary_

    Parameters
    ----------
    text : str
        _description_

    Returns
    -------
    str
        _description_
    """
    # Remove lines starting with ##
    text = re.sub(r'^##.*\n?', '', text, flags=re.MULTILINE)
    # Remove lines/sentences starting with <s>
    text = re.sub(r'^<s>.*?\n?', '', text, flags=re.MULTILINE)
    return text.strip()
        
def match_model(model_name, agent_name, data):
    match model_name:
        case "gpt-3.5-turbo":
            agent_name.create_graph(data)
        case "gpt-4":
            agent_name.create_graph(data)
        case "gemini-pro":
            agent_name.create_graph(data)
        case "llama-base":
            agent_name.create_graph(data)
        case "llama-large":
            agent_name.create_graph(data)
        case "mistral-base":
            agent_name.create_graph(data)
        case "phi-2":
            agent_name.create_graph(data)
        case _:
            print(f"No matching model found for {model_name}.")

def agent_responses(dataset: str, agent_list: list, slice_generator=False):
    #files = glob.glob(dataset + "/**/*.parquet", recursive=True)
    questions = read_bill(dataset) # itertools.chain.from_iterable(read_questions(file) for file in files)
    if slice_generator:
        truthfulqa_responses = {}
        for question in islice(questions, 5):
            if len(question) >= 4097:
                question = question[:2000]
                responses = {}
                for agent in agent_list:
                    agent_name = agent.model_name
                    try:
                        if agent_name == "gpt-3.5-turbo" or agent_name == "gpt-4":
                            initialize = agent.start_chat(f"Please summarize this legislative bill. {question}")
                            responses[agent_name] = agent.give_response(initialize)
                        elif agent_name == "phi-2":
                            responses[agent_name] = agent.generate_text(question, 
                                                            "/p/llmreliability/llm_reliability/models/microsoft/base/microsoft/phi-2")
                        else:
                            responses[agent_name] = agent.start_chat(question)
                    except Exception as e:
                        print(f"Error with agent {agent_name}: {e}.")
                truthfulqa_responses[question] = responses
            print("QA complete.")
            return truthfulqa_responses
    else:
        truthfulqa_responses = {}
        for question in questions:
            responses = {}
            for agent in agent_list:
                agent_name = agent.model_name
                try:
                    if agent_name == "gpt-3.5-turbo" or agent_name == "gpt-4":
                        initialize = agent.start_chat(question)
                        responses[agent_name] = agent.give_response(initialize)
                    elif agent_name == "phi-2":
                        initialize = agent.generate_text(question, 
                                                         "/p/llmreliability/llm_reliability/models/microsoft/base/microsoft/phi-2")
                    else:
                        responses[agent_name] = agent.start_chat(question)
                except Exception as e:
                    print(f"Error with agent {agent_name}: {e}.")
            truthfulqa_responses[question] = responses
        print("QA complete.")
        return truthfulqa_responses

def agent_token_logprobs(dataset_root_path: str, agent_list: list, slice_generator=False):
    files = glob.glob(dataset_root_path + "/**/*.parquet", recursive=True)
    questions = itertools.chain.from_iterable(read_questions(file) for file in files)
    if slice_generator:
        truthfulqa_tokens_dict = {}
        for question in islice(questions, 1):
            token_responses = {}
            for agent in agent_list:
                agent_name = agent.model_name
                try:
                    if agent_name == "gpt-3.5-turbo" or agent_name == "gpt-4":
                        initialize = agent.start_chat(question)
                        token_responses[agent_name] = agent.retrieve_tokens_logprobs(initialize)
                    elif agent_name == "phi-2":
                        initialize = agent.generate_text(question, 
                                                         "/p/llmreliability/llm_reliability/models/microsoft/base/microsoft/phi-2",
                                                         return_raw_outputs=True)
                    else:
                        pass
                except Exception as e:
                    print(f"Error with agent {agent_name}: {e}.")
            truthfulqa_tokens_dict[question] = token_responses
        print("QA token generation complete.")
        return truthfulqa_tokens_dict
    else: 
        truthfulqa_tokens_dict = {}
        for question in questions:
            token_responses = {}
            for agent in agent_list:
                agent_name = agent.model_name
                try:
                    if agent_name == "gpt-3.5-turbo" or agent_name == "gpt-4":
                        initialize = agent.start_chat(question)
                        token_responses[agent_name] = agent.retrieve_tokens_logprobs(initialize)
                    elif agent_name == "phi-2":
                        initialize = agent.generate_text(question, 
                                                         "/p/llmreliability/llm_reliability/models/microsoft/base/microsoft/phi-2",
                                                         return_raw_outputs=True)
                    else:
                        pass
                except Exception as e:
                    print(f"Error with agent {agent_name}: {e}.")
            truthfulqa_tokens_dict[question] = token_responses
        print("QA token generation complete.")
        return truthfulqa_tokens_dict


