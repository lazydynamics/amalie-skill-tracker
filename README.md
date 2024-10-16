# Skill Progress Tracking System

This repository contains the client and server code for a skill-progress tracking system. This is a toy example to showcase the use of `RxInfer.jl` for Bayesian inference and LLMs.

## Server

The server is written in Julia and uses the `RxInfer.jl` package for Bayesian inference.

### Setup

1. Install Julia by following the instructions [here](https://docs.julialang.org/en/v1/manual/getting-started/).

2. Install the required packages:

```julia
using Pkg
Pkg.add(["RxInfer", "JSON3", "Statistics"])
```

3. Run the server:

```bash
cd server
julia skill_model.jl
```

The server will start and listen for connections on port 65432.

## Client

The client is written in Python and uses the `openai` package to interact with the OpenAI API and the server.

### Setup

1. Set up your OpenAI API key:
   - Get your API key [here](https://platform.openai.com/account/api-keys)
   - Export it as an environment variable:

```bash
export OPENAI_API_KEY=<your_api_key>
```

2. Activate your virtual environment:

```bash
source .venv/bin/activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

### Running the Client

To start the client:

```bash
python client.py
```

This will:
1. Connect to the server
2. Start a conversation with the user
3. Send user input to the server for processing
4. Receive and play audio responses from the server

Note: Intermediate information is saved in 'prior_skill.json'. Only prior and posterior data are saved instead of logging the entire performance history.



## Licensing

Lazy Dynamics © 2024. All rights reserved.