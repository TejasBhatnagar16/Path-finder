# Path-finder
Implements A-Star, DFS, BFS and MDP algorithms using OpenAI's Gym library for maze visualisation and generation. 

# Requirements
Please download all the modules in requirements.txt to your virtual environment. The command is usually pip install -r requirements.txt if you're using pip or conda install --yes --file requirements.txt if you're using conda. 

# Running the code. 
The main code is in agent.py. The file takes in 2 command line inputs. 
- agent: Valid options are astar, dfs, bfs, value and policy.
- map: if no argument is provided, a map is chosen at random. Else, you may provide values between 1 and 12.
sample command to run: python agent.py astar 8
