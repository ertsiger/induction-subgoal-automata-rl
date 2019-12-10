# Induction of Subgoal Automata for Reinforcement Learning
Implementation of the ISA (Induction of Subgoal Automata) algorithm presented in [[Furelos-Blanco et al., 2020]](#references).

1. [Installation](#installation)
    1. [Install Python packages](#install-python)
    1. [Install `ILASP` and `clingo` binaries](#install-ilasp-clingo)
1. [Usage](#usage)
    1. [Run the algorithm](#run-isa)
        1. [General flags](#general-flags)
        1. [Reinforcement learning flags](#rl-flags)
        1. [Automata learning flags](#automata-flags)
    1. [Plot the learning curves](#plot-results)
    1. [Collect learning statistics](#collect-stats)
1. [References](#references)

## <a name="installation"></a>Installation
The code only runs on Linux or MacOS computers with Python 3. Firstly, you have to download the repository which can be
done with the following command.
```
git clone https://github.com/ertsiger/induction-subgoal-automata-rl.git
```

The following subsections describe the steps for installing the required [Python packages](#install-python) and the [binaries](#install-ilasp-clingo) related to the
Inductive Logic Programming system.

### <a name="install-python"></a> Install Python packages
To install the required Python packages to run our code, you can use `pip` with the following command:

```
cd induction-subgoal-automata-rl
pip install -r requirements.txt
```

Note that one of the requirements is the package in the `gym-subgoal-automata` repository ([link](https://github.com/ertsiger/gym-subgoal-automata)).
We use the environments implemented in that repository. 

We recommend you to use a [virtual environment](https://docs.python.org/3/tutorial/venv.html) 
since the requirements of our installation may affect your current installation.

### <a name="install-ilasp-clingo"></a> Install `ILASP` and `clingo` binaries
The cloned repository does not include the binaries for the ILASP inductive logic programming system. Therefore, you have
to download the binaries from the following websites and then copy the 
`ILASP` and `clingo` binaries into the `bin` folder:
* [ILASP v3.4.0](https://github.com/marklaw/ILASP-releases/releases/tag/v3.4.0)
* [clingo 5.3.0](https://github.com/potassco/clingo/releases/tag/v5.3.0)

Alternatively, you can run the `install_binaries.sh`, which will
download  and put the files in the `bin` folder for you.

## <a name="usage"></a>Usage
### <a name="running-isa"></a>Run the algorithm
The ISA algorithm can be executed easily by running the `run_isa.py` script:
```
python3 run_isa.py config_file
```

The script receives a single argument `config_file`, which is a JSON configuration file containing the settings with which ISA is executed.
The `config` folder contains some of the configuration files that were used in the paper experiments. A brief description
of the main flags is given below.

#### <a name="general-flags"></a>General flags
* `environments` - A list with the environments used for learning. The file `run_isa.py` contains
a list of the supported environments (e.g., `coffee`, `coffee-mail`, ...).

* `folder_names` - A list of the folders where the exported data (rewards,
automata tasks/solutions/plots, episodes where an automaton is learned). There must be one per environment.

* `use_seed` - Whether to use a seed to generate the environments.

* `num_tasks` - Number of generated tasks for which the learned automata
must generalize and a policy must be learned.

* `debug` - Whether to show information messages during execution.

* `train_model` - Whether to train the model or not.

#### <a name="rl-flags"></a>Reinforcement learning flags
* `num_episodes` - Number of episodes during which the agent trains.

* `max_episode_length` - Maximum number of steps per episode.

* `learning_rate` - Q-Learning learning rate.

* `exploration_rate` - Epsilon factor for epsilon-greedy policy.

* `discount_rate` - Discount factor for the Q-Learning updates.

* `is_tabular_case` - Whether to use tabular Q-Learning or, if false, Deep Q-Learning.

* `use_reward_shaping` - Whether to use the automata for reward shaping.

#### <a name="automata-flags"></a>Automata learning flags
* `interleaved_automaton_learning` - Whether to use ISA (learn an automaton
while a policy is learned). If `false`, the complete automaton is used from the first step
as in [[Toro Icarte et al., 2018]](#references), and the flags below are not used.

* `ilasp_timeout`: Seconds that ILASP has for finding a solution.

* `ilasp_version`: Which version of ILASP to use (`1`, `2`, `2i` or `3`).

* `use_compressed_traces`: Whether to use compressed observation traces or not.

* `starting_num_states`: Number of states to use for learning the automaton (minimum 3 for
    the initial, accepting and rejecting states).

* `use_restricted_observables`: Whether to use only the observables that
    the task requires or use all of them.

* `max_disjunction_size`: Maximum number of conditions that a disjunction can have (i.e.,
    maximum number of edges between two states).

* `learn_acyclic_graph`: Whether constraints that force the learned automata to be
    acyclic are imposed or not.

### <a name="plot-results"></a> Plot the learning curves
To plot the learning curves, you can run the `plot_curves.py` script as follows:
```
python3 plot_curves.py [--plot_individual_tasks] config num_tasks num_runs
```
where:
* `config` - A JSON configuration file with the paths to the folders generated by the `run_isa.py` script. More details
below.
* `num_tasks` - Number of tasks used in the `run_isa.py` script (see flags for that file).
* `num_runs` - Number of times that the `run_isa.py` has been run with the same configuration.
* `plot_individual_tasks` - Whether to plot the learning curves for each of the tasks used to learn the automaton. If not
specified, only the average curve across tasks and runs will be shown.

The configuration file is formed by a list of objects: one for each curve. Each object has three fields:
* `label` - The name that will appear in the legend.
* `folders` - A list of paths to the folders where the results of the algorithm execution are stored. There is a folder for each run.
* `colour` - The colour of the learning curve in hexadecimal format.

The following is an example of a JSON configuration file:
```json
[
  {
    "label": "ISA (S)",
    "folders": [
      "coffee-single-run1",
      "coffee-single-run2"
    ],
    "colour": "#AAAA00"
  },
  {
    "label": "ISA (S+R)",
    "folders": [
      "coffee-single-rs-run1",
      "coffee-single-rs-run2"    
    ],
    "colour": "#EEDD88"
  }
]
```

Then, if the number of tasks is 100 and given that the number of runs is 2, the script would be executed as:
```
python3 plot_curves.py config.json 100 2
```

### <a name="collect-stats"></a> Collect learning statistics
The `collect_stats.py` script produces JSON files containing a summary of the results obtained from the
folders generated by `run_isa.py`. The script can be run as follows:
```
python3 collect_stats.py config_file output_file
```

The configuration file contains a JSON object with one item per setting. Each item consists of a list of result folders
generated by `run_isa.py`. There should be one folder for each run of that setting. The following is an example file:
```json
{
  "ISA (S)": [
    "coffee-single-run1",
    "coffee-single-run2"
  ],
  "ISA (S+R)": [
    "coffee-single-rs-run1",
    "coffee-single-rs-run2"
  ]
}
```

The output is a JSON file with the following fields for each of the settings in the input. All of them provide the
average and the standard error across runs except where noted.
* `num_examples` - Total number of examples.
* `num_pos_examples` - Number of positive examples.
* `num_neg_examples` - Number negative examples.
* `num_inc_examples` - Number incomplete examples.
* `absolute_time` - Total running time (reinforcement learning + automata learning).
* `ilasp_total_time` - ILASP running time.
* `ilasp_percent_time` - Fraction of time during which ILASP runs with respect to ISA's total running time.
* `ilasp_last_time` - ILASP running time for the last automaton.
* `avg_time_per_automaton` - Average and standard error of the time needed for each intermediate automaton solution.
* `max_example_length` - Length of the longest example across runs.
* `example_length` - Average and standard deviation of the example length taking into account the examples from all tasks.

## <a name="references"></a>References
* Furelos-Blanco, D.; Law, M.; Russo, A.; Broda, K.; and Jonsson, A. 2020. [_Induction of Subgoal Automata for Reinforcement Learning_](https://arxiv.org/abs/1911.13152). Proceedings of the 34th AAAI Conference on Artificial Intelligence.
* Toro Icarte, R.; Klassen, T. Q.; Valenzano, R. A.; and McIlraith, S. A. 2018. [_Using Reward Machines for High-Level Task Specification and Decomposition in Reinforcement Learning_](http://proceedings.mlr.press/v80/icarte18a.html). Proceedings of the 35th International Conference on Machine Learning.
