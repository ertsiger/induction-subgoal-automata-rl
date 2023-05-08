# Induction of Subgoal Automata for Reinforcement Learning
Implementation of the ISA (Induction of Subgoal Automata) algorithm presented in [[Furelos-Blanco et al., 2020]](#references) and [[Furelos-Blanco et al., 2021]](#references).

1. [Installation](#installation)
    1. [Install Python packages](#install-python)
    1. [Install `ILASP` and `clingo` binaries](#install-ilasp-clingo)
    1. [Install additional dependencies](#install-additional-dependencies)
1. [Run the algorithm](#running-isa)
1. [Generation of configuration files](#config-file-generator)
1. [Plot the learning curves](#plot-results)
1. [Collect learning statistics](#collect-stats)
1. [Reproducibility of the paper results](#result-reproducibility)
1. [Citation](#citation)
1. [References](#references)

## <a name="installation"></a>Installation
The code only runs on Linux or MacOS computers with Python 3. Firstly, you have to download the repository which can be
done with the following command.
```
git clone https://github.com/ertsiger/induction-subgoal-automata-rl.git
```

The following subsections describe the steps for installing the required [Python packages](#install-python) and the [binaries](#install-ilasp-clingo) related to the
Inductive Logic Programming system we use to learn the automata. Other dependencies you may need to install are listed [here](#install-additional-dependencies).

### <a name="install-python"></a> Install Python packages
To install the required Python packages to run our code, you can use `pip` with the following command:

```
cd induction-subgoal-automata-rl
pip install -r requirements.txt
```

Note that one of the requirements is the package in the `gym-subgoal-automata` repository ([link](https://github.com/ertsiger/gym-subgoal-automata)).
We use the environments implemented in that repository. 

We recommend you to use a [virtual environment](https://docs.python.org/3/tutorial/venv.html) 
since the requirements of our installation may affect your current installation. In our case, we used an Anaconda3
installation and created an environment with Python 3.6.9.

### <a name="install-ilasp-clingo"></a> Install `ILASP` and `clingo` binaries
The cloned repository does not include the binaries for the ILASP inductive logic programming system. Therefore, you have
to download the binaries from the following websites and then copy the 
`ILASP` and `clingo` binaries into the `bin` folder:
* [ILASP v3.6.0](https://github.com/marklaw/ILASP-releases/releases/tag/v3.6.0)
* [clingo 5.4.0](https://github.com/potassco/clingo/releases/tag/v5.4.0)

Alternatively, you can run the `install_binaries.sh`, which will
download and put the files in the `bin` folder for you. If you are using MacOS, you may need to install `wget` first using `brew install wget`.
```
cd induction-subgoal-automata-rl
./install_binaries.sh
```

### <a name="install-additional-dependencies"></a> Install additional dependencies
#### Graphviz
The learned subgoal automata are exported to `.png` using Graphviz. You can follow the instructions in the [official webpage](https://graphviz.org/download/) to install it.

#### SDL2
When installing the [Python packages](#install-python), you may experience the error `fatal error: 'SDL.h' file not found`. To resolve this, you must install SDL2 (Simple Direct Media Layer 2):
* Ubuntu [[link](https://stackoverflow.com/questions/10488775/sdl-h-no-such-file-or-directory-found-when-compiling)]: `sudo apt-get install libsdl2-dev`
* MacOS [[link](https://stackoverflow.com/questions/45992243/pip-install-pygame-sdl-h-file-not-found)]: `brew install sdl sdl_image sdl_mixer sdl_ttf portmidi`

If you use a Conda environment, the following command can also be used to avoid a system-wide installation: `conda install -c conda-forge sdl2` [[link](https://anaconda.org/conda-forge/sdl2)].

#### MacOS dependencies
The code invoking `ILASP` relies on the `timeout` command, which is not available by default in MacOS systems. To install it, you can run:
```
brew install coreutils
```

## <a name="running-isa"></a>Running the algorithm
The ISA algorithm can be executed easily by running the `run_isa.py` script:
```
python3 run_isa.py algorithm config_file
```
where
* `algorithm` can be either `hrl` or `qrm`; and
* `config_file` is the path to a JSON configuration file containing the settings with which ISA is executed.

We provide a script that automatically generates a set of configuration files using some default parameters as well as
other parameters specified through the command line (see [this section](#config-file-generator)).
Alternatively, we provide example configuration files in the `config/examples` folder. In case you want to manually
modify the configuration files, you can find more details about them in the following code files:
* [`LearningAlgorithm`](src/reinforcement_learning/learning_algorithm.py) - Basic learning algorithm fields.
* [`ISAAlgorithmBase`](src/reinforcement_learning/isa_base_algorithm.py) - Specifies basic scheme for interleaving.
Inherits from `Learning Algorithm`.
* [`ISAAlgorithmHRL`](src/reinforcement_learning/isa_hrl_algorithm.py) - Interleaving algorithm using Hierarchical
Reinforcement Learning (HRL). Inherits from `ISAAlgorithmBase`.
* [`ISAAlgorithmQRM`](src/reinforcement_learning/isa_qrm_algorithm.py) - Interleaving algorithm using Q-Learning for
Reward Machines (QRM). Inherits from `ISAAlgorithmBase`.

## <a name="config-file-generator"></a>Generation of configuration files
The [`config/config_generator`](src/config/config_generator.py) script allows to create files similar to the ones we used 
in our experiments. It is executed as follows:
```
python -m config.config_generator [--maximum_episode_length MAXIMUM_EPISODE_LENGTH]
                                  [--num_tasks NUM_TASKS] [--seed SEED]
                                  [--interleaved_learning]
                                  [--use_restricted_observables]
                                  [--max_disj_size MAX_DISJ_SIZE] [--learn_acyclic]
                                  [--symmetry_breaking_method SYMMETRY_BREAKING_METHOD]
                                  [--use_compressed_traces]
                                  [--ignore_empty_observations]
                                  [--prioritize_optimal_solutions]
                                  [--rl_guidance_method RL_GUIDANCE_METHOD]
                                  [--avoid_learning_negative_only_formulas]
                                  [--environments ENVIRONMENTS [ENVIRONMENTS ...]]
                                  [--use_gpu] [--multitask]
                                  domain algorithm num_runs root_experiments_path
                                  experiment_folder_name
```
where:
* `domain` can be `officeworld`, `craftworld` or `waterworld`.
* `algorithm` can be `hrl` or `qrm`.
* `num_runs` is the number of different runs (one folder for each run will be generated, each using a different seed).
* `root_experiments_path` is the path where the experiment folder (below) is created.
* `experiment_folder_name` is the name of the folder containing the experiments.
* `--maximum_episode_length` is used to specify the maximum number of steps per episode.
* `--num_tasks` is the size of the MDP set used to learn the automata.
* `--seed` is the starting seed used to randomly initialize each of the tasks in the MDP set (the first task uses this
value, the second task uses this value plus one, ...).
* `--interleaved_learning` indicates whether an automaton is learned in an interleaved manner (if false, the target
automaton for the tasks is used and no automata learning occurs).
* `--use_restricted_observables` indicates whether only the observables relevant to the task at hand should be used;
* `--max_disj_size` is the maximum number of edges from one state to another.
* `--learn_acyclic` indicates whether to add constraints to enforce the automaton to be acyclic.
* `--symmetry_breaking_method` is the name of the symmetry breaking method (if it is not specific, no symmetry breaking
method is used):
    * `bfs` - direct translation from the SAT encoding into ASP.
    * `bfs-alternative` - ASP encoding of the symmetry breaking method alternative to the direct translation from SAT.
    * `increasing-path` - Method used in the AAAI-20 paper (only works for acyclic automata). 
* `--use_compressed_traces` compresses contiguous equal observations in a trace into a single observation.
* `--ignore_empty_observations` removes empty observations from a trace.
* `--prioritize_optimal_solutions` adds weak constraints to rank equall optimal solutions found by ILASP (still experimental).
* `--rl_guidance_method` is the name of the method uses to provide extra reward signals to the learner (if left empty,
no method is used):
    * In the case of `hrl`, use the name `pseudorewards` (there is only one method).
    * In the case of `qrm`, you can use `max_distance` (currently working for acyclic graphs only) and `min_distance`.
* `--avoid_learning_negative_only_formulas` indicates whether to avoid learning formulas formed only by negated observables.
* `--environments` is a list of the environments for which the configuration files are generated. You should use the aliases
shown in [`run_isa.py`](src/run_isa.py) (e.g., `coffee`, `coffee-mail`). They all must correspond to the `domain` specified before.
* `--use_gpu` indicates whether to use the GPU when deep learning is applied (e.g., in `WaterWorld` tasks).
* `--multitask` indicates whether to enable the multi-task setting (i.e., learn a policy and an automaton for each environment).

## <a name="plot-results"></a> Plot the learning curves
To plot the learning curves, you can run the `plot_curves.py` script as follows:
```
plot_curves.py [--max_episode_length MAX_EPISODE_LENGTH]
               [--plot_task_curves] [--use_greedy_traces]
               [--greedy_evaluation_frequency GREEDY_EVALUATION_FREQUENCY]
               [--use_tex] [--window_size WINDOW_SIZE]
               [--plot_title PLOT_TITLE]
               config num_tasks num_runs num_episodes
```
where:
* `config` is a JSON configuration file with the paths to the folders generated by the `run_isa.py` script. More details
below.
* `num_tasks` is the number of tasks specified in the JSON configuration file given to the `run_isa.py` script.
* `num_runs` is number of runs specified in the JSON configuration file given to the `run_isa.py` script.
* `num_episodes` is the number of episodes to plot.
* `--max_episode_length` is the maximum number of steps that can be run per episode (it should be equal to the value given
in the JSON configuration file).
* `--plot_task_curves` indicates whether to plot the learning curves for each of the tasks used to learn the automaton. 
If not specified, only the average curve across tasks and runs will be shown.
* `--use_greedy_traces` indicates whether to use the evaluations of the greedy policy to plot the curves. Else, the
results obtained by the behavior policy are used (in our case, epsilon-greedy). This will only work if greedy evaluation
was enabled in ISA's execution.
* `--greedy_evaluation_frequency` indicates every how many episodes is the greedy policy evaluated (should have the same
value as in the configuration file of ISA). 
* `--use_tex` indicates whether to use TeX to label the axis and the labels in the plot.
* `--window_size` is the size of the sliding window that averages the reward and the number of steps to make curves smoother.
* `--plot_title` is the title of the plot.

The configuration file is formed by a list of objects: one for each curve. Each object has three fields:
* `label` - The name that will appear in the legend.
* `folders` - A list of paths to the folders where the results of the algorithm execution are stored. There is a folder for each run.
* `colour` - The colour of the learning curve in hexadecimal format.

The following is an example of a JSON configuration file:
```json
[
  {
    "label": "HRL",
    "folders": [
      "hrl-coffee-run1",
      "hrl-coffee-run2"
    ],
    "colour": "#AAAA00"
  },
  {
    "label": "HRL-G",
    "folders": [
      "hrl-g-coffee-run1",
      "hrl-g-coffee-run2"    
    ],
    "colour": "#EEDD88"
  }
]
```

Then, if the number of tasks is 100, the number of runs is 2 and we want to plot 1000 episodes, the script would be executed as:
```
python -m plot_utils.plot_curves.py config.json 100 2 1000
```

## <a name="collect-stats"></a> Collect learning statistics
The `collect_stats.py` script produces JSON files containing a summary of the results obtained from the
folders generated by `run_isa.py`. The script can be run as follows:
```
python -m result_processing.collect_stats.py config_file output_file
```

The configuration file contains a JSON object with one item per setting. Each item consists of a list of result folders
generated by `run_isa.py`. There should be one folder for each run of that setting. The following is an example file:
```json
{
  "HRL": [
    "hrl-coffee-run1",
    "hrl-coffee-run2"
  ],
  "HRL-G": [
    "hrl-g-coffee-run1",
    "hrl-g-coffee-run2"
  ]
}
```

The output is a JSON file with the following fields for each of the settings in the input. All of them provide the
average and the standard error across runs except where noted.
* `num_examples` - Total number of examples.
* `num_goal_examples` - Number of goal examples.
* `num_dend_examples` - Number dead-end examples.
* `num_inc_examples` - Number incomplete examples.
* `absolute_time` - Total running time (reinforcement learning + automata learning).
* `num_completed_runs` - Number of runs that have been successfully completed (i.e., without timeouts).
* `num_found_goal_runs` - Number of runs for which the goal has been found at least once (i.e., automata learning has happened).
* `ilasp_total_time` - ILASP running time.
* `ilasp_percent_time` - Fraction of time during which ILASP runs with respect to ISA's total running time.
* `ilasp_last_time` - ILASP running time for the last automaton.
* `avg_time_per_automaton` - Average and standard error of the time needed for each intermediate automaton solution.
* `max_example_length` - Length of the longest example across runs.
* `example_length` - Average and standard deviation of the example length taking into account the examples from all tasks.
* `example_length_goal` - Average and standard deviation of the goal example length taking into account the examples from all tasks. 
* `example_length_dend` - Average and standard deviation of the dead-end example length taking into account the examples from all tasks. 
* `example_length_inc` - Average and standard deviation of the incomplete example length taking into account the examples from all tasks. 

## <a name="result-reproducibility"></a>Reproducibility of the paper results
The experiments ran on 3.40GHz Intel Core i7-6700 processors using Python 3.6.9. The `requirements.txt` file contains 
the versions of the required Python packages.

We provide scripts that generate all the experiments described in the paper as well as scripts that generate plots and 
statistics reports from them. The folder `paper-experiments` in the root of the repository has a folder called
`results` containing JSON files for generating plots and reports. We now describe how to create the experiments and then
generate results out from them.

To generate the experiments, just go to the `src` folder and run the following command:
```
sh config/generate_experiments.sh
```
This will generate an `experiments` folder inside the `paper-experiments` folder introduced before. This folder contains
the configuration files of all the experiments referenced in the paper. There are quite a lot of experiments to run, so
you might want to comment some of the lines at the bottom of the `config/generate_experiments.sh` file. Furthermore, take
into account that checkpointing is enabled, so a lot of heavy files will be generated during the execution of the experiments.

Once all the experiments have run, you can run the following command:
```
sh config/get_results.sh
```
This will fill the `paper-experiments/results` folder with plots and statistics reports. Again, you can comment some of 
the lines at the bottom of the file if not all experiments have run. However, note that the results for some experiments 
depend on the results of others. For example, HRL with guidance is not rerun in the set of experiments where the different 
RL algorithms are evaluated in `OfficeWorld`.

Since it is costly to run all these experiments, we recommend you to use the configuration generator introduced [here](#config-file-generator)
if you want to test the method with some specific experiments.

## <a name="citation"></a>Citation
If you find this repository useful in your work, please use the following citation:
```
@article{FurelosBlancoLJBR21,
  author       = {Daniel Furelos{-}Blanco and
                  Mark Law and
                  Anders Jonsson and
                  Krysia Broda and
                  Alessandra Russo},
  title        = {{Induction and Exploitation of Subgoal Automata for Reinforcement Learning}},
  journal      = {J. Artif. Intell. Res.},
  volume       = {70},
  pages        = {1031--1116},
  year         = {2021}
}
```

## <a name="references"></a>References
* Toro Icarte, R.; Klassen, T. Q.; Valenzano, R. A.; and McIlraith, S. A. 2018. [_Using Reward Machines for High-Level Task Specification and Decomposition in Reinforcement Learning_](http://proceedings.mlr.press/v80/icarte18a.html). Proceedings of the 35th International Conference on Machine Learning.
* Furelos-Blanco, D.; Law, M.; Russo, A.; Broda, K.; and Jonsson, A. 2020. [_Induction of Subgoal Automata for Reinforcement Learning_](https://doi.org/10.1609/aaai.v34i04.5802). Proceedings of the 34th AAAI Conference on Artificial Intelligence.
* Furelos-Blanco, D.; Law, M.; Jonsson, A.; Broda, K.; and Russo, A. 2021. [_Induction and Exploitation of Subgoal Automata for Reinforcement Learning_](https://jair.org/index.php/jair/article/view/12372). J. Artif. Intell. Res., 70, 1031-1116.

