#!/usr/bin/env bash

get_num_episodes() {
  if [ "$1" = "coffee" ]; then
    echo 1000
  elif [ "$1" = "coffee-mail" ]; then
    echo 2500
  elif [ "$1" = "visit-abcd" ]; then
    echo 8000
  elif [ "$1" = "coffee-drop" ]; then
    echo 1000
  elif [ "$1" = "coffee-mail-drop" ]; then
    echo 2500
  elif [ "$1" = "coffee-or-mail" ]; then
    echo 1000
  elif [ "$1" = "make-plank" ]; then
    echo 10000
  elif [ "$1" = "make-stick" ]; then
    echo 10000
  elif [ "$1" = "make-cloth" ]; then
    echo 10000
  elif [ "$1" = "make-rope" ]; then
    echo 10000
  elif [ "$1" = "make-shears" ]; then
    echo 10000
  elif [ "$1" = "make-bridge" ]; then
    echo 10000
  elif [ "$1" = "make-bed" ]; then
    echo 10000
  elif [ "$1" = "make-axe" ]; then
    echo 10000
  elif [ "$1" = "get-gold" ]; then
    echo 10000
  elif [ "$1" = "get-gem" ]; then
    echo 10000
  elif [ "$1" = "water-rgb" ]; then
    echo 50000
  elif [ "$1" = "water-rg-b" ]; then
    echo 50000
  elif [ "$1" = "water-rgbc" ]; then
    echo 50000
  fi
}

get_plot_name() {
  if [ "$1" = "coffee" ]; then
    echo "Coffee"
  elif [ "$1" = "coffee-mail" ]; then
    echo "CoffeeMail"
  elif [ "$1" = "visit-abcd" ]; then
    echo "VisitABCD"
  elif [ "$1" = "coffee-drop" ]; then
    echo "CoffeeDrop"
  elif [ "$1" = "coffee-mail-drop" ]; then
    echo "CoffeeMailDrop"
  elif [ "$1" = "coffee-or-mail" ]; then
    echo "CoffeeOrMail"
  elif [ "$1" = "make-plank" ]; then
    echo "MakePlank"
  elif [ "$1" = "make-stick" ]; then
    echo "MakeStick"
  elif [ "$1" = "make-cloth" ]; then
    echo "MakeCloth"
  elif [ "$1" = "make-rope" ]; then
    echo "MakeRope"
  elif [ "$1" = "make-shears" ]; then
    echo "MakeShears"
  elif [ "$1" = "make-bridge" ]; then
    echo "MakeBridge"
  elif [ "$1" = "make-bed" ]; then
    echo "MakeBed"
  elif [ "$1" = "make-axe" ]; then
    echo "MakeAxe"
  elif [ "$1" = "get-gold" ]; then
    echo "GetGold"
  elif [ "$1" = "get-gem" ]; then
    echo "GetGem"
  elif [ "$1" = "water-rgb" ]; then
    echo "RGB"
  elif [ "$1" = "water-rg-b" ]; then
    echo "RG-B"
  elif [ "$1" = "water-rgbc" ]; then
    echo "RGBC"
  fi
}

BASE_DIR="/data/Software/learning-automata-rl/results_config/jmlr"
PLOT_CMD="python -m plot_utils.plot_curves"
STATS_CMD="python -m result_processing.collect_stats"
NUM_RUNS=20

DEFAULT_OFFICEWORLD_NUM_STEPS=250
DEFAULT_OFFICEWORLD_GREEDY_EVALUATION_FREQUENCY=1
DEFAULT_OFFICEWORLD_DATASET_SIZE=50
DEFAULT_OFFICEWORLD_WINDOW_SIZE=1
DEFAULT_OFFICEWORLD_TASKS="coffee coffee-mail visit-abcd coffee-drop coffee-mail-drop coffee-or-mail"

DEFAULT_CRAFTWORLD_NUM_STEPS=250
DEFAULT_CRAFTWORLD_GREEDY_EVALUATION_FREQUENCY=1
DEFAULT_CRAFTWORLD_DATASET_SIZE=100
DEFAULT_CRAFTWORLD_WINDOW_SIZE=1
DEFAULT_CRAFTWORLD_TASKS="make-plank make-stick make-cloth make-rope make-shears make-bridge make-bed make-axe get-gold get-gem"

DEFAULT_WATERWORLD_NUM_STEPS=150
DEFAULT_WATERWORLD_GREEDY_EVALUATION_FREQUENCY=500
DEFAULT_WATERWORLD_DATASET_SIZE=1
DEFAULT_WATERWORLD_WINDOW_SIZE=1000
DEFAULT_WATERWORLD_TASKS="water-rg-b water-rgb water-rgbc"

# experiment folder names for officeworld
OFFICEWORLD_ROOT_FOLDER="officeworld-experiments"
DATASETS_AND_STEPS_FOLDER=$OFFICEWORLD_ROOT_FOLDER/"01_datasets_and_steps"
RESTRICTED_OBSERVABLES_FOLDER=$OFFICEWORLD_ROOT_FOLDER/"02_restricted_observables"
COMPRESSION_FOLDER=$OFFICEWORLD_ROOT_FOLDER/"03_uncompressed_traces"
CYCLIC_AUTOMATA_FOLDER=$OFFICEWORLD_ROOT_FOLDER/"04_cyclic_automata"
NUM_DISJUNCTIONS_FOLDER=$OFFICEWORLD_ROOT_FOLDER/"05_num_disjunctions"
NO_SYMMETRY_BREAKING_FOLDER=$OFFICEWORLD_ROOT_FOLDER/"06_no_symmetry_breaking"
RL_ALGORITHMS_FOLDER=$OFFICEWORLD_ROOT_FOLDER/"07_rl_algorithms_learning"
RL_ALGORITHMS_HANDCRAFTED_FOLDER=$OFFICEWORLD_ROOT_FOLDER/"08_rl_algorithms_handcrafted"

# experiment folder names for craftworld experiments
CRAFTWORLD_ROOT_FOLDER="craftworld-experiments"
RL_ALGORITHMS_CRAFTWORLD_FOLDER=$CRAFTWORLD_ROOT_FOLDER/"01_rl_algorithms_learning"
RL_ALGORITHMS_HANCRAFTED_CRAFTWORLD_FOLDER=$CRAFTWORLD_ROOT_FOLDER/"02_rl_algorithms_handcrafted"

# experiment folder names for waterworld experiments
WATERWORLD_ROOT_FOLDER="waterworld-experiments"
RL_ALGORITHMS_WATERWORLD_FOLDER=$WATERWORLD_ROOT_FOLDER/"01_rl_algorithms_learning"
RL_ALGORITHMS_HANCRAFTED_WATERWORLD_FOLDER=$WATERWORLD_ROOT_FOLDER/"02_rl_algorithms_handcrafted"

get_interleaving_rl_algorithms_experiment_results() {
  tasks=$1
  domain_folder=$2
  dataset_size=$3
  max_episode_length=$4
  greedy_evaluation_frequency=$5
  window_size=$6

  for task in $tasks
  do
    folder=$BASE_DIR/$domain_folder/$task
    if [ -d $folder ]
    then
      num_episodes=$(get_num_episodes $task)
      plot_name=$(get_plot_name $task)
      $PLOT_CMD $folder/plot.json $dataset_size $NUM_RUNS $num_episodes -g --greedy_evaluation_frequency $greedy_evaluation_frequency --max_episode_length $max_episode_length --use_tex -w $window_size -t "\textsc{$plot_name}"
      $STATS_CMD $folder/stats.json $folder/stats_out.json
    fi
  done
}

get_handcrafted_automata_experiment_results() {
  tasks=$1
  domain_folder=$2
  dataset_size=$3
  max_episode_length=$4
  greedy_evaluation_frequency=$5
  window_size=$6

  for task in $tasks
  do
    folder=$BASE_DIR/$domain_folder/$task
    if [ -d $folder ]
    then
      num_episodes=$(get_num_episodes $task)
      plot_name=$(get_plot_name $task)
      $PLOT_CMD $folder/plot_hrl.json $dataset_size $NUM_RUNS $num_episodes -g --greedy_evaluation_frequency $greedy_evaluation_frequency --max_episode_length $max_episode_length --use_tex -w $window_size -t "\textsc{$plot_name}"
      $PLOT_CMD $folder/plot_qrm.json $dataset_size $NUM_RUNS $num_episodes -g --greedy_evaluation_frequency $greedy_evaluation_frequency --max_episode_length $max_episode_length --use_tex -w $window_size -t "\textsc{$plot_name}"
    fi
  done
}

get_officeworld_dataset_and_steps_experiment_results() {
  for task in coffee coffee-mail visit-abcd
  do
    folder=$BASE_DIR/$DATASETS_AND_STEPS_FOLDER/$task
    plot_folder=$folder/plot
    for dataset in 1 2
    do
      for num_tasks in 10 50 100
      do
        input_file=$plot_folder/d$dataset-$num_tasks.json
        num_episodes=$(get_num_episodes $task)
        plot_name=$(get_plot_name $task)
        $PLOT_CMD $input_file $num_tasks $NUM_RUNS $num_episodes -g --use_tex -w 1 -t "\textsc{$plot_name} (\$\mathcal{D}^{$num_tasks}_{$dataset}\$)"
      done
    done
    $STATS_CMD $folder/stats.json $folder/stats_out.json
  done
}

get_officeworld_experiment_results() {
  get_interleaving_rl_algorithms_experiment_results "$DEFAULT_OFFICEWORLD_TASKS" $1 $DEFAULT_OFFICEWORLD_DATASET_SIZE $DEFAULT_OFFICEWORLD_NUM_STEPS $DEFAULT_OFFICEWORLD_GREEDY_EVALUATION_FREQUENCY $DEFAULT_OFFICEWORLD_WINDOW_SIZE
}

get_officeworld_restricted_observables_experiment_results() {
  get_officeworld_experiment_results $RESTRICTED_OBSERVABLES_FOLDER
}

get_officeworld_uncompressed_traces_experiment_results() {
  get_officeworld_experiment_results $COMPRESSION_FOLDER
}

get_officeworld_cyclic_automata_experiment_results() {
  get_officeworld_experiment_results $CYCLIC_AUTOMATA_FOLDER
}

get_officeworld_num_disjunctions_experiment_results() {
  get_officeworld_experiment_results $NUM_DISJUNCTIONS_FOLDER
}

get_officeworld_symmetry_breaking_experiment_results() {
  get_officeworld_experiment_results $NO_SYMMETRY_BREAKING_FOLDER
}

get_officeworld_interleaving_rl_algorithms_experiment_results() {
  get_officeworld_experiment_results $RL_ALGORITHMS_FOLDER

  forgetting_folder=$BASE_DIR/$RL_ALGORITHMS_FOLDER/forgetting-effect
  $PLOT_CMD $forgetting_folder/plot.json $DEFAULT_OFFICEWORLD_DATASET_SIZE 1 1000 -g --greedy_evaluation_frequency $DEFAULT_OFFICEWORLD_GREEDY_EVALUATION_FREQUENCY --max_episode_length $DEFAULT_OFFICEWORLD_NUM_STEPS --use_tex -w $DEFAULT_OFFICEWORLD_WINDOW_SIZE
}

get_officeworld_handcrafted_automata_experiment_results() {
  get_handcrafted_automata_experiment_results "$DEFAULT_OFFICEWORLD_TASKS" $RL_ALGORITHMS_HANDCRAFTED_FOLDER $DEFAULT_OFFICEWORLD_DATASET_SIZE $DEFAULT_OFFICEWORLD_NUM_STEPS $DEFAULT_OFFICEWORLD_GREEDY_EVALUATION_FREQUENCY $DEFAULT_OFFICEWORLD_WINDOW_SIZE
}

get_craftworld_interleaving_rl_algorithms_experiment_results() {
  get_interleaving_rl_algorithms_experiment_results "$DEFAULT_CRAFTWORLD_TASKS" $RL_ALGORITHMS_CRAFTWORLD_FOLDER $DEFAULT_CRAFTWORLD_DATASET_SIZE $DEFAULT_CRAFTWORLD_NUM_STEPS $DEFAULT_CRAFTWORLD_GREEDY_EVALUATION_FREQUENCY $DEFAULT_CRAFTWORLD_WINDOW_SIZE
}

get_craftworld_handcrafted_automata_experiment_results() {
  get_handcrafted_automata_experiment_results "$DEFAULT_CRAFTWORLD_TASKS" $RL_ALGORITHMS_HANCRAFTED_CRAFTWORLD_FOLDER $DEFAULT_CRAFTWORLD_DATASET_SIZE $DEFAULT_CRAFTWORLD_NUM_STEPS $DEFAULT_CRAFTWORLD_GREEDY_EVALUATION_FREQUENCY $DEFAULT_CRAFTWORLD_WINDOW_SIZE
}

get_waterworld_interleaving_rl_algorithms_experiment_results() {
  get_interleaving_rl_algorithms_experiment_results "$DEFAULT_WATERWORLD_TASKS" $RL_ALGORITHMS_WATERWORLD_FOLDER $DEFAULT_WATERWORLD_DATASET_SIZE $DEFAULT_WATERWORLD_NUM_STEPS $DEFAULT_WATERWORLD_GREEDY_EVALUATION_FREQUENCY $DEFAULT_WATERWORLD_WINDOW_SIZE
}

get_waterworld_handcrafted_automata_experiment_results() {
  get_handcrafted_automata_experiment_results "$DEFAULT_WATERWORLD_TASKS" $RL_ALGORITHMS_HANCRAFTED_WATERWORLD_FOLDER $DEFAULT_WATERWORLD_DATASET_SIZE $DEFAULT_WATERWORLD_NUM_STEPS $DEFAULT_WATERWORLD_GREEDY_EVALUATION_FREQUENCY $DEFAULT_WATERWORLD_WINDOW_SIZE
}

# officeworld experiments results
get_officeworld_dataset_and_steps_experiment_results
get_officeworld_restricted_observables_experiment_results
get_officeworld_uncompressed_traces_experiment_results
get_officeworld_cyclic_automata_experiment_results
get_officeworld_num_disjunctions_experiment_results
get_officeworld_symmetry_breaking_experiment_results
get_officeworld_interleaving_rl_algorithms_experiment_results
get_officeworld_handcrafted_automata_experiment_results

# craftworld experiments results
get_craftworld_interleaving_rl_algorithms_experiment_results
get_craftworld_handcrafted_automata_experiment_results

# waterworld experiments results
get_waterworld_interleaving_rl_algorithms_experiment_results
get_waterworld_handcrafted_automata_experiment_results
