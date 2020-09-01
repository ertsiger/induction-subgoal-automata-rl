#!/usr/bin/env bash

CMD="python -m config.config_generator"
ROOT_EXPERIMENTS_PATH="/vol/bitbucket/df618"
NUM_RUNS=20

DEFAULT_OFFICEWORLD_NUM_STEPS=250
DEFAULT_OFFICEWORLD_DATASET_SIZE=50

DEFAULT_CRAFTWORLD_NUM_STEPS=250
DEFAULT_CRAFTWORLD_DATASET_SIZE=100
DEFAULT_CRAFTWORLD_ENVIRONMENTS="make-plank make-stick make-cloth make-rope make-bridge make-bed make-axe make-shears get-gold get-gem"

DEFAULT_WATERWORLD_NUM_STEPS=150
DEFAULT_WATERWORLD_DATASET_SIZE=1
DEFAULT_WATERWORLD_ENVIRONMENTS="water-rg-b water-rgb water-rgbc"

DEFAULT_SEED=0

# experiment folder names for officeworld experiments
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

build_dataset_and_steps_experiments() {
  # Different combinations of dataset sizes and number of steps
  for seed in 0 100
  do
    for dataset_size in 10 50 100
    do
      for num_steps in 100 250 500
      do
        $CMD officeworld hrl $NUM_RUNS $ROOT_EXPERIMENTS_PATH $DATASETS_AND_STEPS_FOLDER/hrl-officeworld-$seed-$dataset_size-$num_steps -m $num_steps -t $dataset_size --seed $seed -iacen -g pseudorewards -s bfs-alternative --timed
      done
    done
  done
}

build_restricted_observables_experiments() {
  $CMD officeworld hrl $NUM_RUNS $ROOT_EXPERIMENTS_PATH $RESTRICTED_OBSERVABLES_FOLDER -m $DEFAULT_OFFICEWORLD_NUM_STEPS -t $DEFAULT_OFFICEWORLD_DATASET_SIZE --seed $DEFAULT_SEED -iracen -g pseudorewards -s bfs-alternative --timed
}

build_uncompressed_traces_experiments() {
  $CMD officeworld hrl $NUM_RUNS $ROOT_EXPERIMENTS_PATH $COMPRESSION_FOLDER -m $DEFAULT_OFFICEWORLD_NUM_STEPS -t $DEFAULT_OFFICEWORLD_DATASET_SIZE --seed $DEFAULT_SEED -ian -g pseudorewards -s bfs-alternative --timed
}

build_cyclic_automata_experiments() {
  $CMD officeworld hrl $NUM_RUNS $ROOT_EXPERIMENTS_PATH $CYCLIC_AUTOMATA_FOLDER/cyclic -m $DEFAULT_OFFICEWORLD_NUM_STEPS -t $DEFAULT_OFFICEWORLD_DATASET_SIZE --seed $DEFAULT_SEED -icen -g pseudorewards -s bfs-alternative --timed --environments coffee coffee-mail visit-abcd coffee-drop coffee-mail-drop
  $CMD officeworld hrl $NUM_RUNS $ROOT_EXPERIMENTS_PATH $CYCLIC_AUTOMATA_FOLDER/acyclic -m $DEFAULT_OFFICEWORLD_NUM_STEPS -t $DEFAULT_OFFICEWORLD_DATASET_SIZE --seed $DEFAULT_SEED -iacen -g pseudorewards -s bfs-alternative --timed --environments coffee-drop coffee-mail-drop
}

build_num_disjunctions_experiments() {
  # Number of disjunctions (2)
  $CMD officeworld hrl $NUM_RUNS $ROOT_EXPERIMENTS_PATH $NUM_DISJUNCTIONS_FOLDER/d2 -d 2 -m $DEFAULT_OFFICEWORLD_NUM_STEPS -t $DEFAULT_OFFICEWORLD_DATASET_SIZE --seed $DEFAULT_SEED -iacen -g pseudorewards -s bfs-alternative --timed --environments coffee coffee-mail visit-abcd coffee-or-mail
  $CMD officeworld hrl $NUM_RUNS $ROOT_EXPERIMENTS_PATH $NUM_DISJUNCTIONS_FOLDER/d1 -d 1 -m $DEFAULT_OFFICEWORLD_NUM_STEPS -t $DEFAULT_OFFICEWORLD_DATASET_SIZE --seed $DEFAULT_SEED -iacen -g pseudorewards -s bfs-alternative --timed --environments coffee coffee-mail visit-abcd coffee-or-mail
}

build_symmetry_breaking_experiments() {
  # Test symmetry breaking with cyclic and acyclic
  $CMD officeworld hrl $NUM_RUNS $ROOT_EXPERIMENTS_PATH $NO_SYMMETRY_BREAKING_FOLDER/acyclic -m $DEFAULT_OFFICEWORLD_NUM_STEPS -t $DEFAULT_OFFICEWORLD_DATASET_SIZE --seed $DEFAULT_SEED -iacen -g pseudorewards --timed
  $CMD officeworld hrl $NUM_RUNS $ROOT_EXPERIMENTS_PATH $NO_SYMMETRY_BREAKING_FOLDER/cyclic -m $DEFAULT_OFFICEWORLD_NUM_STEPS -t $DEFAULT_OFFICEWORLD_DATASET_SIZE --seed $DEFAULT_SEED -icen -g pseudorewards --timed
}

build_interleaving_rl_algorithms_experiments() {
  # Different RL algorithms while learning automata
  $CMD officeworld hrl $NUM_RUNS $ROOT_EXPERIMENTS_PATH $RL_ALGORITHMS_FOLDER/hrl-not-guided -m $DEFAULT_OFFICEWORLD_NUM_STEPS -t $DEFAULT_OFFICEWORLD_DATASET_SIZE --seed $DEFAULT_SEED -iacen -s bfs-alternative --timed
  $CMD officeworld qrm $NUM_RUNS $ROOT_EXPERIMENTS_PATH $RL_ALGORITHMS_FOLDER/qrm-not-guided -m $DEFAULT_OFFICEWORLD_NUM_STEPS -t $DEFAULT_OFFICEWORLD_DATASET_SIZE --seed $DEFAULT_SEED -iacen -s bfs-alternative --timed
  $CMD officeworld qrm $NUM_RUNS $ROOT_EXPERIMENTS_PATH $RL_ALGORITHMS_FOLDER/qrm-min-distance -m $DEFAULT_OFFICEWORLD_NUM_STEPS -t $DEFAULT_OFFICEWORLD_DATASET_SIZE --seed $DEFAULT_SEED -iacen -s bfs-alternative -g min_distance --timed
  $CMD officeworld qrm $NUM_RUNS $ROOT_EXPERIMENTS_PATH $RL_ALGORITHMS_FOLDER/qrm-max-distance -m $DEFAULT_OFFICEWORLD_NUM_STEPS -t $DEFAULT_OFFICEWORLD_DATASET_SIZE --seed $DEFAULT_SEED -iacen -s bfs-alternative -g max_distance --timed
}

build_rl_algorithms_handcrafted_automata_experiments() {
  # RL algorithms with handcrafted automata
  $CMD officeworld hrl $NUM_RUNS $ROOT_EXPERIMENTS_PATH $RL_ALGORITHMS_HANDCRAFTED_FOLDER/hrl-not-guided -m $DEFAULT_OFFICEWORLD_NUM_STEPS -t $DEFAULT_OFFICEWORLD_DATASET_SIZE --seed $DEFAULT_SEED -acen -s bfs-alternative
  $CMD officeworld hrl $NUM_RUNS $ROOT_EXPERIMENTS_PATH $RL_ALGORITHMS_HANDCRAFTED_FOLDER/hrl-guided -m $DEFAULT_OFFICEWORLD_NUM_STEPS -t $DEFAULT_OFFICEWORLD_DATASET_SIZE --seed $DEFAULT_SEED -acen -s bfs-alternative -g pseudorewards
  $CMD officeworld qrm $NUM_RUNS $ROOT_EXPERIMENTS_PATH $RL_ALGORITHMS_HANDCRAFTED_FOLDER/qrm-not-guided -m $DEFAULT_OFFICEWORLD_NUM_STEPS -t $DEFAULT_OFFICEWORLD_DATASET_SIZE --seed $DEFAULT_SEED -acen -s bfs-alternative
  $CMD officeworld qrm $NUM_RUNS $ROOT_EXPERIMENTS_PATH $RL_ALGORITHMS_HANDCRAFTED_FOLDER/qrm-min-distance -m $DEFAULT_OFFICEWORLD_NUM_STEPS -t $DEFAULT_OFFICEWORLD_DATASET_SIZE --seed $DEFAULT_SEED -acen -s bfs-alternative -g min_distance
  $CMD officeworld qrm $NUM_RUNS $ROOT_EXPERIMENTS_PATH $RL_ALGORITHMS_HANDCRAFTED_FOLDER/qrm-max-distance -m $DEFAULT_OFFICEWORLD_NUM_STEPS -t $DEFAULT_OFFICEWORLD_DATASET_SIZE --seed $DEFAULT_SEED -acen -s bfs-alternative -g max_distance
}

build_craftworld_interleaving_rl_algorithms_experiments() {
  $CMD craftworld hrl $NUM_RUNS $ROOT_EXPERIMENTS_PATH $RL_ALGORITHMS_CRAFTWORLD_FOLDER/hrl-guided -m $DEFAULT_CRAFTWORLD_NUM_STEPS -t $DEFAULT_CRAFTWORLD_DATASET_SIZE --seed $DEFAULT_SEED -iacen -s bfs-alternative --timed -g pseudorewards --environments $DEFAULT_CRAFTWORLD_ENVIRONMENTS
  $CMD craftworld hrl $NUM_RUNS $ROOT_EXPERIMENTS_PATH $RL_ALGORITHMS_CRAFTWORLD_FOLDER/hrl-not-guided -m $DEFAULT_CRAFTWORLD_NUM_STEPS -t $DEFAULT_CRAFTWORLD_DATASET_SIZE --seed $DEFAULT_SEED -iacen -s bfs-alternative --timed --environments $DEFAULT_CRAFTWORLD_ENVIRONMENTS
  $CMD craftworld qrm $NUM_RUNS $ROOT_EXPERIMENTS_PATH $RL_ALGORITHMS_CRAFTWORLD_FOLDER/qrm-not-guided -m $DEFAULT_CRAFTWORLD_NUM_STEPS -t $DEFAULT_CRAFTWORLD_DATASET_SIZE --seed $DEFAULT_SEED -iacen -s bfs-alternative --timed --environments $DEFAULT_CRAFTWORLD_ENVIRONMENTS
  $CMD craftworld qrm $NUM_RUNS $ROOT_EXPERIMENTS_PATH $RL_ALGORITHMS_CRAFTWORLD_FOLDER/qrm-min-distance -m $DEFAULT_CRAFTWORLD_NUM_STEPS -t $DEFAULT_CRAFTWORLD_DATASET_SIZE --seed $DEFAULT_SEED -iacen -s bfs-alternative --timed -g min_distance --environments $DEFAULT_CRAFTWORLD_ENVIRONMENTS
  $CMD craftworld qrm $NUM_RUNS $ROOT_EXPERIMENTS_PATH $RL_ALGORITHMS_CRAFTWORLD_FOLDER/qrm-max-distance -m $DEFAULT_CRAFTWORLD_NUM_STEPS -t $DEFAULT_CRAFTWORLD_DATASET_SIZE --seed $DEFAULT_SEED -iacen -s bfs-alternative --timed -g max_distance --environments $DEFAULT_CRAFTWORLD_ENVIRONMENTS
}

build_craftworld_rl_algorithms_handcrafted_experiments() {
  $CMD craftworld hrl $NUM_RUNS $ROOT_EXPERIMENTS_PATH $RL_ALGORITHMS_HANCRAFTED_CRAFTWORLD_FOLDER/hrl-guided -m $DEFAULT_CRAFTWORLD_NUM_STEPS -t $DEFAULT_CRAFTWORLD_DATASET_SIZE --seed $DEFAULT_SEED -acen -g pseudorewards --environments $DEFAULT_CRAFTWORLD_ENVIRONMENTS
  $CMD craftworld hrl $NUM_RUNS $ROOT_EXPERIMENTS_PATH $RL_ALGORITHMS_HANCRAFTED_CRAFTWORLD_FOLDER/hrl-not-guided -m $DEFAULT_CRAFTWORLD_NUM_STEPS -t $DEFAULT_CRAFTWORLD_DATASET_SIZE --seed $DEFAULT_SEED -acen --environments $DEFAULT_CRAFTWORLD_ENVIRONMENTS
  $CMD craftworld qrm $NUM_RUNS $ROOT_EXPERIMENTS_PATH $RL_ALGORITHMS_HANCRAFTED_CRAFTWORLD_FOLDER/qrm-not-guided -m $DEFAULT_CRAFTWORLD_NUM_STEPS -t $DEFAULT_CRAFTWORLD_DATASET_SIZE --seed $DEFAULT_SEED -acen --environments $DEFAULT_CRAFTWORLD_ENVIRONMENTS
  $CMD craftworld qrm $NUM_RUNS $ROOT_EXPERIMENTS_PATH $RL_ALGORITHMS_HANCRAFTED_CRAFTWORLD_FOLDER/qrm-min-distance -m $DEFAULT_CRAFTWORLD_NUM_STEPS -t $DEFAULT_CRAFTWORLD_DATASET_SIZE --seed $DEFAULT_SEED -acen -g min_distance --environments $DEFAULT_CRAFTWORLD_ENVIRONMENTS
  $CMD craftworld qrm $NUM_RUNS $ROOT_EXPERIMENTS_PATH $RL_ALGORITHMS_HANCRAFTED_CRAFTWORLD_FOLDER/qrm-max-distance -m $DEFAULT_CRAFTWORLD_NUM_STEPS -t $DEFAULT_CRAFTWORLD_DATASET_SIZE --seed $DEFAULT_SEED -acen -g max_distance --environments $DEFAULT_CRAFTWORLD_ENVIRONMENTS
}

build_waterworld_interleaving_rl_algorithms_experiments() {
  $CMD waterworld hrl $NUM_RUNS $ROOT_EXPERIMENTS_PATH $RL_ALGORITHMS_WATERWORLD_FOLDER/hrl-guided -m $DEFAULT_WATERWORLD_NUM_STEPS -t $DEFAULT_WATERWORLD_DATASET_SIZE --seed $DEFAULT_SEED -iacen -s bfs-alternative --timed -g pseudorewards --environments $DEFAULT_WATERWORLD_ENVIRONMENTS
  $CMD waterworld hrl $NUM_RUNS $ROOT_EXPERIMENTS_PATH $RL_ALGORITHMS_WATERWORLD_FOLDER/hrl-not-guided -m $DEFAULT_WATERWORLD_NUM_STEPS -t $DEFAULT_WATERWORLD_DATASET_SIZE --seed $DEFAULT_SEED -iacen -s bfs-alternative --timed --environments $DEFAULT_WATERWORLD_ENVIRONMENTS
  $CMD waterworld qrm $NUM_RUNS $ROOT_EXPERIMENTS_PATH $RL_ALGORITHMS_WATERWORLD_FOLDER/qrm-not-guided -m $DEFAULT_WATERWORLD_NUM_STEPS -t $DEFAULT_WATERWORLD_DATASET_SIZE --seed $DEFAULT_SEED -iacen -s bfs-alternative --timed --environments $DEFAULT_WATERWORLD_ENVIRONMENTS
  $CMD waterworld qrm $NUM_RUNS $ROOT_EXPERIMENTS_PATH $RL_ALGORITHMS_WATERWORLD_FOLDER/qrm-min-distance -m $DEFAULT_WATERWORLD_NUM_STEPS -t $DEFAULT_WATERWORLD_DATASET_SIZE --seed $DEFAULT_SEED -iacen -s bfs-alternative --timed -g min_distance --environments $DEFAULT_WATERWORLD_ENVIRONMENTS
  $CMD waterworld qrm $NUM_RUNS $ROOT_EXPERIMENTS_PATH $RL_ALGORITHMS_WATERWORLD_FOLDER/qrm-max-distance -m $DEFAULT_WATERWORLD_NUM_STEPS -t $DEFAULT_WATERWORLD_DATASET_SIZE --seed $DEFAULT_SEED -iacen -s bfs-alternative --timed -g max_distance --environments $DEFAULT_WATERWORLD_ENVIRONMENTS
}

build_waterworld_rl_algorithms_handcrafted_experiments() {
  $CMD waterworld hrl $NUM_RUNS $ROOT_EXPERIMENTS_PATH $RL_ALGORITHMS_HANCRAFTED_WATERWORLD_FOLDER/hrl-guided -m $DEFAULT_WATERWORLD_NUM_STEPS -t $DEFAULT_WATERWORLD_DATASET_SIZE --seed $DEFAULT_SEED -acen -g pseudorewards --environments $DEFAULT_WATERWORLD_ENVIRONMENTS
  $CMD waterworld hrl $NUM_RUNS $ROOT_EXPERIMENTS_PATH $RL_ALGORITHMS_HANCRAFTED_WATERWORLD_FOLDER/hrl-not-guided -m $DEFAULT_WATERWORLD_NUM_STEPS -t $DEFAULT_WATERWORLD_DATASET_SIZE --seed $DEFAULT_SEED -acen --environments $DEFAULT_WATERWORLD_ENVIRONMENTS
  $CMD waterworld qrm $NUM_RUNS $ROOT_EXPERIMENTS_PATH $RL_ALGORITHMS_HANCRAFTED_WATERWORLD_FOLDER/qrm-not-guided -m $DEFAULT_WATERWORLD_NUM_STEPS -t $DEFAULT_WATERWORLD_DATASET_SIZE --seed $DEFAULT_SEED -acen --environments $DEFAULT_WATERWORLD_ENVIRONMENTS
  $CMD waterworld qrm $NUM_RUNS $ROOT_EXPERIMENTS_PATH $RL_ALGORITHMS_HANCRAFTED_WATERWORLD_FOLDER/qrm-min-distance -m $DEFAULT_WATERWORLD_NUM_STEPS -t $DEFAULT_WATERWORLD_DATASET_SIZE --seed $DEFAULT_SEED -acen -g min_distance --environments $DEFAULT_WATERWORLD_ENVIRONMENTS
  $CMD waterworld qrm $NUM_RUNS $ROOT_EXPERIMENTS_PATH $RL_ALGORITHMS_HANCRAFTED_WATERWORLD_FOLDER/qrm-max-distance -m $DEFAULT_WATERWORLD_NUM_STEPS -t $DEFAULT_WATERWORLD_DATASET_SIZE --seed $DEFAULT_SEED -acen -g max_distance --environments $DEFAULT_WATERWORLD_ENVIRONMENTS
}

# run the functions for building the experiments
build_dataset_and_steps_experiments
build_restricted_observables_experiments
build_uncompressed_traces_experiments
build_cyclic_automata_experiments
build_num_disjunctions_experiments
build_symmetry_breaking_experiments
build_interleaving_rl_algorithms_experiments
build_rl_algorithms_handcrafted_automata_experiments

# craftworld experiments
build_craftworld_interleaving_rl_algorithms_experiments
build_craftworld_rl_algorithms_handcrafted_experiments

# waterworld experiments
build_waterworld_interleaving_rl_algorithms_experiments
build_waterworld_rl_algorithms_handcrafted_experiments
