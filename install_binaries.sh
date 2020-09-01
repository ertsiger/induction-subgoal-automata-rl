#!/usr/bin/env bash

install_script_path="$( cd "$(dirname "$0")" ; pwd -P )"
target_install_path="${install_script_path}/src/bin"

function install_binary {
  binary_url=$1
  binary_tar_file=$2
  binary_folder=$3
  binary_name=$4
  strip_components=$5

  if [[ ! -f "${binary_tar_file}" ]]; then
    echo "Downloading ${binary_name}..."
    wget -nv -O ${binary_tar_file} ${binary_url}
  fi

  if [[ ! -d "${binary_folder}" ]]; then
    echo "Extracting ${binary_tar_file}..."
    mkdir -p ${binary_folder}
    tar -xzf ${binary_tar_file} --strip-components=${strip_components} -C ${binary_folder}
  fi

  echo "Moving binary to src/bin..."
  mkdir -p "${target_install_path}"
  mv "${binary_folder}/${binary_name}" "${target_install_path}"

  echo "Removing downloaded and extracted files..."
  rm -rf ${binary_tar_file} ${binary_folder}

  echo "Success installing ${binary_name}!"
}

ilasp_url=""
ilasp_tar_file="ILASP.tar.gz"
ilasp_folder="ILASP"
ilasp_binary="ILASP"

clingo_url=""
clingo_tar_file="clingo.tar.gz"
clingo_folder="clingo"
clingo_binary="clingo"

if [[ "$OSTYPE" == "linux-gnu" ]]; then
  ilasp_url="https://github.com/marklaw/ILASP-releases/releases/download/v3.6.0/ILASP-3.6.0-ubuntu.tar.gz"
  clingo_url="https://github.com/potassco/clingo/releases/download/v5.4.0/clingo-5.4.0-linux-x86_64.tar.gz"
elif [[ "$OSTYPE" == "darwin"* ]]; then
  ilasp_url="https://github.com/marklaw/ILASP-releases/releases/download/v3.6.0/ILASP-3.6.0-OSX.tar.gz"
  clingo_url="https://github.com/potassco/clingo/releases/download/v5.4.0/clingo-5.4.0-macos-x86_64.tar.gz"
else
  echo "Error: Only Linux and MacOS installations are supported."
  exit
fi

if [[ ! -f "${target_install_path}/${ilasp_binary}" ]]; then
  install_binary ${ilasp_url} ${ilasp_tar_file} ${ilasp_folder} ${ilasp_binary} 0
fi

if [[ ! -f "${target_install_path}/${clingo_binary}" ]]; then
  install_binary ${clingo_url} ${clingo_tar_file} ${clingo_folder} ${clingo_binary} 1
fi

