# Environment for nuearth on this machine (AlmaLinux 9, CERN):
# CVMFS LCG_110 view gives python 3.13 + jax 0.10 + node 22; MANGO from the
# user's clone (not pip-installed).   source env.sh
source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc13-opt/setup.sh
export MANGO_HOME=${MANGO_HOME:-$HOME/projects/mango-osc}
export PYTHONPATH=$MANGO_HOME:$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")" && pwd)/server:$PYTHONPATH
export JAX_PLATFORMS=cpu
