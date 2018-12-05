# Satellite Object Detection Case Study
_Insert helpful info here!_


### Directory Structure
* `data` - ignored directory contains images and other data used for model training, evaluation, and testing


## Conda Environments
Separate Conda environments were used to manage packages and separate Python runtimes. Python 3.5 is used where possible, while Python 2.7 is used primarily for Apache Beam SDK compatiblity.

Environment Name | Runtime | Conda Export File
--- | --- | ---
gcp | Python 3.5 | <env-file.yml>
gcp-dataflow | Python 2.7 | <env-file.yml>
gcp-py37 | Python 3.7 | <env-file.yml>
