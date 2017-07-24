# Malware Env for OpenAI Gym
**************************

**This is a malware manipulation environment for OpenAI's ``gym``.** 
[OpenAI Gym](https://gym.openai.com/) is a toolkit for developing and comparing reinforcement 
learning algorithms. This makes it possible to write agents that learn 
to manipulate PE files (e.g., malware) to achieve some objective 
(e.g., bypass AV) based on a reward provided by taking specific manipulation
actions.

Objective
======
Create an AI that learns through reinforcement learning which functionality-preserving transformations to make on a malware sample to break through / bypass machine learning static-analysis malware detection.

![Breakout](https://github.com/matthiasplappert/keras-rl/raw/master/assets/breakout.gif?raw=true
"Breakout")

Basics
======

There are two basic concepts in reinforcement learning: the environment (in our case, the malware sample) and the agent (namely, the algorithm used to change the environment). The agent sends `actions` to the environment, and the environment replies with `observations` and `rewards` (that is, a score).

This repo provides an environment for manipulating PE files and providing rewards that are based around bypassing AV.  An agent can be deployed that have already been written for the rich ``gym`` framework.  For example

* https://github.com/pfnet/chainerrl [recommended]
* https://github.com/matthiasplappert/keras-rl
 
Setup
=====
The EvadeRL framework is built on Python3.6 we recommend first creating a virtualenv (details can be found [here]) with Python3.6 then performing the following actions ensure you have the correct python libraries:

[here]: https://docs.python.org/3/tutorial/venv.html
```sh
pip install -r requirements.txt
```

EvadeRL also leverages a Library to Instrument Executable Formats aptly named [LIEF]. It allows our agent to modify the binary on-the-fly. To add it to your virtualenv just ```pip install``` one of their pre-built packages. Examples below:

[LIEF]: https://github.com/lief-project/LIEF

Linux
```
pip install https://github.com/lief-project/LIEF/releases/download/0.7.0/linux_lief-0.7.0_py3.6.tar.gz
```

OSX
```
pip install https://github.com/lief-project/LIEF/releases/download/0.7.0/osx_lief-0.7.0_py3.6.tar.gz
```

Once completed ensure you've moved malware samples into the 
```
gym_malware/gym_malware/envs/utils/samples/
```

If you are unsure where to acquire malware samples see the **Data Acquisition** section below. If you have samples in the correct directory you can check to see if your environment is correctly setup by running :

```
python test_agent_chainer.py
```

Note that if you are using Anaconda, you may need to
```
conda install libgcc
```
in order for LIEF to operate properly.

Data Acquisition
=====
todo

Gym-Malware Environment
====
EvadeRL pits a reinforcement agent against the malware environment consisting of the following components:

* Action Space
* Independent Malware Classifier
* OpenAI framework malware environment (aka gym-malware)
 
Action Space
----
The moves or actions that can be performed on a malware sample in our environment consist of the following binary manipulations:
* append_zero
* append_random_ascii
* append_random_bytes
* remove_signature
* upx_pack
* upx_unpack
* change_section_names_from_list
* change_section_names_to random
* modify_export
* remove_debug
* break_optional_header_checksum

The agent will randomly select these actions in an attempt to bypass our independent classifier (more information to follow). Over time, the agent learns which combinations lead to the highest rewards, or learns a policy (*like an optimal plan of attack for any given observation*).

Independent Classifier
----
We trained a [gradient boosted trees model] to act as our independent classifier that the reinforcement agent must evade. The classifier is trained on **<fill in>** data with the following features extracted:
* Byte-level data (e.g. histogram and entropy)
* Header
* Section
* Import/Exports


[gradient boosted trees model]: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html


